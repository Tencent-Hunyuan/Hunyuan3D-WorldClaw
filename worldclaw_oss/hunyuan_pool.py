"""Persistent Hunyuan3D daemon pool used by the live pipeline.

The regular worker contract remains file based.  This module only changes the
process lifetime and dispatches independent source images to one daemon per
GPU, so each daemon loads Hunyuan3D once and serves subsequent stages.
"""
from __future__ import annotations

import atexit
import copy
import json
import os
import shlex
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _split_gpu_ids(value: str | None) -> list[str]:
    return [item for item in (value or "").replace(";", ",").split(",") if item.strip()]


def _command_parts(command: str) -> tuple[list[str], dict[str, str]]:
    """Extract executable args and inline env assignments from a worker command."""
    tokens = shlex.split(command, posix=os.name != "nt")
    env: dict[str, str] = {}
    if tokens and tokens[0] == "env":
        tokens = tokens[1:]
    while tokens and "=" in tokens[0] and not tokens[0].startswith(("/", "\\")):
        key, value = tokens.pop(0).split("=", 1)
        if key.isidentifier():
            env[key] = value
        else:
            tokens.insert(0, key + "=" + value)
            break
    return tokens, env


def _worker_script_exists(command: str) -> bool:
    try:
        args, _ = _command_parts(command)
    except ValueError:
        return False
    return any(Path(item).is_file() and Path(item).suffix == ".py" for item in args)


def _discover_free_gpus(base: list[str]) -> list[str]:
    explicit = _split_gpu_ids(os.getenv("WORLDCLAW_HUNYUAN_EXTRA_GPUS"))
    if explicit:
        return list(dict.fromkeys(base + explicit))
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return list(dict.fromkeys(base))
    if completed.returncode != 0:
        return list(dict.fromkeys(base))
    excluded = set(_split_gpu_ids(os.getenv("WORLDCLAW_HUNYUAN_GPU_EXCLUDE")))
    try:
        min_free = int(os.getenv("WORLDCLAW_HUNYUAN_MIN_FREE_MB", "12000"))
    except ValueError:
        min_free = 12000
    result = list(base)
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            continue
        gpu, used, total = fields
        if gpu in excluded or gpu in result:
            continue
        try:
            if int(total) - int(used) >= min_free:
                result.append(gpu)
        except ValueError:
            continue
    return result


def _source_ids(request: dict[str, Any]) -> list[str]:
    """Read stage input ids without importing the Hunyuan environment."""
    stage = request.get("stage")
    work_dir = Path(request["work_dir"])
    if stage == "environment_assets":
        override = request.get("reference_override")
        if override:
            return [
                str(item.get("id")) for item in override
                if isinstance(item, dict) and item.get("id") is not None
            ]
        path = work_dir / "environment_references_response.json"
        key = "images"
    elif stage == "reconstruction":
        path = work_dir / "segmentation_response.json"
        key = "instances"
    elif stage == "refinement_reconstruction":
        raw = request.get("refinement_assets", [])
        return [str(item.get("id")) for item in raw if item.get("id") is not None]
    else:
        return []
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [str(item.get("id")) for item in value.get(key, []) if isinstance(item, dict) and item.get("id") is not None]


class HunyuanDaemonPool:
    """Own persistent Hunyuan processes and dispatch requests across GPUs."""

    _instances: dict[tuple[str, tuple[str, ...], str], "HunyuanDaemonPool"] = {}
    _lock = threading.Lock()
    _registered = False

    def __init__(self, command: str, gpu_ids: list[str], socket_dir: Path):
        self.command = command
        self.gpu_ids = tuple(gpu_ids)
        self.socket_dir = socket_dir
        self.processes: dict[str, subprocess.Popen[str] | None] = {}
        self._started = False

    @classmethod
    def from_environment(cls) -> "HunyuanDaemonPool | None":
        if os.name == "nt":
            # The daemon protocol uses Unix domain sockets; Windows keeps the
            # existing one-shot worker contract.
            return None
        command = os.getenv("WORLDCLAW_IMAGE3D_WORKER", "").strip()
        mode = os.getenv("WORLDCLAW_HUNYUAN_DAEMON_POOL", "auto").lower()
        if mode in {"0", "false", "off", "no"} or not command or not _worker_script_exists(command):
            return None
        try:
            _args, command_env = _command_parts(command)
        except ValueError:
            return None
        configured = _split_gpu_ids(os.getenv("WORLDCLAW_HUNYUAN_GPUS"))
        if not configured:
            configured = _split_gpu_ids(command_env.get("CUDA_VISIBLE_DEVICES"))
        if not configured:
            configured = _split_gpu_ids(os.getenv("CUDA_VISIBLE_DEVICES"))
        # An explicit GPU list is a resource-isolation contract.  Automatic
        # expansion is opt-in so a FLUX/segmentation worker cannot be placed
        # on a GPU already reserved for Hunyuan3D.
        auto_expand = os.getenv("WORLDCLAW_HUNYUAN_AUTO_EXPAND", "0").lower() in {
            "1", "true", "yes", "on",
        }
        gpu_ids = _discover_free_gpus(configured) if (not configured or auto_expand) else list(dict.fromkeys(configured))
        if not gpu_ids:
            return None
        socket_dir = Path(os.getenv("WORLDCLAW_HUNYUAN_SOCKET_DIR", "/tmp/worldclaw-hunyuan"))
        key = (command, tuple(gpu_ids), str(socket_dir))
        with cls._lock:
            pool = cls._instances.get(key)
            if pool is None:
                pool = cls(command, gpu_ids, socket_dir)
                cls._instances[key] = pool
            if not cls._registered:
                atexit.register(cls.close_all)
                cls._registered = True
        pool.ensure_started()
        return pool

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            pools = list(cls._instances.values())
        for pool in pools:
            pool.close()

    def ensure_started(self) -> None:
        if self._started:
            return
        self.socket_dir.mkdir(parents=True, exist_ok=True)
        args, command_env = _command_parts(self.command)
        if not args:
            raise RuntimeError("WORLDCLAW_IMAGE3D_WORKER command is empty")
        for gpu in self.gpu_ids:
            socket_path = self.socket_dir / f"hunyuan3d-gpu{gpu}.sock"
            if socket_path.exists():
                # A pool created by a later stage/process reuses the already
                # mounted daemon instead of unlinking its live socket and
                # loading a second multi-GB copy on the same GPU.
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                        probe.settimeout(0.5)
                        probe.connect(str(socket_path))
                        probe.sendall(b'{"ping":true}\n')
                        probe.recv(128)
                    self.processes[gpu] = None  # type: ignore[assignment]
                    continue
                except (OSError, socket.timeout):
                    try:
                        socket_path.unlink()
                    except OSError:
                        pass
            env = os.environ.copy()
            env.update(command_env)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["HUNYUAN_WORKER_SOCKET"] = str(socket_path)
            process = subprocess.Popen(
                args + ["--server", "--socket", str(socket_path)],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True,
            )
            self.processes[gpu] = process
        deadline = time.monotonic() + float(os.getenv("WORLDCLAW_HUNYUAN_DAEMON_START_TIMEOUT", "120"))
        for gpu in self.gpu_ids:
            socket_path = self.socket_dir / f"hunyuan3d-gpu{gpu}.sock"
            while not socket_path.exists():
                process = self.processes[gpu]
                if process is None:
                    raise RuntimeError(f"shared Hunyuan daemon socket disappeared on GPU {gpu}")
                if process.poll() is not None:
                    raise RuntimeError(f"Hunyuan daemon exited during startup on GPU {gpu}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Hunyuan daemon socket did not appear on GPU {gpu}")
                time.sleep(0.2)
        self._started = True

    @staticmethod
    def _request(socket_path: Path, request: dict[str, Any]) -> dict[str, Any]:
        response_path = Path(request["response_path"])
        payload = {"request": request["request"], "response": str(response_path)}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(7200)
            connection.connect(str(socket_path))
            connection.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                data += chunk
        result = json.loads(data.decode("utf-8")) if data else {"status": "error", "error": "daemon closed connection"}
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Hunyuan daemon request failed"))
        return json.loads(response_path.read_text(encoding="utf-8"))

    def run(self, request: dict[str, Any], request_path: Path, response_path: Path) -> dict[str, Any]:
        self.ensure_started()
        ids = _source_ids(request)
        if len(self.gpu_ids) == 1 or len(ids) <= 1:
            shards = [ids] if ids else [None]
        else:
            shards = [ids[index::len(self.gpu_ids)] for index in range(len(self.gpu_ids))]
            shards = [item for item in shards if item]
        shard_results: list[dict[str, Any]] = []

        def submit(index_and_ids: tuple[int, list[str] | None]) -> dict[str, Any]:
            index, selected_ids = index_and_ids
            payload = copy.deepcopy(request)
            if selected_ids is not None:
                payload["source_ids"] = selected_ids
            shard_request = request_path.with_name(f"{request_path.stem}.daemon{index}.json")
            shard_response = response_path.with_name(f"{response_path.stem}.daemon{index}.json")
            shard_request.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return self._request(
                self.socket_dir / f"hunyuan3d-gpu{self.gpu_ids[index % len(self.gpu_ids)]}.sock",
                {"request": payload, "response_path": shard_response},
            )

        with ThreadPoolExecutor(max_workers=len(shards)) as executor:
            shard_results = list(executor.map(submit, enumerate(shards)))
        merged: dict[str, Any] = {"status": "ok", "assets": []}
        for value in shard_results:
            if value.get("status") != "ok":
                raise RuntimeError(value.get("error", "Hunyuan daemon shard failed"))
            merged["assets"].extend(value.get("assets", []))
            for key in ("model", "fallback"):
                if key in value:
                    merged[key] = value[key]
        if ids:
            order = {asset_id: index for index, asset_id in enumerate(ids)}
            merged["assets"].sort(key=lambda item: order.get(str(item.get("id")), len(order)))
        merged["worker"] = {
            "hunyuan_daemon_pool": True, "persistent": True,
            "gpu_ids": list(self.gpu_ids), "parallel_shards": len(shards),
        }
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        response_path.write_text(json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return merged

    def close(self) -> None:
        for process in self.processes.values():
            if process is not None and process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            if process is None:
                continue
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes.clear()
        self._started = False
