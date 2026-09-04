#!/usr/bin/env python3
"""Record isolated remote environment readiness without importing model weights."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("WORLDCLAW_MODEL_ROOT", "/mnt/data/v-huguangyu/worldclaw-oss"))
ENVS = {
    "orchestrator": ["numpy", "pydantic", "trimesh", "open3d", "httpx"],
    "image-seg": ["torch", "transformers", "diffusers"],
    "llm-vlm": ["vllm", "torch", "transformers"],
    "hunyuan3d": ["torch", "trimesh", "open3d"],
    "trellis": ["torch", "transformers", "diffusers"],
}


def run_python(python: Path, modules: list[str]) -> dict:
    code = f"""
import importlib.util
import json
import sys

mods = {modules!r}
out = {{"python": sys.version, "modules": {{}}}}
for module in mods:
    out["modules"][module] = bool(importlib.util.find_spec(module))
try:
    import torch
    out["torch"] = {{
        "version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "gpus": torch.cuda.device_count(),
    }}
except Exception as error:
    out["torch_error"] = f"{{type(error).__name__}}: {{error}}"
print(json.dumps(out))
"""
    try:
        completed = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True, timeout=90, check=False
        )
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    result = {"status": "ok" if completed.returncode == 0 else "error", "returncode": completed.returncode}
    if completed.stdout.strip():
        try:
            result.update(json.loads(completed.stdout.strip().splitlines()[-1]))
        except json.JSONDecodeError:
            result["stdout"] = completed.stdout[-4000:]
    if completed.stderr.strip():
        result["stderr"] = completed.stderr[-4000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "environment-audit.json")
    args = parser.parse_args()
    report = {
        "schema": "worldclaw-oss-environment-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "paths": {name: str(ROOT / "envs" / name) for name in ENVS},
        "environments": {},
        "locks": {},
        "processes": {},
    }
    for name, modules in ENVS.items():
        prefix = ROOT / "envs" / name
        python = prefix / "bin" / "python"
        report["environments"][name] = {
            "prefix_exists": prefix.is_dir(),
            "python_exists": python.is_file(),
            "audit": run_python(python, modules) if python.is_file() else {"status": "missing"},
        }
        lock = ROOT / "environment-locks" / f"{name}.pip-freeze.txt"
        report["locks"][name] = {"path": str(lock), "exists": lock.is_file(), "bytes": lock.stat().st_size if lock.is_file() else 0}
    for pattern in ("bootstrap_remote_envs", "pip install vllm", "conda.*create"):
        result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True, check=False)
        report["processes"][pattern] = result.stdout.splitlines()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
