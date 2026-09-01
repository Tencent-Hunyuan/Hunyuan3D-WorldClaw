from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--socket", type=Path)
    return parser.parse_args()


def read_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_response(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_stage_response(work_dir: Path, name: str) -> dict[str, Any]:
    path = work_dir / f"{name}_response.json"
    if not path.is_file():
        raise FileNotFoundError(f"required stage response is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "ok":
        raise RuntimeError(f"upstream stage {name} is not successful")
    return value


def load_validated_stage_response(work_dir: Path, name: str) -> dict[str, Any]:
    """Load the asset-level quality-gated response when present."""
    validation_name = "mesh_validation_reconstruct" if name == "reconstruction" else f"mesh_validation_{name}"
    validated = work_dir / f"{validation_name}_response.json"
    if validated.is_file():
        value = json.loads(validated.read_text(encoding="utf-8"))
        if value.get("status") != "ok":
            raise RuntimeError(f"validated stage {name} is not successful")
        return value
    return load_stage_response(work_dir, name)


def artifact_key(source: Path, model_id: str, revision: str, parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(b"\0")
    digest.update(model_id.encode())
    digest.update(b"\0")
    digest.update(revision.encode())
    digest.update(b"\0")
    digest.update(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def image_entries(response: dict[str, Any]) -> list[dict[str, Any]]:
    raw = response.get("images", response.get("outputs", []))
    entries = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            entries.append({"path": item, "id": f"image_{index:03d}"})
        else:
            entries.append(dict(item))
    return entries


def resolve_model_source(record: dict[str, Any]) -> tuple[str, str | None]:
    """Use a complete pinned local snapshot when offline Hub lookup is unavailable."""
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        repo_cache = Path(hf_home) / "hub" / f"models--{record['model_id'].replace('/', '--')}"
        snapshot = repo_cache / "snapshots" / record["revision"]
        if snapshot.is_dir():
            return str(snapshot), None
    return record["model_id"], record["revision"]
