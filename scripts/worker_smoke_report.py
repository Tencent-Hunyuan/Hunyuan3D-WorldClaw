#!/usr/bin/env python3
"""Summarize a non-model worker fixture without claiming model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    response_path = root / "work" / "export_response.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    names = (
        "scene.glb", "scene.blend", "scene.json", "preview.png", "walkthrough.mp4",
    )
    files = {}
    for name in names:
        path = root / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    report = {
        "schema": "worldclaw-oss-worker-smoke-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "non-model contract smoke",
        "status": response.get("status"),
        "export_response": response,
        "files": files,
        "official_implementation": False,
        "synthetic_outputs": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
