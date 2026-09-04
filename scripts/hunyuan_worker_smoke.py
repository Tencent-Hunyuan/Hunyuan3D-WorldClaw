#!/usr/bin/env python3
"""Run a real Hunyuan worker contract smoke on an existing PNG fixture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from worldclaw_oss.models import CommandWorker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    (work / "environment_references_response.json").write_text(
        json.dumps({
            "status": "ok",
            "images": [{
                "id": "hunyuan_fixture",
                "path": str(args.image.resolve()),
                "category": "fixture_asset",
                "region_id": "smoke",
            }],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    request = {
        "work_dir": str(work), "run_dir": str(root), "prompt": "worker smoke",
        "seed": 42, "stage": "environment_assets", "models": lock["models"],
    }
    request_path = work / "hunyuan_request.json"
    response_path = work / "hunyuan_response.json"
    worker = CommandWorker("WORLDCLAW_IMAGE3D_WORKER")
    try:
        response = worker.run(request, request_path, response_path)
    except RuntimeError as error:
        response = json.loads(response_path.read_text(encoding="utf-8")) if response_path.is_file() else {
            "status": "error", "error": str(error),
        }
    report = {
        "status": response.get("status", "error"),
        "response": response,
        "model": lock["models"].get("hunyuan3d"),
        "hf_home": os.getenv("HF_HOME"),
        "token_configured": bool(os.getenv("HF_TOKEN")),
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
