#!/usr/bin/env python3
"""Resumable download of one resolved model entry without exposing credentials."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("models.lock.json"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    record = lock["models"][args.name]
    result: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "model_id": record["model_id"],
        "revision": record["revision"],
        "hf_home": os.getenv("HF_HOME"),
        "token_configured": bool(os.getenv("HF_TOKEN")),
    }
    if record.get("license") == "provider-api" or record.get("revision") == "api":
        result.update({"status": "skipped", "reason": "provider API model; no Hugging Face download"})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    try:
        path = Path(snapshot_download(
            repo_id=record["model_id"],
            revision=record["revision"],
            token=os.getenv("HF_TOKEN") or None,
            resume_download=True,
        ))
        files = [item for item in path.rglob("*") if item.is_file()]
        result.update({
            "status": "ok",
            "path": str(path),
            "files": len(files),
            "materialized_bytes": sum(item.stat().st_size for item in files),
        })
    except Exception as error:
        result.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
