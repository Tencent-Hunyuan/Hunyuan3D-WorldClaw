#!/usr/bin/env python3
"""Download every resolved model lock entry into HF_HOME and record actual sizes."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("models.lock.json"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    results = {}
    for name, record in lock["models"].items():
        if record.get("license") == "provider-api" or record.get("revision") == "api":
            results[name] = {"status": "skipped", "reason": "provider API model; no Hugging Face download"}
            continue
        try:
            path = Path(snapshot_download(
                repo_id=record["model_id"],
                revision=record["revision"],
                token=os.getenv("HF_TOKEN") or None,
            ))
            files = [item for item in path.rglob("*") if item.is_file()]
            results[name] = {
                "status": "ok", "path": str(path),
                "files": len(files), "materialized_bytes": sum(item.stat().st_size for item in files),
            }
        except Exception as error:
            results[name] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
            if name not in ("flux_dev", "hunyuan3d"):
                raise
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hf_home": os.getenv("HF_HOME"),
        "token_configured": bool(os.getenv("HF_TOKEN")),
        "models": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
