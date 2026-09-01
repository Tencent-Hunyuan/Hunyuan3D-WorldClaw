#!/usr/bin/env python3
"""Wait for a resumable HF blob and register it in the revision snapshot."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from download_hf_range import finalize_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--hf-home", type=Path, default=Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser())
    args = parser.parse_args()
    while Path(f"/proc/{args.pid}").exists():
        time.sleep(20)
    if not args.output.is_file() or args.output.stat().st_size != args.size:
        raise RuntimeError(f"download did not finish: {args.output}")
    blob = finalize_snapshot(
        args.hf_home, args.repo, args.revision, args.filename,
        args.output, args.sha256,
    )
    print(f"snapshot finalized: {blob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
