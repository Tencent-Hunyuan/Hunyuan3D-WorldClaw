#!/usr/bin/env python3
"""Resume one HF file using independent HTTP Range requests.

This is for proxies that terminate long chunked transfers. Each request is
freshly resolved so signed CDN URLs can expire without invalidating the run.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import requests


def finalize_snapshot(hf_home: Path, repo: str, revision: str, filename: str,
                      output: Path, sha256: str) -> Path:
    """Promote a verified blob and create the revision snapshot symlink."""
    cache = hf_home / "hub" / f"models--{repo.replace('/', '--')}"
    blob = cache / "blobs" / sha256
    blob.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() != blob.resolve():
        if blob.exists():
            if blob.stat().st_size != output.stat().st_size:
                raise RuntimeError(f"refusing to replace existing blob: {blob}")
            output.unlink()
        else:
            output.replace(blob)
    snapshot_file = cache / "snapshots" / revision / filename
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_file.exists() or snapshot_file.is_symlink():
        if snapshot_file.is_symlink() and snapshot_file.resolve() == blob.resolve():
            return blob
        raise RuntimeError(f"snapshot path already exists and differs: {snapshot_file}")
    snapshot_file.symlink_to(Path(os.path.relpath(blob, snapshot_file.parent)))
    return blob


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--chunk-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument(
        "--finalize-snapshot", action="store_true",
        help="promote the verified blob into HF_HOME and create its revision symlink",
    )
    parser.add_argument("--hf-home", type=Path, default=Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser())
    args = parser.parse_args()
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    url = (
        f"{args.endpoint.rstrip('/')}/{args.repo}/resolve/{args.revision}/"
        f"{args.filename}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.proxies.update(proxies or {})
    while args.output.exists() and args.output.stat().st_size < args.size:
        start = args.output.stat().st_size
        end = min(args.size - 1, start + args.chunk_size - 1)
        expected = end - start + 1
        chunk = args.output.with_suffix(args.output.suffix + ".chunk")
        last_error = None
        for attempt in range(1, 8):
            try:
                with session.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    allow_redirects=True,
                    timeout=(30, 900),
                ) as response:
                    response.raise_for_status()
                    content_length = int(response.headers.get("Content-Length", "-1"))
                    if content_length != expected:
                        raise RuntimeError(
                            f"range {start}-{end} returned status={response.status_code} "
                            f"length={content_length}, expected={expected}"
                        )
                    with chunk.open("wb") as stream:
                        for block in response.iter_content(4 * 1024 * 1024):
                            if block:
                                stream.write(block)
                if chunk.stat().st_size != expected:
                    raise RuntimeError(f"chunk size mismatch: {chunk.stat().st_size} != {expected}")
                with args.output.open("ab") as stream, chunk.open("rb") as source:
                    while block := source.read(4 * 1024 * 1024):
                        stream.write(block)
                chunk.unlink()
                print(f"completed {end + 1}/{args.size}", flush=True)
                break
            except Exception as error:  # network retries are intentionally resumable
                last_error = error
                if chunk.exists():
                    chunk.unlink()
                print(f"range {start}-{end} attempt {attempt} failed: {error}", flush=True)
                time.sleep(min(30, attempt * 3))
        else:
            raise RuntimeError(f"range {start}-{end} exhausted retries: {last_error}")
    if not args.output.exists() or args.output.stat().st_size != args.size:
        raise RuntimeError("final size mismatch")
    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != args.sha256:
        raise RuntimeError(f"sha256 mismatch: {actual} != {args.sha256}")
    if args.finalize_snapshot:
        blob = finalize_snapshot(args.hf_home, args.repo, args.revision, args.filename, args.output, args.sha256)
        print(f"snapshot finalized: {blob}")
    print(f"range download ok: {args.output} {args.size} bytes {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
