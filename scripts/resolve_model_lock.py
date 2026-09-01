#!/usr/bin/env python3
"""Resolve exact Hugging Face commits and repository sizes into models.lock.json."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("models.lock.json"))
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument(
        "--proxy",
        default=os.getenv("WORLDCLAW_HF_PROXY", ""),
        help="Explicit HTTP proxy, for example http://127.0.0.1:7890",
    )
    args = parser.parse_args()
    raw = json.loads(args.lock.read_text(encoding="utf-8"))
    token = os.getenv(args.token_env, "")
    headers = {"User-Agent": "worldclaw-oss-lock/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    handlers = []
    if args.proxy:
        handlers.append(urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy}))
    opener = urllib.request.build_opener(*handlers)
    def fetch_json(request, attempts=5):
        last_error = None
        for attempt in range(attempts):
            try:
                with opener.open(request, timeout=60) as response:
                    return json.load(response)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last_error = error
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"metadata request failed after {attempts} attempts: {last_error}")

    for name, record in raw["models"].items():
        requested_id = record.get("requested_model_id") or record["model_id"]
        request = urllib.request.Request(
            "https://huggingface.co/api/models/" + record["model_id"],
            headers={**headers, "Connection": "close"},
        )
        metadata = fetch_json(request)
        revision = metadata["sha"]
        if len(revision) != 40:
            raise RuntimeError(f"{name}: API did not return a commit SHA")
        record["revision"] = revision
        record["size_bytes"] = int(metadata.get("usedStorage") or 0)
        license_name = (metadata.get("cardData") or {}).get("license")
        if license_name and license_name != "other":
            record["license"] = license_name if isinstance(license_name, str) else ",".join(license_name)
        record["resolved"] = True
        if requested_id != record["model_id"]:
            record["requested_model_id"] = requested_id
    raw["generated_at"] = datetime.now(timezone.utc).isoformat()
    raw["resolution_note"] = (
        "Resolved from Hugging Face API through an explicit proxy when configured; "
        "HF_TOKEN value was not persisted."
    )
    args.lock.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(item["size_bytes"] for item in raw["models"].values())
    print(json.dumps({"models": len(raw["models"]), "total_size_bytes": total}, indent=2))


if __name__ == "__main__":
    main()
