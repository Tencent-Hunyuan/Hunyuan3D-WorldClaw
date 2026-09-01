#!/usr/bin/env python3
"""Run one real planner request and record its contract metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from worldclaw_oss.models import FORCED_PLANNER_MODEL, ModelLock, OpenAIJSONClient, Planner
from worldclaw_oss.schemas import ModelRecord


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Planner smoke follows the production rule: Qwen/vLLM is not used.
    ModelLock(args.lock)  # retain lock-file validation for callers
    record = ModelRecord(
        model_id=FORCED_PLANNER_MODEL, revision="api", license="provider-api",
        size_bytes=0, purpose="intent extraction, scene planning, and asset routing",
    )
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    started = time.monotonic()
    result: dict[str, object] = {
        "schema": "worldclaw-oss-planner-smoke-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": record.model_id,
        "revision": record.revision,
        "environment": os.getenv("WORLDCLAW_ENVIRONMENT", "llm-vlm"),
        "gpu": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "seed": args.seed,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    try:
        intent, plan, repairs = Planner(
            OpenAIJSONClient(), record
        ).plan(prompt, args.seed)
        payload = {"intent": intent.model_dump(), "scene_plan": plan.model_dump(mode="json"), "repairs": repairs}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        result.update({
            "status": "ok",
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(encoded).hexdigest(),
            "repairs": repairs,
            "intent": intent.model_dump(),
            "scene_plan": plan.model_dump(mode="json"),
        })
    except Exception as error:
        result.update({"status": "error", "duration_seconds": round(time.monotonic() - started, 3), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
