#!/usr/bin/env python3
"""Run and validate only the Intent stage for one prompt file.

This deliberately does not call ``Planner.plan`` because that method continues
into Scene Planner. The report is an auditable feasibility check for the
IntentPayload contract only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from worldclaw_oss.models import FORCED_PLANNER_MODEL, IntentPayload, ModelLock, OpenAIJSONClient
from worldclaw_oss.prompts import INTENT_SYSTEM
from worldclaw_oss.schemas import ModelRecord


def validate_intent(value: IntentPayload, prompt: str) -> list[str]:
    """Return feasibility findings without inventing scene-plan constraints."""
    findings: list[str] = []
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        findings.append("prompt_empty")
    if value.verbatim_prompt.strip() != normalized_prompt:
        findings.append("verbatim_prompt_mismatch")
    if not value.constraints:
        findings.append("constraints_empty")
    cleaned = [item.strip() for item in value.constraints]
    if any(not item for item in cleaned):
        findings.append("constraint_empty")
    if len(set(cleaned)) != len(cleaned):
        findings.append("constraint_duplicate")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only WorldClaw Intent extraction")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("models.lock.json"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    record = ModelRecord(
        model_id=FORCED_PLANNER_MODEL,
        revision="api",
        license="provider-api",
        size_bytes=0,
        purpose="intent extraction only",
    )
    # Validate the lock entry without loading any downstream model.
    ModelLock(args.lock).require_resolved(["planner"])
    started = time.monotonic()
    result: dict[str, object] = {
        "schema": "worldclaw-oss-intent-smoke-v1",
        "stage": "intent",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_file": str(args.prompt_file),
        "prompt_sha256": prompt_sha256,
        "model_id": record.model_id,
        "revision": record.revision,
        "seed": args.seed,
        "environment": os.getenv("WORLDCLAW_ENVIRONMENT", "llm-vlm"),
        "downstream_stages_run": [],
    }
    try:
        raw = OpenAIJSONClient().json_chat(
            record,
            INTENT_SYSTEM,
            prompt,
            IntentPayload.model_json_schema(),
            args.seed,
        )
        intent = IntentPayload.model_validate(raw)
        findings = validate_intent(intent, prompt)
        result.update({
            "status": "ok" if not findings else "infeasible",
            "duration_seconds": round(time.monotonic() - started, 3),
            "feasibility_findings": findings,
            "intent": intent.model_dump(),
        })
    except Exception as error:
        result.update({
            "status": "error",
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
