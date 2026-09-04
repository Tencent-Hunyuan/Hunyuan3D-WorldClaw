#!/usr/bin/env python3
"""Diagnose the configured OpenAI-compatible gateway without exposing secrets.

The probe is intentionally independent of the WorldClaw pipeline. It checks
DNS resolution, the model-list endpoint, and minimal Responses/Chat requests,
then writes only non-secret metadata and bounded error details to a report.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from worldclaw_oss.openai_api import OpenAIConfig


def _error_payload(error: BaseException, api_key: str) -> dict[str, object]:
    message = str(error).replace(api_key, "[redacted]") if api_key else str(error)
    status_code = getattr(error, "status_code", None)
    return {
        "type": type(error).__name__,
        "status_code": status_code,
        "message": message[:500],
    }


def _timed_call(name: str, callback, api_key: str) -> dict[str, object]:
    started = time.monotonic()
    try:
        value = callback()
    except Exception as error:  # The report must survive an unavailable check.
        return {
            "name": name,
            "status": "error",
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": _error_payload(error, api_key),
        }
    return {
        "name": name,
        "status": "ok",
        "duration_seconds": round(time.monotonic() - started, 3),
        "result": value,
    }


def _dns_check(base_url: str) -> dict[str, object]:
    parsed = urlparse(base_url)
    if not parsed.hostname:
        raise ValueError("base_url has no hostname")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    unique = sorted({item[4][0] for item in addresses})
    return {"hostname": parsed.hostname, "addresses": unique}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=0.0)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 5:
        parser.error("--attempts must be between 1 and 5")
    if args.timeout <= 0 or args.interval < 0:
        parser.error("--timeout must be positive and --interval non-negative")

    config = OpenAIConfig.from_environment()
    from openai import OpenAI

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=args.timeout,
        max_retries=0,
        default_headers={"User-Agent": "WorldClaw/1.0 gateway-diagnostic"},
    )
    report: dict[str, object] = {
        "schema": "worldclaw-oss-openai-gateway-diagnostic-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": config.base_url,
        "model": args.model,
        "timeout_seconds": args.timeout,
        "attempts": args.attempts,
        "api_key_configured": bool(config.api_key),
        "checks": [],
    }
    checks = report["checks"]
    assert isinstance(checks, list)

    checks.append(_timed_call("dns", lambda: _dns_check(config.base_url), config.api_key))
    checks.append(
        _timed_call(
            "models.list",
            lambda: {"models": sorted(str(item.id) for item in client.models.list().data)},
            config.api_key,
        )
    )
    for attempt in range(1, args.attempts + 1):
        checks.append(
            _timed_call(
                f"chat.completions.create#{attempt}",
                lambda: _chat_probe(client, args.model),
                config.api_key,
            )
        )
        checks.append(
            _timed_call(
                f"responses.create#{attempt}",
                lambda: _responses_probe(client, args.model),
                config.api_key,
            )
        )
        if attempt != args.attempts and args.interval:
            time.sleep(args.interval)

    report["status"] = "ok" if all(item["status"] == "ok" for item in checks) else "error"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


def _chat_probe(client, model: str) -> dict[str, object]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Return exactly JSON: {\"ok\":true}."}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    return {
        "model": getattr(response, "model", None),
        "finish_reason": choice.finish_reason,
        "content_present": bool(getattr(choice.message, "content", None)),
    }


def _responses_probe(client, model: str) -> dict[str, object]:
    response = client.responses.create(
        model=model,
        input="Return exactly JSON: {\"ok\":true}.",
    )
    output_text = getattr(response, "output_text", "")
    return {
        "model": getattr(response, "model", None),
        "output_present": bool(output_text),
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Configuration/import failures are printed without traceback secrets.
        print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        raise
