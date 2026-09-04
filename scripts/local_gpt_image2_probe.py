#!/usr/bin/env python3
"""Make one minimal GPT Image 2 request using the current local Codex config."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from worldclaw_oss.openai_api import OpenAIClient, OpenAIConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    base = OpenAIConfig.from_environment()
    config = replace(base, image_model="gpt-image-2", image_via_responses=False)
    report = {"config": config.public_dict(), "request_model": "gpt-image-2", "status": "error"}
    try:
        result = OpenAIClient(config).generate_image(
            prompt="A single isolated tree on the supplied terrain, preserve the terrain layout, no text",
            output_path=args.output,
            conditioning_image=args.image,
        )
        report.update({"status": "ok", "result": result, "output_bytes": args.output.stat().st_size})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
