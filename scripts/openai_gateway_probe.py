#!/usr/bin/env python3
"""Print non-secret model metadata from the configured OpenAI gateway."""
from __future__ import annotations

from worldclaw_oss.openai_api import OpenAIConfig


def main() -> int:
    from openai import OpenAI

    config = OpenAIConfig.from_environment()
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=20.0)
    try:
        models = client.models.list()
        ids = sorted(str(item.id) for item in models.data)
        print({"base_url": config.base_url, "api_key_configured": True, "models": ids})
    except Exception as error:
        print({"base_url": config.base_url, "api_key_configured": True, "error": f"{type(error).__name__}: {error}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
