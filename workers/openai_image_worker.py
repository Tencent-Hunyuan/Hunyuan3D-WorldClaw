#!/usr/bin/env python3
"""OpenAI GPT Image 2 worker for environment asset reference images."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from worldclaw_oss.openai_api import OpenAIClient, OpenAIConfig


REFERENCE_IMAGE_MODEL = "gpt-image-2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    image_requests = request.get("image_requests")
    if not image_requests:
        image_requests = [{
            "id": f"image_{index:03d}",
            "prompt": prompt,
        } for index, prompt in enumerate(request.get("image_prompts") or [request["prompt"]])]

    # Reference images have a distinct quality/latency contract from regional
    # FLUX composition.  Keep this model explicit so OPENAI_IMAGE_MODEL or a
    # text-model setting cannot silently route the asset reference stage away
    # from GPT Image 2.
    config = replace(
        OpenAIConfig.from_environment(),
        image_model=REFERENCE_IMAGE_MODEL,
        image_via_responses=False,
    )
    client = OpenAIClient(config)
    output_dir = Path(request["work_dir"]) / "openai_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for index, item in enumerate(image_requests):
        output_path = output_dir / f"{request['stage']}_{index:03d}.png"
        condition = item.get("conditioning_image")
        result = client.generate_image(
            prompt=item["prompt"],
            output_path=output_path,
            conditioning_image=Path(condition) if condition else None,
        )
        images.append({
            key: value for key, value in item.items() if key != "conditioning_image"
        } | result)
    args.response.write_text(json.dumps({
        "status": "ok",
        "images": images,
        "model": REFERENCE_IMAGE_MODEL,
        "provider": "openai-compatible",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
