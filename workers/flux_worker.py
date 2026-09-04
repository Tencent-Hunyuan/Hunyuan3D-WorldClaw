#!/usr/bin/env python3
"""FLUX worker. Run inside the isolated image environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import FluxImg2ImgPipeline, FluxPipeline


def _pretrained_path(record: dict) -> str:
    """Prefer the pinned NFS snapshot so workers never depend on process cwd/cache defaults."""
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot = hf_home / "hub" / f"models--{record['model_id'].replace('/', '--')}" / "snapshots" / record["revision"]
    return str(snapshot) if snapshot.is_dir() else record["model_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    models = request["models"]
    image_requests = request.get("image_requests")
    if not image_requests:
        image_requests = [{"prompt": prompt, "id": f"image_{index:03d}"}
                          for index, prompt in enumerate(request.get("image_prompts") or [request["prompt"]])]
    conditioned = any(item.get("conditioning_image") for item in image_requests)
    pipeline_class = FluxImg2ImgPipeline if conditioned else FluxPipeline
    selected = models["flux_dev"]
    fallback = None
    try:
        pipeline = pipeline_class.from_pretrained(
            _pretrained_path(selected), revision=selected["revision"], torch_dtype=torch.bfloat16
        )
    except (OSError, PermissionError) as exc:
        text = str(exc).lower()
        if not any(word in text for word in ("gated", "401", "403", "authorized", "access")):
            raise
        selected = models["flux_schnell"]
        fallback = {"from": "flux_dev", "to": "flux_schnell", "reason": str(exc)}
        pipeline = pipeline_class.from_pretrained(
            _pretrained_path(selected), revision=selected["revision"], torch_dtype=torch.bfloat16
        )
    # Model offload is faster, but its first transformer transfer can briefly
    # require the whole block on the selected device.  Sequential offload is
    # useful on shared GPUs where that peak would exceed the available memory.
    offload_mode = os.getenv("WORLDCLAW_FLUX_OFFLOAD", "model").strip().lower()
    if offload_mode in {"sequential", "layer", "layers"}:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()
    output_dir = Path(request["work_dir"]) / "flux"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    is_dev = selected["model_id"].endswith("FLUX.1-dev")
    img2img = pipeline if conditioned else None
    for index, item in enumerate(image_requests):
        prompt = item["prompt"]
        generator = torch.Generator("cpu").manual_seed(int(request["seed"]) + index)
        if item.get("conditioning_image"):
            if img2img is None:
                img2img = FluxImg2ImgPipeline.from_pipe(pipeline)
            from PIL import Image
            condition = Image.open(item["conditioning_image"]).convert("RGB")
            image = img2img(
                prompt=prompt, image=condition, strength=float(item.get("strength", 0.62)),
                generator=generator, num_inference_steps=28 if is_dev else 4,
                guidance_scale=3.5 if is_dev else 0.0,
            ).images[0]
        else:
            image = pipeline(
                prompt, generator=generator, num_inference_steps=28 if is_dev else 4,
                guidance_scale=3.5 if is_dev else 0.0,
            ).images[0]
        path = output_dir / f"{request['stage']}_{index:03d}.png"
        image.save(path)
        images.append({key: value for key, value in item.items() if key not in ("prompt", "conditioning_image")} | {
            "path": str(path), "prompt": prompt,
        })
    args.response.write_text(json.dumps({
        "status": "ok", "images": images, "model": selected, "fallback": fallback,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
