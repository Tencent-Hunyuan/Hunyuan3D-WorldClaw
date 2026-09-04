#!/usr/bin/env python3
"""GPT Vision object localization worker with local polygon rasterization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from workers.common import image_entries, load_stage_response, read_request, worker_args, write_response
from workers.segmentation_worker import deduplicate
from worldclaw_oss.openai_api import OpenAIClient


INSTANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "instances": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "bbox": {
                        "type": "array", "minItems": 4, "maxItems": 4,
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "polygon": {
                        "type": "array", "minItems": 3,
                        "items": {
                            "type": "array", "minItems": 2, "maxItems": 2,
                            "items": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                },
                "required": ["category", "confidence", "bbox", "polygon"],
            },
        },
    },
    "required": ["instances"],
}


def _mask_from_polygon(polygon: list[list[float]], width: int, height: int) -> np.ndarray:
    points = [
        (
            int(np.clip(point[0], 0.0, 1.0) * max(width - 1, 1)),
            int(np.clip(point[1], 0.0, 1.0) * max(height - 1, 1)),
        )
        for point in polygon
        if len(point) == 2
    ]
    mask = Image.new("L", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _categories(work_dir: Path) -> list[str]:
    plan = json.loads((work_dir / "scene_plan.json").read_text(encoding="utf-8"))
    return sorted({
        obj["category"]
        for region in plan["regions"]
        for obj in region.get("objects", [])
        if obj.get("count", 0) > 0
    })


def main() -> None:
    args = worker_args()
    request = read_request(args.request)
    work_dir = Path(request["work_dir"])
    composition = load_stage_response(work_dir, "region_composition")
    entries = image_entries(composition)
    if not entries:
        raise RuntimeError("region composition produced no images")
    categories = request.get("categories") or _categories(work_dir)
    if not categories:
        raise RuntimeError("scene plan contains no detectable object categories")

    client = OpenAIClient()
    output_root = work_dir / "segmentation"
    output_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for image_index, entry in enumerate(entries):
        source_path = Path(entry["path"])
        image = Image.open(source_path).convert("RGB")
        width, height = image.size
        user = (
            "Find only these requested object categories in the image: "
            + ", ".join(categories)
            + ". Return one instance per visible object. Coordinates must be "
            "normalized to [0,1], with x increasing rightward and y downward. "
            "Use a tight bbox and a polygon following the visible silhouette. "
            "Do not invent objects or return terrain, water, sky, or shadows."
        )
        result = client.vision_json(
            system="You are a precise object-localization annotator. Return JSON only.",
            user=user,
            image_paths=[source_path],
            schema=INSTANCE_SCHEMA,
        )
        for item in result.get("instances", []):
            mask = _mask_from_polygon(item["polygon"], width, height)
            bounds = _bbox_from_mask(mask)
            if bounds is None:
                continue
            candidates.append({
                "category": str(item["category"]),
                "score": float(item["confidence"]),
                "mask": mask,
                "source_image": str(source_path),
                "region_id": entry.get("region_id"),
                "terrain_camera": entry.get("camera"),
                "source_size": [width, height],
            })

    results = []
    for instance_index, item in enumerate(deduplicate(candidates, 0.8)):
        mask = item.pop("mask")
        bounds = _bbox_from_mask(mask)
        if bounds is None:
            continue
        bx0, by0, bx1, by1 = bounds
        ys, xs = np.nonzero(mask)
        instance_id = f"image_instance_{instance_index:04d}"
        instance_dir = output_root / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        mask_path = instance_dir / "mask.png"
        crop_path = instance_dir / "crop.png"
        Image.fromarray(mask * 255, mode="L").save(mask_path)
        crop = Image.open(item["source_image"]).convert("RGB").crop((bx0, by0, bx1, by1))
        crop.putalpha(Image.fromarray(mask[by0:by1, bx0:bx1] * 255, mode="L"))
        crop.save(crop_path)
        results.append({
            "id": instance_id,
            "category": item["category"],
            "score": item["score"],
            "bbox_xyxy": [bx0, by0, bx1, by1],
            "centroid_xy": [float(xs.mean()), float(ys.mean())],
            "mask": str(mask_path),
            "crop": str(crop_path),
            "source_image": item["source_image"],
            "crop_to_source_affine": [
                [1.0, 0.0, float(bx0)],
                [0.0, 1.0, float(by0)],
                [0.0, 0.0, 1.0],
            ],
            "region_id": item.get("region_id"),
            "terrain_camera": item.get("terrain_camera"),
            "source_size": item.get("source_size"),
        })
    write_response(args.response, {
        "status": "ok",
        "instances": results,
        "model": client.config.vision_model,
        "provider": "openai-compatible",
    })


if __name__ == "__main__":
    main()
