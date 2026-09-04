#!/usr/bin/env python3
"""Grounding DINO + sliding-window SAM 2.1 instance extraction worker."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from workers.common import image_entries, load_stage_response, read_request, resolve_model_source, seed_everything, worker_args, write_response
from worldclaw_oss.asset_semantics import is_structural_feature


def windows(width: int, height: int, tile: int, overlap: float):
    stride = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, width - tile + 1), stride))
    ys = list(range(0, max(1, height - tile + 1), stride))
    if not xs or xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if not ys or ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            yield x, y, min(width, x + tile), min(height, y + tile)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def deduplicate(instances: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept = []
    for candidate in sorted(instances, key=lambda item: item["score"], reverse=True):
        duplicate = False
        for current in kept:
            if candidate["category"] == current["category"] and mask_iou(candidate["mask"], current["mask"]) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def categories_from_plan(work_dir: Path) -> list[str]:
    import json
    plan = json.loads((work_dir / "scene_plan.json").read_text(encoding="utf-8"))
    return sorted({
        obj["category"]
        for region in plan["regions"]
        for obj in region.get("objects", [])
        if obj.get("count", 0) > 0 and not is_structural_feature(obj.get("category", ""), obj.get("asset_role", obj.get("asset_type")))
    })


def select_sam_mask(processor, outputs, inputs) -> np.ndarray:
    import torch
    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu()
    )[0]
    scores = outputs.iou_scores.detach().cpu().reshape(-1)
    masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    return (masks[int(torch.argmax(scores))].numpy() > 0).astype(np.uint8)


def move_model_inputs(batch, device, dtype):
    """Move processor tensors and align floating inputs with half-precision models."""
    if hasattr(batch, "to"):
        batch = batch.to(device)
    else:
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    for key, value in batch.items():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            batch[key] = value.to(dtype=dtype)
    return batch


def main():
    import torch
    from PIL import Image
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )
    args = worker_args()
    request = read_request(args.request)
    seed_everything(int(request["seed"]))
    work_dir = Path(request["work_dir"])
    composition = load_stage_response(work_dir, "region_composition")
    entries = image_entries(composition)
    if not entries:
        raise RuntimeError("region composition produced no images")
    categories = request.get("categories") or categories_from_plan(work_dir)
    if not categories:
        raise RuntimeError("scene plan contains no detectable object categories")

    dino_record = request["models"]["grounding_dino"]
    sam_record = request["models"]["sam2"]
    dino_source, dino_revision = resolve_model_source(dino_record)
    sam_source, sam_revision = resolve_model_source(sam_record)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Grounding DINO's text-enhancer path emits FP32 tensors in this
    # Transformers revision. Keep both models and processor inputs in FP32 to
    # avoid an internal matmul dtype mismatch; the 32 GB target GPU has room.
    dtype = torch.float32
    dino_processor = AutoProcessor.from_pretrained(
        dino_source, **({"revision": dino_revision} if dino_revision else {})
    )
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_source, torch_dtype=dtype,
        **({"revision": dino_revision} if dino_revision else {})
    ).to(device).eval()
    sam_processor = Sam2Processor.from_pretrained(
        sam_source, **({"revision": sam_revision} if sam_revision else {})
    )
    sam = Sam2Model.from_pretrained(
        sam_source, torch_dtype=dtype,
        **({"revision": sam_revision} if sam_revision else {})
    ).to(device).eval()

    tile_size = int(request.get("tile_size", 1024))
    tile_overlap = float(request.get("tile_overlap", 0.25))
    box_threshold = float(request.get("box_threshold", 0.28))
    text_threshold = float(request.get("text_threshold", 0.22))
    output_root = work_dir / "segmentation"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    prompt_text = ". ".join(categories) + "."

    for image_index, entry in enumerate(entries):
        source_path = Path(entry["path"])
        image = Image.open(source_path).convert("RGB")
        width, height = image.size
        candidates = []
        for x0, y0, x1, y1 in windows(width, height, tile_size, tile_overlap):
            tile_image = image.crop((x0, y0, x1, y1))
            inputs = move_model_inputs(
                dino_processor(images=tile_image, text=prompt_text, return_tensors="pt"),
                device, dtype,
            )
            with torch.inference_mode():
                outputs = dino(**inputs)
            detected = dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(tile_image.height, tile_image.width)],
            )[0]
            labels = detected.get("text_labels", detected.get("labels", []))
            for box, score, label in zip(detected["boxes"], detected["scores"], labels):
                local_box = [float(value) for value in box.detach().cpu().tolist()]
                sam_inputs = move_model_inputs(sam_processor(
                    images=tile_image,
                    input_boxes=[[[local_box[0], local_box[1], local_box[2], local_box[3]]]],
                    return_tensors="pt",
                ), device, dtype)
                with torch.inference_mode():
                    sam_outputs = sam(**sam_inputs, multimask_output=True)
                local_mask = select_sam_mask(sam_processor, sam_outputs, sam_inputs)
                global_mask = np.zeros((height, width), dtype=np.uint8)
                global_mask[y0:y1, x0:x1] = local_mask[: y1 - y0, : x1 - x0]
                category = str(label)
                candidates.append({
                    "category": category,
                    "score": float(score.detach().cpu()),
                    "bbox": [
                        local_box[0] + x0, local_box[1] + y0,
                        local_box[2] + x0, local_box[3] + y0,
                    ],
                    "mask": global_mask,
                })

        for instance_index, item in enumerate(deduplicate(candidates, float(request.get("dedup_iou", 0.8)))):
            mask = item.pop("mask")
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            instance_id = f"image_{image_index:03d}_instance_{instance_index:04d}"
            instance_dir = output_root / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            mask_path = instance_dir / "mask.png"
            crop_path = instance_dir / "crop.png"
            Image.fromarray(mask * 255, mode="L").save(mask_path)
            crop = image.crop((bx0, by0, bx1, by1))
            crop.putalpha(Image.fromarray(mask[by0:by1, bx0:bx1] * 255, mode="L"))
            crop.save(crop_path)
            results.append({
                "id": instance_id,
                "category": item["category"],
                "score": item["score"],
                "source_image": str(source_path),
                "mask": str(mask_path),
                "crop": str(crop_path),
                "bbox_xyxy": [bx0, by0, bx1, by1],
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
                "crop_to_source_affine": [[1.0, 0.0, float(bx0)], [0.0, 1.0, float(by0)], [0.0, 0.0, 1.0]],
                "source_size": [width, height],
                "region_id": entry.get("region_id"),
                "terrain_camera": entry.get("camera"),
            })

    write_response(args.response, {
        "status": "ok",
        "instances": results,
        "models": {"detector": dino_record, "segmenter": sam_record},
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "categories": categories,
    })


if __name__ == "__main__":
    main()
