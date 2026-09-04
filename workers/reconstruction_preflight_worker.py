#!/usr/bin/env python3
"""Pre-Hunyuan reconstructability review for references and object crops.

The worker performs cheap local gates first, then optionally asks the configured
vision model for a structured action. It never reconstructs meshes itself; the
pipeline applies the returned input edits before invoking Hunyuan3D.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from workers.common import load_stage_response, read_request, worker_args, write_response
from worldclaw_oss.asset_semantics import effective_asset_type, is_structural_feature
from worldclaw_oss.asset_types import AssetType
from worldclaw_oss.openai_api import OpenAIClient


PREFLIGHT_VERSION = "reconstructability-v1"
_PROCEDURAL_REPRESENTATIONS = {
    AssetType.PROCEDURAL_NATIVE.value: "vegetation_card",
    AssetType.LINEAR_STRUCTURE.value: "terrain_strip",
    AssetType.SURFACE_REGION_FEATURE.value: "terrain_feature",
    AssetType.STRUCTURAL_FEATURE.value: "structural_feature_branch",
}
_ACTIONS = {"reconstruct", "rewrite_prompt", "recrop", "resegment", "reroute", "drop"}


def _review_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "action": {"type": "string", "enum": sorted(_ACTIONS)},
                        "representation": {"type": "string"},
                        "reason": {"type": "string"},
                        "rewritten_prompt": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["id", "action", "representation", "reason", "rewritten_prompt", "confidence"],
                },
            },
        },
        "required": ["decisions"],
    }


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError, ImportError):
        return None


def _mask_ratio(path: Path) -> tuple[float, tuple[int, int, int, int] | None] | None:
    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L")) > 0
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return 0.0, None
        return float(mask.mean()), (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    except (OSError, ValueError, ImportError):
        return None


def _crop_quality(item: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(item.get("source_image", "")))
    crop = Path(str(item.get("crop", "")))
    mask = Path(str(item.get("mask", "")))
    source_size = _image_size(source) if source.is_file() else None
    crop_size = _image_size(crop) if crop.is_file() else None
    mask_info = _mask_ratio(mask) if mask.is_file() else None
    bbox = item.get("bbox_xyxy")
    valid_bbox = False
    bbox_area = 0.0
    source_area = float(source_size[0] * source_size[1]) if source_size else 0.0
    if isinstance(bbox, list) and len(bbox) == 4 and source_size:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        valid_bbox = x1 > x0 and y1 > y0 and x0 >= 0 and y0 >= 0 and x1 <= source_size[0] and y1 <= source_size[1]
        bbox_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    defects: list[str] = []
    if source_size is None:
        defects.append("missing_source_image")
    if crop_size is None:
        defects.append("missing_crop")
    elif min(crop_size) < 16:
        defects.append("crop_too_small")
    if mask_info is None:
        defects.append("missing_mask")
    elif mask_info[0] <= 0.001 or mask_info[0] >= 0.98:
        defects.append("mask_empty_or_full_frame")
    if not valid_bbox:
        defects.append("invalid_bbox")
    elif source_area and not 0.0005 <= bbox_area / source_area <= 0.85:
        defects.append("bbox_area_implausible")
    return {
        "source_size": list(source_size) if source_size else None,
        "crop_size": list(crop_size) if crop_size else None,
        "mask_ratio": mask_info[0] if mask_info else None,
        "mask_bbox": list(mask_info[1]) if mask_info and mask_info[1] else None,
        "bbox_area_ratio": bbox_area / source_area if source_area else None,
        "defects": defects,
    }


def _recrop(item: dict[str, Any], quality: dict[str, Any]) -> bool:
    """Rebuild a crop from the source and mask without invoking a model."""
    try:
        from PIL import Image
        source = Path(str(item["source_image"]))
        mask = Path(str(item["mask"]))
        crop_path = Path(str(item["crop"]))
        if not source.is_file() or not mask.is_file():
            return False
        with Image.open(source) as source_image, Image.open(mask) as mask_image:
            source_image = source_image.convert("RGBA")
            mask_image = mask_image.convert("L")
            bounds = quality.get("mask_bbox")
            if not bounds:
                return False
            x0, y0, x1, y1 = (int(value) for value in bounds)
            cropped = source_image.crop((x0, y0, x1, y1))
            cropped.putalpha(mask_image.crop((x0, y0, x1, y1)))
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(crop_path)
        item["bbox_xyxy"] = [x0, y0, x1, y1]
        item["crop_to_source_affine"] = [[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]]
        return True
    except (OSError, ValueError, KeyError, ImportError):
        return False


def _local_decision(item: dict[str, Any], source_stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    asset_type = effective_asset_type(item.get("category", ""), item.get("asset_type"))
    representation = _PROCEDURAL_REPRESENTATIONS.get(asset_type.value, "hunyuan3d")
    if is_structural_feature(item.get("category", ""), item.get("asset_type")):
        return ({
            "id": str(item.get("id", "")), "action": "reroute", "representation": representation,
            "reason": "route-native terrain representation is preferable to single-image reconstruction",
            "rewritten_prompt": "", "confidence": 1.0,
        }, {"defects": ["non_object_route"]})
    if source_stage == "environment_assets":
        path = Path(str(item.get("path", "")))
        if not path.is_file() or path.stat().st_size == 0:
            return ({
                "id": str(item.get("id", "")), "action": "rewrite_prompt", "representation": representation,
                "reason": "reference image is missing or empty", "rewritten_prompt": "", "confidence": 1.0,
            }, {"defects": ["missing_reference"]})
        return ({
            "id": str(item.get("id", "")), "action": "reconstruct", "representation": representation,
            "reason": "reference image passed local gates", "rewritten_prompt": "", "confidence": 0.5,
        }, {"defects": []})
    quality = _crop_quality(item)
    if "missing_source_image" in quality["defects"] or "missing_crop" in quality["defects"]:
        action = "resegment"
    elif quality["defects"]:
        action = "recrop" if _recrop(item, quality) else "resegment"
    else:
        action = "reconstruct"
    return ({
        "id": str(item.get("id", "")), "action": action, "representation": representation,
        "reason": "; ".join(quality["defects"]) or "crop, mask, and bbox passed local gates",
        "rewritten_prompt": "", "confidence": 1.0 if quality["defects"] else 0.5,
    }, quality)


def _vlm_review(request: dict[str, Any], source_stage: str, items: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paths = []
    for item in items[:32]:
        value = item.get("path") if source_stage == "environment_assets" else item.get("crop")
        path = Path(str(value or ""))
        if path.is_file() and path.stat().st_size:
            paths.append(path)
    if not paths or os.getenv("WORLDCLAW_RECON_PREFLIGHT_LOCAL_ONLY", "0").lower() in {"1", "true", "yes"}:
        return {}, {"provider": "local", "status": "skipped", "images": len(paths)}
    provider = os.getenv("WORLDCLAW_VALIDATION_PROVIDER", "openai").lower()
    user = (
        "Review these candidate inputs before single-image Image-to-3D reconstruction. "
        "For each item choose reconstruct, rewrite_prompt, recrop, resegment, reroute, or drop. "
        "Reject image inputs with severe occlusion, multiple merged objects, flat artwork, "
        "backdrop-only content, or unusable silhouettes. Prefer vegetation_card, terrain_strip, "
        "or terrain_feature for route-native representations. Items are listed in order:\n"
        + json.dumps([{
            "id": item.get("id"), "category": item.get("category"),
            "asset_type": item.get("asset_type"), "bbox_xyxy": item.get("bbox_xyxy"),
        } for item in items[:32]], ensure_ascii=False)
    )
    try:
        if provider == "openai":
            client = OpenAIClient()
            value = client.vision_json(
                system="You are a strict 3D reconstructability reviewer. Return only the requested JSON.",
                user=user, image_paths=paths, schema=_review_schema(), model=client.config.validation_model,
            )
        else:
            import requests
            url = os.getenv("WORLDCLAW_VLM_URL", os.getenv("WORLDCLAW_VLLM_URL", "")).rstrip("/")
            if not url:
                return {}, {"provider": provider, "status": "unavailable", "reason": "VLM URL is unset"}
            content = [{"type": "text", "text": user}]
            for path in paths:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
            response = requests.post(url + "/v1/chat/completions", json={
                "model": request["models"]["vlm"]["model_id"],
                "messages": [{"role": "system", "content": "You are a strict 3D reconstructability reviewer."}, {"role": "user", "content": content}],
                "temperature": 0.0, "response_format": {"type": "json_schema", "json_schema": {"name": "reconstructability_review", "strict": True, "schema": _review_schema()}},
            }, timeout=900)
            response.raise_for_status()
            value = json.loads(response.json()["choices"][0]["message"]["content"])
        decisions = {
            str(item["id"]): item for item in value.get("decisions", [])
            if isinstance(item, dict) and item.get("id")
        }
        return decisions, {"provider": provider, "status": "ok", "images": len(paths), "model": request.get("models", {}).get("vlm", {}).get("model_id")}
    except Exception as error:
        return {}, {"provider": provider, "status": "error", "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    args = worker_args()
    request = read_request(args.request)
    work = Path(request["work_dir"])
    source_stage = str(request.get("source_stage", "environment_assets"))
    upstream = load_stage_response(work, "environment_references" if source_stage == "environment_assets" else "segmentation")
    key = "images" if source_stage == "environment_assets" else "instances"
    items = [dict(item) for item in upstream.get(key, []) if isinstance(item, dict)]
    for item in items:
        # Archived responses may carry one of the retired route names. Keep
        # the input readable, but make every downstream response canonical.
        item["asset_type"] = effective_asset_type(
            item.get("category", ""), item.get("asset_type")
        ).value
    local_decisions: dict[str, dict[str, Any]] = {}
    local_checks: dict[str, dict[str, Any]] = {}
    for item in items:
        decision, quality = _local_decision(item, source_stage)
        local_decisions[str(item.get("id", ""))] = decision
        local_checks[str(item.get("id", ""))] = quality
    vlm_decisions, vlm_metadata = _vlm_review(request, source_stage, items)
    rewritten_requests = []
    selected = []
    dropped_ids = []
    resegment_required = False
    final_decisions = []
    for item in items:
        item_id = str(item.get("id", ""))
        decision = dict(local_decisions[item_id])
        model_decision = vlm_decisions.get(item_id)
        if model_decision and decision["action"] not in {"resegment", "recrop"}:
            candidate_action = str(model_decision.get("action", "")).lower()
            if source_stage != "environment_assets" and candidate_action == "rewrite_prompt":
                # Cropped reconstruction inputs have no independent prompt
                # generation step; make the requested visual correction a
                # deterministic crop repair instead.
                candidate_action = "recrop"
            if candidate_action in _ACTIONS:
                decision.update({
                    "action": candidate_action,
                    "representation": str(model_decision.get("representation") or decision["representation"]),
                    "reason": str(model_decision.get("reason") or decision["reason"]),
                    "confidence": float(model_decision.get("confidence", decision["confidence"])),
                })
        # Tree prototypes are reusable volumetric objects. A VLM suggestion to
        # use vegetation cards must not silently change that semantic route.
        # Otherwise crossed cards can be generated and later shared with reeds.
        category_route = effective_asset_type(item.get("category", ""), None)
        if (
            category_route == AssetType.REUSABLE_PROTOTYPE
            and decision["action"] == "reroute"
            and decision["representation"] == "vegetation_card"
        ):
            decision["action"] = "reconstruct"
            decision["representation"] = "hunyuan3d"
            decision["reason"] = (
                str(decision.get("reason", ""))
                + "; semantic reusable_prototype route enforced"
            ).strip("; ")
            item["asset_type"] = AssetType.REUSABLE_PROTOTYPE.value
        if decision["action"] == "rewrite_prompt":
            base = str(item.get("prompt", "")).strip()
            rewritten = str(model_decision.get("rewritten_prompt", "")).strip() if model_decision else ""
            item["prompt"] = rewritten or (base + "; one isolated object, clear full-body silhouette, visible side profile, no occlusion, no text, no backdrop plane")
            rewritten_requests.append(item)
            selected.append(item)
        elif decision["action"] == "recrop":
            if not _recrop(item, local_checks[item_id]):
                resegment_required = True
            selected.append(item)
        elif decision["action"] == "resegment":
            resegment_required = True
            selected.append(item)
        elif decision["action"] == "reroute":
            representation = decision["representation"]
            if representation == "vegetation_card":
                item["asset_type"] = AssetType.PROCEDURAL_NATIVE.value
                selected.append(item)
            else:
                dropped_ids.append(item_id)
        elif decision["action"] == "drop":
            dropped_ids.append(item_id)
        else:
            selected.append(item)
        final_decisions.append(decision)
    output = {
        "status": "ok", "preflight_version": PREFLIGHT_VERSION, "source_stage": source_stage,
        "decisions": final_decisions, "local_checks": local_checks,
        "vlm": vlm_metadata, "rewritten_requests": rewritten_requests,
        "resegment_required": resegment_required, "dropped_ids": dropped_ids,
        key: selected,
    }
    write_response(args.response, output)


if __name__ == "__main__":
    main()
