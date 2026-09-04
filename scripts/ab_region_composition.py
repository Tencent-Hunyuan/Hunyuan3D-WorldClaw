#!/usr/bin/env python3
"""A/B test regional FLUX composition against GPT Image 2.

The experiment keeps the scene plan, terrain condition, prompt, seed,
segmentation worker, and reconstructability reviewer fixed. Only the regional
image worker changes. It is intentionally an experiment harness rather than a
change to the production routing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from worldclaw_oss.layout import deterministic_layout, synthetic_plan
from worldclaw_oss.models import CommandWorker
from worldclaw_oss.terrain import generate_terrain


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def make_fixture(root: Path, prompt: str, seed: int, resolution: int) -> dict[str, Any]:
    plan = synthetic_plan(prompt)
    layout = deterministic_layout(plan, resolution=resolution)
    terrain = generate_terrain(plan, layout, seed)
    work = root / "fixture"
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "scene_plan.json", plan.model_dump(mode="json"))
    np.save(work / "layout_labels.npy", layout.labels)
    np.save(work / "layout_weights.npy", layout.weights)
    np.savez(work / "terrain.npz", height=terrain.height, vertices=terrain.vertices, triangles=terrain.triangles)

    from worldclaw_oss.export import write_png

    normalized = (terrain.height - terrain.height.min()) / max(float(terrain.height.max() - terrain.height.min()), 1e-8)
    palette = np.asarray([
        [75, 116, 66], [177, 143, 80], [67, 112, 154],
        [125, 104, 82], [92, 124, 91],
    ], dtype=np.float32)
    condition = np.clip(palette[layout.labels % len(palette)] * (0.55 + normalized[..., None] * 0.45), 0, 255).astype(np.uint8)
    condition_path = work / "terrain_condition.png"
    write_png(condition_path, condition)
    h, w = condition.shape[:2]
    camera_height = max(plan.world_size_m) * 1.2
    focal = camera_height * w / plan.world_size_m[0]
    camera = {
        "intrinsics": [[focal, 0.0, w / 2], [0.0, focal, h / 2], [0.0, 0.0, 1.0]],
        "camera_to_world": [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, camera_height], [0.0, 0.0, 0.0, 1.0]],
        "image_size": [w, h],
        "convention": "pinhole, camera +Z forward, source world Z-up",
    }
    requests = []
    for region in plan.regions:
        categories = ", ".join(obj.category for obj in region.objects if obj.count > 0)
        requests.append({
            "id": f"composition_{region.id}",
            "region_id": region.id,
            "camera": camera,
            "conditioning_image": str(condition_path),
            "strength": 0.58,
            "prompt": (
                f"Terrain-faithful regional composition for {region.function}; "
                f"preserve all terrain boundaries and camera geometry; include {categories}; "
                f"{region.appearance}; no text, no frame"
            ),
        })
    return {"plan": plan, "layout": layout, "terrain": terrain, "condition": condition_path, "requests": requests}


def run_worker(env_name: str, request: dict[str, Any], work: Path, name: str) -> dict[str, Any]:
    request_path = work / f"{name}_request.json"
    response_path = work / f"{name}_response.json"
    try:
        return CommandWorker(env_name).run(request, request_path, response_path, timeout=14400)
    except Exception as error:
        value = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
        if response_path.is_file():
            try:
                persisted = json.loads(response_path.read_text(encoding="utf-8"))
                value.update(persisted)
            except (OSError, json.JSONDecodeError):
                pass
        write_json(response_path, value)
        return value


def planned_categories(plan) -> list[str]:
    return sorted({obj.category for region in plan.regions for obj in region.objects if obj.count > 0})


def image_entries(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in response.get("images", []) if isinstance(item, dict) and item.get("path")]


def segmentation_metrics(plan, response: dict[str, Any]) -> dict[str, Any]:
    instances = [item for item in response.get("instances", []) if isinstance(item, dict)]
    planned = {category: 0 for category in planned_categories(plan)}
    for region in plan.regions:
        for obj in region.objects:
            if obj.count > 0:
                planned[obj.category] += int(obj.count)
    detected = {category: 0 for category in planned}
    for item in instances:
        category = str(item.get("category", ""))
        if category in detected:
            detected[category] += 1
    valid = 0
    for item in instances:
        bbox = item.get("bbox_xyxy")
        mask = Path(str(item.get("mask", "")))
        crop = Path(str(item.get("crop", "")))
        if isinstance(bbox, list) and len(bbox) == 4 and mask.is_file() and crop.is_file():
            valid += 1
    categories_present = sum(value > 0 for value in detected.values())
    return {
        "planned_counts": planned,
        "detected_counts": detected,
        "planned_category_recall": categories_present / max(len(planned), 1),
        "instance_count": len(instances),
        "valid_crop_mask_bbox_fraction": valid / max(len(instances), 1),
    }


def preflight_metrics(response: dict[str, Any]) -> dict[str, Any]:
    decisions = [item for item in response.get("decisions", []) if isinstance(item, dict)]
    actions = {}
    for item in decisions:
        action = str(item.get("action", "unknown"))
        actions[action] = actions.get(action, 0) + 1
    reconstruct = sum(actions.get(action, 0) for action in ("reconstruct", "recrop"))
    return {
        "decision_count": len(decisions),
        "actions": actions,
        "reconstruct_or_recrop_fraction": reconstruct / max(len(decisions), 1),
        "resegment_required": bool(response.get("resegment_required", False)),
        "dropped_ids": response.get("dropped_ids", []),
        "vlm": response.get("vlm", {}),
    }


def evaluate_visual(condition: Path, composition: list[Path], arm: str) -> dict[str, Any]:
    """Use the same structured reviewer for both arms when configured."""
    try:
        from worldclaw_oss.openai_api import OpenAIClient

        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "terrain_layout_fidelity": {"type": "number", "minimum": 1, "maximum": 5},
                "planned_object_presence": {"type": "number", "minimum": 1, "maximum": 5},
                "reason": {"type": "string"},
            },
            "required": ["terrain_layout_fidelity", "planned_object_presence", "reason"],
        }
        prompt = (
            "Compare the first image (deterministic terrain condition) with the following generated "
            f"regional image(s) for the {arm} arm. Score terrain/layout fidelity and planned object "
            "presence from 1 to 5. Judge spatial boundaries and whether requested categories appear; "
            "do not reward artistic quality. Return JSON only."
        )
        result = OpenAIClient().vision_json(system="You are a strict visual A/B evaluator.", user=prompt, image_paths=[condition, *composition], schema=schema)
        return {"status": "ok", **result}
    except Exception as error:
        return {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}


def run_arm(arm: str, worker_env: str, fixture: dict[str, Any], root: Path, lock: dict[str, Any], seed: int) -> dict[str, Any]:
    arm_root = root / arm
    work = arm_root / "work"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture["condition"], work / "terrain_condition.png")
    shutil.copy2(root / "fixture" / "scene_plan.json", work / "scene_plan.json")
    requests = []
    for item in fixture["requests"]:
        copied = dict(item)
        copied["conditioning_image"] = str(work / "terrain_condition.png")
        requests.append(copied)
    request = {"work_dir": str(work), "run_dir": str(arm_root), "prompt": "A/B regional composition", "seed": seed, "stage": f"region_composition_{arm}", "models": lock["models"], "image_requests": requests}
    started = time.monotonic()
    composition = run_worker(worker_env, request, work, "region_composition")
    composition_paths = [Path(item["path"]) for item in image_entries(composition) if Path(item["path"]).is_file()]
    write_json(work / "region_composition_response.json", composition | {"images": image_entries(composition)})
    if not composition_paths:
        return {"status": "unavailable", "composition": composition, "elapsed_seconds": round(time.monotonic() - started, 3)}

    categories = planned_categories(fixture["plan"])
    seg_request = {"work_dir": str(work), "run_dir": str(arm_root), "prompt": "A/B segmentation", "seed": seed, "stage": "segmentation", "models": lock["models"], "categories": categories, "tile_size": 1024}
    segmentation = run_worker("WORLDCLAW_SEGMENT_WORKER", seg_request, work, "segmentation")
    write_json(work / "segmentation_response.json", segmentation)
    preflight_request = {"work_dir": str(work), "run_dir": str(arm_root), "prompt": "A/B reconstructability", "seed": seed, "stage": "reconstruction_preflight", "models": lock["models"], "source_stage": "reconstruction"}
    preflight = run_worker("WORLDCLAW_RECON_PREFLIGHT_WORKER", preflight_request, work, "reconstruction_preflight")
    write_json(work / "reconstruction_preflight_response.json", preflight)

    # Reconstruct at most one crop per detected category so the expensive
    # Hunyuan stage remains a diagnostic sample, not a full-scene run.
    selected = []
    for item in segmentation.get("instances", []):
        if item.get("category") not in {entry.get("category") for entry in selected}:
            selected.append(item)
    selected = selected[:3]
    hunyuan = {"status": "skipped", "reason": "no valid segmented samples"}
    if selected:
        sampled = dict(segmentation)
        sampled["instances"] = selected
        write_json(work / "segmentation_response.json", sampled)
        hy_request = {"work_dir": str(work), "run_dir": str(arm_root), "prompt": "A/B Hunyuan sample", "seed": seed, "stage": "reconstruction", "models": lock["models"], "source_ids": [item.get("id") for item in selected]}
        hunyuan = run_worker("WORLDCLAW_IMAGE3D_WORKER", hy_request, work, "reconstruction")
    metrics = {
        "status": "ok", "composition": composition, "visual": evaluate_visual(fixture["condition"], composition_paths, arm),
        "segmentation": segmentation_metrics(fixture["plan"], segmentation), "segmentation_worker": segmentation.get("worker", {}),
        "preflight": preflight_metrics(preflight), "hunyuan": hunyuan, "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(arm_root / "metrics.json", metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--prompt", default="forest lake")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--arm", choices=("all", "flux", "gpt_image_2"), default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads(args.lock.resolve().read_text(encoding="utf-8"))
    fixture = make_fixture(root, args.prompt, args.seed, args.resolution)
    write_json(root / "experiment.json", {"prompt": args.prompt, "seed": args.seed, "resolution": args.resolution, "arms": ["flux", "gpt_image_2"], "models": {key: lock["models"].get(key) for key in ("flux_dev", "flux_schnell", "reference_image", "grounding_dino", "sam2", "vlm", "hunyuan3d")}})
    flux = run_arm("flux", "WORLDCLAW_FLUX_WORKER", fixture, root, lock, args.seed) if args.arm in {"all", "flux"} else json.loads((root / "flux" / "metrics.json").read_text(encoding="utf-8")) if (root / "flux" / "metrics.json").is_file() else {"status": "missing"}
    gpt = run_arm("gpt_image_2", "WORLDCLAW_REFERENCE_IMAGE_WORKER", fixture, root, lock, args.seed) if args.arm in {"all", "gpt_image_2"} else json.loads((root / "gpt_image_2" / "metrics.json").read_text(encoding="utf-8")) if (root / "gpt_image_2" / "metrics.json").is_file() else {"status": "missing"}
    summary = {"experiment": str(root), "flux": flux, "gpt_image_2": gpt}
    write_json(root / "ab_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
