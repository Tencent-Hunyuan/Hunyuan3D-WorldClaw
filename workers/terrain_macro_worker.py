#!/usr/bin/env python3
"""Deterministic command worker for planner-owned macro terrain geometry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldclaw_oss.terrain_macro import (
    TerrainMacroPlan,
    generate_macro_height,
    macro_vertices,
    validate_macro_height,
    write_validation_bundle,
)


def run(request: dict) -> dict:
    plan_path = Path(request["terrain_macro_plan"])
    output_dir = Path(request["output_dir"])
    resolution = int(request.get("resolution", 512))
    plan = TerrainMacroPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))

    planner_input = Path(request["planner_input"]) if request.get("planner_input") else None

    def input_path(key: str, filename: str) -> Path | None:
        value = request.get(key)
        if value:
            return Path(value)
        if planner_input is not None:
            for candidate in (planner_input / "layout" / filename, planner_input / filename):
                if candidate.is_file():
                    return candidate
        return None

    def read_json_input(key: str, filename: str) -> dict | None:
        path = input_path(key, filename)
        if path is None:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{filename} must contain a JSON object")
        return value

    world_spec = read_json_input("world_spec", "world_spec.json")
    if world_spec is not None and "world_size_m" in world_spec:
        authored_size = tuple(float(value) for value in world_spec["world_size_m"])
        if authored_size != tuple(float(value) for value in plan.world_size_m):
            raise ValueError("world_spec.world_size_m must match terrain_macro_plan.world_size_m")
    terrain_constraints = read_json_input("terrain_constraints", "terrain_constraints.json")
    if terrain_constraints is not None and not isinstance(terrain_constraints.get("explicit_constraints", []), list):
        raise ValueError("terrain_constraints.explicit_constraints must be a list")

    labels = None
    labels_path = input_path("layout_labels", "layout_labels.npy")
    if labels_path:
        labels = np.load(labels_path)
        if labels.ndim != 2 or labels.shape != (resolution, resolution):
            raise ValueError("layout_labels shape must be (resolution, resolution)")
    weights = None
    weights_path = input_path("layout_weights", "layout_weights.npy")
    region_ids = tuple(str(value) for value in request.get("region_ids", [])) or None
    if weights_path:
        weights = np.load(weights_path)
        if labels is not None and weights.shape[1:] != labels.shape:
            raise ValueError("layout_weights shape must match layout_labels shape")
        if weights.ndim != 3 or weights.shape[1:] != (resolution, resolution):
            raise ValueError("layout_weights shape must be (region_count, resolution, resolution)")
        if region_ids is None or len(region_ids) != weights.shape[0]:
            raise ValueError("layout_weights requires one region_id per weight plane")
    height = generate_macro_height(plan, resolution, layout_weights=weights, region_ids=region_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "macro_height.npy", height)
    vertices, triangles = macro_vertices(height, plan.world_size_m, plan)
    np.savez_compressed(output_dir / "terrain_macro.npz", height=height, vertices=vertices, triangles=triangles)
    metrics = validate_macro_height(height, plan, labels, weights, region_ids or ())
    validation_dir = output_dir / "validation"
    write_validation_bundle(validation_dir, height, labels, metrics)
    (output_dir / "terrain_macro_validation.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not metrics["valid"]:
        raise ValueError(f"macro terrain validation failed: {metrics}")
    return {
        "status": "ok",
        "resolution": resolution,
        "macro_height": str(output_dir / "macro_height.npy"),
        "validation": str(validation_dir),
        "inputs": {
            "layout_labels": str(labels_path) if labels_path else None,
            "layout_weights": str(weights_path) if weights_path else None,
            "world_spec": str(input_path("world_spec", "world_spec.json")) if world_spec is not None else None,
            "terrain_constraints": str(input_path("terrain_constraints", "terrain_constraints.json")) if terrain_constraints is not None else None,
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(json.loads(args.request.read_text(encoding="utf-8")))
    except Exception as error:
        result = {"status": "error", "error": str(error)}
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if result.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
