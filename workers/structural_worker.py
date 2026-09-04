"""Executable structural worker used by the structural-side workflow.

The worker consumes the materialized ``structural_agent_input`` tree and
executes planning (or loads the already adjudicated GPT plan), numerical mesh
generation, and terrain integration.  It intentionally delegates the common
geometry implementation to ``worldclaw_oss.structural``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldclaw_oss.structural import build_structural_geometry, structural_features_from_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute structural planning, generation, and terrain integration")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structural-plan", type=Path)
    return parser.parse_args()


def run(input_dir: Path, output_dir: Path, seed: int = 0, structural_plan_path: Path | None = None) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    plan = json.loads((input_dir / "scene_plan.json").read_text(encoding="utf-8"))
    selected_plan = structural_plan_path or (input_dir / "structural_plan.json")
    if selected_plan.is_file():
        structural_plan = json.loads(selected_plan.read_text(encoding="utf-8"))
        features = structural_plan.get("features", [])
        if not isinstance(features, list):
            raise ValueError("structural_plan.json features must be a list")
    else:
        features = structural_features_from_plan(plan)
        structural_plan = {"status": "ok", "provider": "deterministic_fixture", "features": features, "deterministic": True}
    terrain_path = input_dir / "terrain" / "terrain.npz"
    terrain = np.load(terrain_path)
    if "height" not in terrain or "vertices" not in terrain or "triangles" not in terrain:
        raise ValueError("terrain.npz must contain height, vertices, and triangles")
    world_size = tuple(float(value) for value in plan["world_size_m"])
    layout_weights = np.load(input_dir / "layout" / "layout_weights.npy")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_structural_geometry(plan, terrain["height"], world_size, output_dir, seed, features=features, layout_weights=layout_weights)
    # Persist the integrated surface, not only a height diagnostic.  Export
    # consumes vertices, so keeping their Z values aligned is what makes the
    # basin/channel operation visible in the final scene.
    modified = np.asarray(np.load(output_dir / "terrain_structural.npz")["height"], dtype=np.float32)
    vertices = np.asarray(terrain["vertices"], dtype=np.float32).copy()
    if len(vertices) != modified.size:
        raise ValueError("terrain vertex count does not match height grid")
    vertices[:, 2] = modified.reshape(-1)
    np.savez_compressed(
        output_dir / "terrain_structural.npz",
        height=modified,
        vertices=vertices,
        triangles=np.asarray(terrain["triangles"], dtype=np.uint32),
    )
    (output_dir / "structural_plan.json").write_text(json.dumps(structural_plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    result["worker"] = {"planning": True, "generation": True, "terrain_integration": True, "input_dir": str(input_dir), "output_dir": str(output_dir)}
    return result


def main() -> None:
    args = parse_args()
    result = run(args.input, args.output, args.seed, args.structural_plan)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
