"""Create a tiny non-model fixture for the placement/export worker smoke test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


def main(root: Path) -> None:
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    (root / "diagnostic_views").mkdir(exist_ok=True)
    vertices = np.array([
        [-10.0, -10.0, 0.0], [10.0, -10.0, 0.0],
        [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    np.savez_compressed(work / "terrain.npz", vertices=vertices, triangles=triangles, height=np.zeros((2, 2)))

    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    mesh.apply_translation([0.0, 0.0, 1.0])
    mesh_path = work / "asset.glb"
    mesh.export(mesh_path, file_type="glb")

    regions = [
        {"id": "a", "function": "keep", "center": {"x": -5.0, "y": 0.0},
         "polygon": {"points": [{"x": -10.0, "y": -10.0}, {"x": 0.0, "y": -10.0}, {"x": 0.0, "y": 10.0}]},
         "coverage": 0.34, "neighbors": ["b"]},
        {"id": "b", "function": "keep", "center": {"x": 5.0, "y": 0.0},
         "polygon": {"points": [{"x": 0.0, "y": -10.0}, {"x": 10.0, "y": -10.0}, {"x": 10.0, "y": 10.0}]},
         "coverage": 0.33, "neighbors": ["a", "c"]},
        {"id": "c", "function": "keep", "center": {"x": 0.0, "y": 0.0},
         "polygon": {"points": [{"x": -10.0, "y": 0.0}, {"x": 10.0, "y": 0.0}, {"x": 0.0, "y": 10.0}]},
         "coverage": 0.33, "neighbors": ["b"]},
    ]
    plan = {
        "theme": "worker-smoke", "world_size_m": [20.0, 20.0], "regions": regions,
        "terrain": [{"region_id": item["id"], "mask_key": "smoke", "base_height": 0.0,
                      "noise_octaves": [1.0], "operators": [], "boundary_blend": 0.1} for item in regions],
        "materials": {"default": "green"}, "explicit_constraints": [],
    }
    (work / "scene_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    asset = {
        "id": "smoke_asset_000", "category": "box", "source_image": "smoke://box",
        "mask": "smoke", "mesh": str(mesh_path), "material": "box",
        "transform_z_up": np.eye(4).tolist(), "region_id": "a", "contact_ratio": 1.0,
        "source_model": "smoke-model",
    }
    (work / "placement_response.json").write_text(json.dumps({"status": "ok", "assets": [asset]}), encoding="utf-8")
    request = {"work_dir": str(work), "run_dir": str(root), "prompt": "worker smoke", "seed": 42, "models": {}}
    (work / "request.json").write_text(json.dumps(request), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
