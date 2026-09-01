#!/usr/bin/env python3
"""Run a real Grounding DINO + SAM2 contract smoke on a fixture image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldclaw_oss.models import CommandWorker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    source = args.image.resolve()
    (work / "region_composition_response.json").write_text(
        json.dumps({
            "status": "ok",
            "images": [{"id": "segmentation_fixture", "path": str(source), "region_id": "smoke"}],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (work / "scene_plan.json").write_text(json.dumps({
        "theme": "segmentation-smoke", "world_size_m": [20.0, 20.0],
        "regions": [{
            "id": "smoke", "function": "fixture", "center": {"x": 0.0, "y": 0.0},
            "polygon": {"points": [{"x": -10.0, "y": -10.0}, {"x": 10.0, "y": -10.0}, {"x": 0.0, "y": 10.0}]},
            "coverage": 1.0, "neighbors": [],
            "objects": [{"category": "castle", "count": 1, "density": 0.1, "appearance": ""}],
        }],
        "terrain": [{"region_id": "smoke", "mask_key": "smoke", "base_height": 0.0,
                      "noise_octaves": [1.0], "operators": [], "boundary_blend": 0.1}],
        "materials": {"default": "green"}, "explicit_constraints": [],
    }, indent=2) + "\n", encoding="utf-8")
    request = {
        "work_dir": str(work), "run_dir": str(root), "prompt": "worker smoke",
        "seed": 42, "stage": "segmentation", "models": lock["models"],
        "categories": ["castle"], "tile_size": 1024,
    }
    request_path = work / "segmentation_request.json"
    response_path = work / "segmentation_response.json"
    try:
        response = CommandWorker("WORLDCLAW_SEGMENT_WORKER").run(
            request, request_path, response_path,
        )
    except RuntimeError as error:
        response = json.loads(response_path.read_text(encoding="utf-8")) if response_path.is_file() else {
            "status": "error", "error": str(error),
        }
    report = {"status": response.get("status", "error"), "response": response,
              "model": {key: lock["models"][key] for key in ("grounding_dino", "sam2")}}
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
