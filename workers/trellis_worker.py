#!/usr/bin/env python3
"""TRELLIS fallback worker; invoke only after a verified Hunyuan3D load failure."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from workers.common import (
    artifact_key,
    image_entries,
    load_stage_response,
    read_request,
    seed_everything,
    worker_args,
    write_response,
)


def source_entries(request: dict, work_dir: Path) -> list[dict]:
    stage = request.get("stage", "reconstruction")
    if stage == "environment_assets":
        return image_entries(load_stage_response(work_dir, "environment_references"))
    if stage == "refinement_reconstruction":
        return list(request.get("refinement_assets", []))
    return load_stage_response(work_dir, "segmentation").get("instances", [])


def main():
    args = worker_args()
    request = read_request(args.request)
    seed = int(request["seed"])
    seed_everything(seed)
    record = request["models"]["trellis"]
    work_dir = Path(request["work_dir"])
    sources = source_entries(request, work_dir)
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    pipeline = TrellisImageTo3DPipeline.from_pretrained(record["model_id"], revision=record["revision"])
    output_root = work_dir / "reconstruction" / "trellis"
    output_root.mkdir(parents=True, exist_ok=True)
    assets = []
    for index, source in enumerate(sources):
        image_path = Path(source["crop"])
        key = artifact_key(image_path, record["model_id"], record["revision"], {"seed": seed})
        glb_path = output_root / f"{key}.glb"
        if not glb_path.exists():
            outputs = pipeline.run(Image.open(image_path).convert("RGB"), seed=seed + index)
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024
            )
            glb.export(glb_path)
        assets.append({
            "id": source.get("id", f"asset_{index:04d}"),
            "category": source.get("category", "asset"), "source_image": str(image_path),
            "mesh": str(glb_path), "bbox_xyxy": source.get("bbox_xyxy"),
            "centroid_xy": source.get("centroid_xy"), "region_id": source.get("region_id"),
            "model": record,
        })
    write_response(args.response, {"status": "ok", "assets": assets, "model": record, "fallback": "trellis"})


if __name__ == "__main__":
    main()
