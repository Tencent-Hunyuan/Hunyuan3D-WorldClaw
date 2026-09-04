"""Repair legacy GPT segmentation responses with geometry fields required by reconstruction."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    response = args.run.resolve() / "work" / "segmentation_response.json"
    if not response.is_file():
        raise SystemExit(f"missing response: {response}")
    value = json.loads(response.read_text(encoding="utf-8"))
    if all("centroid_xy" in item and "crop_to_source_affine" in item
           for item in value.get("instances", [])):
        print(json.dumps({"status": "already_repaired", "instances": len(value.get("instances", []))}))
        return 0
    backup = response.with_name("segmentation_response.pre_contract_repair.json")
    if not backup.exists():
        shutil.copy2(response, backup)
    repaired = 0
    for item in value.get("instances", []):
        mask_path = Path(item["mask"])
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        bx0, by0, bx1, by1 = item["bbox_xyxy"]
        item["centroid_xy"] = [float(xs.mean()), float(ys.mean())]
        item["crop_to_source_affine"] = [
            [1.0, 0.0, float(bx0)],
            [0.0, 1.0, float(by0)],
            [0.0, 0.0, 1.0],
        ]
        repaired += 1
    value["contract_repair"] = {
        "kind": "add_centroid_and_crop_affine",
        "source_backup": str(backup),
        "instances_repaired": repaired,
    }
    response.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "repaired", "instances": repaired, "backup": str(backup)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
