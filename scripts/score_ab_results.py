#!/usr/bin/env python3
"""Add deterministic terrain/layout proxy scores to an A/B result directory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = np.asarray([[75, 116, 66], [177, 143, 80], [67, 112, 154], [125, 104, 82], [92, 124, 91]], dtype=np.float32)


def nearest_labels(image: np.ndarray) -> np.ndarray:
    distances = ((image[..., None, :].astype(np.float32) - PALETTE[None, None, ...]) ** 2).sum(axis=-1)
    return distances.argmin(axis=-1)


def score_arm(root: Path, condition: Path) -> dict:
    response = root / "region_composition_response.json"
    if not response.is_file():
        return {"status": "unavailable", "reason": "missing composition response"}
    value = json.loads(response.read_text(encoding="utf-8"))
    cond = np.asarray(Image.open(condition).convert("RGB"), dtype=np.uint8)
    expected = nearest_labels(cond)
    rows = []
    for item in value.get("images", []):
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            path = root / "flux" / Path(str(item.get("path", ""))).name
        if not path.is_file():
            continue
        image = np.asarray(Image.open(path).convert("RGB").resize((cond.shape[1], cond.shape[0])), dtype=np.uint8)
        actual = nearest_labels(image)
        agreement = float((actual == expected).mean())
        expected_edges = np.concatenate([(expected[:, 1:] != expected[:, :-1]).ravel(), (expected[1:, :] != expected[:-1, :]).ravel()])
        actual_edges = np.concatenate([(actual[:, 1:] != actual[:, :-1]).ravel(), (actual[1:, :] != actual[:-1, :]).ravel()])
        edge_overlap = float((expected_edges & actual_edges).sum() / max(expected_edges.sum(), 1))
        rows.append({"id": item.get("id"), "palette_region_agreement": agreement, "boundary_edge_recall_proxy": edge_overlap})
    return {"status": "ok", "images": rows, "mean_palette_region_agreement": float(np.mean([r["palette_region_agreement"] for r in rows])) if rows else None, "mean_boundary_edge_recall_proxy": float(np.mean([r["boundary_edge_recall_proxy"] for r in rows])) if rows else None}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    condition = root / "fixture" / "terrain_condition.png"
    summary = json.loads((root / "ab_summary.json").read_text(encoding="utf-8")) if (root / "ab_summary.json").is_file() else {}
    for arm in ("flux", "gpt_image_2"):
        summary.setdefault(arm, {})["terrain_layout_proxy"] = score_arm(root / arm / "work", condition)
    (root / "ab_summary_scored.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
