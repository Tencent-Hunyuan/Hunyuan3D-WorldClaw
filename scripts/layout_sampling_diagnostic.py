#!/usr/bin/env python3
"""Persist seeded Layout sampling and reproducibility evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worldclaw_oss.layout import sample_layout, synthetic_plan
from worldclaw_oss.schemas import ScenePlan


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene-plan", type=Path)
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seed-a", type=int, default=42)
    parser.add_argument("--seed-b", type=int, default=43)
    parser.add_argument("--candidate-count", type=int, default=4)
    args = parser.parse_args()

    if args.scene_plan is not None:
        plan = ScenePlan.model_validate_json(args.scene_plan.read_text(encoding="utf-8"))
        source_name = str(args.scene_plan)
    else:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        plan = synthetic_plan(prompt)
        source_name = str(args.prompt_file)

    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for seed in (args.seed_a, args.seed_b):
        layout = sample_layout(
            plan, args.resolution, seed=seed, candidate_count=args.candidate_count,
        )
        np.save(args.output / f"layout_labels_seed{seed}.npy", layout.labels)
        np.save(args.output / f"layout_weights_seed{seed}.npy", layout.weights)
        (args.output / f"layout_generation_seed{seed}.json").write_text(
            json.dumps(layout.generation, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        results[str(seed)] = layout

    first, second = results[str(args.seed_a)], results[str(args.seed_b)]
    report = {
        "schema": "worldclaw-oss-layout-sampling-diagnostic-v1",
        "source": source_name,
        "resolution": args.resolution,
        "candidate_count": args.candidate_count,
        "seeds": [args.seed_a, args.seed_b],
        "same_seed_reproducible": bool(
            np.array_equal(first.labels, sample_layout(
                plan, args.resolution, seed=args.seed_a,
                candidate_count=args.candidate_count,
            ).labels)
        ),
        "different_seed_label_difference_ratio": float(
            np.mean(first.labels != second.labels)
        ),
        "topology_valid": {
            str(args.seed_a): bool(first.generation["topology_valid"]),
            str(args.seed_b): bool(second.generation["topology_valid"]),
        },
        "artifacts": sorted(path.name for path in args.output.iterdir() if path.is_file()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
