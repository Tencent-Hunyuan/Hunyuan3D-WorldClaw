"""Executable deterministic structural geometry validator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldclaw_oss.structural import validate_structural_branch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic structural geometry validation")
    parser.add_argument("--input", type=Path, required=True, help="structural output directory")
    parser.add_argument("--terrain", type=Path, required=True, help="terrain_structural.npz")
    parser.add_argument("--output", type=Path, required=True, help="validation JSON")
    return parser.parse_args()


def run(input_dir: Path, terrain_path: Path, output_path: Path) -> dict:
    branch_path = input_dir / "structural_branch.json"
    if not branch_path.is_file():
        raise FileNotFoundError(branch_path)
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    with np.load(terrain_path) as data:
        terrain_height = np.asarray(data["height"])
    report = validate_structural_branch(branch, terrain_height)
    report["validator"] = {"type": "deterministic_geometry", "source": str(branch_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.input, args.terrain, args.output), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
