#!/usr/bin/env python3
"""Run a same-condition Live workflow into a fresh directory and compare it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from compare_terrain_runs import compare_runs


DEFAULT_LIVE_PROMPT_FILE = _SCRIPT_DIR.parent / "prompts" / "forest_lake.txt"


_REQUIRED_CANDIDATE_ARTIFACTS = (
    "run_manifest.json",
    "metrics.json",
    "mesh_reuse_manifest.json",
    "scene.glb",
    "scene.blend",
    "preview.png",
    "work/scene_plan.json",
    "work/terrain_macro_plan.json",
    "work/macro_height.npy",
    "work/terrain.npz",
    "work/terrain_structural.npz",
    "work/structural_plan.json",
    "work/assets.json",
    "work/terrain_validation/terrain_metrics.json",
    "work/terrain_visual_validation.json",
    "work/terrain_regional_validation.json",
)
_REQUIRED_LIVE_STAGES = frozenset({
    "intent", "plan", "layout", "terrain_macro_plan", "terrain_macro_generate",
    "terrain", "terrain_visual_validate", "structural_input", "structural_generate",
    "structural_replan", "structural_integrate", "structural_validate",
    "environment_assets", "mesh_validation_environment_assets", "region_composition",
    "segment", "reconstruct", "mesh_validation_reconstruct", "place", "refine", "export",
    "structural_view_plan", "structural_render", "structural_visual_validate",
    "structural_adaptive_render", "structural_final_validate", "validate",
})


def _manifest(run: Path) -> dict[str, Any]:
    path = run / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"baseline manifest missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("run_manifest.json must contain an object")
    return value


def validate_regression_identity(baseline: str | Path, prompt_file: str | Path, seed: int, output: str | Path) -> dict[str, Any]:
    baseline_path = Path(baseline).resolve()
    output_path = Path(output).resolve()
    prompt_path = Path(prompt_file).resolve()
    manifest = _manifest(baseline_path)
    prompt_sha = hashlib.sha256(prompt_path.read_text(encoding="utf-8").strip().encode()).hexdigest()
    if manifest.get("prompt_sha256") != prompt_sha:
        raise ValueError("prompt SHA does not match baseline run_manifest.json")
    if manifest.get("seed") != seed:
        raise ValueError("seed does not match baseline run_manifest.json")
    if output_path == baseline_path:
        raise ValueError("candidate output must be an independent directory")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"candidate output must be empty or absent: {output_path}")
    return {
        "baseline_run": str(baseline_path),
        "prompt_file": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "seed": seed,
        "candidate_run": str(output_path),
        "baseline_models_present": bool(manifest.get("models")),
    }


def validate_candidate_artifacts(output: str | Path) -> dict[str, Any]:
    """Require a self-contained candidate Run before comparing it."""
    output_path = Path(output).resolve()
    missing: list[str] = []
    external: list[str] = []
    for relative in _REQUIRED_CANDIDATE_ARTIFACTS:
        path = output_path / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(output_path):
            external.append(relative)
    if missing or external:
        details = {"missing": missing, "external_paths": external}
        raise ValueError(f"candidate artifacts are incomplete or external: {json.dumps(details, sort_keys=True)}")
    manifest = _manifest(output_path)
    stage_results = manifest.get("stage_results", {})
    incomplete_stages = sorted(
        name for name in _REQUIRED_LIVE_STAGES
        if not isinstance(stage_results.get(name), dict)
        or stage_results[name].get("status") != "complete"
    )
    if manifest.get("mode") != "live" or manifest.get("synthetic_outputs") is True or incomplete_stages:
        details = {
            "mode": manifest.get("mode"),
            "synthetic_outputs": manifest.get("synthetic_outputs"),
            "incomplete_stages": incomplete_stages,
        }
        raise ValueError(f"candidate manifest is not a complete Live Run: {json.dumps(details, sort_keys=True)}")
    return {
        "required": list(_REQUIRED_CANDIDATE_ARTIFACTS),
        "required_stages": sorted(_REQUIRED_LIVE_STAGES),
        "verified": True,
    }


def run_regression(*, baseline: str | Path, prompt_file: str | Path, output: str | Path, seed: int, models_lock: str | Path, cache_root: str | Path | None = None) -> dict[str, Any]:
    identity = validate_regression_identity(baseline, prompt_file, seed, output)
    output_path = Path(output).resolve()
    command = [
        sys.executable, "-m", "worldclaw_oss", "generate",
        "--prompt-file", str(Path(prompt_file).resolve()), "--seed", str(seed),
        "--mode", "live", "--output", str(output_path),
        "--models-lock", str(Path(models_lock).resolve()),
    ]
    if cache_root is not None:
        command.extend(("--cache-root", str(Path(cache_root).resolve())))
    environment = os.environ.copy()
    environment["WORLDCLAW_BASELINE_RUN"] = str(Path(baseline).resolve())
    completed = subprocess.run(command, check=False, env=environment, text=True, capture_output=True)
    result: dict[str, Any] = identity | {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        result["status"] = "pipeline_failed"
        return result
    try:
        result["candidate_artifacts"] = validate_candidate_artifacts(output_path)
    except ValueError as error:
        result["status"] = "invalid_candidate_artifacts"
        result["artifact_error"] = str(error)
        return result
    comparison = compare_runs(baseline, output_path)
    comparison_path = output_path / "terrain_regression_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result["comparison"] = comparison
    result["comparison_artifact"] = str(comparison_path)
    same_condition = comparison.get("regression_identity", {}).get("same_condition") is True
    result["status"] = "comparison_ready" if same_condition else "comparison_not_same_condition"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--prompt-file", type=Path, default=DEFAULT_LIVE_PROMPT_FILE,
        help="prompt file (defaults to prompts/forest_lake.txt)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models-lock", type=Path, default=Path("models.lock.json"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run_regression(
        baseline=args.baseline, prompt_file=args.prompt_file, output=args.output,
        seed=args.seed, models_lock=args.models_lock, cache_root=args.cache_root,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(args.report.resolve())}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "comparison_ready" else 1)


if __name__ == "__main__":
    main()
