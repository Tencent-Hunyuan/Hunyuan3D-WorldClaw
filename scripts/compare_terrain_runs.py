#!/usr/bin/env python3
"""Compare two independent Terrain runs without treating missing evidence as pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(run: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    for candidate in (run / "work" / name, run / name, run / "terrain" / name):
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, str(candidate)
        return value if isinstance(value, dict) else None, str(candidate)
    return None, None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(run: Path) -> dict[str, str | None]:
    return {
        name: _sha256(path)
        for name, path in {
            "macro_height": run / "work" / "macro_height.npy",
            "terrain": run / "work" / "terrain.npz",
            "scene_glb": run / "scene.glb",
            "scene_blend": run / "scene.blend",
        }.items()
    }


def _model_identity(manifest: dict[str, Any] | None) -> dict[str, tuple[Any, Any]] | None:
    """Return stable model identifiers used to establish same-condition runs."""
    models = manifest.get("models") if manifest else None
    if not isinstance(models, dict) or not models:
        return None
    return {
        str(name): (record.get("model_id"), record.get("revision"))
        for name, record in sorted(models.items())
        if isinstance(record, dict)
    }


def _numeric_differences(baseline: dict[str, Any], candidate: dict[str, Any], fields: tuple[str, ...]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for field in fields:
        left, right = baseline.get(field), candidate.get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[field] = {"baseline": float(left), "candidate": float(right), "delta": float(right - left)}
        else:
            result[field] = {"baseline": left if isinstance(left, (int, float)) else None, "candidate": right if isinstance(right, (int, float)) else None, "delta": None}
    return result


def compare_runs(baseline: str | Path, candidate: str | Path) -> dict[str, Any]:
    baseline_path = Path(baseline).resolve()
    candidate_path = Path(candidate).resolve()
    if baseline_path == candidate_path:
        raise ValueError("baseline and candidate must be independent run directories")

    macro_names = ("terrain_macro_validation.json", "terrain_regional_validation.json")
    baseline_macro = {name: _read_json(baseline_path, name) for name in macro_names}
    candidate_macro = {name: _read_json(candidate_path, name) for name in macro_names}
    baseline_metrics, _ = _read_json(baseline_path, "metrics.json")
    candidate_metrics, _ = _read_json(candidate_path, "metrics.json")
    baseline_manifest, _ = _read_json(baseline_path, "run_manifest.json")
    candidate_manifest, _ = _read_json(candidate_path, "run_manifest.json")
    baseline_visual, _ = _read_json(baseline_path, "terrain_visual_validation.json")
    candidate_visual, _ = _read_json(candidate_path, "terrain_visual_validation.json")
    baseline_reuse, _ = _read_json(baseline_path, "mesh_reuse_manifest.json")
    candidate_reuse, _ = _read_json(candidate_path, "mesh_reuse_manifest.json")

    missing: list[str] = []
    for label, value in (
        ("baseline.terrain_macro_validation", baseline_macro[macro_names[0]][0]),
        ("candidate.terrain_macro_validation", candidate_macro[macro_names[0]][0]),
        ("baseline.metrics", baseline_metrics),
        ("candidate.metrics", candidate_metrics),
        ("baseline.terrain_visual_validation", baseline_visual),
        ("candidate.terrain_visual_validation", candidate_visual),
        ("baseline.mesh_reuse_manifest", baseline_reuse),
        ("candidate.mesh_reuse_manifest", candidate_reuse),
        ("baseline.run_manifest", baseline_manifest),
        ("candidate.run_manifest", candidate_manifest),
    ):
        if value is None:
            missing.append(label)

    baseline_macro_metrics = baseline_macro[macro_names[0]][0] or {}
    candidate_macro_metrics = candidate_macro[macro_names[0]][0] or {}
    baseline_regional = baseline_macro[macro_names[1]][0] or {}
    candidate_regional = candidate_macro[macro_names[1]][0] or {}
    baseline_run_metrics = baseline_metrics or {}
    candidate_run_metrics = candidate_metrics or {}
    report: dict[str, Any] = {
        "schema_version": "worldclaw-oss-terrain-regression-v1",
        "status": "comparison_ready",
        "baseline_run": str(baseline_path),
        "candidate_run": str(candidate_path),
        "independent_directory": baseline_path != candidate_path,
        "regression_identity": {
            "baseline": {
                "prompt_sha256": baseline_manifest.get("prompt_sha256") if baseline_manifest else None,
                "seed": baseline_manifest.get("seed") if baseline_manifest else None,
                "models": baseline_manifest.get("models") if baseline_manifest else None,
            },
            "candidate": {
                "prompt_sha256": candidate_manifest.get("prompt_sha256") if candidate_manifest else None,
                "seed": candidate_manifest.get("seed") if candidate_manifest else None,
                "models": candidate_manifest.get("models") if candidate_manifest else None,
            },
            "prompt_match": bool(baseline_manifest and candidate_manifest and baseline_manifest.get("prompt_sha256") == candidate_manifest.get("prompt_sha256")),
            "seed_match": bool(baseline_manifest and candidate_manifest and baseline_manifest.get("seed") == candidate_manifest.get("seed")),
            "model_records_present": bool(baseline_manifest and candidate_manifest and baseline_manifest.get("models") and candidate_manifest.get("models")),
            "model_identity_match": (
                _model_identity(baseline_manifest) is not None
                and _model_identity(baseline_manifest) == _model_identity(candidate_manifest)
            ),
        },
        "artifacts": {"baseline": _artifact_hashes(baseline_path), "candidate": _artifact_hashes(candidate_path)},
        "terrain": {
            "macro": _numeric_differences(
                baseline_macro_metrics,
                candidate_macro_metrics,
                ("global_relief", "min_elevation", "max_elevation", "max_slope_deg", "p95_slope_deg", "lake_basin_elevation", "ridge_height", "boundary_continuity", "world_boundary_continuity"),
            ),
            "regional": _numeric_differences(
                baseline_regional,
                candidate_regional,
                ("macro_relief_m", "regional_detail_rms_m", "regional_to_macro_rms_ratio", "low_frequency_error_ratio"),
            ),
            "validity": {
                "baseline_macro_valid": baseline_macro_metrics.get("valid"),
                "candidate_macro_valid": candidate_macro_metrics.get("valid"),
                "baseline_macro_preserved": baseline_regional.get("macro_composition_preserved"),
                "candidate_macro_preserved": candidate_regional.get("macro_composition_preserved"),
            },
        },
        "functional_and_downstream": {
            "baseline": {key: baseline_macro_metrics.get(key) for key in ("functional_zone_slope", "buildable_area_ratio", "trail_traversability", "lake_basin_containment")},
            "candidate": {key: candidate_macro_metrics.get(key) for key in ("functional_zone_slope", "buildable_area_ratio", "trail_traversability", "lake_basin_containment")},
            "run_metrics": {"baseline": baseline_run_metrics, "candidate": candidate_run_metrics},
        },
        "visual_validation": {
            "baseline": {key: baseline_visual.get(key) for key in ("status", "confidence", "diagnosis")} if baseline_visual else None,
            "candidate": {key: candidate_visual.get(key) for key in ("status", "confidence", "diagnosis")} if candidate_visual else None,
        },
        "mesh_reuse": {
            "baseline": {key: baseline_reuse.get("reason", {}).get(key) for key in ("reused_count", "reuse_eligible_count", "regenerated_count")} if baseline_reuse else None,
            "candidate": {key: candidate_reuse.get("reason", {}).get(key) for key in ("reused_count", "reuse_eligible_count", "regenerated_count")} if candidate_reuse else None,
        },
        "missing_evidence": missing,
    }
    for scope, hashes in report["artifacts"].items():
        for artifact, digest in hashes.items():
            if digest is None:
                report["missing_evidence"].append(f"{scope}.artifact.{artifact}")
    identity = report["regression_identity"]
    identity["same_condition"] = bool(
        identity["prompt_match"]
        and identity["seed_match"]
        and identity["model_identity_match"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
