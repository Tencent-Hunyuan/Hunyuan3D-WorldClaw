import json
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_terrain_runs.py"
_SPEC = importlib.util.spec_from_file_location("compare_terrain_runs", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
compare_runs = _MODULE.compare_runs


def _run(path, relief, visual_status="PASS"):
    work = path / "work"
    work.mkdir(parents=True)
    (work / "terrain_macro_validation.json").write_text(json.dumps({
        "valid": True, "global_relief": relief, "min_elevation": -2,
        "max_elevation": relief - 2, "max_slope_deg": 20,
        "p95_slope_deg": 12, "lake_basin_elevation": -3,
        "ridge_height": 8, "boundary_continuity": 1,
        "world_boundary_continuity": 1, "functional_zone_slope": {"site": 4},
        "buildable_area_ratio": 0.3, "trail_traversability": True,
        "lake_basin_containment": True,
    }), encoding="utf-8")
    (work / "terrain_regional_validation.json").write_text(json.dumps({
        "macro_relief_m": relief, "regional_detail_rms_m": 1,
        "regional_to_macro_rms_ratio": 0.1, "low_frequency_error_ratio": 0.05,
        "macro_composition_preserved": True,
    }), encoding="utf-8")
    (work / "metrics.json").write_text(json.dumps({"asset_count": 2}), encoding="utf-8")
    (work / "terrain_visual_validation.json").write_text(json.dumps({"status": visual_status, "confidence": "high"}), encoding="utf-8")
    (path / "scene.glb").write_bytes(f"scene-{relief}".encode())
    (path / "run_manifest.json").write_text(json.dumps({
        "prompt_sha256": "same-prompt", "seed": 42, "models": {"planner": {"model_id": "planner-test"}},
    }), encoding="utf-8")


def test_compare_runs_reports_differences_and_independence(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _run(baseline, 20)
    _run(candidate, 24)
    report = compare_runs(baseline, candidate)
    assert report["status"] == "comparison_ready"
    assert report["independent_directory"] is True
    assert report["terrain"]["macro"]["global_relief"]["delta"] == 4.0
    assert report["regression_identity"]["prompt_match"] is True
    assert report["regression_identity"]["seed_match"] is True
    assert report["artifacts"]["baseline"]["scene_glb"] != report["artifacts"]["candidate"]["scene_glb"]
    assert "candidate.mesh_reuse_manifest" in report["missing_evidence"]


def test_compare_runs_rejects_same_directory(tmp_path):
    _run(tmp_path, 20)
    with pytest.raises(ValueError, match="independent"):
        compare_runs(tmp_path, tmp_path)
