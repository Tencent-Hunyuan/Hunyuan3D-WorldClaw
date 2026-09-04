import importlib.util
import json
from pathlib import Path


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_terrain_macro_goal.py"
_SPEC = importlib.util.spec_from_file_location("audit_terrain_macro_goal", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_goal_audit_is_conservative_without_live_run():
    report = _MODULE.audit(Path(__file__).resolve().parents[1])
    assert report["complete"] is False
    assert report["counts"]["pass"] == 23
    assert report["counts"]["fail"] == 1
    assert report["counts"]["partial"] == 3
    assert report["counts"]["unverified"] == 2
    assert len(report["requirements"]) == 29
    assert report["code_signals"]["layout"] is True
    assert report["code_signals"]["replan"] is True
    assert report["code_signals"]["determinism"] is True
    assert report["requirements"][17]["status"] == "pass"


def test_goal_audit_marks_missing_runtime_evidence(tmp_path):
    report = _MODULE.audit(Path(__file__).resolve().parents[1], tmp_path / "empty-run")
    item = next(value for value in report["requirements"] if value["id"] == 27)
    assert item["status"] == "unverified"
    assert report["runtime_signals"]["live_mode"] is False


def test_goal_audit_does_not_accept_failed_stage_as_complete_live_run(tmp_path):
    run = tmp_path / "failed-live"
    run.mkdir()
    stages = {name: {"status": "complete"} for name in {
        "intent", "plan", "layout", "terrain_macro_plan", "terrain_macro_generate",
        "terrain", "terrain_visual_validate", "structural_input", "structural_generate",
        "structural_replan", "structural_integrate", "structural_validate",
        "environment_assets", "mesh_validation_environment_assets", "region_composition",
        "segment", "reconstruct", "mesh_validation_reconstruct", "place", "refine", "export",
        "structural_view_plan", "structural_render", "structural_visual_validate",
        "structural_adaptive_render", "structural_final_validate", "validate",
    }}
    stages["terrain"] = {"status": "failed"}
    (run / "run_manifest.json").write_text(json.dumps({
        "mode": "live", "synthetic_outputs": False, "stage_results": stages,
    }), encoding="utf-8")
    report = _MODULE.audit(Path(__file__).resolve().parents[1], run)
    assert report["runtime_signals"]["live_mode"] is True
    assert report["runtime_signals"]["complete_workflow"] is False
    assert next(value for value in report["requirements"] if value["id"] == 27)["status"] == "unverified"


def test_goal_audit_reads_nested_validation_and_regeneration_evidence(tmp_path):
    run = tmp_path / "synthetic-evidence"
    for relative in (
        "terrain/terrain_macro_plan.json",
        "terrain/macro_height.npy",
        "terrain/validation/terrain_metrics.json",
        "terrain_validation/heightmap.png",
        "terrain_validation/oblique_01.png",
        "terrain_validation/layout_overlay.png",
    ):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"evidence")
    (run / "metrics.json").write_text(json.dumps({"run_validation": {"valid": True}}), encoding="utf-8")
    (run / "terrain_visual_validation.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (run / "mesh_reuse_manifest.json").write_text(json.dumps({
        "reused": [],
        "input_hashes": {
            "scene_plan.json": "a", "terrain_macro_plan.json": "b",
            "layout_labels.npy": "c", "layout_weights.npy": "d",
        },
    }), encoding="utf-8")
    report = _MODULE.audit(Path(__file__).resolve().parents[1], run)
    assert report["runtime_signals"]["validation_pass"] is True
    assert report["runtime_signals"]["macro_visual_pass"] is True
    assert report["runtime_signals"]["goal_outputs_regenerated"] is True


def test_goal_audit_rejects_comparison_with_model_mismatch(tmp_path):
    run = tmp_path / "comparison-evidence"
    run.mkdir()
    (run / "terrain_regression_comparison.json").write_text(json.dumps({
        "status": "comparison_ready",
        "regression_identity": {
            "prompt_match": True,
            "seed_match": True,
            "model_identity_match": False,
            "same_condition": False,
        },
    }), encoding="utf-8")

    report = _MODULE.audit(Path(__file__).resolve().parents[1], run)

    assert report["runtime_signals"]["comparison"] is False
    assert next(value for value in report["requirements"] if value["id"] == 28)["status"] == "partial"
