import importlib.util
import json
from pathlib import Path


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_terrain_runs.py"
_SPEC = importlib.util.spec_from_file_location("compare_terrain_runs", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_compare_marks_different_model_identity_as_not_same_condition(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "run_manifest.json").write_text(json.dumps({
        "prompt_sha256": "prompt", "seed": 42,
        "models": {"planner": {"model_id": "qwen", "revision": "rev-a"}},
    }), encoding="utf-8")
    (candidate / "run_manifest.json").write_text(json.dumps({
        "prompt_sha256": "prompt", "seed": 42,
        "models": {"planner": {"model_id": "gpt", "revision": "api"}},
    }), encoding="utf-8")

    report = _MODULE.compare_runs(baseline, candidate)

    identity = report["regression_identity"]
    assert identity["model_identity_match"] is False
    assert identity["same_condition"] is False
