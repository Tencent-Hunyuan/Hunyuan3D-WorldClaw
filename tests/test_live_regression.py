import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_live_regression.py"
_SPEC = importlib.util.spec_from_file_location("run_live_regression", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
validate_regression_identity = _MODULE.validate_regression_identity
run_regression = _MODULE.run_regression
validate_candidate_artifacts = _MODULE.validate_candidate_artifacts


def _baseline(path, prompt_file, seed=42):
    path.mkdir()
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    (path / "run_manifest.json").write_text(json.dumps({
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "seed": seed, "models": {"planner": {"model_id": "planner-test"}},
    }), encoding="utf-8")


def test_live_regression_identity_requires_same_prompt_seed_and_output(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("same scene", encoding="utf-8")
    baseline = tmp_path / "baseline"
    _baseline(baseline, prompt)
    identity = validate_regression_identity(baseline, prompt, 42, tmp_path / "candidate")
    assert identity["prompt_sha256"] == hashlib.sha256(b"same scene").hexdigest()
    assert identity["baseline_models_present"] is True
    with pytest.raises(ValueError, match="seed"):
        validate_regression_identity(baseline, prompt, 7, tmp_path / "candidate2")


def test_live_regression_default_prompt_is_canonical():
    assert _MODULE.DEFAULT_LIVE_PROMPT_FILE.is_file()
    assert _MODULE.DEFAULT_LIVE_PROMPT_FILE.read_text(encoding="utf-8").strip() == (
        "A forest region with lakes, rivers, cabins and trails"
    )


def test_live_regression_rejects_nonempty_candidate_output(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("same scene", encoding="utf-8")
    baseline = tmp_path / "baseline"
    _baseline(baseline, prompt)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "old.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or absent"):
        validate_regression_identity(baseline, prompt, 42, candidate)


def test_live_regression_persists_comparison_inside_candidate_run(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("same scene", encoding="utf-8")
    baseline = tmp_path / "baseline"
    _baseline(baseline, prompt)
    candidate = tmp_path / "candidate"

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, check, env, text, capture_output):
        candidate.mkdir()
        stages = {name: {"status": "complete"} for name in _MODULE._REQUIRED_LIVE_STAGES}
        (candidate / "run_manifest.json").write_text(json.dumps({
            "mode": "live", "synthetic_outputs": False, "stage_results": stages,
        }), encoding="utf-8")
        for relative in _MODULE._REQUIRED_CANDIDATE_ARTIFACTS:
            path = candidate / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b"candidate")
        return Completed()

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(_MODULE, "compare_runs", lambda _baseline, _candidate: {
        "status": "comparison_ready",
        "regression_identity": {"same_condition": True},
    })
    result = run_regression(
        baseline=baseline, prompt_file=prompt, output=candidate, seed=42,
        models_lock=tmp_path / "models.lock.json",
    )
    assert result["status"] == "comparison_ready"
    comparison_path = candidate / "terrain_regression_comparison.json"
    assert comparison_path.is_file()
    assert json.loads(comparison_path.read_text(encoding="utf-8"))["status"] == "comparison_ready"


def test_candidate_artifact_gate_rejects_missing_outputs(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(ValueError, match="incomplete or external"):
        validate_candidate_artifacts(candidate)
