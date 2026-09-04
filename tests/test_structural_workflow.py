import json
from pathlib import Path

import numpy as np

from worldclaw_oss.pipeline import Pipeline
from worldclaw_oss.structural_workflow import build_validation_bundle, render_additional_views


LOCK = Path(__file__).resolve().parents[1] / "models.lock.json"


def test_structural_trail_lake_workflow_materializes_and_executes(tmp_path):
    run = Pipeline(
        tmp_path / "forest_structural",
        LOCK,
        "synthetic",
        17,
        "forest trail structural",
        ["pytest"],
    ).run()
    work = run / "work"
    plan = json.loads((work / "structural_plan.json").read_text(encoding="utf-8"))
    assert {item["semantic_category"] for item in plan["features"]} == {"trail", "lake"}

    for name in (
        "structural_agent_input/scene_plan.json",
        "structural_agent_input/feature_plan.json",
        "structural_agent_input/layout/feature_mask.npy",
        "structural_agent_input/terrain/terrain.npz",
        "structural_agent_input/existing_structures/structural_response.json",
        "structural_agent_input/structural_task.json",
        "structural_agent_output/structural_plan.json",
        "structural_agent_output/structural_worker.py",
        "structural_agent_output/structural_validator.py",
    ):
        assert (work / name).is_file(), name

    branch = json.loads((work / "structural_branch.json").read_text(encoding="utf-8"))
    assert branch["metrics"]["trail_count"] == 1
    assert branch["metrics"]["lake_count"] == 1
    assert branch["terrain_modified"] is True
    deterministic = json.loads((work / "structural_validation.json").read_text(encoding="utf-8"))
    assert deterministic["status"] == "pass"

    original = np.load(work / "terrain.npz")
    integrated = np.load(work / "terrain_structural.npz")
    assert np.array_equal(integrated["vertices"][:, :2], original["vertices"][:, :2])
    assert not np.array_equal(integrated["vertices"][:, 2], original["vertices"][:, 2])

    bundle = work / "validation_bundle"
    assert (bundle / "renders/fixed/top_down.png").stat().st_size > 0
    assert (bundle / "renders/fixed/oblique_overview.png").stat().st_size > 0
    assert any((bundle / "renders/adaptive").glob("*.png"))
    final = json.loads((work / "structural_final_validation.json").read_text(encoding="utf-8"))
    assert final["status"] == "pass"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["structural_final_validation"]["status"] == "pass"


def test_structural_adaptive_follow_up_persists_requested_views(tmp_path):
    run = Pipeline(
        tmp_path / "forest_structural_follow_up",
        LOCK,
        "synthetic",
        19,
        "forest trail structural",
        ["pytest"],
    ).run()
    work = run / "work"
    views = json.loads((work / "validation_views.json").read_text(encoding="utf-8"))
    deterministic = json.loads((work / "structural_validation.json").read_text(encoding="utf-8"))
    plan = json.loads((work / "structural_plan.json").read_text(encoding="utf-8"))
    scene_plan = json.loads((work / "scene_plan.json").read_text(encoding="utf-8"))
    request = [{
        "id": "lit_trail_shoreline_closeup",
        "feature_ids": ["forest_trail_001", "lake_lake_001"],
        "position": [35.0, 35.0, 12.0],
        "target": [10.0, 10.0, 0.0],
        "reason": "verify the shoreline termination",
    }]
    result = render_additional_views(work, views, request)
    assert result["needed"] is True
    assert (work / "validation_bundle/renders/additional/lit_trail_shoreline_closeup.png").stat().st_size > 0
    views["additional"] = result["requested"]
    build_validation_bundle(
        work, views, deterministic, plan, scene_plan, result,
        result["requested"],
    )
    persisted = json.loads((work / "validation_bundle/metadata/validation_views.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in persisted["additional"]] == ["lit_trail_shoreline_closeup"]
