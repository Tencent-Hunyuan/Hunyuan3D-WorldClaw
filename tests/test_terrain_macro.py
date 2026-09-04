import json

import numpy as np
import pytest

from worldclaw_oss.layout import deterministic_layout, synthetic_plan
from worldclaw_oss.terrain_macro import (
    derive_macro_plan,
    generate_macro_height,
    summarize_layout,
    validate_macro_height,
    visual_validate_terrain,
    write_validation_bundle,
)
from workers.terrain_macro_worker import run as run_macro_worker


def test_layout_summary_is_serializable_and_informative():
    scene = synthetic_plan("forest lake")
    layout = deterministic_layout(scene, 16)
    summary = summarize_layout(layout.labels, layout.weights, layout.region_ids)
    assert summary["shape"] == [16, 16]
    assert summary["region_ids"] == list(layout.region_ids)
    assert len(summary["weights"]) == len(scene.regions)
    assert all("mass" in item for item in summary["weights"])
    json.dumps(summary)


def test_layout_weights_act_as_soft_macro_constraint():
    scene = synthetic_plan("forest lake")
    layout = deterministic_layout(scene, 32)
    plan = derive_macro_plan(scene)
    unconstrained = generate_macro_height(plan, 32)
    constrained = generate_macro_height(
        plan, 32, layout_weights=layout.weights, region_ids=layout.region_ids
    )
    assert constrained.shape == unconstrained.shape
    assert not np.array_equal(constrained, unconstrained)


def test_layout_semantic_compatibility_and_boundary_gate():
    scene = synthetic_plan("forest lake cabins trail")
    layout = deterministic_layout(scene, 128)
    plan = derive_macro_plan(scene)
    height = generate_macro_height(
        plan, 128, layout_weights=layout.weights, region_ids=layout.region_ids
    )
    metrics = validate_macro_height(
        height, plan, layout.labels, layout.weights, layout.region_ids
    )
    assert metrics["layout_shape_valid"] is True
    assert metrics["layout_boundary_max_jump_m"] <= metrics["layout_boundary_jump_limit_m"]
    assert metrics["layout_boundary_p95_slope_deg"] <= metrics["layout_boundary_slope_limit_deg"]
    assert metrics["landform_semantic_alignment"]["lake_basin"]["region_id"] == "lake"
    assert metrics["landform_semantic_alignment"]["lake_basin"]["alignment"] >= 0.5
    assert metrics["valid"] is True


def test_layout_weight_shape_is_rejected():
    scene = synthetic_plan("forest lake")
    layout = deterministic_layout(scene, 16)
    plan = derive_macro_plan(scene)
    with pytest.raises(ValueError, match="layout_weights shape"):
        generate_macro_height(
            plan, 16, layout_weights=layout.weights[:, :-1], region_ids=layout.region_ids
        )


def test_validation_bundle_persists_metrics(tmp_path):
    scene = synthetic_plan("forest lake")
    plan = derive_macro_plan(scene)
    height = generate_macro_height(plan, 32)
    metrics = validate_macro_height(height, plan)
    paths = write_validation_bundle(tmp_path, height, metrics=metrics)
    assert (tmp_path / "terrain_metrics.json").is_file()
    assert json.loads((tmp_path / "terrain_metrics.json").read_text()) == metrics
    assert "terrain_metrics.json" in paths


def test_visual_validator_can_send_render_images_and_persist_io(tmp_path):
    scene = synthetic_plan("castle")
    plan = derive_macro_plan(scene)
    height = generate_macro_height(plan, 16)
    write_validation_bundle(tmp_path, height)

    class VisionClient:
        def __init__(self):
            self.paths = []

        def vision_json(self, **kwargs):
            self.paths = list(kwargs["image_paths"])
            return {"status": "PASS", "failures": [], "recommendations": []}

    client = VisionClient()
    result = visual_validate_terrain(
        tmp_path,
        {"valid": True, "landform_count": 1},
        client=client,
        model=type("Model", (), {"model_id": "vision-test"})(),
        scene_plan=scene.model_dump(mode="json"),
        request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json",
    )
    assert result["status"] == "PASS"
    assert len(client.paths) >= 7
    assert (tmp_path / "request.json").is_file()
    assert json.loads((tmp_path / "response.json").read_text())["status"] == "PASS"
    request = json.loads((tmp_path / "request.json").read_text())
    assert request["stage_contract"]["stage"] == "terrain_base"
    assert "lake basins" in request["stage_contract"]["downstream_recarves"]
    assert "visible lake or river topology" in request["stage_contract"]["do_not_gate_on"]
    assert "base terrain is finite and present" in request["questions"]


def test_macro_worker_consumes_layout_and_planner_inputs(tmp_path):
    scene = synthetic_plan("forest lake")
    layout = deterministic_layout(scene, 16)
    plan = derive_macro_plan(scene)
    planner_input = tmp_path / "planner_input"
    planner_input.mkdir()
    (planner_input / "terrain_macro_plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    layout_dir = planner_input / "layout"
    layout_dir.mkdir()
    np.save(layout_dir / "layout_labels.npy", layout.labels)
    np.save(layout_dir / "layout_weights.npy", layout.weights)
    np.save(layout_dir / "masks_forest.npy", layout.labels == 1)
    (planner_input / "world_spec.json").write_text(
        json.dumps({"world_size_m": list(plan.world_size_m)}), encoding="utf-8"
    )
    (planner_input / "terrain_constraints.json").write_text(
        json.dumps({"explicit_constraints": []}), encoding="utf-8"
    )
    result = run_macro_worker({
        "planner_input": str(planner_input),
        "terrain_macro_plan": str(planner_input / "terrain_macro_plan.json"),
        "output_dir": str(tmp_path / "output"),
        "resolution": 16,
        "region_ids": list(layout.region_ids),
    })
    assert result["status"] == "ok"
    assert result["inputs"]["layout_weights"] is not None
    assert np.load(tmp_path / "output" / "macro_height.npy").shape == (16, 16)
