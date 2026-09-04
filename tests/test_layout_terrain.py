import numpy as np
import json

from worldclaw_oss.layout import (
    deterministic_layout,
    normalize_scene_plan,
    sample_layout,
    stable_seed,
    synthetic_plan,
    target_world_size_from_environment,
)
from worldclaw_oss.terrain import (
    boundary_discontinuity,
    generate_terrain,
    validate_regional_detail_preserves_macro,
)
from worldclaw_oss.terrain_macro import (
    TerrainMacroPlanner,
    coerce_macro_plan,
    derive_macro_plan,
    generate_macro_height,
    macro_vertices,
    macro_plan_quality_issues,
    replan_macro_plan,
    validate_macro_height,
)
from worldclaw_oss.schemas import Polygon, TerrainLandform, TerrainMacroPlan, Vec2
from worldclaw_oss.terrain import TerrainResult
import worldclaw_oss.pipeline as pipeline_module


def test_layout_is_deterministic_and_normalized():
    plan = synthetic_plan("desert city")
    a = deterministic_layout(plan, 64)
    b = deterministic_layout(plan, 64)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.weights, b.weights)
    assert np.allclose(a.weights.sum(axis=0), 1.0)
    assert set(np.unique(a.labels)) == {0, 1, 2}


def test_sample_layout_is_seeded_diverse_and_auditable():
    plan = synthetic_plan("forest lake cabins trail")
    first = sample_layout(plan, 96, seed=42, candidate_count=4)
    replay = sample_layout(plan, 96, seed=42, candidate_count=4)
    alternate = sample_layout(plan, 96, seed=43, candidate_count=4)
    assert np.array_equal(first.labels, replay.labels)
    assert np.array_equal(first.weights, replay.weights)
    assert first.generation == replay.generation
    assert not np.array_equal(first.labels, alternate.labels)
    assert first.generation["seed"] == 42
    assert first.generation["candidate_count"] == 4
    assert len(first.generation["candidate_scores"]) == 4
    assert first.generation["selected_metrics"]["valid"] is True
    assert first.generation["topology_valid"] is True


def test_soft_region_terrain_is_continuous():
    plan = synthetic_plan("castle")
    layout = deterministic_layout(plan, 64)
    a = generate_terrain(plan, layout, stable_seed("castle", 42))
    b = generate_terrain(plan, layout, stable_seed("castle", 42))
    assert np.array_equal(a.height, b.height)
    assert np.isfinite(a.height).all()
    assert boundary_discontinuity(a.height, layout.labels) < 3.0
    assert len(a.triangles) == 2 * 63 * 63


def test_regional_detail_preserves_planner_macro_field():
    plan = synthetic_plan("forest lake")
    layout = deterministic_layout(plan, 64)
    macro_plan = derive_macro_plan(plan)
    macro = generate_macro_height(macro_plan, 64, layout_weights=layout.weights, region_ids=layout.region_ids)
    final = generate_terrain(plan, layout, stable_seed("forest lake", 42), macro_height=macro).height
    metrics = validate_regional_detail_preserves_macro(macro, final)
    assert metrics["macro_composition_preserved"] is True
    assert metrics["regional_to_macro_rms_ratio"] <= metrics["max_detail_ratio"]


def test_regional_terrain_matches_positive_plan_coordinate_frame():
    scene = synthetic_plan("forest lake")
    regions = []
    for region in scene.regions:
        regions.append(region.model_copy(update={
            "center": region.center.model_copy(update={
                "x": region.center.x + 50.0,
                "y": region.center.y + 50.0,
            }),
            "polygon": region.polygon.model_copy(update={
                "points": [point.model_copy(update={"x": point.x + 50.0, "y": point.y + 50.0}) for point in region.polygon.points],
            }),
        }))
    positive_scene = scene.model_copy(update={"regions": regions})
    layout = deterministic_layout(positive_scene, 64)
    macro_plan = derive_macro_plan(positive_scene)
    macro = generate_macro_height(
        macro_plan, 64, layout_weights=layout.weights, region_ids=layout.region_ids,
    )
    final = generate_terrain(
        positive_scene, layout, stable_seed("positive forest lake", 42), macro_height=macro,
    ).height
    metrics = validate_regional_detail_preserves_macro(macro, final)
    assert metrics["macro_composition_preserved"] is True
    terrain = generate_terrain(
        positive_scene, layout, stable_seed("positive terrain frame", 42), macro_height=macro,
    )
    assert np.allclose(terrain.vertices[:, 0].min(), -50.0)
    assert np.allclose(terrain.vertices[:, 0].max(), 50.0)
    assert np.allclose(terrain.vertices[:, 1].min(), -50.0)
    assert np.allclose(terrain.vertices[:, 1].max(), 50.0)


def test_regional_detail_validator_rejects_broad_topology_shift():
    macro = np.zeros((32, 32), dtype=np.float64)
    macro[:, :16] = 10.0
    shifted = np.roll(macro, 8, axis=1)
    metrics = validate_regional_detail_preserves_macro(macro, shifted)
    assert metrics["macro_composition_preserved"] is False


def test_regional_detail_validator_ignores_uniform_datum_shift():
    macro = np.zeros((32, 32), dtype=np.float64)
    macro[:, :16] = 10.0
    shifted = macro + 4.0
    metrics = validate_regional_detail_preserves_macro(macro, shifted)
    assert metrics["macro_composition_preserved"] is True
    assert metrics["global_offset_m"] == 4.0
    assert metrics["low_frequency_error_m"] == 0.0


def test_live_plan_normalization_scales_geometry_terrain_and_counts():
    plan = synthetic_plan("forest lake")
    raw = plan.model_copy(update={
        "world_size_m": (1000.0, 1000.0),
        "regions": [
            region.model_copy(update={
                "center": region.center.model_copy(update={
                    "x": region.center.x * 10,
                    "y": region.center.y * 10,
                }),
                "polygon": region.polygon.model_copy(update={
                    "points": [
                        point.model_copy(update={"x": point.x * 10, "y": point.y * 10})
                        for point in region.polygon.points
                    ]
                }),
                "objects": [obj.model_copy(update={"count": 1200}) for obj in region.objects],
            })
            for region in plan.regions
        ],
        "terrain": [
            spec.model_copy(update={
                "operators": [
                    operator.model_copy(update={"scale": 80.0})
                    for operator in spec.operators
                ]
            })
            for spec in plan.terrain
        ],
    })
    normalized = normalize_scene_plan(raw, (100.0, 100.0))
    assert normalized.world_size_m == (100.0, 100.0)
    assert normalized.regions[0].center == plan.regions[0].center
    assert normalized.regions[0].polygon.points == plan.regions[0].polygon.points
    assert all(obj.count == 12 for region in normalized.regions for obj in region.objects)
    assert all(operator.scale == 8.0 for spec in normalized.terrain for operator in spec.operators)


def test_target_world_size_environment_parser(monkeypatch):
    monkeypatch.setenv("WORLDCLAW_TARGET_WORLD_SIZE_M", "80x120")
    assert target_world_size_from_environment() == (80.0, 120.0)


def test_macro_plan_quality_gate_rejects_schema_valid_placeholder_plan():
    scene = synthetic_plan("forest lake cabins trail")
    plan = derive_macro_plan(scene).model_copy(update={
        "landforms": [
            landform.model_copy(update={"height_m": 0.0, "depth_m": 0.001 if landform.type == "basin" else 0.0, "target_elevation_m": None})
            for landform in derive_macro_plan(scene).landforms
        ]
    })
    failures = macro_plan_quality_issues(plan, scene)
    assert "plan has no meaningful macro relief" in failures
    assert "plan has no meaningful macro relief" not in macro_plan_quality_issues(
        plan, scene, stage="terrain_base"
    )


def test_macro_plan_quality_gate_requires_elevated_framing_for_surrounded_water():
    scene = synthetic_plan("forest lake cabins trail")
    scene = scene.model_copy(update={
        "regions": [
            region.model_copy(update={"spatial_relations": ["surrounded by forest"]})
            if region.id == "lake" else region
            for region in scene.regions
        ]
    })
    base = derive_macro_plan(scene)
    coastal_only = base.model_copy(update={
        "landforms": [
            landform for landform in base.landforms
            if landform.type in {"basin", "bench", "coastal_slope", "saddle"}
        ]
    })
    assert "surrounding-water semantics require elevated framing landforms" in macro_plan_quality_issues(coastal_only, scene)


def test_generic_basin_fallback_adds_framing_for_collapsed_region_centers():
    scene = synthetic_plan("forest lake cabins trail")
    anchor = scene.regions[0].center
    collapsed = scene.model_copy(update={
        "regions": [region.model_copy(update={"center": anchor}) for region in scene.regions]
    })
    plan = derive_macro_plan(collapsed)
    ridges = [landform for landform in plan.landforms if landform.type == "ridge"]
    assert len(ridges) >= 4
    assert max(landform.height_m for landform in ridges) > 0.0
    benches = [landform for landform in plan.landforms if landform.type == "bench"]
    assert benches and max(benches[0].radius_m) <= 32.0
    layout = deterministic_layout(collapsed, 128)
    height = generate_macro_height(plan, 128, layout_weights=layout.weights, region_ids=layout.region_ids)
    assert validate_macro_height(height, plan, layout.labels)["valid"] is True


def test_positive_coordinate_macro_validation_and_vertices_share_plan_frame():
    plan = TerrainMacroPlan(
        world_size_m=(100.0, 100.0),
        landforms=[
            TerrainLandform(
                id="lake_basin", type="basin", center=Vec2(x=50.0, y=50.0),
                radius_m=20.0, depth_m=5.0,
            ),
            TerrainLandform(
                id="north_ridge", type="ridge",
                control_points=[Vec2(x=0.0, y=85.0), Vec2(x=100.0, y=85.0)],
                width_m=10.0, height_m=10.0,
            ),
        ],
    )
    height = generate_macro_height(plan, 64)
    metrics = validate_macro_height(height, plan)
    vertices, _ = macro_vertices(height, plan.world_size_m, plan)
    assert metrics["lake_basin_containment"] is True
    assert metrics["ridge_height"] > 0.0
    assert vertices[:, 0].min() >= 0.0
    assert vertices[:, 1].min() >= 0.0


def test_polygon_basin_floor_cap_is_smooth_at_boundary():
    plan = TerrainMacroPlan(
        world_size_m=(100.0, 100.0),
        landforms=[TerrainLandform(
            id="lake_basin", type="basin", center=Vec2(x=50.0, y=50.0),
            polygon=Polygon(points=[
                Vec2(x=25.0, y=35.0), Vec2(x=40.0, y=25.0),
                Vec2(x=70.0, y=30.0), Vec2(x=78.0, y=55.0),
                Vec2(x=55.0, y=75.0), Vec2(x=30.0, y=65.0),
            ]),
            depth_m=6.0, falloff_m=8.0,
        )],
    )
    height = generate_macro_height(plan, 128)
    adjacent = np.concatenate((
        np.abs(np.diff(height, axis=0)).ravel(),
        np.abs(np.diff(height, axis=1)).ravel(),
    ))
    assert float(adjacent.max()) < 0.8


def test_multiple_basins_use_nearest_external_shore_when_local_ring_is_empty():
    def rectangle(x0, x1):
        return Polygon(points=[
            Vec2(x=x0, y=20.0), Vec2(x=x1, y=20.0),
            Vec2(x=x1, y=80.0), Vec2(x=x0, y=80.0),
        ])

    plan = TerrainMacroPlan(
        world_size_m=(100.0, 100.0),
        landforms=[
            TerrainLandform(
                id="basin_a", type="basin", center=Vec2(x=27.5, y=50.0),
                polygon=rectangle(10.0, 45.0), depth_m=8.0, falloff_m=0.1,
            ),
            TerrainLandform(
                id="basin_b", type="basin", center=Vec2(x=72.5, y=50.0),
                polygon=rectangle(55.0, 90.0), depth_m=8.0, falloff_m=0.1,
            ),
        ],
    )
    height = generate_macro_height(plan, 32)
    metrics = validate_macro_height(height, plan)
    assert metrics["lake_basin_containment"] is True
    assert [item["shore_reference_source"] for item in metrics["basin_elevation_checks"]] == [
        "nearest_external", "nearest_external",
    ]


def test_basin_containment_remains_invalid_without_any_external_land():
    plan = TerrainMacroPlan(
        world_size_m=(100.0, 100.0),
        landforms=[TerrainLandform(
            id="world_basin", type="basin", center=Vec2(x=50.0, y=50.0),
            polygon=Polygon(points=[
                Vec2(x=0.0, y=0.0), Vec2(x=100.0, y=0.0),
                Vec2(x=100.0, y=100.0), Vec2(x=0.0, y=100.0),
            ]), depth_m=4.0, falloff_m=2.0,
        )],
    )
    height = generate_macro_height(plan, 32)
    metrics = validate_macro_height(height, plan)
    assert metrics["lake_basin_containment"] is False
    assert metrics["basin_elevation_checks"][0]["shore_reference_source"] == "unavailable"


def test_macro_replan_promotes_zero_amplitudes_without_seed_change():
    scene = synthetic_plan("forest lake cabins trail")
    plan = derive_macro_plan(scene).model_copy(update={
        "landforms": [
            landform.model_copy(update={"height_m": 0.0, "depth_m": 0.0})
            for landform in derive_macro_plan(scene).landforms
        ]
    })
    replanned = replan_macro_plan(plan, {"status": "REPLAN", "failures": ["regional_detail_preservation"]})
    assert any(max(abs(lf.height_m), abs(lf.depth_m)) > 0.0 for lf in replanned.landforms)
    assert replanned.composition_constraints[-1] == "targeted replan applied"


def test_radial_replan_uses_short_asymmetric_segments_and_local_hills():
    scene = synthetic_plan("forest lake cabins trail")
    plan = derive_macro_plan(scene)
    original_ridges = {
        landform.id: landform
        for landform in plan.landforms
        if landform.type == "ridge" and len(landform.control_points) >= 2
    }
    replanned = replan_macro_plan(
        plan,
        {"status": "REPLAN", "failures": ["near-concentric radial basin/ridge form"]},
    )
    assert any(landform.id.startswith("replan_hill_") for landform in replanned.landforms)
    changed_ridges = [
        landform for landform in replanned.landforms
        if landform.id in original_ridges
        and len(landform.control_points) >= 2
        and len(original_ridges[landform.id].control_points) >= 2
    ]
    assert changed_ridges
    assert all(
        np.hypot(
            ridge.control_points[-1].x - ridge.control_points[0].x,
            ridge.control_points[-1].y - ridge.control_points[0].y,
        )
        < np.hypot(
            original_ridges[ridge.id].control_points[-1].x
            - original_ridges[ridge.id].control_points[0].x,
            original_ridges[ridge.id].control_points[-1].y
            - original_ridges[ridge.id].control_points[0].y,
        )
        for ridge in changed_ridges
    )
    hills = [landform for landform in replanned.landforms if landform.id.startswith("replan_hill_")]
    assert all(
        hill.center is not None
        and any(
            np.hypot(
                hill.center.x - (ridge.control_points[0].x + ridge.control_points[-1].x) * 0.5,
                hill.center.y - (ridge.control_points[0].y + ridge.control_points[-1].y) * 0.5,
            ) > 0.0
            for ridge in changed_ridges
        )
        for hill in hills
    )


def test_gpt_placeholder_plan_uses_explicit_generic_fallback():
    scene = synthetic_plan("forest lake cabins trail")
    base = derive_macro_plan(scene)
    placeholder = base.model_copy(update={
        "landforms": [
            landform.model_copy(update={"height_m": 0.0, "depth_m": 0.001 if landform.type == "basin" else 0.0})
            for landform in base.landforms
        ]
    }).model_dump(mode="json")

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def json_chat(self, *args, **kwargs):
            self.calls += 1
            return placeholder

    client = FakeClient()
    plan, metadata = TerrainMacroPlanner(client, type("Model", (), {"model_id": "test"})()).plan(scene, 42)
    assert client.calls == 2
    assert metadata["provider"] == "semantic_fallback_after_gpt_quality_gate"
    assert max(abs(lf.height_m) for lf in plan.landforms) > 0.0


def test_macro_plan_migrates_structured_geometry_and_elevation_variants():
    payload = {
        "world_size_m": [100, 100],
        "landforms": [
            {
                "region_id": "lake_region",
                "geometry": {"polygon": [{"x": 25, "y": 25}, {"x": 75, "y": 25}, {"x": 75, "y": 75}]},
                "elevation": {"floor_m": 0},
            },
            {
                "region_id": "outer_ridge",
                "geometry": {"centerline": [{"x": 0, "y": 20}, {"x": 100, "y": 20}]},
                "elevation": {"relative_to": "plateau", "direction": "above"},
                "role": "elevated framing",
            },
        ],
    }
    plan = coerce_macro_plan(payload)
    assert plan.landforms[0].type == "basin"
    assert plan.landforms[0].depth_m > 0
    assert plan.landforms[1].type == "ridge"
    assert len(plan.landforms[1].control_points) == 2
    assert plan.landforms[1].height_m > 0


def test_gpt_replan_receives_original_context_and_visual_feedback(tmp_path):
    scene = synthetic_plan("forest lake cabins trail")
    current = derive_macro_plan(scene)
    calls = []

    class FakeClient:
        def json_chat(self, model, system, request, schema, seed):
            calls.append(json.loads(request))
            return current.model_dump(mode="json", exclude_none=True)

    planner = TerrainMacroPlanner(FakeClient(), type("Model", (), {"model_id": "test"})())
    candidate, metadata = planner.replan(
        scene,
        current,
        {"status": "REPLAN", "failures": ["river too straight"], "recommendations": ["use a meandering channel"], "metrics": {"p95_slope": 42}},
        seed=42,
        layout_summary={"shape": [8, 8], "region_ids": ["forest", "lake", "clearing"]},
        planner_context={
            "scene_plan": {"theme": scene.theme},
            "world_spec": {"world_size_m": [100, 100]},
            "response_json_hard_constraints": [{
                "path": "terrain_base_validation_recheck/response.json",
                "payload": {"failures": ["edge jump"], "recommendations": ["smooth edge"]},
            }],
        },
        request_path=tmp_path / "replan_request.json",
        response_path=tmp_path / "replan_response.json",
    )

    assert metadata["provider"] == "gpt_replan"
    assert candidate.world_size_m == current.world_size_m
    payload = calls[0]
    assert payload["original_terrain_inputs"]["world_spec"]["world_size_m"] == [100, 100]
    assert payload["current_macro_plan"] == current.model_dump(mode="json", exclude_none=True)
    assert payload["visual_feedback"]["failures"] == ["river too straight"]
    assert payload["visual_feedback"]["recommendations"] == ["use a meandering channel"]
    assert payload["hard_constraints"]["response_json_is_mandatory"] is True
    assert payload["hard_constraints"]["response_json"][0]["payload"]["failures"] == ["edge jump"]
    assert (tmp_path / "replan_request.json").is_file()
    assert (tmp_path / "replan_response.json").is_file()


def test_replan_context_includes_first_planner_exchange(tmp_path):
    from worldclaw_oss.pipeline import _terrain_planner_context

    scene = synthetic_plan("forest lake")
    planner_input = tmp_path / "terrain_planner_input"
    planner_input.mkdir()
    (planner_input / "scene_plan.json").write_text(
        scene.model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "terrain_macro_planner_request.json").write_text(
        json.dumps({"scene_plan": {"theme": scene.theme}, "layout": {"shape": [8, 8]}}),
        encoding="utf-8",
    )
    (tmp_path / "terrain_macro_planner_response.json").write_text(
        json.dumps({"world_size_m": [100, 100], "landforms": []}),
        encoding="utf-8",
    )
    validation_dir = tmp_path / "terrain_base_validation_recheck_20260904"
    validation_dir.mkdir()
    (validation_dir / "response.json").write_text(
        json.dumps({"status": "REPLAN", "failures": ["edge jump"], "recommendations": ["smooth edge"]}),
        encoding="utf-8",
    )
    (tmp_path / "terrain_visual_validation_response.json").write_text(
        json.dumps({"status": "REPLAN", "failures": ["diagonal step"], "recommendations": ["smooth transition"]}),
        encoding="utf-8",
    )
    context = _terrain_planner_context(tmp_path, scene, {"shape": [8, 8]})
    assert context["terrain_macro_planner_request"]["layout"]["shape"] == [8, 8]
    assert context["terrain_macro_planner_response"]["world_size_m"] == [100, 100]
    assert context["response_json_hard_constraints"][0]["payload"]["failures"] == ["edge jump"]
    assert any(
        item["payload"].get("failures") == ["diagonal step"]
        for item in context["response_json_hard_constraints"]
    )


def test_regional_recovery_keeps_resolution_seed_and_uses_generic_fallback(tmp_path, monkeypatch):
    scene = synthetic_plan("forest lake cabins trail")
    layout = deterministic_layout(scene, 16)
    labels, weights = layout.labels, layout.weights
    region_ids = tuple(layout.region_ids)
    macro = np.zeros((16, 16), dtype=np.float32)
    (tmp_path / "terrain").mkdir()
    (tmp_path / "terrain_macro_plan.json").write_text(
        derive_macro_plan(scene).model_dump_json(), encoding="utf-8"
    )

    calls = []
    validations = iter((False, False, True))

    def fake_generate(scene_value, layout_value, seed, macro_height=None):
        calls.append((seed, macro_height.shape))
        height = np.asarray(macro_height, dtype=np.float32)
        vertices = np.zeros((height.size, 3), dtype=np.float32)
        triangles = np.zeros((0, 3), dtype=np.uint32)
        return TerrainResult(height, vertices, triangles)

    def fake_validate(macro_height, final_height):
        preserved = next(validations)
        return {
            "macro_composition_preserved": preserved,
            "regional_to_macro_rms_ratio": 0.1 if preserved else 0.9,
            "low_frequency_error_ratio": 0.05 if preserved else 0.8,
        }

    replan_resolutions = []
    fallback_resolutions = []

    def fake_replan(work, labels_value, weights_value, ids_value, resolution, feedback):
        replan_resolutions.append(resolution)
        plan = TerrainMacroPlan.model_validate_json(
            (work / "terrain_macro_plan.json").read_text(encoding="utf-8")
        ).model_copy(update={"composition_constraints": ["targeted replan applied"]})
        (work / "terrain_macro_plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
        updated = np.ones((resolution, resolution), dtype=np.float32)
        np.save(work / "macro_height.npy", updated)
        return plan, updated, {"valid": True}

    def fake_fallback(work, scene_value, labels_value, weights_value, ids_value, resolution):
        fallback_resolutions.append(resolution)
        plan = derive_macro_plan(scene_value).model_copy(update={"composition_constraints": ["generic fallback"]})
        (work / "terrain_macro_plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
        updated = np.full((resolution, resolution), 2.0, dtype=np.float32)
        np.save(work / "macro_height.npy", updated)
        return plan, updated, {"valid": True}

    monkeypatch.setattr(pipeline_module, "generate_terrain", fake_generate)
    monkeypatch.setattr(pipeline_module, "validate_regional_detail_preserves_macro", fake_validate)
    monkeypatch.setattr(pipeline_module, "_replan_macro_artifacts", fake_replan)
    monkeypatch.setattr(pipeline_module, "_fallback_macro_artifacts", fake_fallback)

    value, recovered_macro, preservation = pipeline_module._recover_regional_detail(
        tmp_path, scene, layout, labels, weights, region_ids, 12345, macro,
    )

    assert value.height.shape == (16, 16)
    assert recovered_macro.shape == (16, 16)
    assert preservation["macro_composition_preserved"] is True
    assert replan_resolutions == [16]
    assert fallback_resolutions == [16]
    assert [seed for seed, _ in calls] == [12345, 12345, 12345]
    history = json.loads((tmp_path / "terrain" / "logs" / "terrain_retry_history.json").read_text(encoding="utf-8"))
    assert len(history["attempts"]) == 2
    assert history["seed_changes"] == 0
    final_plan = json.loads((tmp_path / "terrain_macro_plan.json").read_text(encoding="utf-8"))
    assert final_plan["composition_constraints"] == ["generic fallback"]
