import numpy as np

from worldclaw_oss.layout import (
    deterministic_layout,
    normalize_scene_plan,
    stable_seed,
    synthetic_plan,
    target_world_size_from_environment,
)
from worldclaw_oss.terrain import boundary_discontinuity, generate_terrain


def test_layout_is_deterministic_and_normalized():
    plan = synthetic_plan("desert city")
    a = deterministic_layout(plan, 64)
    b = deterministic_layout(plan, 64)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.weights, b.weights)
    assert np.allclose(a.weights.sum(axis=0), 1.0)
    assert set(np.unique(a.labels)) == {0, 1, 2}


def test_soft_region_terrain_is_continuous():
    plan = synthetic_plan("castle")
    layout = deterministic_layout(plan, 64)
    a = generate_terrain(plan, layout, stable_seed("castle", 42))
    b = generate_terrain(plan, layout, stable_seed("castle", 42))
    assert np.array_equal(a.height, b.height)
    assert np.isfinite(a.height).all()
    assert boundary_discontinuity(a.height, layout.labels) < 3.0
    assert len(a.triangles) == 2 * 63 * 63


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
