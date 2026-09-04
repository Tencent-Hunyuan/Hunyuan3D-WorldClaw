import json

import numpy as np

from worldclaw_oss.asset_semantics import generation_class, is_structural_feature, structural_subtype
from worldclaw_oss.structural import (
    StructuralFeatureAgent,
    _clip_centerline_to_world,
    _clip_mesh_xy_to_world,
    build_structural_geometry,
    validate_structural_branch,
)


def _plan():
    return {
        "world_size_m": [20.0, 20.0],
        "regions": [{
            "id": "valley", "function": "woodland valley",
            "center": {"x": 0.0, "y": 0.0},
            "polygon": {"points": [{"x": -8.0, "y": -8.0}, {"x": 8.0, "y": -8.0}, {"x": 8.0, "y": 8.0}, {"x": -8.0, "y": 8.0}]},
            "objects": [
                {"category": "river", "count": 1, "asset_role": "structural_feature", "instance_strategy": "path"},
                {"category": "lake", "count": 1, "asset_role": "structural_feature", "instance_strategy": "region"},
                {"category": "tree", "count": 2, "asset_role": "reusable_prototype", "instance_strategy": "scatter"},
            ],
        }],
    }


def test_structural_routes_collapse_legacy_categories():
    assert is_structural_feature("trail")
    assert is_structural_feature("lake")
    assert generation_class("river") == "structural_feature"
    assert structural_subtype("trail") == "linear"
    assert structural_subtype("river") == "regional_surface"
    assert structural_subtype("lake") == "regional_surface"


def test_linear_feature_centerline_reserves_swept_width():
    centerline = np.asarray([[-20.0, -20.0], [20.0, 20.0]], dtype=float)
    clipped = _clip_centerline_to_world(centerline, 8.0, (20.0, 20.0))
    assert np.all(clipped >= -6.0)
    assert np.all(clipped <= 6.0)


def test_linear_mesh_is_clipped_to_world_bounds():
    vertices = np.asarray([[-12.0, 11.0, 2.0], [12.0, -11.0, 3.0]], dtype=np.float32)
    clipped = _clip_mesh_xy_to_world(vertices, (20.0, 20.0))
    assert np.all(clipped[:, :2] >= -10.0)
    assert np.all(clipped[:, :2] <= 10.0)


def test_structural_agent_plans_without_mesh_generation():
    result = StructuralFeatureAgent().plan(_plan())
    assert result["agent_calls"] == 1
    assert {item["semantic_category"] for item in result["features"]} == {"river", "lake"}
    assert all(item["generation_class"] == "structural_feature" for item in result["features"])


def test_structural_geometry_integrates_terrain_and_masks(tmp_path):
    height = np.ones((64, 64), dtype=np.float32)
    result = build_structural_geometry(_plan(), height, (20.0, 20.0), tmp_path / "structural")
    assert result["metrics"]["hunyuan_calls_avoided"] == 2
    modified = np.load(tmp_path / "structural" / "terrain_structural.npz")["height"]
    assert np.any(modified < height)
    exclusion = np.load(tmp_path / "structural" / "structural_exclusion_mask.npy")
    assert exclusion.any()
    water = np.load(tmp_path / "structural" / "water_surface_mask.npy")
    assert water.dtype == np.bool_
    assert water.any()
    assert len(result["structural_meshes"]) == 2
    assert validate_structural_branch(result, modified)["status"] == "pass"
    json.loads((tmp_path / "structural" / "structural_branch.json").read_text())


def test_river_mesh_uses_carved_terrain_height(tmp_path):
    result = build_structural_geometry(_plan(), np.ones((64, 64), dtype=np.float32), (20.0, 20.0), tmp_path / "structural")
    river = next(item for item in result["structural_meshes"] if item["category"] == "river")
    with np.load(river["mesh"]) as data:
        vertices = np.asarray(data["vertices"])
    # The river is carved by 1.5 m before its bed/water meshes are sampled;
    # its highest vertices therefore remain below the original unit terrain.
    assert float(vertices[:, 2].max()) < 0.0
    assert np.all(vertices[:, 0] >= -10.0)
    assert np.all(vertices[:, 0] <= 10.0)
    assert np.all(vertices[:, 1] >= -10.0)
    assert np.all(vertices[:, 1] <= 10.0)


def test_river_surface_keeps_cross_section_level_on_steep_terrain(tmp_path):
    rows, cols = 64, 64
    # A strong lateral slope reproduces the macro-field condition that used to
    # twist the independently sampled river bank vertices.
    x = np.linspace(-10.0, 10.0, cols)
    height = np.broadcast_to(4.0 * x[None, :] / 10.0, (rows, cols)).astype(np.float32)
    result = build_structural_geometry(_plan(), height, (20.0, 20.0), tmp_path / "structural")
    river = next(item for item in result["structural_meshes"] if item["category"] == "river")
    with np.load(river["mesh"]) as data:
        vertices = np.asarray(data["vertices"])
    left, right = vertices[::2], vertices[1::2]
    assert np.max(np.abs(left[:, 2] - right[:, 2])) <= 1e-6
    modified = np.load(tmp_path / "structural" / "terrain_structural.npz")["height"]
    # The corridor is capped to the shared bed profile rather than retaining
    # the original 8 m cross-section difference.
    assert float(np.max(modified) - np.min(modified)) > 0.0


def test_regional_surface_preserves_irregular_region_footprint(tmp_path):
    plan = _plan()
    plan["regions"][0]["polygon"] = {"points": [
        {"x": -8.0, "y": -7.0}, {"x": 3.0, "y": -8.0},
        {"x": 8.0, "y": -1.0}, {"x": 5.0, "y": 8.0},
        {"x": -4.0, "y": 6.0},
    ]}
    result = build_structural_geometry(plan, np.ones((64, 64), dtype=np.float32), (20.0, 20.0), tmp_path / "structural")
    lake = next(item for item in result["structural_meshes"] if item["category"] == "lake")
    with np.load(lake["mesh"]) as data:
        vertices = np.asarray(data["vertices"])
        triangles = np.asarray(data["triangles"])
    assert len(vertices) == 6  # centroid plus five authored boundary points
    assert len(triangles) == 5
    assert len(np.unique(vertices[1:, :2], axis=0)) == 5
    authored = np.asarray([
        [-8.0, -7.0], [3.0, -8.0], [8.0, -1.0], [5.0, 8.0], [-4.0, 6.0],
    ], dtype=np.float32)
    assert np.array_equal(vertices[1:, :2], authored)


def test_regional_surface_organicizes_layout_box(tmp_path):
    result = build_structural_geometry(
        _plan(), np.ones((64, 64), dtype=np.float32), (20.0, 20.0), tmp_path / "structural"
    )
    lake = next(item for item in result["structural_meshes"] if item["category"] == "lake")
    with np.load(lake["mesh"]) as data:
        boundary = np.asarray(data["vertices"])[1:, :2]
    # A rectangular planner region is converted to a bounded natural outline,
    # so the exported regional surface cannot retain four straight corners.
    assert len(boundary) > 4
    assert len(np.unique(np.round(boundary[:, 0], 4))) > 2
    assert float(boundary[:, 0].min()) >= -8.0
    assert float(boundary[:, 0].max()) <= 8.0
    assert float(boundary[:, 1].min()) >= -8.0
    assert float(boundary[:, 1].max()) <= 8.0
