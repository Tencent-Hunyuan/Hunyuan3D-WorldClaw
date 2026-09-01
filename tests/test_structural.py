import json

import numpy as np

from worldclaw_oss.asset_semantics import generation_class, is_structural_feature, structural_subtype
from worldclaw_oss.structural import StructuralFeatureAgent, build_structural_geometry, validate_structural_branch


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
    assert structural_subtype("lake") == "regional_surface"


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
    assert len(result["structural_meshes"]) == 2
    assert validate_structural_branch(result, modified)["status"] == "pass"
    json.loads((tmp_path / "structural" / "structural_branch.json").read_text())


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
