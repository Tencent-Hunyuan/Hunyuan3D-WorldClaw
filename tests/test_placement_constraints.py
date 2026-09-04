import numpy as np

from worldclaw_oss.placement_constraints import (
    apply_hard_gate,
    footprint_intersects_mask,
    prepare_surface_masks,
    requirements_from_spec,
)


def test_generic_dry_support_gate_rejects_water_without_asset_name_logic():
    water = np.zeros((5, 5), dtype=bool)
    water[2, 2] = True
    gate = apply_hard_gate(
        np.ones_like(water),
        requirements_from_spec({"category": "Asset_123"}),
        {"water": water},
    )
    assert not gate[2, 2]
    assert gate[0, 0]


def test_profile_can_allow_water_and_require_a_named_surface():
    water = np.zeros((5, 5), dtype=bool)
    water[2, 2] = True
    shore = np.zeros((5, 5), dtype=bool)
    shore[1, 2] = True
    profile = {
        "category": "WaterFeature_42",
        "placement_profile": {
            "requires_dry_support": False,
            "avoid_structural_exclusion": False,
            "required_support_surfaces": ["shore"],
        },
    }
    gate = apply_hard_gate(
        np.ones_like(water), requirements_from_spec(profile),
        {"water": water, "shore": shore},
    )
    assert gate[1, 2]
    assert not gate[2, 2]


def test_named_floor_alias_uses_generic_dry_terrain_without_asset_logic():
    water = np.zeros((5, 5), dtype=bool)
    water[2, 2] = True
    masks = prepare_surface_masks({"water": water})
    profile = {
        "category": "Asset_123",
        "placement_profile": {"allowed_support_surfaces": ["forest_floor"]},
    }
    gate = apply_hard_gate(
        np.ones_like(water), requirements_from_spec(profile), masks,
    )
    assert gate[0, 0]
    assert not gate[2, 2]


def test_dry_terrain_has_a_shape_based_fallback_when_no_masks_exist():
    masks = prepare_surface_masks({}, (3, 4))
    assert masks["dry_terrain"].shape == (3, 4)
    assert masks["dry_terrain"].all()


def test_footprint_gate_is_area_aware():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    assert footprint_intersects_mask((-1, -1, 1, 1), mask, (10.0, 10.0))
