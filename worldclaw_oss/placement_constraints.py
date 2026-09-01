"""Generic, data-driven hard placement constraints.

The placement worker should decide feasibility from a profile and scene masks,
not from the display name of an asset or region.  This module is shared by the
live worker and the deterministic synthetic path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PlacementRequirements:
    requires_dry_support: bool = True
    avoid_structural_exclusion: bool = True
    required_support_surfaces: tuple[str, ...] = ()
    forbidden_support_surfaces: tuple[str, ...] = ()
    allowed_support_surfaces: tuple[str, ...] = ()
    distance_preferences: tuple[tuple[str, float], ...] = ()
    footprint_overlap_threshold: float | None = None


def requirements_from_spec(spec: Mapping[str, Any]) -> PlacementRequirements:
    """Read a serialized placement profile, tolerating archived plans."""
    raw = spec.get("placement_profile") or {}
    if not isinstance(raw, Mapping):
        raw = {}

    def names(key: str) -> tuple[str, ...]:
        value = raw.get(key, ())
        if isinstance(value, str):
            value = (value,)
        if not isinstance(value, (list, tuple, set)):
            return ()
        return tuple(dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip()))

    preferences = raw.get("distance_preferences", {})
    if not isinstance(preferences, Mapping):
        preferences = {}
    parsed_preferences = []
    for key, value in preferences.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if str(key).strip() and np.isfinite(numeric) and numeric > 0.0:
            parsed_preferences.append((str(key).strip().lower(), numeric))
    try:
        overlap_threshold = (
            None if raw.get("footprint_overlap_threshold") in (None, "")
            else float(raw["footprint_overlap_threshold"])
        )
    except (TypeError, ValueError):
        overlap_threshold = None
    return PlacementRequirements(
        requires_dry_support=bool(raw.get("requires_dry_support", True)),
        avoid_structural_exclusion=bool(raw.get("avoid_structural_exclusion", True)),
        required_support_surfaces=names("required_support_surfaces"),
        forbidden_support_surfaces=names("forbidden_support_surfaces"),
        allowed_support_surfaces=names("allowed_support_surfaces"),
        distance_preferences=tuple(parsed_preferences),
        footprint_overlap_threshold=overlap_threshold,
    )


def apply_hard_gate(
    base_gate: np.ndarray,
    requirements: PlacementRequirements,
    surface_masks: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Apply generic support/structure compatibility to candidate cells."""
    gate = np.asarray(base_gate, dtype=bool).copy()

    def mask_for(name: str) -> np.ndarray | None:
        value = surface_masks.get(name)
        if value is None:
            return None
        value = np.asarray(value, dtype=bool)
        if value.shape != gate.shape:
            raise ValueError(f"surface mask {name!r} shape does not match placement grid")
        return value

    if requirements.avoid_structural_exclusion:
        exclusion = mask_for("structural_exclusion")
        if exclusion is not None:
            gate &= ~exclusion
    if requirements.requires_dry_support:
        water = mask_for("water")
        if water is not None:
            gate &= ~water

    forbidden = set(requirements.forbidden_support_surfaces)
    if requirements.requires_dry_support:
        forbidden.add("water")
    for name in forbidden:
        mask = mask_for(name)
        if mask is not None:
            gate &= ~mask

    required_names = set(requirements.required_support_surfaces)
    allowed_names = set(requirements.allowed_support_surfaces)
    if required_names:
        required = [mask_for(name) for name in required_names]
        required = [mask for mask in required if mask is not None]
        if required:
            gate &= np.logical_or.reduce(required)
        else:
            gate &= False
    if allowed_names:
        allowed = [mask_for(name) for name in allowed_names]
        allowed = [mask for mask in allowed if mask is not None]
        if allowed:
            gate &= np.logical_or.reduce(allowed)
        else:
            gate &= False
    return gate


def footprint_intersects_mask(
    footprint: tuple[float, float, float, float],
    mask: np.ndarray,
    world_size: tuple[float, float],
) -> bool:
    """Check a whole XY footprint against a raster hard-exclusion mask."""
    width, depth = world_size
    rows, columns = mask.shape
    col0 = int(np.clip(np.floor((footprint[0] + width / 2) / width * (columns - 1)), 0, columns - 1))
    col1 = int(np.clip(np.ceil((footprint[2] + width / 2) / width * (columns - 1)), 0, columns - 1))
    row0 = int(np.clip(np.floor((footprint[1] + depth / 2) / depth * (rows - 1)), 0, rows - 1))
    row1 = int(np.clip(np.ceil((footprint[3] + depth / 2) / depth * (rows - 1)), 0, rows - 1))
    return bool(np.any(np.asarray(mask, dtype=bool)[row0:row1 + 1, col0:col1 + 1]))
