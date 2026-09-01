"""Shared semantic rules for environment asset references and placement."""
from __future__ import annotations

import hashlib
import json

from .asset_types import AssetType


SPATIAL_STRUCTURE_CATEGORIES = frozenset({
    "trail", "trails", "winding_trail", "winding_trails",
    "road", "roads", "river", "rivers", "stream", "streams",
})

SURFACE_FEATURE_CATEGORIES = frozenset({
    "lake", "lakes", "water", "shore", "shoreline", "mud", "mudflat",
    "forest_ground", "ground", "terrain", "sand", "snow", "grassland",
})

# Structural features are world-building operations, not isolated assets.
STRUCTURAL_LINEAR_CATEGORIES = frozenset({
    "trail", "trails", "winding_trail", "winding_trails", "road", "roads",
    "river", "rivers", "river_path", "river_paths", "stream", "streams",
})
STRUCTURAL_SURFACE_CATEGORIES = frozenset({
    "lake", "lakes", "water", "water_surface", "shore", "shoreline",
    "grassland",
})

VEGETATION_CATEGORIES = frozenset({
    "reed", "reeds", "water_reed", "water_reeds", "grass", "grasses", "fern", "ferns",
    "bush", "bushes", "shrub", "shrubs", "shoreline_vegetation", "vegetation",
    "groundcover", "underbrush",
})

SCATTER_DETAIL_CATEGORIES = frozenset({
    "pebble", "pebbles", "small_stone", "small_stones", "stone", "stones",
    "leaf", "leaves", "debris", "litter", "detail",
})

TREE_CATEGORIES = frozenset({
    "tree", "trees", "dense_trees", "forest_dense_trees", "pine", "pines",
    "birch", "birches", "broadleaf", "broadleaf_tree", "palm", "palms",
})

SOLID_ENTITY_CATEGORIES = frozenset({
    "rock", "rocks", "log", "logs", "stump", "stumps", "cabin", "cabins",
    "house", "houses", "building", "buildings", "castle",
})

# Planner-facing vocabulary is deliberately smaller than the legacy category
# vocabulary.  Modifiers such as ``dense`` describe distribution, not the
# identity of the reusable entity.
CATEGORY_ALIASES = {
    "trees": "tree",
    "dense_trees": "tree",
    "forest_dense_trees": "tree",
    "rocks": "rock",
    "cabins": "cabin",
    "houses": "house",
    "reeds": "reed",
    "grasses": "grass",
    "ferns": "fern",
    "bushes": "bush",
    "shrubs": "shrub",
    "pebbles": "pebble",
    "stones": "stone",
    "pines": "pine",
    "birches": "birch",
    "broadleaf_tree": "broadleaf",
    "palms": "palm",
}

_DENSITY_WORDS = {"sparse", "medium", "dense", "scattered", "clustered", "lush"}
_STRATEGY_SUFFIXES = {"cluster", "clusters", "grove", "groves", "stand", "stands", "group", "groups"}
# Keep the retired route-shaped category spelling intact when reopening old
# plans; new plans use a semantic category plus a canonical asset role.
_LEGACY_ROUTE_CATEGORIES = {"vegetation_cluster", "vegetation_clusters"}


def canonical_category(category: str) -> str:
    """Return the stable semantic entity name used by a normalized plan."""
    key = str(category).strip().lower()
    return CATEGORY_ALIASES.get(key, key)


def normalize_object_semantics(
    category: str,
    *,
    asset_role: AssetType | str | None = None,
    instance_strategy: str | None = None,
    density: object = None,
) -> dict[str, object]:
    """Split common distribution words from a planner entity name.

    This is a compatibility normalizer for planner output.  The prompt is the
    primary source of this contract; this function keeps a malformed or older
    response from reintroducing compound categories downstream.
    """
    raw = str(category).strip().lower().replace("-", "_").replace(" ", "_")
    inferred_density = density_label(density, category=raw)
    words = [part for part in raw.split("_") if part]
    if words and words[0] in _DENSITY_WORDS:
        if words[0] in {"dense", "lush"}:
            inferred_density = "dense"
        elif words[0] in {"sparse", "scattered"}:
            inferred_density = "sparse"
        words = words[1:]
    strategy = str(instance_strategy or "").strip().lower()
    if words and words[-1] in _STRATEGY_SUFFIXES and raw not in _LEGACY_ROUTE_CATEGORIES:
        strategy = "cluster"
        words = words[:-1]
    base = canonical_category("_".join(words) or raw)
    known_semantic_category = base in (
        SPATIAL_STRUCTURE_CATEGORIES
        | SURFACE_FEATURE_CATEGORIES
        | VEGETATION_CATEGORIES
        | SCATTER_DETAIL_CATEGORIES
        | TREE_CATEGORIES
        | SOLID_ENTITY_CATEGORIES
    )
    # A model-provided role cannot turn a known rock/tree/path into another
    # semantic family. Unknown custom entities may still provide an explicit
    # role for the router to use.
    role = (
        AssetType.STRUCTURAL_FEATURE if structural_subtype(base) is not None
        else (infer_asset_type(base) if known_semantic_category else effective_asset_type(base, asset_role))
    )
    if strategy not in {"single", "scatter", "cluster", "path", "region"}:
        strategy = default_instance_strategy(base, role)
    return {
        "category": base,
        "asset_role": role,
        "instance_strategy": strategy,
        "density": inferred_density,
    }


def density_label(value: object, *, category: str = "") -> str | None:
    """Convert explicit legacy density values without inventing a modifier."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"sparse", "medium", "dense"}:
            return key
    if isinstance(value, (int, float)):
        number = float(value)
        if number <= 1.0 / 3.0:
            return "sparse"
        if number <= 2.0 / 3.0:
            return "medium"
        return "dense"
    if str(category).strip().lower() in {"dense_trees", "forest_dense_trees"}:
        return "dense"
    return None


def default_instance_strategy(category: str, asset_role: AssetType | str | None = None) -> str:
    """Choose how plan entities are distributed without choosing geometry."""
    route = effective_asset_type(category, asset_role)
    if route in {AssetType.STRUCTURAL_FEATURE, AssetType.LINEAR_STRUCTURE} and structural_subtype(category) == "linear":
        return "path"
    if route in {AssetType.STRUCTURAL_FEATURE, AssetType.SURFACE_REGION_FEATURE} and structural_subtype(category) == "regional_surface":
        return "region"
    if route == AssetType.PROCEDURAL_NATIVE:
        return "cluster"
    if route == AssetType.SOLID_OBJECT and str(category).strip().lower() in {
        "cabin", "cabins", "house", "houses", "building", "castle",
    }:
        return "scatter"
    return "scatter"


def default_asset_role(category: str) -> AssetType:
    """Return the planner role used as a routing hint, not a mesh decision."""
    return infer_asset_type(category)

INSTANCE_PROMPTS = {
    "dense_trees": "single mature tree, one trunk and one canopy",
    "forest_dense_trees": "single mature tree, one trunk and one canopy",
    "tree": "single mature tree, one trunk and one canopy",
    "rocks": "single natural rock",
    "rock": "single natural rock",
    "shoreline_rocks": "single shoreline rock",
    "shoreline_vegetation": "single shoreline plant or reed",
    "water_reeds": "single reed plant",
    "reeds": "single reed plant",
    "cabin": "single small cabin building",
    "cabins": "single small cabin building",
    "house": "single small house building",
    "houses": "single small house building",
}

# Values are physical scale ranges in metres applied after a prototype has
# been normalized to a one-unit maximum extent.
SEMANTIC_SCALE_RANGES = {
    "dense_trees": (3.0, 8.0),
    "forest_dense_trees": (3.0, 8.0),
    "tree": (3.0, 8.0),
    "rocks": (0.8, 2.5),
    "rock": (0.8, 2.5),
    "shoreline_rocks": (0.6, 2.0),
    "shoreline_vegetation": (0.8, 2.5),
    "water_reeds": (0.8, 2.5),
    "reeds": (0.8, 2.5),
    "cabin": (4.0, 8.0),
    "cabins": (4.0, 8.0),
    "house": (5.0, 10.0),
    "houses": (5.0, 10.0),
    "building": (5.0, 12.0),
    "castle": (12.0, 20.0),
    "palm": (2.0, 5.0),
}


def infer_asset_type(category: str) -> AssetType:
    """Return a conservative local route for old plans or offline fixtures."""
    key = category.strip().lower()
    if key in SPATIAL_STRUCTURE_CATEGORIES:
        return AssetType.LINEAR_STRUCTURE
    if key in SURFACE_FEATURE_CATEGORIES:
        return AssetType.SURFACE_REGION_FEATURE
    if key in VEGETATION_CATEGORIES:
        return AssetType.PROCEDURAL_NATIVE
    if key in SCATTER_DETAIL_CATEGORIES:
        return AssetType.SCATTER_DETAIL
    if key in TREE_CATEGORIES:
        return AssetType.REUSABLE_PROTOTYPE
    return AssetType.SOLID_OBJECT


def structural_subtype(category: str) -> str | None:
    """Return the structural branch subtype for a semantic category."""
    key = str(category).strip().lower().replace("-", "_").replace(" ", "_")
    if key in STRUCTURAL_LINEAR_CATEGORIES:
        return "linear"
    if key in STRUCTURAL_SURFACE_CATEGORIES:
        return "regional_surface"
    return None


def is_structural_feature(category: str, asset_role: AssetType | str | None = None) -> bool:
    """Structural routing is category-authoritative and legacy-role tolerant."""
    if structural_subtype(category) is not None:
        return True
    try:
        return effective_asset_type(category, asset_role) in {
            AssetType.LINEAR_STRUCTURE, AssetType.SURFACE_REGION_FEATURE,
        }
    except (TypeError, ValueError):
        return False


def generation_class(category: str, asset_role: AssetType | str | None = None) -> str:
    """Collapse legacy routes into the three high-level generation classes."""
    if is_structural_feature(category, asset_role):
        return "structural_feature"
    route = effective_asset_type(category, asset_role)
    if route in {AssetType.REUSABLE_PROTOTYPE, AssetType.PROCEDURAL_NATIVE, AssetType.SCATTER_DETAIL}:
        return "prototype_asset"
    return "reconstructable_asset"


def effective_asset_type(category: str, asset_type: AssetType | str | None = None) -> AssetType:
    if asset_type is None or str(asset_type).strip() == "":
        return infer_asset_type(category)
    try:
        return asset_type if isinstance(asset_type, AssetType) else AssetType(str(asset_type))
    except ValueError:
        return infer_asset_type(category)


def needs_reference_asset(asset_type: AssetType | str | None) -> bool:
    return effective_asset_type("", asset_type) not in {
        AssetType.STRUCTURAL_FEATURE, AssetType.LINEAR_STRUCTURE, AssetType.SURFACE_REGION_FEATURE,
    }


def uses_hunyuan_asset(asset_type: AssetType | str | None) -> bool:
    return effective_asset_type("", asset_type) in {
        AssetType.SOLID_OBJECT, AssetType.REUSABLE_PROTOTYPE, AssetType.SCATTER_DETAIL,
    }


def is_spatial_structure(category: str) -> bool:
    return infer_asset_type(category) == AssetType.LINEAR_STRUCTURE


def reference_subject(category: str, asset_type: AssetType | str | None = None) -> str:
    key = category.strip().lower()
    subject = INSTANCE_PROMPTS.get(key, f"single {category} object")
    route = effective_asset_type(category, asset_type)
    if route == AssetType.PROCEDURAL_NATIVE:
        return f"{subject}; small simplified procedural-native vegetation, alpha-card friendly silhouette"
    if route == AssetType.SCATTER_DETAIL:
        return f"small reusable {category} detail mesh, isolated single piece"
    return subject


def semantic_scale_range(category: str) -> tuple[float, float]:
    key = canonical_category(category)
    return SEMANTIC_SCALE_RANGES.get(key, (1.0, 4.0))


def appearance_key(region_id: str, category: str, appearance: str) -> str:
    payload = json.dumps(
        {"region_id": region_id, "category": category, "appearance": appearance or ""},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
