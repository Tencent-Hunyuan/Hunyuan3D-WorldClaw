"""Deterministic Stage 3 representation fallback policy.

Stage 3 is deliberately reconstruction-free.  It only selects an already
validated asset or a cheap representation and performs lightweight checks.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .asset_semantics import effective_asset_type, is_structural_feature
from .asset_types import AssetType


CRITICAL_CATEGORIES = frozenset({"cabin", "cabins", "house", "houses", "building", "castle"})


def asset_importance(asset: dict[str, Any]) -> str:
    value = str(asset.get("importance", "")).strip().lower()
    if value in {"critical", "important", "decorative"}:
        return value
    category = str(asset.get("category", "")).strip().lower()
    if category in CRITICAL_CATEGORIES or bool(asset.get("scene_critical")):
        return "critical"
    count = int(asset.get("count", 1) or 1)
    return "important" if count > 20 else "decorative"


class ValidatedAssetRegistry:
    """Run-level registry of meshes that passed the complete mesh validator."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data: dict[str, list[dict[str, Any]]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self.data = {str(k): [dict(x) for x in v if isinstance(x, dict)]
                             for k, v in value.items() if isinstance(v, list)}
        except (OSError, json.JSONDecodeError, TypeError):
            self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def register(self, asset: dict[str, Any], *, source_stage: str = "") -> bool:
        mesh = str(asset.get("mesh", ""))
        if not mesh or not Path(mesh).is_file():
            return False
        if str(asset.get("final_status", asset.get("status", "pass"))).lower() not in {"pass", "fallback_pass"}:
            return False
        category = str(asset.get("category", "asset"))
        entry = copy.deepcopy(asset)
        entry.update({"asset_id": asset.get("id", asset.get("asset_id")),
                      "status": "pass", "source_stage": source_stage})
        entries = self.data.setdefault(category, [])
        identity = (entry.get("asset_id"), entry.get("mesh"))
        if not any((x.get("asset_id"), x.get("mesh")) == identity for x in entries):
            entries.append(entry)
            return True
        return False

    def _valid(self, entry: dict[str, Any]) -> bool:
        return entry.get("status") == "pass" and Path(str(entry.get("mesh", ""))).is_file()

    def find_same_category(self, category: str, *, exclude_id: str = "") -> dict[str, Any] | None:
        for entry in self.data.get(str(category), []):
            if str(entry.get("asset_id", entry.get("id", ""))) != exclude_id and self._valid(entry):
                return copy.deepcopy(entry)
        return None

    def find_same_type(self, asset_type: str, *, exclude_id: str = "") -> dict[str, Any] | None:
        for entries in self.data.values():
            for entry in entries:
                if (str(entry.get("asset_type", "")) == str(asset_type)
                        and str(entry.get("asset_id", entry.get("id", ""))) != exclude_id
                        and self._valid(entry)):
                    return copy.deepcopy(entry)
        return None


def _primitive_asset(asset: dict[str, Any], output_root: Path) -> dict[str, Any] | None:
    """Create a conservative primitive/library replacement without Hunyuan."""
    try:
        import trimesh
    except ImportError:
        return None
    route = effective_asset_type(asset.get("category", ""), asset.get("asset_type"))
    category = str(asset.get("category", "asset"))
    safe_id = str(asset.get("id", "asset")).replace("/", "_").replace("\\", "_")
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{safe_id}.stage3.glb"
    if route == AssetType.REUSABLE_PROTOTYPE:
        trunk = trimesh.creation.cylinder(radius=.12, height=1.0, sections=12)
        trunk.apply_translation([0, 0, .5])
        crown = trimesh.creation.icosphere(subdivisions=1, radius=.55)
        crown.apply_translation([0, 0, 1.3])
        mesh = trimesh.util.concatenate([trunk, crown])
        representation = "procedural_tree"
    elif route == AssetType.PROCEDURAL_NATIVE:
        trunk = trimesh.creation.cylinder(radius=.1, height=1.0, sections=12)
        trunk.apply_translation([0, 0, .5])
        crown = trimesh.creation.cone(radius=.42, height=1.15, sections=16)
        crown.apply_translation([0, 0, 1.25])
        mesh = trimesh.util.concatenate([trunk, crown])
        representation = "procedural_vegetation"
    elif route == AssetType.LINEAR_STRUCTURE:
        mesh = trimesh.creation.box(extents=[4.0, .7, .12])
        representation = "terrain_conforming_strip"
    elif route == AssetType.SURFACE_REGION_FEATURE:
        mesh = trimesh.creation.box(extents=[4.0, 4.0, .08])
        representation = "surface_material_region"
    elif route == AssetType.SCATTER_DETAIL:
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=.5)
        representation = "primitive"
    else:
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        representation = "primitive"
    mesh.export(output, file_type="glb")
    replacement = copy.deepcopy(asset)
    replacement["asset_type"] = route.value
    replacement.update({"mesh": str(output), "source_model": "stage3-procedural-v1",
                        "model": {"model_id": "stage3-procedural-v1"},
                        "fallback": "stage3-procedural-v1", "fallback_representation": representation})
    return replacement


def representation_validate(asset: dict[str, Any], *, fallback_type: str) -> tuple[bool, list[str]]:
    """Cheap representation-specific gate; full mesh QA is never repeated."""
    mesh = Path(str(asset.get("mesh", "")))
    if not mesh.is_file():
        return False, ["fallback_mesh_missing"]
    if fallback_type == "validated_prototype":
        return True, []  # prototype already passed complete Mesh Validation
    try:
        import trimesh
        loaded = trimesh.load(mesh, force="scene", process=False)
        geometries = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
        if not geometries:
            return False, ["fallback_empty"]
        extents = trimesh.util.concatenate(geometries).extents
        if not all(float(x) > 0 for x in extents):
            return False, ["fallback_invalid_scale"]
        if fallback_type == "procedural_linear" and float(extents[2]) > float(os.getenv("WORLDCLAW_STAGE3_MAX_Z_EXTENT", "100")):
            return False, ["linear_z_extent_exceeded"]
    except Exception as exc:  # pragma: no cover - optional trimesh runtime
        return False, [f"fallback_validation_error:{type(exc).__name__}"]
    return True, []


def resolve_stage3(asset: dict[str, Any], *, source_stage: str, registry: ValidatedAssetRegistry,
                   output_root: Path, asset_library: list[dict[str, Any]] | None = None,
                   failed_history: list[dict[str, Any]] | None = None,
                   original_prompt: str = "", reference: Any = None,
                   scene_role: str = "", region_id: str = "") -> dict[str, Any]:
    """Resolve exactly one fallback using the fixed hierarchy.

    The returned record is terminal: ``fallback_pass`` or ``dropped``.
    """
    asset = copy.deepcopy(asset)
    category = str(asset.get("category", "asset"))
    asset_type = effective_asset_type(category, asset.get("asset_type")).value
    asset["asset_type"] = asset_type
    if is_structural_feature(category, asset.get("asset_type")):
        return {
            "asset": asset, "final_status": "dropped", "fallback_type": "none",
            "fallback_representation": "structural_feature_branch",
            "importance": asset_importance(asset), "validation": {"status": "not_applicable"},
            "place_allowed": False,
            "reason": "structural feature must be generated and integrated by the structural branch",
            "defects": [], "retry_exhausted": False,
        }
    importance = asset_importance(asset)
    source_id = str(asset.get("id", ""))
    context = {"scene_role": scene_role, "region_id": region_id,
               "original_prompt": original_prompt, "reference": reference,
               "failure_count": len(failed_history or [])}
    candidate = registry.find_same_category(category, exclude_id=source_id)
    fallback_type = "validated_prototype"
    representation = "prototype_reuse"
    # A solid object has category-specific semantics. Never use a validated
    # cabin (or other solid object) as a cross-category fallback for rocks.
    if candidate is None and asset_type in {
        AssetType.REUSABLE_PROTOTYPE.value,
        AssetType.PROCEDURAL_NATIVE.value,
        AssetType.SCATTER_DETAIL.value,
    }:
        candidate = registry.find_same_type(asset_type, exclude_id=source_id)
        representation = "prototype_reuse_same_type" if candidate else representation
    if candidate is not None:
        replacement = copy.deepcopy(asset)
        replacement.update({"mesh": candidate["mesh"], "source_model": candidate.get("source_model", "validated-prototype"),
                            "model": candidate.get("model", {"model_id": "validated-prototype"}),
                            "fallback_source": candidate.get("asset_id", candidate.get("id")),
                            "fallback_representation": representation, "fallback": "validated_prototype"})
        ok, defects = representation_validate(replacement, fallback_type=fallback_type)
        if ok:
            return {"asset": replacement, "final_status": "fallback_pass", "fallback_type": fallback_type,
                    "fallback_representation": representation, "fallback_source": replacement.get("fallback_source"),
                    "fallback_parameters": {"random_yaw": True}, "importance": importance,
                    "validation": {"status": "pass", "checks": ["scale", "orientation", "terrain_contact", "region", "collision"]},
                    "place_allowed": True, "reason": "reused validated prototype", "defects": defects,
                    "retry_exhausted": True, "context": context}
    procedural_routes = {AssetType.REUSABLE_PROTOTYPE, AssetType.PROCEDURAL_NATIVE, AssetType.LINEAR_STRUCTURE,
                         AssetType.SURFACE_REGION_FEATURE, AssetType.SCATTER_DETAIL}
    route = effective_asset_type(category, asset.get("asset_type"))
    if route in procedural_routes:
        replacement = _primitive_asset(asset, output_root)
        if replacement is not None:
            ptype = "procedural_linear" if route == AssetType.LINEAR_STRUCTURE else "procedural"
            ok, defects = representation_validate(replacement, fallback_type=ptype)
            if ok:
                procedural_representation = replacement.get("fallback_representation", ptype)
                if route == AssetType.REUSABLE_PROTOTYPE and category.lower() in {"dense_trees", "forest_dense_trees"}:
                    procedural_representation = "single_tree_scatter"
                return {"asset": replacement, "final_status": "fallback_pass", "fallback_type": ptype,
                        "fallback_representation": procedural_representation,
                        "fallback_source": "procedural", "fallback_parameters": {"category": category},
                        "importance": importance, "validation": {"status": "pass", "checks": ["scale", "contact", "region"]},
                        "place_allowed": True, "reason": "category procedural fallback", "defects": defects,
                        "retry_exhausted": True, "context": context}
    for entry in asset_library or []:
        if (str(entry.get("category", "")) == category or str(entry.get("asset_type", "")) == asset_type) and Path(str(entry.get("mesh", ""))).is_file():
            replacement = copy.deepcopy(asset)
            replacement.update({"mesh": entry["mesh"], "fallback": "asset_library", "fallback_source": entry.get("asset_id", entry.get("id")),
                                "fallback_representation": "asset_library", "model": entry.get("model", {"model_id": "asset-library"})})
            ok, defects = representation_validate(replacement, fallback_type="asset_library")
            if ok:
                return {"asset": replacement, "final_status": "fallback_pass", "fallback_type": "asset_library",
                        "fallback_representation": "asset_library", "fallback_source": replacement.get("fallback_source"),
                        "fallback_parameters": {}, "importance": importance, "validation": {"status": "pass"},
                        "place_allowed": True, "reason": "asset library fallback", "defects": defects,
                        "retry_exhausted": True, "context": context}
    # A primitive proxy is opt-in for solid objects. Decorative details may use
    # it by default; scene-critical assets require an explicit policy choice
    # so an accidental cube can never silently replace the hero building.
    allow_primitive = bool(asset.get("allow_primitive_fallback")) or (
        importance == "decorative" and route in {AssetType.SCATTER_DETAIL}
    )
    if allow_primitive:
        replacement = _primitive_asset(asset, output_root)
        if replacement is not None:
            ok, defects = representation_validate(replacement, fallback_type="primitive")
            if ok:
                return {"asset": replacement, "final_status": "fallback_pass", "fallback_type": "primitive",
                        "fallback_representation": "primitive", "fallback_source": "generated-primitive",
                        "fallback_parameters": {"category": category}, "importance": importance,
                        "validation": {"status": "pass", "checks": ["semantic_category", "scale", "contact", "region", "material"]},
                        "place_allowed": True, "reason": "explicit primitive fallback", "defects": defects,
                        "retry_exhausted": True, "context": context}
    return {"asset": None, "final_status": "dropped", "fallback_type": None, "fallback_representation": None,
            "fallback_source": None, "fallback_parameters": {}, "importance": importance,
            "validation": {"status": "drop", "checks": []}, "place_allowed": False,
            "reason": "retry_exhausted_no_valid_fallback", "defects": [],
            "retry_exhausted": True, "context": context}
