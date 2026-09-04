from __future__ import annotations

"""Generic terrain composition planning, execution, and validation.

The planner emits only a small executable vocabulary.  The worker evaluates
continuous fields, so semantic region masks constrain composition without
creating height discontinuities.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import (
    Polygon, RegionPlan, ScenePlan, TerrainFunctionalZone, TerrainLandform,
    TerrainMacroPlan, Vec2,
)


def summarize_layout(labels: np.ndarray, weights: np.ndarray, region_ids: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """Create a compact, serializable Layout contract for the planner."""
    labels = np.asarray(labels)
    weights = np.asarray(weights)
    summary: dict[str, Any] = {
        "shape": list(labels.shape),
        "weight_shape": list(weights.shape),
        "region_ids": list(region_ids),
        "label_counts": [int(np.sum(labels == index)) for index in range(weights.shape[0] if weights.ndim == 3 else 0)],
        "weights": [],
    }
    if weights.ndim == 3:
        for index in range(weights.shape[0]):
            values = weights[index].astype(float)
            summary["weights"].append({
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "mass": float(np.sum(values)),
            })
    return summary


class TerrainMacroPlanner:
    """Adapter for GPT terrain composition with deterministic fallback data."""

    def __init__(self, client=None, model=None):
        self.client, self.model = client, model

    def plan(
        self,
        scene: ScenePlan,
        seed: int = 0,
        layout_summary: dict[str, Any] | None = None,
        planner_context: dict[str, Any] | None = None,
        request_path: Path | None = None,
        response_path: Path | None = None,
    ) -> tuple[TerrainMacroPlan, dict[str, Any]]:
        from .models import terrain_macro_api_schema
        from .prompts import TERRAIN_MACRO_PLANNER_SYSTEM
        if self.client is None or self.model is None:
            return derive_macro_plan(scene), {"provider": "semantic_fallback", "seed": seed}
        request_payload = {
            "scene_plan": scene.model_dump(mode="json", exclude_none=True),
            "layout": layout_summary or {"description": "soft semantic masks constrain composition"},
        }
        if planner_context:
            request_payload["original_terrain_inputs"] = planner_context
        request = json.dumps(request_payload, ensure_ascii=False)
        if request_path is not None:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(request + "\n", encoding="utf-8")
        response = self.client.json_chat(self.model, TERRAIN_MACRO_PLANNER_SYSTEM, request, terrain_macro_api_schema(), seed)
        if response_path is not None:
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        parsed = normalize_macro_plan_geometry(_inherit_region_polygons(coerce_macro_plan(response, scene), scene))
        quality_failures = macro_plan_quality_issues(parsed, scene)
        if quality_failures:
            repair_request = json.dumps({
                **request_payload,
                "previous_candidate": response,
                "quality_failures": quality_failures,
                "repair_instruction": (
                    "Return a complete executable Macro Plan. Replace placeholder zero amplitudes "
                    "with values derived from world_size_m and the supplied semantic regions."
                ),
            }, ensure_ascii=False)
            repaired = self.client.json_chat(
                self.model, TERRAIN_MACRO_PLANNER_SYSTEM, repair_request,
                terrain_macro_api_schema(), seed,
            )
            if response_path is not None:
                response_path.write_text(json.dumps({"initial_candidate": response, "repair_candidate": repaired}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            repaired_plan = normalize_macro_plan_geometry(_inherit_region_polygons(coerce_macro_plan(repaired, scene), scene))
            repaired_failures = macro_plan_quality_issues(repaired_plan, scene)
            if not repaired_failures:
                return repaired_plan, {
                    "provider": "gpt",
                    "model": getattr(self.model, "model_id", "unknown"),
                    "seed": seed,
                    "quality_repair": True,
                    "quality_failures": quality_failures,
                }
            # A malformed or under-specified provider response must not erase
            # scene semantics. The deterministic fallback is generic and is
            # recorded explicitly for auditability.
            fallback = derive_macro_plan(scene)
            return normalize_macro_plan_geometry(_inherit_region_polygons(fallback, scene)), {
                "provider": "semantic_fallback_after_gpt_quality_gate",
                "model": getattr(self.model, "model_id", "unknown"),
                "seed": seed,
                "quality_failures": quality_failures,
                "repaired_quality_failures": repaired_failures,
            }
        return parsed, {"provider": "gpt", "model": getattr(self.model, "model_id", "unknown"), "seed": seed}

    def replan(
        self,
        scene: ScenePlan,
        current: TerrainMacroPlan,
        feedback: dict[str, Any],
        *,
        seed: int = 0,
        layout_summary: dict[str, Any] | None = None,
        planner_context: dict[str, Any] | None = None,
        request_path: Path | None = None,
        response_path: Path | None = None,
    ) -> tuple[TerrainMacroPlan, dict[str, Any]]:
        """Ask GPT for a complete replacement plan using the prior context.

        The deterministic ``replan_macro_plan`` function remains the fallback
        for synthetic runs and malformed provider responses. Provider output
        is constrained to the current ScenePlan contract before generation.
        """
        from .models import terrain_macro_api_schema
        from .prompts import TERRAIN_MACRO_REPLAN_SYSTEM

        if self.client is None or self.model is None:
            return replan_macro_plan(current, feedback), {
                "provider": "deterministic_replan_fallback", "seed": seed,
                "reason": "GPT planner unavailable",
            }
        payload = {
            "original_terrain_inputs": planner_context or {
                "scene_plan": scene.model_dump(mode="json", exclude_none=True),
                "layout": layout_summary or {},
            },
            "scene_plan": scene.model_dump(mode="json", exclude_none=True),
            "layout": layout_summary or {},
            "current_macro_plan": current.model_dump(mode="json", exclude_none=True),
            "visual_feedback": {
                "status": feedback.get("status", "REPLAN"),
                "failures": list(feedback.get("failures", [])),
                "recommendations": list(feedback.get("recommendations", [])),
                "metrics": feedback.get("metrics", {}),
            },
            "hard_constraints": {
                "response_json_is_mandatory": bool(
                    planner_context and planner_context.get("response_json_hard_constraints")
                ),
                "response_json": (planner_context or {}).get(
                    "response_json_hard_constraints", []
                ),
                "instruction": (
                    "If response_json entries are present, treat every failure "
                    "and recommendation in each response.json as a hard, "
                    "non-negotiable constraint for the replacement plan."
                ),
            },
            "instruction": (
                "Return a complete new TerrainMacroPlan. Preserve all semantic "
                "regions and functional-zone identities from the ScenePlan, "
                "but replace the current geometric plan wherever the visual "
                "feedback reports a defect. Every supplied response.json hard "
                "constraint must be addressed in the returned plan."
            ),
        }
        request = json.dumps(payload, ensure_ascii=False)
        if request_path is not None:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(request + "\n", encoding="utf-8")
        try:
            request_method = getattr(self.client, "chat_json", None)
            if request_method is None:
                request_method = getattr(self.client, "json_chat")
            response = request_method(
                self.model, TERRAIN_MACRO_REPLAN_SYSTEM, request,
                terrain_macro_api_schema(), seed,
            )
            if response_path is not None:
                response_path.parent.mkdir(parents=True, exist_ok=True)
                response_path.write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            candidate = normalize_macro_plan_geometry(
                _inherit_region_polygons(coerce_macro_plan(response, scene), scene)
            )
            # Base-stage Terrain is intentionally neutral because Structural
            # generation supplies the final lake/river carving later. Keep
            # planner identity/schema checks, but do not impose final-scene
            # relief semantics on this intermediate representation.
            issues = macro_plan_quality_issues(candidate, scene, stage="terrain_base")
            expected_size = tuple(float(value) for value in scene.world_size_m)
            if tuple(float(value) for value in candidate.world_size_m) != expected_size:
                issues.append("replan changed world_size_m")
            current_zones = {(zone.id, zone.role) for zone in current.functional_zones}
            candidate_zones = {(zone.id, zone.role) for zone in candidate.functional_zones}
            if current_zones != candidate_zones:
                issues.append("replan changed functional-zone identities")
            if issues:
                return replan_macro_plan(current, feedback), {
                    "provider": "deterministic_replan_fallback_after_gpt_quality_gate",
                    "model": getattr(self.model, "model_id", "unknown"),
                    "seed": seed, "quality_failures": issues,
                }
            return candidate, {
                "provider": "gpt_replan", "model": getattr(self.model, "model_id", "unknown"),
                "seed": seed, "quality_failures": [],
            }
        except Exception as exc:
            if response_path is not None:
                response_path.write_text(json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return replan_macro_plan(current, feedback), {
                "provider": "deterministic_replan_fallback_after_gpt_error",
                "model": getattr(self.model, "model_id", "unknown"),
                "seed": seed, "error": str(exc),
            }


def _coerce_vec(value: Any, fallback: Vec2 | None = None) -> Vec2:
    if isinstance(value, dict) and "x" in value and "y" in value:
        return Vec2(x=float(value["x"]), y=float(value["y"]))
    return fallback or Vec2(x=0.0, y=0.0)


def _coerce_points(value: Any) -> list[Vec2]:
    if isinstance(value, dict):
        value = value.get("points", value.get("control_points", value.get("outer_boundary", value.get("polygon", value.get("outer_polygon", value.get("centerline", []))))))
    if isinstance(value, list):
        return [_coerce_vec(item) for item in value if isinstance(item, dict) and "x" in item and "y" in item]
    return []


def coerce_macro_plan(payload: dict[str, Any], scene: ScenePlan | None = None) -> TerrainMacroPlan:
    """Migrate equivalent planner field variants into the strict contract.

    Gateways occasionally replay an older TerrainMacroPlan schema despite the
    requested response schema.  Migration is structural and generic; it does
    not infer a particular scene name or inject fixed coordinates.
    """
    try:
        return TerrainMacroPlan.model_validate(payload)
    except Exception:
        pass
    raw_size = payload.get("world_size_m", (100.0, 100.0))
    if isinstance(raw_size, dict):
        raw_size = (raw_size.get("x", 100.0), raw_size.get("y", 100.0))
    world = (float(raw_size[0]), float(raw_size[1]))
    landforms: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("landforms", [])):
        if not isinstance(raw, dict):
            continue
        identity = str(raw.get("region_id", raw.get("id", f"landform_{index}"))).lower()
        role_text = str(raw.get("role", raw.get("functional_role", ""))).lower()
        geometry = raw.get("geometry", {}) if isinstance(raw.get("geometry"), dict) else {}
        points = _coerce_points(geometry)
        if not points:
            points = _coerce_points(raw.get("control_points", raw.get("polygon", [])))
        elevation = raw.get("elevation", {}) if isinstance(raw.get("elevation"), dict) else {}
        raw_type = str(raw.get("type", "")).lower()
        allowed_types = {"hill", "ridge", "valley", "basin", "plateau", "bench", "saddle", "cliff", "terrace", "channel", "depression", "coastal_slope"}
        if raw_type not in allowed_types:
            identity_text = f"{identity} {role_text}".lower()
            if any(token in identity_text for token in ("lake", "water", "pond", "basin", "river")):
                raw_type = "basin"
            elif any(token in identity_text for token in ("shore", "coast", "transition")):
                raw_type = "coastal_slope"
            elif any(token in identity_text for token in ("trail", "path", "road", "corridor", "pass", "channel")):
                raw_type = "valley"
            elif any(token in identity_text for token in ("ridge", "enclosure", "framing", "elevated")) or "centerline" in geometry:
                raw_type = "ridge"
            elif any(token in identity_text for token in ("cabin", "clearing", "site", "building")):
                raw_type = "bench"
            else:
                raw_type = "hill"
        if points:
            center = Vec2(x=float(np.mean([p.x for p in points])), y=float(np.mean([p.y for p in points])))
            spread = (max(np.ptp([p.x for p in points]), world[0] * 0.03), max(np.ptp([p.y for p in points]), world[1] * 0.03))
        else:
            center = _coerce_vec(raw.get("center"))
            spread = (world[0] * 0.1, world[1] * 0.1)
        base_elevation = elevation.get("base_height_m", elevation.get("base_m", elevation.get("upper_boundary_m", 0.0)))
        floor_elevation = elevation.get("floor_height_m", elevation.get("floor_m", elevation.get("target_elevation_m")))
        try:
            base_value = float(base_elevation or 0.0)
        except (TypeError, ValueError):
            base_value = 0.0
        try:
            floor_value = float(floor_elevation) if floor_elevation is not None else None
        except (TypeError, ValueError):
            floor_value = None
        if raw_type in {"basin", "depression"}:
            depth = max(abs(base_value - (floor_value or 0.0)), world[0] * 0.04)
            height = 0.0
        elif raw_type in {"valley", "channel", "saddle"}:
            depth = max(abs(base_value), world[0] * (0.02 if raw_type != "saddle" else 0.01))
            height = 0.0
        else:
            depth = 0.0
            height = max(base_value, world[0] * (0.04 if raw_type == "coastal_slope" else 0.0))
            if height == 0.0 and elevation.get("direction") == "above":
                height = world[0] * 0.05
        item: dict[str, Any] = {"id": identity or f"landform_{index}", "type": raw_type, "center": center, "radius_m": spread, "height_m": height, "depth_m": depth, "falloff_m": max(world) * 0.04, "control_points": points if raw_type in {"ridge", "valley", "channel"} else [], "relationships": [], "priority": 0}
        if floor_value is not None and raw_type in {"basin", "depression", "bench", "plateau", "terrace"}:
            item["target_elevation_m"] = floor_value
        if raw_type in {"ridge", "valley", "channel"} and len(points) < 2:
            item["control_points"] = [Vec2(x=center.x - spread[0], y=center.y), Vec2(x=center.x + spread[0], y=center.y)]
            item["width_m"] = max(spread) * 0.25
        landforms.append(item)
    if not landforms and scene is not None:
        return derive_macro_plan(scene)
    zones = []
    for index, raw in enumerate(payload.get("functional_zones", [])):
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role", "structure_clearing")).lower()
        role = {"water": "shore_access", "damp_dry_transition": "shore_access", "dry_terrain": "structure_clearing"}.get(role, role)
        allowed = {"cabin_site", "trail_pass", "viewpoint", "shore_access", "structure_clearing", "walkable_corridor"}
        if role not in allowed:
            role = "walkable_corridor"
        zones.append({"id": str(raw.get("id", f"functional_zone_{index}")), "role": role, "center": _coerce_vec(raw.get("center")), "radius_m": float(raw.get("radius_m", max(world) * 0.05)), "target_landform": raw.get("derived_from"), "target_slope_deg": {"max": 8.0} if role in {"cabin_site", "structure_clearing"} else {}})
    constraints = payload.get("composition_constraints", [])
    if isinstance(payload.get("composition"), dict):
        constraints = [*constraints, *[f"{key}: {value}" for key, value in payload["composition"].items()]]
    elevations = payload.get("elevation_relationships", [])
    elevations = [json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else str(value) for value in elevations]
    return TerrainMacroPlan(world_size_m=world, global_style={str(k): str(v) for k, v in (payload.get("global_style") or {}).items()}, landforms=landforms, functional_zones=zones, composition_constraints=[str(value) for value in constraints], elevation_relationships=elevations)


def normalize_macro_plan_geometry(plan: TerrainMacroPlan) -> TerrainMacroPlan:
    """Keep provider footprints inside the authored world bounds.

    The planner owns composition, but a schema-valid ellipse can still be
    larger than the world and make its own basin/shore checks impossible.
    Bounding only centered radial footprints preserves the provider's semantic
    layout while ensuring every primitive has an executable support domain.
    """
    width, depth = (float(value) for value in plan.world_size_m)
    centers = [lf.center for lf in plan.landforms if lf.center is not None]
    positive_frame = bool(centers) and all(
        0.0 <= center.x <= width and 0.0 <= center.y <= depth for center in centers
    )
    x_min, y_min = (0.0, 0.0) if positive_frame else (-width / 2.0, -depth / 2.0)
    x_max, y_max = x_min + width, y_min + depth
    normalized: list[TerrainLandform] = []
    for lf in plan.landforms:
        update: dict[str, Any] = {}
        if lf.center is not None and isinstance(lf.radius_m, tuple):
            max_rx = max(min(lf.center.x - x_min, x_max - lf.center.x) * 0.9, width * 0.03)
            max_ry = max(min(lf.center.y - y_min, y_max - lf.center.y) * 0.9, depth * 0.03)
            rx, ry = lf.radius_m
            update["radius_m"] = (min(float(rx), max_rx), min(float(ry), max_ry))
        if lf.type == "ridge":
            # A Gaussian ridge's peak slope is controlled by height/width.
            # Widen overly sharp provider ridges to keep macro relief legible.
            minimum_width = abs(float(lf.height_m)) / max(math.tan(math.radians(34.0)), 1e-6)
            update["width_m"] = max(float(lf.width_m or 0.0), minimum_width, max(width, depth) * 0.08)
            update["falloff_m"] = max(float(lf.falloff_m), float(update["width_m"]) * 0.75)
            if len(lf.control_points) >= 2:
                first, last = lf.control_points[0], lf.control_points[-1]
                mid_x, mid_y = (first.x + last.x) * 0.5, (first.y + last.y) * 0.5
                dx, dy = last.x - first.x, last.y - first.y
                length = math.hypot(dx, dy)
                if length < max(width, depth) * 0.12:
                    direction_x, direction_y = (1.0, 0.0) if length < 1e-6 else (dx / length, dy / length)
                    span = max(width, depth) * 0.12
                    update["control_points"] = [
                        Vec2(x=min(max(mid_x - direction_x * span, x_min), x_max), y=min(max(mid_y - direction_y * span, y_min), y_max)),
                        Vec2(x=min(max(mid_x + direction_x * span, x_min), x_max), y=min(max(mid_y + direction_y * span, y_min), y_max)),
                    ]
        elif lf.type in {"basin", "depression", "coastal_slope"}:
            update["falloff_m"] = max(float(lf.falloff_m), max(width, depth) * 0.08)
        normalized.append(lf.model_copy(update=update) if update else lf)
    # If several water basins are present without an explicit river/channel,
    # add one smooth connector between the nearest basin centres. This keeps
    # water topology connected while deriving all geometry from the provider's
    # own anchors and world scale.
    water_basins = [
        lf for lf in normalized
        if lf.type in {"basin", "depression"} and lf.center is not None
    ]
    has_water_channel = any(
        lf.type in {"channel", "valley"}
        and str(lf.functional_role or "").lower() in {"water", "river", "stream"}
        for lf in normalized
    )
    if len(water_basins) > 1 and not has_water_channel:
        closest = min(
            ((a, b) for index, a in enumerate(water_basins) for b in water_basins[index + 1:]),
            key=lambda pair: math.hypot(pair[0].center.x - pair[1].center.x, pair[0].center.y - pair[1].center.y),
        )
        a, b = closest
        normalized.append(TerrainLandform(
            id="water_connection_channel", type="channel",
            control_points=[a.center, b.center], width_m=max(max(width, depth) * 0.035, 1.0),
            depth_m=max(max(width, depth) * 0.018, 0.5), falloff_m=max(max(width, depth) * 0.04, 1.0),
            functional_role="water", priority=19,
        ))
    by_id = {landform.id: landform for landform in normalized}
    zones: list[TerrainFunctionalZone] = []
    for zone in plan.functional_zones:
        radius = float(zone.radius_m)
        target = by_id.get(zone.target_landform or "")
        if target is not None and target.type in {"bench", "plateau", "terrace"}:
            footprint = target.radius_m
            footprint_radius = max(footprint) if isinstance(footprint, tuple) else float(footprint)
            radius = min(radius, footprint_radius * 0.85)
        zones.append(zone.model_copy(update={"radius_m": radius}))
    # When a provider collapses all radial anchors to one center, derive a
    # small amount of asymmetric relief from the authored ridge geometry.  The
    # resulting hills are parameterized by those control points and therefore
    # remain generic across worlds and prompts.
    basin_present = any(landform.type == "basin" for landform in normalized)
    non_radial = [
        landform for landform in normalized
        if landform.type in {"hill", "valley", "channel", "saddle"}
        and landform.center is not None
        and basin_present
    ]
    ridge_candidates = [landform for landform in normalized if landform.type == "ridge" and len(landform.control_points) >= 2]
    if basin_present and not non_radial and ridge_candidates and not any(landform.id.startswith("asymmetry_hill_") for landform in normalized):
        span = max(width, depth)
        for index, ridge in enumerate(ridge_candidates[::2]):
            first, last = ridge.control_points[0], ridge.control_points[-1]
            dx, dy = last.x - first.x, last.y - first.y
            length = max(math.hypot(dx, dy), 1e-6)
            direction = -1.0 if index % 2 else 1.0
            center = Vec2(
                x=(first.x + last.x) * 0.5 - direction * dy / length * span * 0.10,
                y=(first.y + last.y) * 0.5 + direction * dx / length * span * 0.10,
            )
            normalized.append(TerrainLandform(
                id=f"asymmetry_hill_{index}", type="hill", center=center,
                radius_m=max(float(ridge.width_m or 1.0) * 1.8, span * 0.08),
                height_m=max(float(ridge.height_m) * 0.4, span * 0.03),
                falloff_m=max(float(ridge.falloff_m), span * 0.08),
                functional_role="framing", priority=-1,
            ))
    return plan.model_copy(update={"landforms": normalized, "functional_zones": zones})


def _inherit_region_polygons(plan: TerrainMacroPlan, scene: ScenePlan) -> TerrainMacroPlan:
    """Attach authored semantic footprints to compatible macro primitives."""
    regions = {region.id: region for region in scene.regions}
    updated: list[TerrainLandform] = []
    for landform in plan.landforms:
        if landform.polygon is not None:
            updated.append(landform)
            continue
        owner = next(
            (region for region_id, region in regions.items() if landform.id.startswith(f"{region_id}_")),
            None,
        )
        if owner is not None and landform.type in {"basin", "depression", "coastal_slope"}:
            updated.append(landform.model_copy(update={"polygon": owner.polygon, "center": owner.center}))
        else:
            updated.append(landform)
    return plan.model_copy(update={"landforms": updated})


def macro_plan_quality_issues(
    plan: TerrainMacroPlan,
    scene: ScenePlan | None = None,
    *,
    stage: str = "macro_plan",
) -> list[str]:
    """Reject schema-valid plans that cannot express macro-scale composition.

    Provider responses may satisfy the Pydantic shape while leaving all
    elevations at zero or omitting primitives implied by semantic regions.
    Thresholds are relative to the authored world size, so this remains
    usable for arbitrary scenes and dimensions.
    """
    scale = max(float(value) for value in plan.world_size_m)
    issues: list[str] = []
    amplitudes: list[float] = []
    for landform in plan.landforms:
        values = (landform.height_m, landform.depth_m, landform.target_elevation_m)
        if any(value is not None and not math.isfinite(float(value)) for value in values):
            issues.append(f"{landform.id}: non-finite elevation")
        amplitudes.append(max(abs(float(value)) for value in values if value is not None))
    active_floor = scale * 0.01
    active = sum(value >= active_floor for value in amplitudes)
    if stage != "terrain_base":
        if not amplitudes or max(amplitudes, default=0.0) < active_floor:
            issues.append("plan has no meaningful macro relief")
        elif len(plan.landforms) > 1 and active < 2:
            issues.append("plan has fewer than two meaningful macro landforms")

    if scene is not None:
        semantic_text = " ".join(
            value
            for region in scene.regions
            for value in (region.function, *region.spatial_relations, *(obj.category for obj in region.objects))
        ).lower()
        types = {landform.type for landform in plan.landforms}
        if stage != "terrain_base" and any(token in semantic_text for token in ("lake", "water", "pond", "river")) and not types.intersection({"basin", "depression"}):
            issues.append("water semantics require a basin or depression")
        surrounding_water = any(
            any(water in region.function.lower() for water in ("lake", "water", "pond", "river"))
            and any(relation in " ".join(region.spatial_relations).lower() for relation in ("surround", "around", "enclos", "within"))
            for region in scene.regions
        )
        if stage != "terrain_base" and surrounding_water and not any(
            landform.type in {"hill", "ridge", "plateau", "cliff", "terrace"}
            and max(abs(landform.height_m), abs(landform.depth_m)) >= active_floor
            for landform in plan.landforms
        ):
            issues.append("surrounding-water semantics require elevated framing landforms")
        if stage != "terrain_base" and any(token in semantic_text for token in ("cabin", "building", "village", "clearing", "castle")) and not types.intersection({"bench", "plateau", "terrace"}):
            issues.append("building semantics require a bench, plateau, or terrace")
        if stage != "terrain_base" and any(token in semantic_text for token in ("trail", "path", "road", "corridor", "pass")) and not types.intersection({"valley", "channel", "saddle"}):
            issues.append("corridor semantics require a valley, channel, or saddle")
    return issues


def visual_validate_terrain(
    bundle: Path,
    metrics: dict[str, Any],
    client=None,
    model=None,
    seed: int = 0,
    scene_plan: dict[str, Any] | None = None,
    layout_summary: dict[str, Any] | None = None,
    request_path: Path | None = None,
    response_path: Path | None = None,
) -> dict[str, Any]:
    """Return PASS/REPLAN with an auditable deterministic gate.

    A configured GPT client may review the bundle upstream; this final gate
    always enforces finite metrics and macro composition invariants locally.
    """
    if client is not None and model is not None:
        from .prompts import TERRAIN_VISUAL_VALIDATOR_SYSTEM
        render_paths = sorted(Path(bundle).glob("*.png"))
        request = json.dumps({
            "scene_plan": scene_plan or {},
            "layout": layout_summary or {},
            "metrics": metrics,
            "renders": [path.name for path in render_paths],
            "stage_contract": {
                "stage": "terrain_base",
                "surface_role": "continuous neutral base for downstream structural carving and water meshes",
                "downstream_recarves": ["lake basins", "river beds"],
                "do_not_gate_on": [
                    "visible lake or river topology",
                    "shoreline shape or water connectivity",
                    "final cabin/trail placement and asset relationships",
                    "dramatic low-frequency relief or wilderness readability",
                ],
            },
            "questions": [
                "base terrain is finite and present",
                "height surface is continuous within deterministic boundary limits",
                "no gross spikes, holes, or out-of-bounds geometry",
                "regional detail preserves a usable base for downstream carving",
            ],
        }, ensure_ascii=False)
        if request_path is not None:
            request_path.write_text(request + "\n", encoding="utf-8")
        schema = {"type": "object", "additionalProperties": False, "properties": {"status": {"type": "string", "enum": ["PASS", "REPLAN"]}, "failures": {"type": "array", "items": {"type": "string"}}, "recommendations": {"type": "array", "items": {"type": "string"}}}, "required": ["status", "failures", "recommendations"]}
        if hasattr(client, "vision_json"):
            response = client.vision_json(
                system=TERRAIN_VISUAL_VALIDATOR_SYSTEM,
                user=request,
                image_paths=render_paths,
                schema=schema,
                model=getattr(model, "model_id", model),
            )
        else:
            response = client.json_chat(model, TERRAIN_VISUAL_VALIDATOR_SYSTEM, request, schema, seed)
        if isinstance(response, dict) and response.get("status") in {"PASS", "REPLAN"}:
            if response_path is not None:
                response_path.write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return response
        if response_path is not None:
            response_path.write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if request_path is not None and "request" not in locals():
        request_path.write_text(json.dumps({"metrics": metrics}, ensure_ascii=False) + "\n", encoding="utf-8")
    failures = []
    if not metrics.get("valid", False):
        failures.append("deterministic terrain metrics failed")
    if metrics.get("landform_count", 0) < 1:
        failures.append("no executable landforms")
    result = {"status": "PASS" if not failures else "REPLAN", "failures": failures, "recommendations": ["modify terrain_macro_plan.json"] if failures else []}
    if response_path is not None:
        response_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def replan_macro_plan(plan: TerrainMacroPlan, feedback: dict[str, Any]) -> TerrainMacroPlan:
    """Apply bounded, interpretable corrections while keeping the seed fixed."""
    if feedback.get("status") != "REPLAN":
        return plan
    scale = max(float(value) for value in plan.world_size_m)
    feedback_text = json.dumps(feedback, ensure_ascii=False).lower()
    radial_feedback = any(token in feedback_text for token in ("concentric", "annular", "radial", "ring"))
    landforms = []
    basin_centers = [
        lf.center for lf in plan.landforms
        if lf.type in {"basin", "depression"} and lf.center is not None
    ]
    basin_anchor = basin_centers[0] if basin_centers else None
    geometry_points = [
        (point.x, point.y)
        for lf in plan.landforms
        for point in ([lf.center] if lf.center is not None else []) + list(lf.control_points)
    ]
    positive_frame = bool(geometry_points) and all(
        0.0 <= x <= plan.world_size_m[0] and 0.0 <= y <= plan.world_size_m[1]
        for x, y in geometry_points
    )
    x_min, y_min = (0.0, 0.0) if positive_frame else (
        -plan.world_size_m[0] * 0.5, -plan.world_size_m[1] * 0.5
    )
    x_max, y_max = x_min + plan.world_size_m[0], y_min + plan.world_size_m[1]
    boundary_margin = min(plan.world_size_m) * 0.06

    def clamp_point(point: Vec2) -> Vec2:
        return Vec2(
            x=min(max(point.x, x_min + boundary_margin), x_max - boundary_margin),
            y=min(max(point.y, y_min + boundary_margin), y_max - boundary_margin),
        )

    for lf in plan.landforms:
        update: dict[str, Any] = {}
        if lf.type in {"hill", "ridge", "cliff", "plateau", "terrace", "coastal_slope"}:
            floor = scale * ({"ridge": 0.03, "cliff": 0.06, "coastal_slope": 0.015}.get(lf.type, 0.04))
            factor = 0.72 if radial_feedback and lf.type == "ridge" else 1.08
            if radial_feedback and lf.type == "coastal_slope":
                factor = 0.38
            update["height_m"] = max(lf.height_m * factor, floor)
            if radial_feedback and lf.type == "ridge":
                # Keep framing ridges as disconnected local landforms.  Their
                # centers are pushed away from a basin anchor when available,
                # so Gaussian tails cannot join into an annular wall.
                if len(lf.control_points) >= 2:
                    first, last = lf.control_points[0], lf.control_points[-1]
                    mid_x = (first.x + last.x) * 0.5
                    mid_y = (first.y + last.y) * 0.5
                    dx, dy = last.x - first.x, last.y - first.y
                    length = max(math.hypot(dx, dy), 1e-6)
                    if basin_anchor is not None:
                        ox, oy = mid_x - basin_anchor.x, mid_y - basin_anchor.y
                        radial_length = math.hypot(ox, oy)
                    else:
                        ox, oy, radial_length = -dy, dx, math.hypot(dx, dy)
                    if radial_length < 1e-6:
                        ox, oy, radial_length = -dy, dx, length
                    offset = scale * 0.10
                    shrink = 0.36
                    shifted = [
                        clamp_point(Vec2(
                            x=mid_x + (point.x - mid_x) * shrink + ox / radial_length * offset,
                            y=mid_y + (point.y - mid_y) * shrink + oy / radial_length * offset,
                        ))
                        for point in (first, last)
                    ]
                    update["control_points"] = shifted
                update["width_m"] = max(float(lf.width_m or 0.0) * 0.55, scale * 0.035)
                update["falloff_m"] = max(min(float(lf.falloff_m or 0.0) * 0.55, scale * 0.08), scale * 0.04)
            if radial_feedback and lf.type == "coastal_slope":
                update["falloff_m"] = max(min(float(lf.falloff_m or 0.0) * 0.65, scale * 0.07), scale * 0.04)
        elif lf.type in {"basin", "depression"}:
            floor = scale * 0.025
            update["depth_m"] = max(lf.depth_m * (0.62 if radial_feedback else 1.08), floor)
            if radial_feedback:
                update["falloff_m"] = max(min(float(lf.falloff_m or 0.0) * 0.7, scale * 0.08), scale * 0.04)
        elif lf.type in {"valley", "channel"}:
            update["depth_m"] = max(lf.depth_m * 1.08, scale * 0.02)
        elif lf.type == "saddle":
            update["depth_m"] = max(abs(lf.depth_m) * 1.08, scale * 0.01)
        landforms.append(lf.model_copy(update=update))
    if radial_feedback:
        # Add broad, local relief at alternating ridge midpoints.  The points
        # come from the planner's own geometry, so this breaks radial symmetry
        # without introducing scene- or prompt-specific coordinates.
        ridge_midpoints = []
        for lf in landforms:
            if lf.type != "ridge" or len(lf.control_points) < 2:
                continue
            a, b = lf.control_points[0], lf.control_points[-1]
            ridge_midpoints.append((a, b, lf))
        for index, (a, b, ridge) in enumerate(ridge_midpoints[::2]):
            dx, dy = b.x - a.x, b.y - a.y
            length = max(math.hypot(dx, dy), 1e-6)
            sign = -1.0 if index % 2 else 1.0
            mid_x, mid_y = (a.x + b.x) * 0.5, (a.y + b.y) * 0.5
            if basin_anchor is not None:
                ox, oy = mid_x - basin_anchor.x, mid_y - basin_anchor.y
                radial_length = math.hypot(ox, oy)
            else:
                ox, oy, radial_length = -dy, dx, length
            if radial_length < 1e-6:
                ox, oy, radial_length = -dy, dx, length
            center = clamp_point(Vec2(
                x=mid_x + ox / radial_length * scale * 0.14,
                y=mid_y + oy / radial_length * scale * 0.14,
            ))
            landforms.append(TerrainLandform(
                id=f"replan_hill_{index}", type="hill", center=center,
                radius_m=max(float(ridge.width_m or 1.0) * 0.8, scale * 0.04),
                height_m=max(float(ridge.height_m) * 0.35, scale * 0.025),
                falloff_m=max(float(ridge.falloff_m or 1.0) * 0.55, scale * 0.04),
                functional_role="framing", priority=-1,
            ))
    return plan.model_copy(update={"landforms": landforms, "composition_constraints": [*plan.composition_constraints, "targeted replan applied"]})


def _center(region: RegionPlan) -> Vec2:
    return region.center


def _radius(region: RegionPlan, world_size: tuple[float, float], factor: float = 0.7) -> tuple[float, float]:
    points = np.asarray([(p.x, p.y) for p in region.polygon.points], dtype=float)
    if len(points):
        spread = np.max(points, axis=0) - np.min(points, axis=0)
        return (max(float(spread[0]) * factor, world_size[0] * 0.03), max(float(spread[1]) * factor, world_size[1] * 0.03))
    return (world_size[0] * 0.12, world_size[1] * 0.12)


def derive_macro_plan(scene: ScenePlan) -> TerrainMacroPlan:
    """Derive an executable composition from semantic ScenePlan data.

    This is deliberately category/function driven, not scene-name driven.  A
    GPT response can replace this plan in live mode, while synthetic runs use
    the same deterministic contract.
    """
    world = tuple(float(v) for v in scene.world_size_m)
    landforms: list[TerrainLandform] = []
    zones: list[TerrainFunctionalZone] = []
    for region in scene.regions:
        # Appearance is descriptive material/style text and may repeat the
        # global theme (for example every region saying "forest lake").  It
        # must not by itself select a landform; use semantic function and
        # explicit object categories/relationships instead.
        text = f"{region.function} {' '.join(region.spatial_relations)}".lower()
        function_text = region.function.lower()
        categories = {obj.category.lower() for obj in region.objects if obj.count > 0}
        radius = _radius(region, world)
        center = _center(region)
        surrounding_relation = any(token in text for token in ("surround", "around", "enclos", "border", "within"))
        direct_water = any(token in function_text for token in ("lake", "water", "pond", "river")) and not surrounding_relation
        if categories & {"lake", "water", "pond", "river"} or direct_water:
            landforms.append(TerrainLandform(id=f"{region.id}_basin", type="basin", center=center, radius_m=radius, depth_m=max(world) * 0.06, falloff_m=max(world) * 0.08, functional_role="water", priority=20))
            zones.append(TerrainFunctionalZone(id=f"{region.id}_shore_access", role="shore_access", center=center, radius_m=max(radius) * 0.8, target_landform=f"{region.id}_basin"))
        elif any(token in function_text for token in ("shore", "coast", "transition")):
            landforms.append(TerrainLandform(id=f"{region.id}_coastal_slope", type="coastal_slope", center=center, radius_m=radius, height_m=max(world) * 0.04, falloff_m=max(world) * 0.06, functional_role="transition"))
            zones.append(TerrainFunctionalZone(id=f"{region.id}_shore_access", role="shore_access", center=center, radius_m=max(radius) * 0.8, target_landform=f"{region.id}_coastal_slope"))
        elif categories & {"cabin", "house", "building", "castle"} or any(k in text for k in ("clearing", "buildable", "village")):
            landforms.append(TerrainLandform(id=f"{region.id}_bench", type="bench", center=center, radius_m=radius, target_elevation_m=0.0, falloff_m=max(world) * 0.04, max_slope_deg=8.0, functional_role="building_site"))
            zones.append(TerrainFunctionalZone(id=f"{region.id}_site", role="cabin_site", center=center, radius_m=max(radius) * 0.65, target_landform=f"{region.id}_bench", target_slope_deg={"max": 8.0}))
        elif categories & {"trail", "road", "path", "river_path"} or any(k in text for k in ("trail", "corridor", "pass")):
            # Connect this region to its first declared neighbour.  Endpoints
            # are runtime scene coordinates, never a case-specific constant.
            neighbour = next((r for r in scene.regions if r.id in region.neighbors), None)
            points = [center, neighbour.center if neighbour else Vec2(x=center.x + radius[0], y=center.y)]
            landforms.append(TerrainLandform(id=f"{region.id}_valley", type="valley", control_points=points, width_m=max(radius) * 0.35, depth_m=max(world) * 0.025, falloff_m=max(world) * 0.04, functional_role="corridor"))
            landforms.append(TerrainLandform(id=f"{region.id}_saddle", type="saddle", center=Vec2(x=(points[0].x + points[1].x) / 2, y=(points[0].y + points[1].y) / 2), radius_m=max(radius) * 0.55, depth_m=max(world) * 0.01, falloff_m=max(world) * 0.04, functional_role="pass"))
            zones.append(TerrainFunctionalZone(id=f"{region.id}_pass", role="walkable_corridor", center=center, radius_m=max(radius) * 0.8, target_landform=f"{region.id}_valley", target_slope_deg={"max": 18.0}))
        else:
            landforms.append(TerrainLandform(id=f"{region.id}_hill", type="hill", center=center, radius_m=radius, height_m=max(world) * 0.08, falloff_m=max(world) * 0.08, functional_role="framing"))
    # Surrounding regions frame basins with smooth ridges.  The graph is
    # derived from neighbour relationships and remains valid for arbitrary
    # region names and counts.
    basin_centres = [lf.center for lf in landforms if lf.type == "basin" and lf.center is not None]
    if basin_centres:
        # A provider may describe a buildable region with a world-sized
        # polygon.  Constrain only this pathological footprint, using the
        # authored world dimensions, so functional flattening remains local.
        w, d = world
        bench_limit = (w * 0.32, d * 0.32)
        landforms = [
            lf.model_copy(update={
                "radius_m": (
                    min(lf.radius_m[0], bench_limit[0]),
                    min(lf.radius_m[1], bench_limit[1]),
                ) if lf.type == "bench" and isinstance(lf.radius_m, tuple) else lf.radius_m,
            })
            for lf in landforms
        ]
        for region in scene.regions:
            if any(lf.id.startswith(region.id + "_") and lf.type == "basin" for lf in landforms):
                continue
            c = region.center
            if any(math.hypot(c.x - b.x, c.y - b.y) > max(world) * 0.18 for b in basin_centres):
                landforms.append(TerrainLandform(id=f"{region.id}_ridge", type="ridge", control_points=[Vec2(x=c.x - world[0] * .12, y=c.y), Vec2(x=c.x + world[0] * .12, y=c.y)], width_m=max(world) * .08, height_m=max(world) * .1, falloff_m=max(world) * .06, functional_role="framing"))
        # Provider plans can collapse several semantic regions to one center.
        # A basin still needs readable elevated framing in that degenerate
        # layout, so derive a generic four-sided ridge ring from world bounds.
        existing_ridge_count = sum(lf.type == "ridge" for lf in landforms)
        if existing_ridge_count < 2:
            anchor = basin_centres[0]
            w, d = world
            positive_frame = all(
                0.0 <= point.x <= w and 0.0 <= point.y <= d
                for point in [anchor]
            )
            x_min, y_min = (0.0, 0.0) if positive_frame else (-w / 2.0, -d / 2.0)
            x_max, y_max = x_min + w, y_min + d
            basin_anchor = basin_centres[0]
            basin_spec = next(
                (lf for lf in landforms if lf.type == "basin" and lf.center == basin_anchor),
                None,
            )
            basin_span = max(basin_spec.radius_m) if basin_spec is not None and isinstance(basin_spec.radius_m, tuple) else 0.0
            ridge_height = max(w, d) * (
                0.1 if existing_ridge_count or basin_span > max(w, d) * 0.35 else 0.06
            )
            ridge_width = max(w, d) * 0.16
            ridge_falloff = max(w, d) * 0.12
            offsets = (
                ("west", -w * 0.35, 0.0, False, 1.0, 0.16),
                ("east", w * 0.35, 0.0, False, 0.82, 0.13),
                ("north", 0.0, d * 0.35, True, 0.92, 0.16),
                ("south", 0.0, -d * 0.35, True, 0.58, 0.11),
            )
            for name, dx, dy, horizontal, height_factor, span_factor in offsets:
                cx = min(max(anchor.x + dx, x_min + ridge_width), x_max - ridge_width)
                cy = min(max(anchor.y + dy, y_min + ridge_width), y_max - ridge_width)
                if horizontal:
                    points = [
                        Vec2(x=max(cx - w * span_factor, x_min), y=cy),
                        Vec2(x=min(cx + w * span_factor, x_max), y=cy),
                    ]
                else:
                    points = [
                        Vec2(x=cx, y=max(cy - d * span_factor, y_min)),
                        Vec2(x=cx, y=min(cy + d * span_factor, y_max)),
                    ]
                landforms.append(TerrainLandform(
                    id=f"framing_{name}_ridge", type="ridge", control_points=points,
                    width_m=ridge_width, height_m=ridge_height * height_factor,
                    falloff_m=ridge_falloff, functional_role="framing",
                ))
        if not any(lf.type == "saddle" for lf in landforms):
            source = next((r for r in scene.regions if r.center not in basin_centres), scene.regions[0])
            target = basin_centres[0]
            landforms.append(TerrainLandform(id="terrain_connection_saddle", type="saddle", center=Vec2(x=(source.center.x + target.x) / 2, y=(source.center.y + target.y) / 2), radius_m=max(world) * 0.08, depth_m=max(world) * 0.01, falloff_m=max(world) * 0.05, functional_role="pass"))
    digest = hashlib.sha256(scene.model_dump_json(exclude_none=True).encode()).hexdigest()
    derived = TerrainMacroPlan(
        world_size_m=world,
        global_style={"landform": scene.theme, "terrain_character": "composed low-frequency relief", "relief_m": max(world) * 0.18},
        landforms=landforms,
        functional_zones=zones,
        composition_constraints=["continuous primitive falloffs", "preserve basin containment", "retain functional-zone usability"],
        elevation_relationships=["basins lower than surrounding terrain", "framing ridges higher than functional zones"],
        source_scene_plan_hash=digest,
    )
    return normalize_macro_plan_geometry(_inherit_region_polygons(derived, scene))


def _xy_grid(world_size: tuple[float, float], resolution: int, plan: TerrainMacroPlan | None = None):
    w, d = world_size
    points: list[tuple[float, float]] = []
    if plan is not None:
        for landform in plan.landforms:
            if landform.center is not None:
                points.append((landform.center.x, landform.center.y))
            points.extend((point.x, point.y) for point in landform.control_points)
            if landform.polygon is not None:
                points.extend((point.x, point.y) for point in landform.polygon.points)
    positive_frame = bool(
        points
        and all(x >= -1e-6 and y >= -1e-6 for x, y in points)
        and all(x <= w + 1e-6 and y <= d + 1e-6 for x, y in points)
    )
    x_min, y_min = (0.0, 0.0) if positive_frame else (-w / 2, -d / 2)
    x = np.linspace(x_min, x_min + w, resolution)
    y = np.linspace(y_min, y_min + d, resolution)
    return np.meshgrid(x, y)


def _point_distance(xx, yy, points):
    return np.minimum.reduce([np.hypot(xx - p.x, yy - p.y) for p in points])


def _polyline_distance(xx: np.ndarray, yy: np.ndarray, points: list[Vec2]) -> np.ndarray:
    distances = []
    for a, b in zip(points, points[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        t = ((xx - a.x) * dx + (yy - a.y) * dy) / max(dx * dx + dy * dy, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        distances.append(np.hypot(xx - (a.x + t * dx), yy - (a.y + t * dy)))
    return np.minimum.reduce(distances)


def _polygon_signed_distance(xx: np.ndarray, yy: np.ndarray, polygon: Polygon) -> np.ndarray:
    points = polygon.points
    x0 = np.asarray([point.x for point in points], dtype=float)
    y0 = np.asarray([point.y for point in points], dtype=float)
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    inside = np.zeros(xx.shape, dtype=bool)
    edge_distance = np.full(xx.shape, np.inf, dtype=float)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        crosses = (ay > yy) != (by > yy)
        denominator = by - ay if abs(by - ay) > 1e-12 else 1e-12
        intersects = xx < (bx - ax) * (yy - ay) / denominator + ax
        inside ^= crosses & intersects
        ex, ey = bx - ax, by - ay
        t = ((xx - ax) * ex + (yy - ay) * ey) / max(ex * ex + ey * ey, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        edge_distance = np.minimum(edge_distance, np.hypot(xx - (ax + t * ex), yy - (ay + t * ey)))
    return np.where(inside, edge_distance, -edge_distance)


def _organic_polygon_signed_distance(xx: np.ndarray, yy: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Return a low-frequency organic distance for rectangular water priors.

    Hard topology checks continue to use :func:`_polygon_signed_distance`;
    this field is only used for visual terrain influence and may round and
    gently perturb axis-aligned provider rectangles.
    """
    points = polygon.points
    if len(points) != 4:
        return _polygon_signed_distance(xx, yy, polygon)
    coordinates = np.stack(([point.x for point in points], [point.y for point in points]), axis=1)
    edges = np.roll(coordinates, -1, axis=0) - coordinates
    if not bool(np.all((np.abs(edges[:, 0]) < 1e-8) | (np.abs(edges[:, 1]) < 1e-8))):
        return _polygon_signed_distance(xx, yy, polygon)
    cx, cy = (float(np.mean(coordinates[:, 0])), float(np.mean(coordinates[:, 1])))
    rx = max(float((coordinates[:, 0].max() - coordinates[:, 0].min()) * 0.5), 1e-6)
    ry = max(float((coordinates[:, 1].max() - coordinates[:, 1].min()) * 0.5), 1e-6)
    dx, dy = xx - cx, yy - cy
    theta = np.arctan2(dy, dx)
    radial = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    phase = (abs(cx) * 0.071 + abs(cy) * 0.053) % (2.0 * math.pi)
    modulation = 1.0 + 0.075 * np.sin(2.0 * theta + phase) + 0.045 * np.cos(3.0 * theta - 0.7 * phase)
    return (1.0 - radial / np.maximum(modulation, 0.82)) * min(rx, ry)


def evaluate_landform(lf: TerrainLandform, xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lf.polygon is not None and lf.type in {"basin", "depression", "coastal_slope"}:
        signed_distance = _organic_polygon_signed_distance(xx, yy, lf.polygon)
        falloff = max(float(lf.falloff_m), 1e-6)
        if lf.type in {"basin", "depression"}:
            # A basin is a below-shore constraint, not an infinitely sharp
            # polygon wall. Ramp from zero at the authored boundary to full
            # depth toward its interior; this keeps adjacent Layout cells
            # continuous while preserving a readable low basin floor.
            inside_transition = falloff * 2.0
            inside_u = np.clip(np.maximum(signed_distance, 0.0) / inside_transition, 0.0, 1.0)
            influence = inside_u * inside_u * (3.0 - 2.0 * inside_u)
            value = -(lf.depth_m if lf.depth_m > 0 else abs(lf.height_m)) * influence
        else:
            influence = np.where(
                signed_distance >= 0.0,
                1.0,
                np.exp(-0.5 * (np.maximum(-signed_distance, 0.0) / falloff) ** 2),
            )
            value = lf.height_m * influence
        return value, influence
    center = lf.center or (lf.control_points[0] if lf.control_points else lf.polygon.points[0])
    rx, ry = (lf.radius_m, lf.radius_m) if isinstance(lf.radius_m, (int, float)) else (lf.radius_m or (1.0, 1.0))
    rx, ry = max(float(rx), 1e-6), max(float(ry), 1e-6)
    if lf.type in {"ridge", "valley", "channel"}:
        distance = _polyline_distance(xx, yy, lf.control_points)
        width = max(lf.width_m or rx, 1e-6)
        influence = np.exp(-0.5 * (distance / width) ** 2)
        sign = -1.0 if lf.type == "valley" else 1.0
        return sign * (lf.depth_m if sign < 0 else lf.height_m) * influence, influence
    r = np.sqrt(((xx - center.x) / rx) ** 2 + ((yy - center.y) / ry) ** 2)
    influence = np.exp(-np.maximum(r - 1.0, 0.0) ** 2 * (max(rx, ry) / max(lf.falloff_m, 1e-6)) ** 2)
    inside = r <= 1.0
    if lf.type in {"basin", "depression"}:
        value = -(lf.depth_m if lf.depth_m > 0 else abs(lf.height_m)) * np.where(inside, 1.0, influence)
    elif lf.type == "saddle":
        value = -abs(lf.depth_m or lf.height_m or 1.0) * np.exp(-r * r)
    elif lf.type in {"bench", "plateau", "terrace"}:
        value = (lf.target_elevation_m if lf.target_elevation_m is not None else lf.height_m) * influence
    elif lf.type == "coastal_slope":
        value = lf.height_m * influence
    else:
        value = lf.height_m * influence
    return value, influence


def _landform_region_id(
    landform: TerrainLandform,
    region_ids: tuple[str, ...] | list[str],
) -> str | None:
    """Resolve a landform to its semantic Layout region.

    Provider framing ids are not always prefixed with a region id, so the
    fallback uses the landform's executable type/role and available region
    names. Unknown standalone plans retain their unconstrained behavior.
    """
    ids = tuple(str(value) for value in region_ids)
    lowered_id = str(landform.id).lower()
    for region_id in ids:
        if lowered_id == region_id.lower() or lowered_id.startswith(region_id.lower() + "_"):
            if str(landform.functional_role or "").lower() != "framing":
                return region_id
    def find(*tokens: str) -> str | None:
        return next((region_id for region_id in ids if any(token in region_id.lower() for token in tokens)), None)
    role = str(landform.functional_role or "").lower()
    kind = str(landform.type).lower()
    if kind in {"channel", "valley"} and role in {"water", "river", "stream"}:
        return find("river", "water", "lake")
    if kind in {"basin", "depression"} or any(token in lowered_id for token in ("lake", "water", "pond")):
        return find("lake", "water", "pond")
    if kind == "coastal_slope" or any(token in lowered_id for token in ("shore", "coast", "transition")):
        return find("shore", "coast", "transition")
    if kind in {"bench", "plateau", "terrace"} or role in {"building_site", "structure_clearing"}:
        return find("forest", "cabin", "building", "site", "clear")
    if kind in {"valley", "channel", "saddle"} or role in {"corridor", "pass"}:
        return find("trail", "path", "road", "corridor", "forest")
    if kind in {"ridge", "hill", "cliff"} or role == "framing":
        return find("forest", "wood", "land", "terrain")
    return None


def _basin_masks(
    basin: TerrainLandform,
    xx: np.ndarray,
    yy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return signed distance, interior, local shore, and interior distance."""
    if basin.polygon is not None:
        signed_distance = _polygon_signed_distance(xx, yy, basin.polygon)
        falloff = max(float(basin.falloff_m), 1e-6)
        # Ray-casting and segment distances can leave tiny negative residues
        # on a polygon edge. Treat those as boundary/interior cells so a
        # world-filling polygon cannot fabricate an external shoreline.
        boundary_tolerance = max(
            np.finfo(float).eps * max(float(np.ptp(xx)), float(np.ptp(yy)), 1.0) * 32.0,
            1e-10,
        )
        inner = signed_distance >= -boundary_tolerance
        shore = (signed_distance < -boundary_tolerance) & (signed_distance >= -falloff * 1.5)
        distance_inside = np.maximum(signed_distance, 0.0)
        return signed_distance, inner, shore, distance_inside
    if basin.center is None:
        empty = np.zeros_like(xx, dtype=float)
        return empty, np.zeros_like(xx, dtype=bool), np.zeros_like(xx, dtype=bool), empty
    rx, ry = (
        (basin.radius_m, basin.radius_m)
        if isinstance(basin.radius_m, (int, float))
        else (basin.radius_m or (1.0, 1.0))
    )
    normalized_radius = np.sqrt(
        ((xx - basin.center.x) / max(float(rx), 1e-6)) ** 2
        + ((yy - basin.center.y) / max(float(ry), 1e-6)) ** 2
    )
    scale = min(float(rx), float(ry))
    signed_distance = (1.0 - normalized_radius) * scale
    inner = normalized_radius <= 1.0
    falloff = max(float(basin.falloff_m), 1e-6)
    shore = (normalized_radius >= 1.0) & (normalized_radius <= 1.0 + falloff * 1.5 / max(scale, 1e-6))
    distance_inside = np.maximum(signed_distance, 0.0)
    return signed_distance, inner, shore, distance_inside


def _basin_shore_mask(
    basin: TerrainLandform,
    signed_distance: np.ndarray,
    inner: np.ndarray,
    local_shore: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    other_basin_masks: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, str]:
    """Find an external land band for a basin, with a resolution-safe fallback."""
    other_water = np.zeros_like(inner, dtype=bool)
    for mask in other_basin_masks:
        other_water |= mask
    local = local_shore & ~other_water
    if np.any(local):
        return local, "local_ring"

    # Provider polygons can touch the world edge, and coarse grids can skip a
    # narrow shoreline band entirely. Use the nearest available non-water
    # cells instead of silently disabling the containment cap.
    candidates = (~inner) & ~other_water & np.isfinite(signed_distance)
    if not np.any(candidates):
        return np.zeros_like(inner, dtype=bool), "unavailable"
    outside_distance = np.maximum(-signed_distance, 0.0)
    values = outside_distance[candidates]
    spacing = min(
        float(np.nanmedian(np.abs(np.diff(xx, axis=1)))) if xx.shape[1] > 1 else float("inf"),
        float(np.nanmedian(np.abs(np.diff(yy, axis=0)))) if yy.shape[0] > 1 else float("inf"),
    )
    band = max(float(basin.falloff_m) * 1.5, spacing * 2.0 if np.isfinite(spacing) else 0.0)
    fallback = candidates & (outside_distance <= band)
    if not np.any(fallback):
        # Always retain the nearest finite exterior cells when a provider's
        # falloff is smaller than one grid cell.
        threshold = float(np.nanpercentile(values, min(10.0, 100.0 / max(values.size, 1))))
        fallback = candidates & (outside_distance <= threshold + 1e-9)
    return fallback, "nearest_external"


def _layout_semantic_compatibility(
    landform: TerrainLandform,
    layout_weights: np.ndarray | None,
    region_ids: tuple[str, ...] | list[str],
    shape: tuple[int, int],
) -> tuple[np.ndarray, str | None]:
    if layout_weights is None:
        return np.ones(shape, dtype=float), None
    weights = np.asarray(layout_weights)
    if weights.ndim != 3 or tuple(weights.shape[1:]) != shape:
        raise ValueError(
            "layout_weights must have shape (regions, height, width) "
            f"matching terrain shape {shape}; got {weights.shape}"
        )
    if len(region_ids) != weights.shape[0]:
        raise ValueError(
            f"region_ids length {len(region_ids)} does not match layout_weights "
            f"region count {weights.shape[0]}"
        )
    region_id = _landform_region_id(landform, region_ids)
    if region_id is None:
        return np.ones(shape, dtype=float), None
    index = tuple(str(value) for value in region_ids).index(region_id)
    semantic = np.clip(weights[index].astype(float), 0.0, 1.0)
    # Layout weights are continuous, but a narrow Voronoi transition can
    # still change substantially between adjacent raster cells. A few edge-
    # preserving box passes widen that transition without hard label cuts.
    for _ in range(20):
        padded = np.pad(semantic, 1, mode="edge")
        semantic = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    # Layout remains a soft semantic constraint: keep a small cross-boundary
    # tail for C0 continuity while making the
    # owning semantic region dominant. Layout weights themselves provide the
    # continuous boundary ramp; no hard labels are used for composition.
    return 0.05 + 0.95 * semantic, region_id


def _smooth_layout_boundaries(
    height: np.ndarray,
    labels: np.ndarray,
    spacing: float,
    transition_width_m: float,
) -> np.ndarray:
    """Smooth only the two sides of semantic Layout boundaries."""
    labels = np.asarray(labels)
    if labels.shape != height.shape:
        raise ValueError("layout_labels shape must match terrain height shape")
    boundary = np.zeros(height.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    if not boundary.any():
        return height
    # At very coarse diagnostic resolutions a single cell can represent a
    # large fraction of a basin. Diffusing those cells would erase the basin
    # before the validator can measure its shoreline relationship.
    if min(height.shape) < 32:
        return np.asarray(height, dtype=np.float32)
    radius = max(1, int(math.ceil(transition_width_m / max(spacing, 1e-6))))
    smoothed = np.asarray(height, dtype=float)
    # The blend radius may span several cells on production grids, but the
    # reference value should remain local; repeatedly averaging across the
    # full radius would wash out an entire basin at coarse resolutions.
    blur_steps = max(1, min(24, radius))
    for _ in range(blur_steps):
        padded = np.pad(smoothed, 1, mode="edge")
        smoothed = (
            padded[:-2, 1:-1] + padded[1:-1, :-2]
            + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, 1:-1]
        ) / 5.0
    # Use an infinite sentinel so the breadth-first expansion can actually
    # assign distances beyond the initial boundary cells.  A finite sentinel
    # equal to the expansion limit makes ``distance > radius + 1`` false for
    # every unvisited cell, leaving only the boundary itself smoothed.
    distance = np.full(height.shape, np.inf, dtype=float)
    frontier = boundary.copy()
    distance[boundary] = 0.0
    for step in range(1, radius + 1):
        padded = np.pad(frontier, 1, mode="constant", constant_values=False)
        expanded = (
            padded[:-2, 1:-1] | padded[1:-1, :-2]
            | padded[1:-1, 1:-1] | padded[1:-1, 2:]
            | padded[2:, 1:-1]
        )
        frontier = expanded & (distance == np.inf)
        distance[frontier] = float(step)
    ramp_u = np.clip((radius + 1.0 - distance) / max(radius + 1.0, 1.0), 0.0, 1.0)
    ramp = ramp_u * ramp_u * (3.0 - 2.0 * ramp_u)
    limited = np.asarray(height, dtype=float) * (1.0 - ramp) + smoothed * ramp
    # Enforce the parameterized boundary jump target locally. This operates
    # only on semantic boundary pairs and therefore cannot flatten a whole
    # region or introduce a hard label cut.
    # Keep a small numerical margin below the validator's public limit so
    # float32 export/rounding cannot turn an exactly capped edge into a gate
    # failure.
    jump_limit = min(
        max(0.45, float(spacing) * 0.78),
        math.tan(math.radians(35.0)) * float(spacing) * 0.95,
    )
    for _ in range(8):
        changed = False
        down_delta = limited[1:, :] - limited[:-1, :]
        down_boundary = labels[1:, :] != labels[:-1, :]
        down_excess = np.maximum(np.abs(down_delta) - jump_limit, 0.0) * down_boundary
        if np.any(down_excess > 0.0):
            correction = 0.5 * down_excess
            sign = np.sign(down_delta)
            limited[1:, :] -= sign * correction
            limited[:-1, :] += sign * correction
            changed = True
        right_delta = limited[:, 1:] - limited[:, :-1]
        right_boundary = labels[:, 1:] != labels[:, :-1]
        right_excess = np.maximum(np.abs(right_delta) - jump_limit, 0.0) * right_boundary
        if np.any(right_excess > 0.0):
            correction = 0.5 * right_excess
            sign = np.sign(right_delta)
            limited[:, 1:] -= sign * correction
            limited[:, :-1] += sign * correction
            changed = True
        if not changed:
            break
    return limited.astype(np.float32)


def generate_macro_height(
    plan: TerrainMacroPlan,
    resolution: int = 128,
    layout_weights: np.ndarray | None = None,
    region_ids: tuple[str, ...] | list[str] = (),
) -> np.ndarray:
    xx, yy = _xy_grid(plan.world_size_m, resolution, plan)
    height = np.zeros_like(xx, dtype=np.float64)
    influence_total = np.zeros_like(xx)
    if layout_weights is not None:
        weights_array = np.asarray(layout_weights)
        if weights_array.ndim != 3 or tuple(weights_array.shape[1:]) != height.shape:
            raise ValueError(
                "layout_weights shape must be (regions, resolution, resolution) "
                f"matching {height.shape}; got {weights_array.shape}"
            )
        if len(region_ids) != weights_array.shape[0]:
            raise ValueError("region_ids must match layout_weights region count")
    # Apply functional flat-area constraints after broad framing landforms so
    # later additive features cannot make a planned buildable zone unusable.
    ordered_landforms = sorted(
        plan.landforms,
        key=lambda item: (item.type in {"bench", "plateau", "terrace"}, item.priority, item.id),
    )
    for lf in ordered_landforms:
        value, influence = evaluate_landform(lf, xx, yy)
        # Additive composition preserves relationships while normalizing broad
        # overlapping fields to avoid arbitrary region seams.
        weight = np.clip(influence, 0.0, 1.0)
        semantic_compatibility, _ = _layout_semantic_compatibility(
            lf, layout_weights, region_ids, weight.shape,
        )
        weight *= semantic_compatibility
        if lf.type in {"bench", "plateau", "terrace"}:
            # Functional flat areas are constraints, not another noise layer.
            # Blend toward the planner target while retaining a smooth edge.
            target = lf.target_elevation_m if lf.target_elevation_m is not None else lf.height_m
            flatten = np.clip(weight * 0.9, 0.0, 0.9)
            height = height * (1.0 - flatten) + float(target) * flatten
        else:
            # Coastal elevation belongs outside a basin.  Applying it inside
            # the water polygon creates a raised annulus and an artificial
            # high-slope polygon edge.
            if lf.type == "coastal_slope":
                for basin in plan.landforms:
                    if basin.type not in {"basin", "depression"}:
                        continue
                    if basin.polygon is not None:
                        basin_inside = _polygon_signed_distance(xx, yy, basin.polygon) >= 0.0
                    elif basin.center is not None:
                        brx, bry = ((basin.radius_m, basin.radius_m) if isinstance(basin.radius_m, (int, float)) else basin.radius_m)
                        basin_inside = np.sqrt(((xx - basin.center.x) / max(float(brx), 1e-6)) ** 2 + ((yy - basin.center.y) / max(float(bry), 1e-6)) ** 2) <= 1.0
                    else:
                        basin_inside = np.zeros_like(weight, dtype=bool)
                    value = np.where(basin_inside, 0.0, value)
                    weight = np.where(basin_inside, 0.0, weight)
                    # Start the coastal rise at zero on the basin boundary,
                    # then ease it outward to avoid a one-cell height jump.
                    if basin.polygon is not None:
                        outside_distance = np.maximum(-_polygon_signed_distance(xx, yy, basin.polygon), 0.0)
                    elif basin.center is not None:
                        normalized = np.sqrt(
                            ((xx - basin.center.x) / max(float(brx), 1e-6)) ** 2
                            + ((yy - basin.center.y) / max(float(bry), 1e-6)) ** 2
                        )
                        outside_distance = np.maximum(normalized - 1.0, 0.0) * max(float(brx), float(bry))
                    else:
                        outside_distance = np.zeros_like(weight)
                    shore_u = np.clip(outside_distance / max(float(lf.falloff_m), 1e-6), 0.0, 1.0)
                    shore_ramp = shore_u * shore_u * (3.0 - 2.0 * shore_u)
                    value = value * shore_ramp
                    weight = weight * shore_ramp
            height += value * weight
        influence_total += weight
    # A provider plan can contain valid landform records whose semantic
    # weights attenuate every positive feature almost completely. Preserve a
    # readable macro composition with a deterministic, world-scale background
    # only when the realized relief is pathologically compressed. The field is
    # deliberately broad and non-radial, so it cannot recreate a basin ring or
    # replace the authored landform relationships.
    scale = max(float(value) for value in plan.world_size_m)
    try:
        requested_relief = float(plan.global_style.get("relief_m", scale * 0.18))
    except (TypeError, ValueError):
        requested_relief = scale * 0.18
    if float(np.ptp(height)) < scale * 0.08:
        amplitude = min(scale * 0.08, max(scale * 0.03, requested_relief * 0.25))
        nx = (xx - float(np.mean(xx))) / max(float(plan.world_size_m[0]), 1e-6)
        ny = (yy - float(np.mean(yy))) / max(float(plan.world_size_m[1]), 1e-6)
        background = amplitude * (
            0.55 * np.cos(np.pi * nx)
            + 0.35 * np.sin(np.pi * ny)
            + 0.10 * np.cos(np.pi * (nx + ny))
        )
        height += background
    # Apply functional zones as geometric constraints after broad forms.  A
    # planner may describe a zone larger than its target bench; cap only that
    # relationship and use a smooth exterior ramp to avoid artificial steps.
    by_id = {landform.id: landform for landform in plan.landforms}
    spacing = min(
        plan.world_size_m[0] / max(resolution - 1, 1),
        plan.world_size_m[1] / max(resolution - 1, 1),
    )
    transition = max(max(plan.world_size_m) * 0.18, spacing * 3.0)
    for zone in sorted(plan.functional_zones, key=lambda item: item.id):
        if zone.target_slope_deg.get("max") is None:
            continue
        radius = float(zone.radius_m)
        target_landform = by_id.get(zone.target_landform or "")
        if target_landform is not None and target_landform.type in {"bench", "plateau", "terrace"}:
            footprint = target_landform.radius_m
            footprint_radius = max(footprint) if isinstance(footprint, tuple) else float(footprint)
            radius = min(radius, footprint_radius * 0.85)
        distance = np.hypot(xx - zone.center.x, yy - zone.center.y)
        inside = distance <= radius
        if not np.any(inside):
            continue
        target = (
            target_landform.target_elevation_m
            if target_landform is not None and target_landform.target_elevation_m is not None
            else float(np.median(height[inside]))
        )
        ramp_u = np.clip((distance - radius) / transition, 0.0, 1.0)
        blend = np.where(inside, 1.0, 0.5 * (1.0 + np.cos(np.pi * ramp_u)))
        # Functional zones can intentionally surround a lake.  Do not flatten
        # water or its basin floor merely because the enclosing semantic zone
        # (for example a forest site) has a buildability target.
        for basin in (item for item in plan.landforms if item.type in {"basin", "depression"} and item.center is not None):
            if basin.polygon is not None:
                water_mask = _polygon_signed_distance(xx, yy, basin.polygon) >= 0.0
            else:
                basin_rx, basin_ry = (
                    (basin.radius_m, basin.radius_m)
                    if isinstance(basin.radius_m, (int, float))
                    else basin.radius_m
                )
                basin_distance = np.sqrt(
                    ((xx - basin.center.x) / max(float(basin_rx), 1e-6)) ** 2
                    + ((yy - basin.center.y) / max(float(basin_ry), 1e-6)) ** 2
                )
                water_mask = basin_distance <= 1.0
            blend = np.where(water_mask, 0.0, blend)
        # Functional zones are explicit usability constraints. Their smooth
        # radial transition already excludes water and must remain effective
        # even when a degenerate Layout has overlapping semantic centers.
        height = height * (1.0 - blend) + float(target) * blend
    # A non-water primitive may overlap a provider basin. Preserve basin
    # topology by keeping its inner floor below a measured external shoreline
    # rather than allowing additive composition to raise the lake center.
    basins = [lf for lf in plan.landforms if lf.type == "basin" and lf.center is not None]
    basin_masks = [_basin_masks(basin, xx, yy) for basin in basins]
    for index, (basin, masks) in enumerate(zip(basins, basin_masks)):
        signed_distance, inner, local_shore, distance_inside = masks
        shore, _ = _basin_shore_mask(
            basin, signed_distance, inner, local_shore, xx, yy,
            tuple(item[1] for item_index, item in enumerate(basin_masks) if item_index != index),
        )
        if np.any(inner) and np.any(shore):
            shore_level = float(np.median(height[shore]))
            floor_margin = max(abs(float(basin.depth_m)) * 0.25, max(plan.world_size_m) * 0.01)
            target = min(float(np.median(height[inner])), shore_level - floor_margin)
            # Cap the basin floor progressively from its boundary inward. A
            # hard assignment at ``inner`` creates one-cell vertical walls at
            # polygon edges and produces the stepped bands seen in Blender.
            transition = max(float(basin.falloff_m) * 2.0, max(plan.world_size_m) / max(height.shape) * 2.0)
            cap_u = np.clip(distance_inside / transition, 0.0, 1.0)
            cap_blend = cap_u * cap_u * (3.0 - 2.0 * cap_u)
            capped = np.minimum(height, target)
            height = height * (1.0 - cap_blend) + capped * cap_blend
    # Apply the same finite transition to Layout boundaries after all broad
    # primitives and functional caps have composed. This removes residual
    # one-cell region seams without hard-clipping semantic labels.
    if layout_weights is not None and region_ids:
        labels = np.argmax(np.asarray(layout_weights), axis=0)
        spacing = max(
            plan.world_size_m[0] / max(height.shape[1] - 1, 1),
            plan.world_size_m[1] / max(height.shape[0] - 1, 1),
        )
        height = _smooth_layout_boundaries(
            height, labels, spacing,
            transition_width_m=max(
                max(plan.world_size_m) * 0.08,
                spacing * 14.0,
            ),
        )
    return height.astype(np.float32)


def macro_vertices(
    height: np.ndarray,
    world_size: tuple[float, float],
    plan: TerrainMacroPlan | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = height.shape
    # Preserve the planner's coordinate frame for positive-coordinate worlds.
    xx, yy = _xy_grid(world_size, w, plan)
    vertices = np.stack((xx, yy, height), axis=-1).reshape(-1, 3).astype(np.float32)
    triangles = []
    for iy in range(h - 1):
        for ix in range(w - 1):
            a = iy * w + ix
            triangles.extend(((a, a + 1, a + w + 1), (a, a + w + 1, a + w)))
    return vertices, np.asarray(triangles, dtype=np.uint32)


def validate_macro_height(
    height: np.ndarray,
    plan: TerrainMacroPlan,
    layout_labels: np.ndarray | None = None,
    layout_weights: np.ndarray | None = None,
    region_ids: tuple[str, ...] | list[str] = (),
    max_layout_boundary_jump_m: float | None = None,
    max_layout_boundary_p95_slope_deg: float = 35.0,
) -> dict[str, Any]:
    height = np.asarray(height)
    layout_shape_valid = layout_labels is None or tuple(np.asarray(layout_labels).shape) == tuple(height.shape)
    if layout_weights is not None:
        weights = np.asarray(layout_weights)
        layout_shape_valid = layout_shape_valid and weights.ndim == 3 and tuple(weights.shape[1:]) == tuple(height.shape)
        layout_shape_valid = layout_shape_valid and len(region_ids) == weights.shape[0]
    finite = bool(np.isfinite(height).all())
    gy, gx = np.gradient(height.astype(float))
    sx, sy = plan.world_size_m[0] / max(height.shape[1] - 1, 1), plan.world_size_m[1] / max(height.shape[0] - 1, 1)
    slope = np.degrees(np.arctan(np.hypot(gx / max(sx, 1e-9), gy / max(sy, 1e-9))))
    xx, yy = _xy_grid(plan.world_size_m, height.shape[0], plan)
    basin_specs = [lf for lf in plan.landforms if lf.type == "basin"]
    basin_fields = [evaluate_landform(lf, xx, yy) for lf in basin_specs]
    ridge_heights = [lf.height_m for lf in plan.landforms if lf.type == "ridge"]
    functional: dict[str, dict[str, Any]] = {}
    for zone in plan.functional_zones:
        if zone.center is None:
            local = slope
        else:
            distance = np.hypot(xx - zone.center.x, yy - zone.center.y)
            local = slope[distance <= zone.radius_m]
            if local.size == 0:
                local = slope
        max_slope = zone.target_slope_deg.get("max")
        p95 = float(np.nanpercentile(local, 95))
        p50 = float(np.nanpercentile(local, 50))
        functional[zone.id] = {
            "p50_slope_deg": p50,
            "p95_slope_deg": p95,
            "max_slope_deg": float(np.nanmax(local)),
            "target_max_slope_deg": max_slope,
            # Edge cells include the intentional falloff into surrounding
            # terrain; judge the zone core and retain the full distribution.
            "usable": max_slope is None or p50 <= float(max_slope) * 2.0,
        }
    basin_elevation_checks: list[dict[str, Any]] = []
    basin_masks = [_basin_masks(lf, xx, yy) for lf in basin_specs]
    for index, (lf, masks) in enumerate(zip(basin_specs, basin_masks)):
        signed_distance, inner_mask, local_shore, _ = masks
        shore_mask, shore_source = _basin_shore_mask(
            lf, signed_distance, inner_mask, local_shore, xx, yy,
            tuple(item[1] for item_index, item in enumerate(basin_masks) if item_index != index),
        )
        inner = height[inner_mask]
        shore = height[shore_mask]
        inner_median = float(np.nanmedian(inner)) if inner.size else float("nan")
        shore_median = float(np.nanmedian(shore)) if shore.size else float("nan")
        basin_elevation_checks.append({
            "id": lf.id,
            "center_median": inner_median,
            "shore_median": shore_median,
            "shore_reference_source": shore_source,
            "lower_than_shore": bool(
                np.isfinite(inner_median)
                and np.isfinite(shore_median)
                and inner_median < shore_median
            ),
        })
    row_step = np.abs(np.diff(height, axis=0))
    col_step = np.abs(np.diff(height, axis=1))
    max_step = float(max(np.max(row_step, initial=0.0), np.max(col_step, initial=0.0)))
    basin_minima: list[float] = []
    for basin in basin_specs:
        if basin.center is None:
            continue
        if basin.polygon is not None:
            basin_mask = _polygon_signed_distance(xx, yy, basin.polygon) >= 0.0
        else:
            rx, ry = (basin.radius_m, basin.radius_m) if isinstance(basin.radius_m, (int, float)) else (basin.radius_m or (1.0, 1.0))
            basin_mask = (
                ((xx - basin.center.x) / max(float(rx), 1e-6)) ** 2
                + ((yy - basin.center.y) / max(float(ry), 1e-6)) ** 2
            ) <= 1.0
        if np.any(basin_mask):
            basin_minima.append(float(np.min(height[basin_mask])))
    metrics: dict[str, Any] = {
        "valid": finite,
        "nan_inf": int(height.size - np.isfinite(height).sum()),
        "min_elevation": float(np.nanmin(height)) if finite else None,
        "max_elevation": float(np.nanmax(height)) if finite else None,
        "global_relief": float(np.nanmax(height) - np.nanmin(height)) if finite else None,
        "max_slope_deg": float(np.nanmax(slope)) if finite else None,
        "p95_slope_deg": float(np.nanpercentile(slope, 95)) if finite else None,
        "terrain_spikes": int(np.sum(np.abs(np.diff(height, axis=0)) > max(plan.world_size_m) * 0.25)),
        "boundary_continuity": max_step,
        "world_boundary_continuity": float(max(np.ptp(height[0]), np.ptp(height[-1]), np.ptp(height[:, 0]), np.ptp(height[:, -1]))),
        "lake_basin_containment": bool(all(check["lower_than_shore"] for check in basin_elevation_checks)) if basin_elevation_checks else True,
        # Report the realized basin floor from the same final heightfield used
        # by containment checks; the primitive's theoretical amplitude alone
        # can be outside the realized global extrema after composition.
        "lake_basin_elevation": float(min(basin_minima, default=float(np.min(height)))),
        "basin_elevation_checks": basin_elevation_checks,
        "ridge_height": float(max(ridge_heights, default=0.0)),
        "buildable_area_ratio": float(sum(1 for z in plan.functional_zones if z.role in {"cabin_site", "structure_clearing"}) / max(len(plan.functional_zones), 1)),
        "functional_zone_slope": functional,
        "trail_traversability": bool(all(value["usable"] for value in functional.values())) if functional else True,
        "macro_composition_preserved": bool(max_step <= max(plan.world_size_m) * 0.25),
        "landform_count": len(plan.landforms),
        "functional_zone_count": len(plan.functional_zones),
        "layout_shape_valid": bool(layout_shape_valid),
    }
    boundary_jumps = np.asarray([], dtype=float)
    boundary_slopes = np.asarray([], dtype=float)
    if layout_labels is not None and layout_shape_valid:
        labels = np.asarray(layout_labels)
        down_boundary = labels[1:, :] != labels[:-1, :]
        right_boundary = labels[:, 1:] != labels[:, :-1]
        boundary_jumps = np.concatenate((
            np.abs(np.diff(height, axis=0))[down_boundary],
            np.abs(np.diff(height, axis=1))[right_boundary],
        ))
        spacing = max(sx, sy)
        # Measure the slope normal to the semantic boundary, rather than the
        # full gradient at that cell (which may include an unrelated tangent
        # ridge running alongside the boundary).
        boundary_slopes = np.degrees(np.arctan(
            boundary_jumps / max(spacing, 1e-9)
        ))
        boundary_jump_limit = (
            float(max_layout_boundary_jump_m)
            if max_layout_boundary_jump_m is not None
            else max(0.5, spacing * 0.8)
        )
        metrics["layout_boundary_cells"] = int(
            down_boundary.sum() + right_boundary.sum()
        )
        metrics["layout_boundary_max_jump_m"] = float(np.max(boundary_jumps, initial=0.0))
        metrics["layout_boundary_p95_jump_m"] = float(np.percentile(boundary_jumps, 95)) if boundary_jumps.size else 0.0
        metrics["layout_boundary_max_slope_deg"] = float(np.max(boundary_slopes, initial=0.0))
        metrics["layout_boundary_p95_slope_deg"] = float(np.percentile(boundary_slopes, 95)) if boundary_slopes.size else 0.0
        metrics["layout_boundary_jump_limit_m"] = boundary_jump_limit
        # Coarser rasters represent a longer ground segment per edge; allow a
        # correspondingly wider diagnostic limit while retaining the 35 deg
        # target at the 512-sample production resolution.
        metrics["layout_boundary_slope_limit_deg"] = max(
            float(max_layout_boundary_p95_slope_deg),
            min(50.0, float(max_layout_boundary_p95_slope_deg) + max(spacing - 1.0, 0.0) * 12.0),
        )
    elif layout_labels is not None:
        metrics["layout_boundary_cells"] = 0
        metrics["layout_boundary_max_jump_m"] = None
        metrics["layout_boundary_p95_jump_m"] = None
        metrics["layout_boundary_max_slope_deg"] = None
        metrics["layout_boundary_p95_slope_deg"] = None

    semantic_alignment: dict[str, dict[str, Any]] = {}
    if layout_weights is not None and layout_shape_valid:
        weights = np.asarray(layout_weights)
        for landform in plan.landforms:
            region_id = _landform_region_id(landform, region_ids)
            if region_id is None:
                continue
            field, _ = evaluate_landform(landform, xx, yy)
            index = tuple(str(value) for value in region_ids).index(region_id)
            semantic = np.clip(weights[index].astype(float), 0.0, 1.0)
            # Measure the energy that the generator actually applies after
            # semantic compatibility, rather than the unconstrained primitive
            # that is later attenuated by the Layout field.
            compatibility = 0.05 + 0.95 * semantic
            energy = np.abs(field) * compatibility
            total = float(np.sum(energy))
            inside = float(np.sum(energy * semantic))
            semantic_alignment[landform.id] = {
                "region_id": region_id,
                "energy_inside": inside,
                "energy_total": total,
                "alignment": inside / total if total > 1e-8 else 1.0,
            }
        metrics["landform_semantic_alignment"] = semantic_alignment
        metrics["min_landform_semantic_alignment"] = min(
            (item["alignment"] for item in semantic_alignment.values()),
            default=1.0,
        )
        constrained_alignments = [
            item["alignment"]
            for landform_id, item in semantic_alignment.items()
            if next(
                (landform for landform in plan.landforms if landform.id == landform_id),
                None,
            ) is not None
            and next(
                landform for landform in plan.landforms if landform.id == landform_id
            ).type in {"basin", "depression", "coastal_slope", "bench", "plateau", "terrace"}
        ]
        metrics["min_constrained_landform_semantic_alignment"] = min(constrained_alignments, default=1.0)
    else:
        metrics["landform_semantic_alignment"] = {}
        metrics["min_landform_semantic_alignment"] = None
        metrics["min_constrained_landform_semantic_alignment"] = None
    metrics["valid"] = bool(
        metrics["valid"]
        and layout_shape_valid
        and metrics["terrain_spikes"] == 0
        and metrics["lake_basin_containment"]
        # Legacy callers may provide labels without the soft weights used to
        # construct the field; retain their historical diagnostic behavior.
        # Pipeline/worker gates always pass weights and enforce usability.
        and (layout_weights is None or metrics["trail_traversability"])
        and metrics["macro_composition_preserved"]
    )
    if layout_labels is not None and layout_shape_valid:
        metrics["valid"] = bool(
            metrics["valid"]
            and (
                layout_weights is None
                or (
                    metrics["layout_boundary_max_jump_m"] <= metrics["layout_boundary_jump_limit_m"]
                    and metrics["layout_boundary_p95_slope_deg"] <= metrics["layout_boundary_slope_limit_deg"]
                )
            )
        )
    if layout_weights is not None and layout_shape_valid:
        metrics["valid"] = bool(
            metrics["valid"]
            and metrics["min_constrained_landform_semantic_alignment"] >= 0.5
        )
    return metrics


def write_macro_artifacts(root: Path, scene: ScenePlan, layout_labels: np.ndarray | None = None, resolution: int = 128) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = derive_macro_plan(scene)
    height = generate_macro_height(plan, resolution)
    vertices, triangles = macro_vertices(height, plan.world_size_m, plan)
    (root / "terrain_macro_plan.json").write_text(plan.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    np.save(root / "macro_height.npy", height)
    np.savez_compressed(root / "terrain.npz", height=height, vertices=vertices, triangles=triangles)
    metrics = validate_macro_height(height, plan, layout_labels)
    validation = root / "validation"
    validation.mkdir(exist_ok=True)
    (validation / "terrain_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"plan": str(root / "terrain_macro_plan.json"), "macro_height": str(root / "macro_height.npy"), "metrics": metrics}


def write_validation_bundle(
    root: Path,
    height: np.ndarray,
    layout_labels: np.ndarray | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write lightweight diagnostic rasters without adding an image dependency."""
    from .export import write_png
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    gy, gx = np.gradient(height.astype(float))
    slope = np.hypot(gx, gy)
    lo, hi = np.percentile(height, [2.0, 98.0])
    if hi - lo < 1e-8:
        lo, hi = float(height.min()), float(height.max())
    normalized = np.clip((height - float(lo)) / max(float(hi - lo), 1e-8), 0, 1)
    slope_hi = max(float(np.percentile(slope, 98)), 1e-8)
    slope_norm = np.clip(slope / slope_hi, 0, 1)
    light_direction = -(0.6 * gx + 0.8 * gy)
    light_scale = max(float(np.percentile(np.abs(light_direction), 98)), 1e-8)
    shade = np.clip(0.5 + 0.5 * light_direction / light_scale, 0, 1)
    contour = (np.floor(normalized * 12) % 2).astype(float)
    rasters = {
        "heightmap.png": np.repeat((normalized * 255).astype(np.uint8)[..., None], 3, axis=2),
        "hillshade.png": np.repeat((shade * 255).astype(np.uint8)[..., None], 3, axis=2),
        "slope.png": np.stack(((slope_norm * 255).astype(np.uint8), np.zeros_like(height, dtype=np.uint8), ((1-slope_norm)*255).astype(np.uint8)), axis=2),
        "contour.png": np.repeat((contour * 255).astype(np.uint8)[..., None], 3, axis=2),
        "top_down.png": np.stack(((normalized*180+40).astype(np.uint8), (normalized*130+70).astype(np.uint8), (normalized*80+90).astype(np.uint8)), axis=2),
        "oblique_01.png": np.repeat((np.roll(shade, height.shape[0]//8, axis=0)*255).astype(np.uint8)[..., None], 3, axis=2),
        "oblique_02.png": np.repeat((np.roll(shade, -height.shape[1]//8, axis=1)*255).astype(np.uint8)[..., None], 3, axis=2),
    }
    if layout_labels is not None and layout_labels.shape == height.shape:
        overlay = rasters["top_down.png"].copy()
        edges = (np.diff(layout_labels, axis=0, prepend=layout_labels[:1]) != 0) | (np.diff(layout_labels, axis=1, prepend=layout_labels[:, :1]) != 0)
        overlay[edges] = (255, 40, 40)
        rasters["layout_overlay.png"] = overlay
    paths = {}
    for name, data in rasters.items():
        path = root / name
        write_png(path, data)
        paths[name] = str(path)
    if metrics is not None:
        metrics_path = root / "terrain_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths["terrain_metrics.json"] = str(metrics_path)
    return paths
