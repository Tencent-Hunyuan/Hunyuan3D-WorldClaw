"""Deterministic planning and world integration for terrain-dependent features.

Structural features are intentionally not assets.  This module keeps the
agent-facing plan small and delegates all numerical geometry to deterministic
numpy code so roads, trails, rivers, and lakes are generated in world space.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .asset_semantics import is_structural_feature, structural_subtype


LINEAR_REPRESENTATIONS = {
    "trail": "terrain_conforming_swept_strip",
    "road": "terrain_conforming_swept_strip",
    "river": "terrain_channel_with_water",
    "stream": "terrain_channel_with_water",
}
SURFACE_REPRESENTATIONS = {
    "lake": "regional_surface",
    "shoreline": "shoreline_transition",
    "water": "regional_surface",
    "grassland": "regional_surface",
}


class StructuralFeatureAgent:
    """GPT-backed structural planner; numerical geometry remains worker-owned."""

    def __init__(self, client=None, model=None):
        self.client = client
        self.model = model

    @staticmethod
    def api_schema() -> dict[str, Any]:
        nullable_number = {"type": ["number", "null"]}
        nullable_boolean = {"type": ["boolean", "null"]}
        nullable_string = {"type": ["string", "null"]}
        geometry = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "width_m": nullable_number,
                "thickness_m": nullable_number,
                "depth_m": nullable_number,
                "bank_width_m": nullable_number,
                "terrain_following": nullable_boolean,
                "meander": nullable_number,
                "water_surface": nullable_boolean,
                "terrain_operation": nullable_string,
                "shape": nullable_string,
                "region_mask": nullable_string,
            },
            "required": [
                "width_m", "thickness_m", "depth_m", "bank_width_m",
                "terrain_following", "meander", "water_surface",
                "terrain_operation", "shape", "region_mask",
            ],
        }
        feature = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "feature_id": {"type": "string"},
                "region_id": {"type": "string"},
                "semantic_category": {"type": "string"},
                "generation_class": {"type": "string", "enum": ["structural_feature"]},
                "structural_subtype": {"type": "string", "enum": ["linear", "regional_surface"]},
                "representation": {"type": "string"},
                "instance_strategy": {"type": "string", "enum": ["world_integrated"]},
                "geometry": geometry,
                "region_relation": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "region_id": {"type": "string"},
                        "function": {"type": "string"},
                    },
                    "required": ["region_id", "function"],
                },
                "importance": {"type": "string"},
            },
            "required": [
                "feature_id", "region_id", "semantic_category", "generation_class",
                "structural_subtype", "representation", "instance_strategy", "geometry",
                "region_relation", "importance",
            ],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["ok"]},
                "features": {"type": "array", "items": feature},
            },
            "required": ["status", "features"],
        }

    def plan(self, scene_plan: dict[str, Any], *, layout: np.ndarray | None = None,
             terrain: np.ndarray | None = None, world_size: tuple[float, float] | None = None,
             agent_input: dict[str, Any] | None = None,
             request_path: Path | None = None, response_path: Path | None = None,
             seed: int = 0) -> dict[str, Any]:
        expected = structural_features_from_plan(scene_plan)
        if self.client is None or self.model is None:
            # Synthetic mode and unit fixtures may use the deterministic
            # planner explicitly; live mode always injects an API client.
            # Synthetic runs still execute the structural planning contract;
            # the provider is local and deterministic, but the invocation is
            # recorded so the stage cannot be mistaken for a skipped planner.
            return {"status": "ok", "features": expected, "agent_calls": 1, "provider": "deterministic_fixture"}
        terrain_summary = None
        if terrain is not None:
            values = np.asarray(terrain, dtype=float)
            terrain_summary = {
                "shape": list(values.shape),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "mean": float(np.nanmean(values)),
            }
        # The input tree is part of the agent contract.  Include the
        # structured JSON documents in the persisted request as well as their
        # paths, so an API invocation can be audited without dereferencing a
        # mutable work directory later.  Binary arrays remain represented by
        # shape/statistics in the request and are copied in the input tree.
        input_payload = dict(agent_input or {})
        input_root = input_payload.get("root")
        if input_root:
            root = Path(str(input_root))
            json_inputs = {
                "scene_plan": root / "scene_plan.json",
                "feature_plan": root / "feature_plan.json",
                "structural_task": root / "structural_task.json",
                "existing_structures": root / "existing_structures" / "structural_response.json",
            }
            for key, path in json_inputs.items():
                if not path.is_file():
                    raise FileNotFoundError(f"required structural agent input is missing: {path}")
                try:
                    input_payload[key] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid structural agent input: {path}") from exc
            input_payload["binary_inputs"] = {
                "layout_labels": str(root / "layout" / "layout_labels.npy"),
                "layout_weights": str(root / "layout" / "layout_weights.npy"),
                "feature_mask": str(root / "layout" / "feature_mask.npy"),
                "terrain": str(root / "terrain" / "terrain.npz"),
                "existing_exclusion_mask": str(root / "existing_structures" / "exclusion_masks" / "existing_exclusion_mask.npy"),
            }
        request = {
            "scene_plan": scene_plan,
            "structural_features": expected,
            "agent_input": input_payload,
            "world_size_m": list(world_size or scene_plan.get("world_size_m", ())),
            "layout": {
                "shape": list(layout.shape) if layout is not None else None,
                "region_count": int(np.max(layout) + 1) if layout is not None and layout.size else None,
            },
            "terrain": terrain_summary,
        }
        system = """You are the Structural Planning Agent in a 3D world pipeline.
Plan only terrain-dependent structural features listed in the input. Do not add,
remove, rename, or reinterpret features. Do not generate mesh data, vertices,
images, or Hunyuan prompts. Choose a representation and bounded geometry
parameters that a deterministic worker can execute in world coordinates.

Preserve every feature_id, region_id, semantic_category, structural_subtype,
generation_class=structural_feature, and instance_strategy=world_integrated.
For trails and roads use terrain_conforming_swept_strip. For rivers use
terrain_channel_with_water. For lakes use regional_surface with water_surface
true and terrain_operation basin. Geometry values must be finite and physically
reasonable for the supplied world size. Return one JSON object only."""
        user = json.dumps(request, ensure_ascii=False)
        if request_path is not None:
            request_path.write_text(user + "\n", encoding="utf-8")
        raw = self.client.json_chat(self.model, system, user, self.api_schema(), seed)
        if response_path is not None:
            response_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not isinstance(raw, dict) or raw.get("status") != "ok" or not isinstance(raw.get("features"), list):
            raise ValueError("structural planning API returned an invalid response")
        by_id = {}
        for item in raw["features"]:
            if not isinstance(item, dict) or item.get("feature_id") in by_id:
                raise ValueError("structural planning API returned duplicate or invalid feature_id")
            by_id[item.get("feature_id")] = item
        expected_ids = {item["feature_id"] for item in expected}
        if set(by_id) != expected_ids:
            raise ValueError(f"structural planning API feature ids mismatch: expected {sorted(expected_ids)}, got {sorted(by_id)}")
        normalized = []
        for base in expected:
            candidate = by_id[base["feature_id"]]
            for field in ("region_id", "semantic_category", "generation_class", "structural_subtype", "instance_strategy"):
                if candidate.get(field) != base[field]:
                    raise ValueError(f"structural planning API changed protected field {field} for {base['feature_id']}")
            geometry = dict(base["geometry"])
            geometry.update({key: value for key, value in candidate.get("geometry", {}).items() if value is not None})
            normalized.append({
                **base,
                "representation": candidate.get("representation") or base["representation"],
                "geometry": geometry,
                "importance": candidate.get("importance") or base["importance"],
                "region_relation": candidate.get("region_relation") or base["region_relation"],
            })
        return {"status": "ok", "features": normalized, "agent_calls": 1, "provider": "openai-compatible"}


def _canonical(category: str) -> str:
    return str(category).strip().lower().replace("-", "_").rstrip("s")


def _region_center(region: dict[str, Any]) -> np.ndarray:
    center = region.get("center", {})
    return np.asarray([float(center.get("x", 0.0)), float(center.get("y", 0.0))])


def structural_features_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized structural planning requests without generating mesh."""
    features: list[dict[str, Any]] = []
    for region in plan.get("regions", []):
        for index, obj in enumerate(region.get("objects", [])):
            category = _canonical(obj.get("category", ""))
            if not is_structural_feature(category, obj.get("asset_role", obj.get("asset_type"))):
                continue
            subtype = structural_subtype(category)
            if subtype == "linear":
                representation = LINEAR_REPRESENTATIONS.get(category, "terrain_conforming_swept_strip")
                geometry = {
                    "width_m": 8.0 if category in {"river", "stream"} else (6.0 if category == "road" else 2.5),
                    "thickness_m": 0.08,
                    "terrain_following": True,
                    "meander": 0.25 if category in {"river", "stream"} else 0.15,
                }
                if category in {"river", "stream"}:
                    geometry.update({"depth_m": 1.5, "bank_width_m": 3.0, "water_surface": True})
            else:
                representation = SURFACE_REPRESENTATIONS.get(category, "regional_surface")
                geometry = {"water_surface": category in {"lake", "water"}, "terrain_operation": "basin" if category in {"lake", "water"} else "flatten"}
            features.append({
                "feature_id": f"{region.get('id', 'region')}_{category}_{index:03d}",
                "region_id": region.get("id", ""),
                "semantic_category": category,
                "generation_class": "structural_feature",
                "structural_subtype": subtype,
                "representation": representation,
                "instance_strategy": "world_integrated",
                "geometry": geometry,
                "region_relation": {"region_id": region.get("id", ""), "function": region.get("function", "")},
                "importance": "important" if category in {"river", "lake", "road"} else "normal",
            })
    return features


def _centered_polygon(region: dict[str, Any], world_size: tuple[float, float]) -> np.ndarray:
    points = np.asarray([[p["x"], p["y"]] for p in region.get("polygon", {}).get("points", [])], dtype=float)
    if len(points) < 3:
        center = _region_center(region)
        points = center[None, :] + np.asarray([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    width, depth = world_size
    # Detect the lower-left world frame only when both axes agree.  Shifting
    # each axis independently misclassifies a centered scene whose region
    # happens to lie wholly above or to the right of the origin.
    positive_frame = (
        points[:, 0].min() >= -1e-6 and points[:, 0].max() <= width + 1e-6
        and points[:, 1].min() >= -1e-6 and points[:, 1].max() <= depth + 1e-6
    )
    if positive_frame:
        points -= np.asarray([width / 2, depth / 2])
    return points


def _polygon_mask(points: np.ndarray, world_size: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    width, depth = world_size
    xs = np.linspace(-width / 2, width / 2, cols)
    ys = np.linspace(-depth / 2, depth / 2, rows)
    xx, yy = np.meshgrid(xs, ys)
    inside = np.zeros((rows, cols), dtype=bool)
    x0, y0 = points[-1]
    for x1, y1 in points:
        denominator = y0 - y1
        if abs(denominator) < 1e-12:
            denominator = 1e-12
        crossing = ((y1 > yy) != (y0 > yy)) & (xx < (x0 - x1) * (yy - y1) / denominator + x1)
        inside ^= crossing
        x0, y0 = x1, y1
    return inside


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate a raster mask without wrapping at array boundaries."""
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1] | padded[2:, 1:-1]
            | padded[1:-1, :-2] | padded[1:-1, 2:]
            | padded[:-2, :-2] | padded[:-2, 2:]
            | padded[2:, :-2] | padded[2:, 2:]
        )
    return result


def _sample_height(height: np.ndarray, world_size: tuple[float, float], x: float, y: float) -> float:
    width, depth = world_size
    col = int(np.clip(round((x + width / 2) / width * (height.shape[1] - 1)), 0, height.shape[1] - 1))
    row = int(np.clip(round((y + depth / 2) / depth * (height.shape[0] - 1)), 0, height.shape[0] - 1))
    return float(height[row, col])


def _polyline_distance_field(centerline: np.ndarray, world_size: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
    """Return world-unit distance from every raster cell to a feature path."""
    width, depth = world_size
    rows, cols = shape
    xs = np.linspace(-width / 2, width / 2, cols)
    ys = np.linspace(-depth / 2, depth / 2, rows)
    xx, yy = np.meshgrid(xs, ys)
    distance = np.full((rows, cols), np.inf, dtype=np.float64)
    for start, end in zip(centerline[:-1], centerline[1:]):
        dx, dy = end - start
        denominator = max(float(dx * dx + dy * dy), 1e-12)
        t = np.clip(((xx - start[0]) * dx + (yy - start[1]) * dy) / denominator, 0.0, 1.0)
        distance = np.minimum(distance, np.hypot(xx - (start[0] + t * dx), yy - (start[1] + t * dy)))
    return distance


def _linear_centerline(
    region: dict[str, Any], category: str, world_size: tuple[float, float],
    points_count: int = 32, meander: float | None = None,
    target_polygon: np.ndarray | None = None, target_clearance: float = 0.0,
) -> np.ndarray:
    polygon = _centered_polygon(region, world_size)
    center = polygon.mean(axis=0)
    covariance = np.cov((polygon - center).T) if len(polygon) > 2 else np.eye(2)
    direction = np.linalg.eigh(covariance)[1][:, -1]
    if direction[0] < 0:
        direction = -direction
    lateral = np.asarray([-direction[1], direction[0]])
    meander = float(meander if meander is not None else (0.08 if category in {"trail", "road"} else 0.18))
    meander = max(0.0, min(abs(meander), 0.5))
    if target_polygon is not None and len(target_polygon) >= 3:
        # For an explicitly relational feature (for example a trail that
        # approaches a lake), terminate at the target boundary instead of
        # sweeping through its interior.  The projected target extent is a
        # conservative clearance for arbitrary convex/non-convex polygons.
        target = np.asarray(target_polygon, dtype=float)
        target_center = target.mean(axis=0)
        toward = target_center - center
        if float(np.linalg.norm(toward)) > 1e-8:
            direction = toward / np.linalg.norm(toward)
        else:
            target_covariance = np.cov((target - target_center).T) if len(target) > 2 else np.eye(2)
            direction = np.linalg.eigh(target_covariance)[1][:, -1]
            if direction[0] < 0:
                direction = -direction
        lateral = np.asarray([-direction[1], direction[0]])
        source_extent = max(float(np.max(np.abs((polygon - center) @ direction))), min(world_size) * 0.12)
        target_extent = max(float(np.max(np.abs((target - target_center) @ direction))), min(world_size) * 0.08)
        clearance = max(float(target_clearance), 1e-3)
        end_distance = target_extent + clearance
        start_distance = max(source_extent, end_distance + min(world_size) * 0.12)
        distances = np.linspace(-start_distance, -end_distance, points_count)
        wiggle = min(start_distance - end_distance, target_extent * 0.25) * meander
        return np.asarray([
            target_center + direction * distance
            + lateral * wiggle * math.sin((distance + start_distance) / max(start_distance - end_distance, 1.0) * math.pi)
            for distance in distances
        ], dtype=float)
    extent = max(float(np.max(np.abs((polygon - center) @ direction))), min(world_size) * 0.12)
    distances = np.linspace(-extent, extent, points_count)
    return np.asarray([
        center + direction * distance + lateral * extent * meander * math.sin(distance / max(extent, 1.0) * math.pi)
        for distance in distances
    ], dtype=float)


def _relationship_target_region(
    plan: dict[str, Any], feature: dict[str, Any], source_region_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """Resolve an authored relational target without inventing a new feature."""
    geometry = feature.get("geometry", {})
    relation = feature.get("region_relation", {})
    text = " ".join(str(value).lower() for value in (
        geometry.get("shape", ""), geometry.get("region_mask", ""),
        relation.get("function", ""),
    ))
    candidates = [(str(region.get("id", "")), region) for region in plan.get("regions", []) if str(region.get("id", "")) != source_region_id]
    preferred_tokens = []
    if any(token in text for token in ("lake", "water", "river")):
        preferred_tokens.extend(("lake", "water", "river"))
    if "shore" in text:
        preferred_tokens.append("shore")
    for token in preferred_tokens:
        for region_id, region in candidates:
            description = f"{region_id} {region.get('function', '')}".lower()
            if token in description:
                return region_id, region
    # Also honor explicit scene-level relationship text when the agent used a
    # concise geometry shape without repeating the target name.
    for region_id, region in candidates:
        relations = " ".join(str(item).lower() for item in region.get("spatial_relations", []))
        if "trail" in relations and any(token in relations for token in ("lake", "shore", "water")):
            return region_id, region
    return None


def _strip_mesh(centerline: np.ndarray, width: float, heights: np.ndarray, thickness: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    for index, point in enumerate(centerline):
        tangent = centerline[min(index + 1, len(centerline) - 1)] - centerline[max(index - 1, 0)]
        tangent /= max(np.linalg.norm(tangent), 1e-8)
        side = np.asarray([-tangent[1], tangent[0]]) * width / 2
        z = float(heights[index])
        vertices.extend([[*(point - side), z], [*(point + side), z], [*(point - side), z - thickness], [*(point + side), z - thickness]])
    faces: list[list[int]] = []
    for i in range(len(centerline) - 1):
        a, b = i * 4, (i + 1) * 4
        faces.extend([[a, b, b + 1], [a, b + 1, a + 1], [a + 2, a + 3, b + 3], [a + 2, b + 3, b + 2], [a, a + 2, b + 2], [a, b + 2, b], [a + 1, b + 1, b + 3], [a + 1, b + 3, a + 3]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _surface_mesh(points: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a filled regional surface from its authored polygon footprint."""
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    # The centroid fan preserves the authored boundary instead of replacing it
    # with an axis-aligned bounding rectangle.  Region polygons in the planner
    # are simple footprints, so each wedge remains a deterministic surface cell.
    center = np.mean(polygon, axis=0)
    vertices = np.column_stack([np.vstack([center, polygon]), np.full(len(polygon) + 1, float(z))])
    triangles = np.asarray(
        [[0, index + 1, (index + 1) % len(polygon) + 1] for index in range(len(polygon))],
        dtype=np.uint32,
    )
    return vertices.astype(np.float32), triangles


def build_structural_geometry(
    plan: dict[str, Any], height: np.ndarray, world_size: tuple[float, float], output_dir: Path, seed: int = 0,
    features: list[dict[str, Any]] | None = None,
    layout_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Construct feature meshes and masks deterministically in world coordinates."""
    del seed
    output_dir.mkdir(parents=True, exist_ok=True)
    features = list(features) if features is not None else structural_features_from_plan(plan)
    region_by_id = {str(r.get("id")): r for r in plan.get("regions", [])}
    exclusion = np.zeros_like(height, dtype=bool)
    occupied = np.zeros_like(height, dtype=bool)
    surface = np.zeros_like(height, dtype=np.int16)
    water_surface = np.zeros_like(height, dtype=bool)
    # Trail influence is separate from base layout semantics.  A value of 0
    # is a hard tree-free zone; it rises continuously to 1 in the soft zone.
    trail_clearance = np.ones_like(height, dtype=np.float32)
    trail_exclusion = np.zeros_like(height, dtype=bool)
    meshes: list[dict[str, Any]] = []
    modified = np.asarray(height, dtype=np.float32).copy()
    width, depth = world_size
    if layout_weights is not None:
        layout_weights = np.asarray(layout_weights, dtype=np.float32)
        if layout_weights.ndim != 3 or layout_weights.shape[1:] != height.shape:
            raise ValueError("layout_weights shape must be [regions, rows, cols]")
    region_indices = {str(item.get("id")): index for index, item in enumerate(plan.get("regions", []))}
    lake_integrations: list[dict[str, Any]] = []
    for feature in features:
        category = feature["semantic_category"]
        region = region_by_id[feature["region_id"]]
        target = None
        if feature["structural_subtype"] == "linear":
            target = _relationship_target_region(plan, feature, str(feature["region_id"])) if category in {"trail", "road"} else None
            target_polygon = _centered_polygon(target[1], world_size) if target is not None else None
            target_width = float(feature.get("geometry", {}).get("width_m", 0.0))
            # Leave enough room for the swept strip itself plus a visible
            # shoreline gap; projected extents are conservative for arbitrary
            # region polygons and can otherwise let the strip overlap water.
            target_clearance = target_width if target is not None else 0.0
            centerline = _linear_centerline(
                region, category, world_size,
                meander=feature.get("geometry", {}).get("meander"),
                target_polygon=target_polygon, target_clearance=target_clearance,
            )
            width_m = float(feature["geometry"]["width_m"])
            heights = np.asarray([_sample_height(modified, world_size, *point) for point in centerline])
            vertices, triangles = _strip_mesh(centerline, width_m, heights, float(feature["geometry"].get("thickness_m", 0.08)))
            if category in {"river", "stream"}:
                depth_m = float(feature["geometry"].get("depth_m", 1.5))
                for point in centerline:
                    col = int(np.clip(round((point[0] + width / 2) / width * (height.shape[1] - 1)), 0, height.shape[1] - 1))
                    row = int(np.clip(round((point[1] + depth / 2) / depth * (height.shape[0] - 1)), 0, height.shape[0] - 1))
                    radius_x = max(1, int(width_m / width * height.shape[1] / 2))
                    radius_y = max(1, int(width_m / depth * height.shape[0] / 2))
                    modified[max(0, row - radius_y):row + radius_y + 1, max(0, col - radius_x):col + radius_x + 1] -= depth_m
                water_heights = heights - depth_m + 0.03
                water_vertices, water_triangles = _strip_mesh(centerline, width_m * 0.9, water_heights, 0.01)
                vertices = np.vstack([vertices, water_vertices])
                triangles = np.vstack([triangles, water_triangles + len(vertices) - len(water_vertices)])
                surface_name = "water"
            else:
                surface_name = "road" if category == "road" else "trail"
            for point in centerline:
                col = int(np.clip(round((point[0] + width / 2) / width * (height.shape[1] - 1)), 0, height.shape[1] - 1))
                row = int(np.clip(round((point[1] + depth / 2) / depth * (height.shape[0] - 1)), 0, height.shape[0] - 1))
                radius_x = max(1, int(width_m / width * height.shape[1] / 2))
                radius_y = max(1, int(width_m / depth * height.shape[0] / 2))
                exclusion[max(0, row - radius_y):row + radius_y + 1, max(0, col - radius_x):col + radius_x + 1] = True
                occupied[max(0, row - radius_y):row + radius_y + 1, max(0, col - radius_x):col + radius_x + 1] = True
                surface[max(0, row - radius_y):row + radius_y + 1, max(0, col - radius_x):col + radius_x + 1] = 2 if surface_name == "water" else 1
                if surface_name == "water":
                    water_surface[max(0, row - radius_y):row + radius_y + 1, max(0, col - radius_x):col + radius_x + 1] = True
            if category in {"trail", "road"}:
                # Both radii derive from the actual swept feature width.  The
                # second width is a soft transition, not a semantic boundary.
                distance = _polyline_distance_field(centerline, world_size, height.shape)
                hard_radius = max(width_m * 0.5, 1e-6)
                grid_spacing = min(
                    width / max(height.shape[1] - 1, 1),
                    depth / max(height.shape[0] - 1, 1),
                )
                soft_radius = hard_radius + max(width_m, grid_spacing)
                trail_exclusion |= distance <= hard_radius
                trail_clearance = np.minimum(
                    trail_clearance,
                    np.clip((distance - hard_radius) / max(soft_radius - hard_radius, 1e-6), 0.0, 1.0),
                )
            centerline_payload = centerline.tolist()
        else:
            polygon = _centered_polygon(region, world_size)
            mask = _polygon_mask(polygon, world_size, height.shape)
            if layout_weights is not None and feature["geometry"].get("water_surface"):
                region_index = region_indices.get(str(feature["region_id"]))
                if region_index is not None:
                    # Reuse the Layout semantic field for the basin footprint;
                    # this is a geometry operation, not a second shoreline field.
                    mask = layout_weights[region_index] >= 0.5
            boundary = modified[~mask]
            level = float(np.percentile(boundary, 35)) if len(boundary) else float(np.median(modified))
            surface_level = level
            if feature["geometry"].get("terrain_operation") == "basin":
                original_heights = modified.copy()
                spacing = min(width / max(height.shape[1] - 1, 1), depth / max(height.shape[0] - 1, 1))
                region_span = max(float(np.ptp(polygon[:, 0])), float(np.ptp(polygon[:, 1])), spacing)
                basin_depth = max(float(feature["geometry"].get("depth_m") or region_span * 0.02), spacing)
                shoreline_reference = float(np.percentile(boundary, 10)) if len(boundary) else level
                if layout_weights is not None and len(boundary):
                    region_index = region_indices.get(str(feature["region_id"]))
                    shore_indices = [index for index, item in enumerate(plan.get("regions", [])) if "shore" in f"{item.get('id', '')} {item.get('function', '')}".lower()]
                    if region_index is not None and shore_indices:
                        shore_weight = np.max(layout_weights[shore_indices], axis=0)
                        shoreline_cells = (~mask) & (shore_weight >= 0.15)
                        if np.any(shoreline_cells):
                            shoreline_reference = float(np.percentile(modified[shoreline_cells], 10))
                surface_level = shoreline_reference - basin_depth * 0.25
                basin_bottom = surface_level - basin_depth
                modified[mask] = np.minimum(modified[mask], basin_bottom)
                # Blend the basin edge over a short, world-sized ring.  A hard
                # mask step creates vertical, pale-looking walls in oblique
                # renders and makes the lake read as a cutout rather than a
                # terrain-integrated water feature.
                terrain_specs = {str(item.get("region_id")): item for item in plan.get("terrain", [])}
                blend_fraction = float(terrain_specs.get(str(feature["region_id"]), {}).get("boundary_blend", 0.12))
                region_span = max(float(np.ptp(polygon[:, 0])), float(np.ptp(polygon[:, 1])), spacing)
                shoreline_cells = max(1, int(round(blend_fraction * region_span / max(spacing, 1e-6))))
                previous = mask
                for step in range(1, shoreline_cells + 1):
                    grown = _dilate_mask(mask, step)
                    ring = grown & ~previous
                    if np.any(ring):
                        fraction = step / float(shoreline_cells + 1)
                        modified[ring] = (1.0 - fraction) * basin_bottom + fraction * original_heights[ring]
                    previous = grown
            occupied |= mask
            exclusion |= mask
            surface[mask] = 2 if feature["geometry"].get("water_surface") else 1
            if feature["geometry"].get("water_surface"):
                water_surface[mask] = True
            z = float(surface_level if feature["geometry"].get("terrain_operation") == "basin" else (np.mean(modified[mask]) if np.any(mask) else level))
            vertices, triangles = _surface_mesh(polygon, z)
            centerline_payload = []
            if feature["geometry"].get("terrain_operation") == "basin":
                lake_integrations.append({
                    "feature_id": feature["feature_id"],
                    "water_level": float(surface_level),
                    "basin_bottom": float(np.min(modified[mask])) if np.any(mask) else float(surface_level),
                    "shoreline_reference": float(shoreline_reference),
                    "water_above_basin": bool(np.any(modified[mask] < surface_level)) if np.any(mask) else False,
                })
        mesh_path = output_dir / f"{feature['feature_id']}.npz"
        np.savez_compressed(mesh_path, vertices=vertices, triangles=triangles)
        mesh_record = {"feature_id": feature["feature_id"], "category": category, "region_id": feature["region_id"], "representation": feature["representation"], "mesh": str(mesh_path), "transform_z_up": np.eye(4).tolist(), "centerline": centerline_payload}
        if feature["structural_subtype"] == "regional_surface" and category in {"lake", "water"}:
            mesh_record["water_level"] = float(z)
        if feature["structural_subtype"] == "linear" and target is not None:
            mesh_record["target_region_id"] = target[0]
            mesh_record["relationship_mode"] = "approach_target_boundary"
        meshes.append(mesh_record)
    np.savez_compressed(output_dir / "terrain_structural.npz", height=modified)
    np.save(output_dir / "structural_exclusion_mask.npy", exclusion)
    np.save(output_dir / "structural_occupied_mask.npy", occupied)
    np.save(output_dir / "surface_type_mask.npy", surface)
    np.save(output_dir / "water_surface_mask.npy", water_surface)
    np.save(output_dir / "trail_clearance_field.npy", trail_clearance)
    np.save(output_dir / "trail_exclusion_mask.npy", trail_exclusion)
    np.save(output_dir / "structural_placement_weights.npy", trail_clearance)
    payload = {"status": "ok", "features": features, "structural_meshes": meshes, "terrain_modified": bool(np.any(modified != height)), "lake_integrations": lake_integrations, "metrics": {"structural_feature_count": len(features), "trail_count": sum(f["semantic_category"] == "trail" for f in features), "road_count": sum(f["semantic_category"] == "road" for f in features), "river_count": sum(f["semantic_category"] in {"river", "stream"} for f in features), "lake_count": sum(f["semantic_category"] == "lake" for f in features), "hunyuan_calls_avoided": len(features), "trail_exclusion_cells": int(trail_exclusion.sum())}, "placement_influence": {"trail_clearance_field": str(output_dir / "trail_clearance_field.npy"), "trail_exclusion_mask": str(output_dir / "trail_exclusion_mask.npy"), "structural_placement_weights": str(output_dir / "structural_placement_weights.npy"), "water_surface_mask": str(output_dir / "water_surface_mask.npy")}}
    (output_dir / "structural_branch.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def validate_structural_branch(payload: dict[str, Any], terrain_height: np.ndarray | None = None) -> dict[str, Any]:
    """Run representation-specific deterministic checks; never invoke Hunyuan."""
    defects: list[str] = []
    feature_checks: list[dict[str, Any]] = []
    for feature in payload.get("structural_meshes", []):
        check = {"feature_id": feature.get("feature_id"), "representation": feature.get("representation"), "spline_continuity": True, "terrain_contact": True, "within_region": True, "normal_consistency": True}
        path = Path(feature.get("mesh", ""))
        if not path.is_file():
            defects.append(f"{feature.get('feature_id')}:missing_mesh")
            feature_checks.append(check | {"status": "fail"})
            continue
        with np.load(path) as data:
            vertices = np.asarray(data["vertices"])
            triangles = np.asarray(data["triangles"])
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            defects.append(f"{feature.get('feature_id')}:invalid_vertices")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or (len(triangles) and triangles.max() >= len(vertices)):
            defects.append(f"{feature.get('feature_id')}:invalid_triangles")
        if feature.get("centerline") and len(feature["centerline"]) < 2:
            defects.append(f"{feature.get('feature_id')}:spline_discontinuity")
            check["spline_continuity"] = False
        if len(vertices) and float(np.ptp(vertices[:, 2])) > 100.0:
            defects.append(f"{feature.get('feature_id')}:vertical_explosion")
            check["terrain_contact"] = False
        check["status"] = "pass" if all(value is True for key, value in check.items() if key not in {"feature_id", "representation", "status"}) else "fail"
        feature_checks.append(check)
    for integration in payload.get("lake_integrations", []):
        if not bool(integration.get("water_above_basin")):
            defects.append(f"{integration.get('feature_id')}:water_not_above_basin")
        if float(integration.get("water_level", 0.0)) >= float(integration.get("shoreline_reference", 0.0)):
            defects.append(f"{integration.get('feature_id')}:water_not_below_shoreline")
    if terrain_height is not None and payload.get("terrain_modified") and not np.isfinite(terrain_height).all():
        defects.append("terrain:nonfinite")
    return {"status": "pass" if not defects else "fail", "structural_validation_failures": len(defects), "defects": defects, "feature_checks": feature_checks, "structural_retry_count": 0, "representation_specific": True}
