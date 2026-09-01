from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

import numpy as np

from .asset_semantics import infer_asset_type, normalize_object_semantics
from .schemas import Polygon, RegionPlan, ScenePlan, TerrainOperator, TerrainSpec, Vec2, ObjectSpec


@dataclass(frozen=True)
class LayoutResult:
    labels: np.ndarray
    weights: np.ndarray
    region_ids: tuple[str, ...]


def deterministic_layout(plan: ScenePlan, resolution: int = 128, softness: float = 0.12) -> LayoutResult:
    """Create semantic boundaries from region polygons, never from an image model.

    Planner coordinates are allowed to be either centered around zero (the
    synthetic fixture convention) or in the ``[0, world_size]`` frame used by
    the live planner.  Polygon membership is the source of hard boundaries;
    center distances remain a deterministic fallback for malformed/degenerate
    polygons.
    """
    width, height = plan.world_size_m
    polygons = [np.asarray([(point.x, point.y) for point in r.polygon.points], dtype=np.float64) for r in plan.regions]
    all_points = np.concatenate(polygons, axis=0) if polygons else np.empty((0, 2), dtype=np.float64)
    # Live planner polygons commonly use a lower-left origin while synthetic
    # plans use a centered world.  Select the frame from the authored bounds.
    positive_frame = bool(
        len(all_points)
        and np.all(all_points[:, 0] >= -1e-6)
        and np.all(all_points[:, 1] >= -1e-6)
        and np.max(all_points[:, 0]) <= width + 1e-6
        and np.max(all_points[:, 1]) <= height + 1e-6
    )
    x_min, y_min = (0.0, 0.0) if positive_frame else (-width / 2, -height / 2)
    xs = np.linspace(x_min, x_min + width, resolution, dtype=np.float64)
    ys = np.linspace(y_min, y_min + height, resolution, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    signed_distances = []
    areas = []
    for region, polygon in zip(plan.regions, polygons):
        if len(polygon) < 3:
            dx = (xx - region.center.x) / max(width, 1e-9)
            dy = (yy - region.center.y) / max(height, 1e-9)
            signed_distances.append(-(dx * dx + dy * dy))
            areas.append(float("inf"))
            continue
        x0, y0 = polygon[:, 0], polygon[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        inside = np.zeros(xx.shape, dtype=bool)
        for edge_x0, edge_y0, edge_x1, edge_y1 in zip(x0, y0, x1, y1):
            crosses = (edge_y0 > yy) != (edge_y1 > yy)
            denominator = edge_y1 - edge_y0
            safe_denominator = denominator if abs(denominator) > 1e-12 else 1e-12
            intersects = xx < (edge_x1 - edge_x0) * (yy - edge_y0) / safe_denominator + edge_x0
            inside ^= crosses & intersects

        # Minimum point-to-segment distance gives a stable soft boundary
        # without requiring scipy or a remote rasterization dependency.
        edge_distance = np.full(xx.shape, np.inf, dtype=np.float64)
        for edge_x0, edge_y0, edge_x1, edge_y1 in zip(x0, y0, x1, y1):
            ex, ey = edge_x1 - edge_x0, edge_y1 - edge_y0
            t = ((xx - edge_x0) * ex + (yy - edge_y0) * ey) / max(ex * ex + ey * ey, 1e-12)
            t = np.clip(t, 0.0, 1.0)
            distance = np.hypot(xx - (edge_x0 + t * ex), yy - (edge_y0 + t * ey))
            edge_distance = np.minimum(edge_distance, distance)
        signed_distances.append(np.where(inside, edge_distance, -edge_distance) / max(width, height, 1e-9))
        areas.append(abs(float(np.dot(x0, y1) - np.dot(y0, x1))) * 0.5)

    signed = np.stack(signed_distances, axis=0)
    containing = signed >= 0.0
    area_order = np.asarray(areas, dtype=np.float64)
    area_order[~np.isfinite(area_order)] = np.finfo(np.float64).max
    # Nested authored polygons (lake inside shoreline inside forest) should
    # select the most specific, smallest containing polygon.
    priority = np.broadcast_to(area_order[:, None, None], signed.shape)
    containing_priority = np.where(containing, priority, np.finfo(np.float64).max)
    selected = np.argmin(containing_priority, axis=0)
    has_containing = containing.any(axis=0)
    selected = np.where(has_containing, selected, np.argmax(signed, axis=0)).astype(np.int16)

    # Convert the priority result into explicit mutually-exclusive masks.  A
    # nested broad polygon (for example forest around a lake) must not retain
    # a global softmax advantage merely because it is far from its outer edge.
    labels = selected
    one_hot = np.zeros_like(signed, dtype=np.float64)
    rows, cols = np.indices(labels.shape)
    one_hot[labels, rows, cols] = 1.0

    # Only cells close to their selected region boundary receive a soft blend.
    # The selected region always keeps at least 55% mass, so labels and
    # argmax(weights) are definitionally consistent.
    # ``TerrainSpec.boundary_blend`` is authored per semantic region and is
    # the source of truth for local transition width.  Keep the function
    # argument as a compatibility fallback for plans without a matching
    # terrain specification.
    terrain_blend = {spec.region_id: float(spec.boundary_blend) for spec in plan.terrain}
    region_temperature = np.asarray(
        [max(terrain_blend.get(region.id, softness), 1e-4) for region in plan.regions],
        dtype=np.float64,
    )
    selected_temperature = region_temperature[labels]
    selected_distance = signed[labels, rows, cols]
    blend = 0.45 * np.clip(1.0 - np.abs(selected_distance) / selected_temperature, 0.0, 1.0)
    alternatives = np.exp(
        np.clip(-np.abs(signed) / region_temperature[:, None, None], -60.0, 0.0)
    )
    alternatives[labels, rows, cols] = 0.0
    alternative_sum = alternatives.sum(axis=0, keepdims=True)
    alternatives /= np.maximum(alternative_sum, 1e-12)
    weights = one_hot * (1.0 - blend[None, ...]) + alternatives * blend[None, ...]
    return LayoutResult(labels, weights.astype(np.float32), tuple(r.id for r in plan.regions))


def _box(cx: float, cy: float, rx: float, ry: float) -> Polygon:
    return Polygon(points=[Vec2(x=cx-rx, y=cy-ry), Vec2(x=cx+rx, y=cy-ry), Vec2(x=cx+rx, y=cy+ry), Vec2(x=cx-rx, y=cy+ry)])


def synthetic_plan(prompt: str) -> ScenePlan:
    """Deterministic test fixture. It is never used by live generation."""
    p = prompt.lower()
    if "desert" in p or "沙漠" in prompt:
        theme, names, centers, kinds = "desert city", ["walled_city", "dunes", "oasis"], [(0, 5), (0, 30), (-30, -24)], ["terrace", "dune", "erosion"]
        objects = [[("building", 12)], [("rock", 6)], [("palm", 8)]]
    elif "forest" in p or "lake" in p or "森林" in prompt or "湖" in prompt:
        theme, names, centers, kinds = "forest lake", ["lake", "forest", "cabins"], [(10, 0), (-25, 18), (28, -22)], ["erosion", "peak", "terrace"]
        objects = [[("rock", 4)], [("tree", 16)], [("cabin", 5)]]
    else:
        theme, names, centers, kinds = "medieval castle", ["castle_hill", "village", "farmland"], [(0, 25), (25, 0), (-28, -20)], ["peak", "erosion", "terrace"]
        objects = [[("castle", 1), ("tree", 4)], [("house", 10)], [("tree", 5), ("rock", 3)]]
    coverages = [0.34, 0.34, 0.32]
    regions = []
    for i, (name, (x, y)) in enumerate(zip(names, centers)):
        regions.append(RegionPlan(id=name, function=name.replace("_", " "), center=Vec2(x=x, y=y), polygon=_box(x, y, 18, 18), coverage=coverages[i], neighbors=[names[(i+1)%3]], objects=[ObjectSpec(category=c, count=n, density=min(1, n/20), appearance=theme, asset_type=infer_asset_type(c)) for c,n in objects[i]], appearance=theme))
    # Keep the ordinary fixtures stable while allowing an explicit structural
    # fixture prompt to exercise the complete trail/lake branch end to end.
    if any(token in p for token in ("trail", "structural")):
        lake_region = regions[0]
        forest_region = regions[1]
        regions[0] = lake_region.model_copy(update={
            "objects": [*lake_region.objects, ObjectSpec(
                category="lake", count=1, instance_strategy="region",
                appearance=theme, asset_type=infer_asset_type("lake"),
            )]
        })
        regions[1] = forest_region.model_copy(update={
            "objects": [*forest_region.objects, ObjectSpec(
                category="trail", count=1, instance_strategy="path",
                appearance="dirt trail", asset_type=infer_asset_type("trail"),
            )]
        })
    terrain = [TerrainSpec(region_id=n, mask_key=n, base_height=0.0, operators=[TerrainOperator(kind=k, strength=8.0 if k == "peak" else 3.0, scale=20.0)]) for n,k in zip(names,kinds)]
    return ScenePlan(theme=theme, world_size_m=(100.0, 100.0), regions=regions, terrain=terrain, materials={n: ("sand" if "desert" in theme else "grass") for n in names}, explicit_constraints=[prompt])


def stable_seed(prompt: str, seed: int) -> int:
    raw = hashlib.sha256(f"{seed}\0{prompt}".encode()).digest()
    return int.from_bytes(raw[:8], "little")


def target_world_size_from_environment() -> tuple[float, float]:
    """Read the horizontal world target used by live planning."""
    raw = os.getenv("WORLDCLAW_TARGET_WORLD_SIZE_M", "100,100")
    parts = [part.strip() for part in raw.replace("x", ",").split(",")]
    if len(parts) != 2:
        raise ValueError("WORLDCLAW_TARGET_WORLD_SIZE_M must be WIDTH,DEPTH in metres")
    try:
        width, depth = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError("WORLDCLAW_TARGET_WORLD_SIZE_M must contain numeric values") from error
    if not math.isfinite(width) or not math.isfinite(depth) or width <= 0 or depth <= 0:
        raise ValueError("WORLDCLAW_TARGET_WORLD_SIZE_M values must be finite and positive")
    return width, depth


def normalize_plan_semantics(plan: ScenePlan) -> ScenePlan:
    """Normalize object identity fields while preserving authored dimensions."""
    regions = []
    for region in plan.regions:
        objects = []
        for obj in region.objects:
            semantics = normalize_object_semantics(
                obj.category,
                asset_role=obj.asset_role,
                instance_strategy=obj.instance_strategy,
                density=obj.density,
            )
            objects.append(obj.model_copy(update=semantics))
        regions.append(region.model_copy(update={"objects": objects}))
    return plan.model_copy(update={"regions": regions})


def normalize_scene_plan(
    plan: ScenePlan,
    target_world_size: tuple[float, float] = (100.0, 100.0),
) -> ScenePlan:
    """Scale planner output into a compact world while preserving semantics.

    Coordinates may be centered or use a lower-left origin, so scaling around
    zero preserves either convention. Operational object counts are area-scaled
    because a compact world cannot accommodate thousands of large instances.
    """
    target_width, target_depth = target_world_size
    if (
        not math.isfinite(target_width)
        or not math.isfinite(target_depth)
        or target_width <= 0
        or target_depth <= 0
    ):
        raise ValueError("target_world_size must contain finite positive values")
    source_width, source_depth = plan.world_size_m
    scale_x = target_width / source_width
    scale_y = target_depth / source_depth
    area_ratio = scale_x * scale_y
    horizontal_scale = math.sqrt(area_ratio)

    def point(value: Vec2) -> Vec2:
        return Vec2(x=value.x * scale_x, y=value.y * scale_y)

    plan = normalize_plan_semantics(plan)
    regions = []
    for region in plan.regions:
        objects = []
        for obj in region.objects:
            objects.append(obj.model_copy(update={
                "count": max(1, int(round(obj.count * area_ratio))) if obj.count > 0 else 0,
            }))
        regions.append(region.model_copy(update={
            "center": point(region.center),
            "polygon": Polygon(points=[point(item) for item in region.polygon.points]),
            "objects": objects,
        }))

    terrain = []
    for spec in plan.terrain:
        operators = [
            operator.model_copy(update={"scale": operator.scale * horizontal_scale})
            for operator in spec.operators
        ]
        terrain.append(spec.model_copy(update={"operators": operators}))

    return plan.model_copy(update={
        "world_size_m": (float(target_width), float(target_depth)),
        "regions": regions,
        "terrain": terrain,
    })
