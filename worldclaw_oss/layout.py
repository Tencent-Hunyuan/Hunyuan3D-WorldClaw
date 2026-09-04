from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .asset_semantics import infer_asset_type, normalize_object_semantics
from .schemas import Polygon, RegionPlan, ScenePlan, TerrainOperator, TerrainSpec, Vec2, ObjectSpec


@dataclass(frozen=True)
class LayoutResult:
    labels: np.ndarray
    weights: np.ndarray
    region_ids: tuple[str, ...]
    generation: dict[str, Any] = field(default_factory=dict)


def _rasterize_layout(
    plan: ScenePlan,
    resolution: int = 128,
    softness: float = 0.12,
    polygons: list[np.ndarray] | None = None,
) -> LayoutResult:
    """Create semantic boundaries from region polygons, never from an image model.

    Planner coordinates are allowed to be either centered around zero (the
    synthetic fixture convention) or in the ``[0, world_size]`` frame used by
    the live planner.  Polygon membership is the source of hard boundaries;
    center distances remain a deterministic fallback for malformed/degenerate
    polygons.
    """
    width, height = plan.world_size_m
    polygons = polygons or [
        np.asarray([(point.x, point.y) for point in r.polygon.points], dtype=np.float64)
        for r in plan.regions
    ]
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

    # A provider may reuse the water polygon for a shoreline transition.  In
    # that case area-priority selection gives the later shoreline region zero
    # hard cells.  Recover a narrow, deterministic exterior annulus from the
    # water boundary while leaving authored polygons and all other regions
    # untouched.
    semantic_text = [
        f"{region.id} {region.function}".lower() for region in plan.regions
    ]
    semantic_tokens = [set(re.findall(r"[a-z0-9]+", text)) for text in semantic_text]
    water_indices = [
        index for index, tokens in enumerate(semantic_tokens)
        if tokens.intersection({"lake", "water", "pond", "river"})
        or any(
            str(obj.category).lower() in {"lake", "water", "pond", "river"}
            for obj in plan.regions[index].objects
        )
    ]
    transition_indices = [
        index for index, tokens in enumerate(semantic_tokens)
        if tokens.intersection({"shore", "shoreline", "coast", "coastal", "transition", "littoral"})
    ]
    annulus_width = max(min(width, height) * 0.01, max(width, height) / max(resolution, 1) * 2.0)
    for transition_index in transition_indices:
        if np.any(selected == transition_index) or not water_indices:
            continue
        water_index = min(
            water_indices,
            key=lambda index: abs(float(areas[index]) - float(areas[transition_index])),
        )
        # signed distances are normalized by the larger world dimension.
        water_boundary = signed[water_index]
        annulus = (
            (water_boundary < 0.0)
            & (water_boundary >= -annulus_width / max(width, height, 1e-9))
        )
        if not np.any(annulus):
            # At very low raster resolutions, guarantee at least a small
            # semantic footprint by selecting the nearest outside cell.
            annulus = (water_boundary < 0.0) & (
                water_boundary >= np.max(water_boundary[water_boundary < 0.0])
            )
        selected[annulus] = transition_index

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


def deterministic_layout(plan: ScenePlan, resolution: int = 128, softness: float = 0.12) -> LayoutResult:
    """Backward-compatible rasterization of the authored geometry prior.

    New generation code should use :func:`sample_layout`, while this helper
    remains useful for callers that explicitly need the unperturbed prior.
    """
    return _rasterize_layout(plan, resolution, softness)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return point-in-polygon using the same even/odd rule as rasterization."""
    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        if (y0 > y) != (y1 > y):
            denominator = y1 - y0
            crossing = (x1 - x0) * (y - y0) / (denominator if abs(denominator) > 1e-12 else 1e-12) + x0
            if x < crossing:
                inside = not inside
    return inside


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(float(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                     - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) * 0.5)


def _perturb_polygon(
    polygon: np.ndarray,
    rng: np.random.Generator,
    world_size: tuple[float, float],
    strength: float,
    frame_min: tuple[float, float],
) -> np.ndarray:
    """Apply bounded low-frequency geometry variation to a polygon prior."""
    if len(polygon) < 3:
        return polygon.copy()
    width, height = world_size
    centroid = polygon.mean(axis=0)
    vectors = polygon - centroid
    radius = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    # Low-frequency harmonics keep boundaries smooth and avoid pixel noise.
    amplitudes = rng.normal(0.0, strength, size=3)
    phases = rng.uniform(-math.pi, math.pi, size=3)
    radial_scale = 1.0
    for harmonic, (amplitude, phase) in zip((1, 2, 3), zip(amplitudes, phases)):
        radial_scale += amplitude * np.sin(harmonic * angles + phase)
    radial_scale = np.clip(radial_scale, 0.82, 1.18)
    center_shift = rng.normal(0.0, strength * 0.18, size=2) * np.asarray((width, height))
    varied = centroid + center_shift + vectors * radial_scale[:, None]
    x_min, y_min = frame_min
    x_max, y_max = x_min + width, y_min + height
    return np.column_stack((
        np.clip(varied[:, 0], x_min, x_max),
        np.clip(varied[:, 1], y_min, y_max),
    ))


def _component_count(mask: np.ndarray) -> int:
    """Count 4-connected components without requiring scipy."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    remaining = mask.copy()
    components = 0
    rows, cols = remaining.shape
    while remaining.any():
        start = tuple(np.argwhere(remaining)[0])
        queue = deque([start])
        remaining[start] = False
        components += 1
        while queue:
            row, col = queue.popleft()
            for next_row, next_col in (
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ):
                if 0 <= next_row < rows and 0 <= next_col < cols and remaining[next_row, next_col]:
                    remaining[next_row, next_col] = False
                    queue.append((next_row, next_col))
    return components


def _containment_pairs(plan: ScenePlan, polygons: list[np.ndarray]) -> list[tuple[int, int]]:
    """Infer authored containment from polygon area and centroid geometry."""
    areas = [_polygon_area(polygon) for polygon in polygons]
    by_id = {region.id: index for index, region in enumerate(plan.regions)}
    explicit: set[tuple[int, int]] = set()
    for child, region in enumerate(plan.regions):
        for parent_id in region.topology.inside:
            if parent_id in by_id:
                explicit.add((by_id[parent_id], child))
        for child_id in (*region.topology.contains, *region.topology.surrounds):
            if child_id in by_id:
                explicit.add((child, by_id[child_id]))
    pairs: list[tuple[int, int]] = []
    for child, child_polygon in enumerate(polygons):
        if len(child_polygon) < 3 or not areas[child]:
            continue
        child_center = child_polygon.mean(axis=0)
        for parent, parent_polygon in enumerate(polygons):
            if child == parent or areas[parent] <= areas[child] or len(parent_polygon) < 3:
                continue
            if _point_in_polygon(child_center, parent_polygon):
                pairs.append((parent, child))
    return sorted(set(pairs) | explicit)


def _adjacency_pairs(plan: ScenePlan) -> list[tuple[int, int]]:
    """Collect explicit hard adjacency declarations.

    ``neighbors`` predates the topology contract and is an approximate planning
    hint; treating it as a hard contact requirement would reject valid legacy
    plans whose polygons intentionally leave a navigable gap.
    """
    by_id = {region.id: index for index, region in enumerate(plan.regions)}
    pairs: set[tuple[int, int]] = set()
    for index, region in enumerate(plan.regions):
        for other_id in region.topology.adjacent_to:
            other = by_id.get(other_id)
            if other is not None and other != index:
                pairs.add(tuple(sorted((index, other))))
    return sorted(pairs)


def _repair_containment(
    polygons: list[np.ndarray],
    pairs: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Shrink child geometry only as needed to retain authored containment."""
    repaired = [polygon.copy() for polygon in polygons]
    for parent_index, child_index in pairs:
        parent = repaired[parent_index]
        child = repaired[child_index]
        if len(parent) < 3 or len(child) < 3:
            continue
        child_center = child.mean(axis=0)
        for _ in range(10):
            if all(_point_in_polygon(point, parent) for point in child):
                break
            child = child_center + (child - child_center) * 0.88
        repaired[child_index] = child
    return repaired


def _candidate_metrics(
    plan: ScenePlan,
    layout: LayoutResult,
    baseline: LayoutResult,
    polygons: list[np.ndarray],
    containment_pairs: list[tuple[int, int]],
    adjacency_pairs: list[tuple[int, int]],
    coverage_tolerance: float,
    score_weights: dict[str, float],
) -> dict[str, Any]:
    labels = layout.labels
    counts = np.bincount(labels.ravel(), minlength=len(plan.regions)).astype(np.float64)
    coverage = counts / max(float(labels.size), 1.0)
    coverage_error = float(np.mean(np.abs(coverage - np.asarray([r.coverage for r in plan.regions]))))
    components = [_component_count(labels == index) for index in range(len(plan.regions))]
    topology_valid = all(component == 1 for component in components)
    containment_valid = all(
        all(_point_in_polygon(point, polygons[parent]) for point in polygons[child])
        for parent, child in containment_pairs
    )
    topology_valid = topology_valid and containment_valid
    adjacency_contacts = []
    for first, second in adjacency_pairs:
        contacts = int(np.count_nonzero(
            ((labels[:-1, :] == first) & (labels[1:, :] == second))
            | ((labels[:-1, :] == second) & (labels[1:, :] == first))
            | ((labels[:, :-1] == first) & (labels[:, 1:] == second))
            | ((labels[:, :-1] == second) & (labels[:, 1:] == first))
        ))
        adjacency_contacts.append(contacts)
    adjacency_valid = all(contact > 0 for contact in adjacency_contacts)
    topology_valid = topology_valid and adjacency_valid
    boundary = (
        np.diff(labels, axis=0, prepend=labels[:1]) != 0
    ) | (
        np.diff(labels, axis=1, prepend=labels[:, :1]) != 0
    )
    boundary_smoothness = 1.0 - min(float(boundary.mean()) * 4.0, 1.0)
    diversity = float(np.mean(labels != baseline.labels))
    coverage_score = max(0.0, 1.0 - coverage_error * 8.0)
    topology_score = 1.0 if topology_valid else 0.0
    total_score = (
        score_weights["topology"] * topology_score
        + score_weights["coverage"] * coverage_score
        + score_weights["connectivity"] * (sum(component == 1 for component in components) / max(len(components), 1))
        + score_weights["adjacency"] * (sum(contact > 0 for contact in adjacency_contacts) / max(len(adjacency_contacts), 1))
        + score_weights["boundary_smoothness"] * boundary_smoothness
        + score_weights["diversity"] * diversity
    )
    return {
        "topology_valid": topology_valid,
        "containment_valid": containment_valid,
        "coverage": coverage.tolist(),
        "coverage_error": coverage_error,
        "connected_components": components,
        "connectivity_score": float(sum(component == 1 for component in components) / max(len(components), 1)),
        "adjacency_contacts": adjacency_contacts,
        "adjacency_valid": adjacency_valid,
        "adjacency_score": float(sum(contact > 0 for contact in adjacency_contacts) / max(len(adjacency_contacts), 1)),
        "boundary_smoothness_score": boundary_smoothness,
        "diversity_score": diversity,
        "topology_score": topology_score,
        "coverage_score": coverage_score,
        "total_score": total_score,
        "valid": topology_valid and coverage_error <= coverage_tolerance,
    }


def sample_layout(
    plan: ScenePlan,
    resolution: int = 128,
    seed: int = 0,
    candidate_count: int = 4,
    softness: float = 0.12,
    variation_strength: float = 0.065,
    coverage_tolerance: float = 0.20,
    score_weights: dict[str, float] | None = None,
) -> LayoutResult:
    """Sample, validate, score, and select a seeded semantic layout candidate."""
    if resolution < 4:
        raise ValueError("resolution must be at least 4")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not 0.0 <= variation_strength <= 0.5:
        raise ValueError("variation_strength must be between 0 and 0.5")
    if not 0.0 <= coverage_tolerance <= 1.0:
        raise ValueError("coverage_tolerance must be between 0 and 1")
    weights = {
        "topology": 5.0,
        "coverage": 2.0,
        "connectivity": 1.0,
        "adjacency": 1.0,
        "boundary_smoothness": 1.0,
        "diversity": 1.0,
    }
    if score_weights is not None:
        weights.update({str(key): float(value) for key, value in score_weights.items()})
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("score_weights must contain finite non-negative values")
    base_polygons = [
        np.asarray([(point.x, point.y) for point in region.polygon.points], dtype=np.float64)
        for region in plan.regions
    ]
    all_points = np.concatenate(base_polygons, axis=0) if base_polygons else np.empty((0, 2))
    positive_frame = bool(
        len(all_points)
        and np.all(all_points[:, 0] >= -1e-6)
        and np.all(all_points[:, 1] >= -1e-6)
        and np.max(all_points[:, 0]) <= plan.world_size_m[0] + 1e-6
        and np.max(all_points[:, 1]) <= plan.world_size_m[1] + 1e-6
    )
    frame_min = (0.0, 0.0) if positive_frame else (
        -plan.world_size_m[0] / 2.0, -plan.world_size_m[1] / 2.0,
    )
    baseline = _rasterize_layout(plan, resolution, softness, base_polygons)
    pairs = _containment_pairs(plan, base_polygons)
    adjacency_pairs = _adjacency_pairs(plan)
    candidates: list[tuple[LayoutResult, dict[str, Any]]] = []
    for candidate_id in range(candidate_count):
        rng = np.random.default_rng(np.random.SeedSequence([
            int(seed) & 0xFFFFFFFFFFFFFFFF, candidate_id,
        ]))
        strength = 0.0 if candidate_id == 0 else variation_strength
        polygons = [
            _perturb_polygon(polygon, rng, plan.world_size_m, strength, frame_min)
            for polygon in base_polygons
        ]
        polygons = _repair_containment(polygons, pairs)
        layout = _rasterize_layout(plan, resolution, softness, polygons)
        metrics = _candidate_metrics(
            plan, layout, baseline, polygons, pairs, adjacency_pairs,
            coverage_tolerance, weights,
        )
        metrics["candidate_id"] = candidate_id
        candidates.append((layout, metrics))
    valid = [(layout, metrics) for layout, metrics in candidates if metrics["valid"]]
    selected_layout, selected_metrics = max(
        valid or candidates,
        key=lambda item: (bool(item[1]["valid"]), float(item[1]["total_score"]), -int(item[1]["candidate_id"])),
    )
    generation = {
        "seed": int(seed),
        "candidate_count": int(candidate_count),
        "variation_strength": float(variation_strength),
        "coverage_tolerance": float(coverage_tolerance),
        "score_weights": weights,
        "selected_candidate": int(selected_metrics["candidate_id"]),
        "candidate_scores": [metrics for _, metrics in candidates],
        "topology_valid": bool(selected_metrics["topology_valid"]),
        "selected_metrics": selected_metrics,
    }
    return LayoutResult(
        selected_layout.labels, selected_layout.weights, selected_layout.region_ids, generation,
    )


def _box(cx: float, cy: float, rx: float, ry: float) -> Polygon:
    return Polygon(points=[Vec2(x=cx-rx, y=cy-ry), Vec2(x=cx+rx, y=cy-ry), Vec2(x=cx+rx, y=cy+ry), Vec2(x=cx-rx, y=cy+ry)])


def synthetic_plan(prompt: str) -> ScenePlan:
    """Load a deterministic synthetic fixture from data, never from Live input."""
    config_path = Path(__file__).with_name("synthetic_fixtures.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalized = prompt.casefold()
    templates = config.get("templates", [])
    template = next(
        (
            value for value in templates
            if any(str(token).casefold() in normalized for token in value.get("match_tokens", []))
        ),
        next(value for value in templates if value.get("default")),
    )
    theme = str(template["theme"])
    regions_config = list(template["regions"])
    names = [str(value["id"]) for value in regions_config]
    coverages = [float(value) for value in template.get("coverages", [0.34, 0.34, 0.32])]
    regions: list[RegionPlan] = []
    for index, value in enumerate(regions_config):
        x, y = (float(item) for item in value["center"])
        objects = [
            ObjectSpec(
                category=str(item["category"]), count=int(item["count"]),
                density=min(1.0, int(item["count"]) / 20.0), appearance=theme,
                asset_type=infer_asset_type(str(item["category"])),
            )
            for item in value.get("objects", [])
        ]
        half_extent = float(value.get("half_extent_m", 18.0))
        regions.append(RegionPlan(
            id=str(value["id"]), function=str(value.get("function", value["id"])).replace("_", " "),
            center=Vec2(x=x, y=y), polygon=_box(x, y, half_extent, half_extent),
            coverage=coverages[index], neighbors=[names[(index + 1) % len(names)]],
            objects=objects, appearance=theme,
        ))
    for feature in template.get("augmentations", []):
        if not any(str(token).casefold() in normalized for token in feature.get("match_tokens", [])):
            continue
        for item in feature.get("objects", []):
            region_index = next(index for index, region in enumerate(regions) if region.id == item["region_id"])
            object_spec = ObjectSpec(
                category=str(item["category"]), count=int(item.get("count", 1)),
                instance_strategy=str(item.get("instance_strategy", "scatter")),
                appearance=str(item.get("appearance", theme)),
                asset_type=infer_asset_type(str(item["category"])),
            )
            region = regions[region_index]
            regions[region_index] = region.model_copy(update={"objects": [*region.objects, object_spec]})
    terrain = [
        TerrainSpec(
            region_id=str(value["id"]), mask_key=str(value["id"]), base_height=0.0,
            operators=[TerrainOperator(
                kind=str(value["terrain"]["kind"]),
                strength=float(value["terrain"].get("strength", 3.0)),
                scale=float(value["terrain"].get("scale_m", 20.0)),
            )],
        )
        for value in regions_config
    ]
    materials = {name: str(template.get("materials", {}).get(name, "grass")) for name in names}
    world_size = tuple(float(value) for value in config.get("world_size_m", [100.0, 100.0]))
    return ScenePlan(theme=theme, world_size_m=world_size, regions=regions, terrain=terrain, materials=materials, explicit_constraints=[prompt])


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
