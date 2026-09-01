#!/usr/bin/env python3
"""Camera-ray terrain placement with projected-scale and contact optimization."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from workers.common import load_validated_stage_response, read_request, seed_everything, worker_args, write_response
from worldclaw_oss.asset_semantics import effective_asset_type, is_structural_feature, semantic_scale_range
from worldclaw_oss.asset_types import AssetType
from worldclaw_oss.geometry import cropped_intrinsics, gltf_vertices_to_z_up, pixel_ray, projected_bbox_scale
from worldclaw_oss.placement_constraints import (
    apply_hard_gate,
    footprint_intersects_mask,
    prepare_surface_masks,
    requirements_from_spec,
)
from worldclaw_oss.schemas import AssetInstance


# Process workers build these once in their initializer.  Open3D scenes are
# not reliably picklable, so only the terrain archive path crosses the process
# boundary and each worker owns its BVH/cache.
_PLACEMENT_TERRAIN_SCENE = None
_PLACEMENT_TERRAIN_SAMPLE = None
_PLACEMENT_TOLERANCE_FRACTION = 0.05
_PLACEMENT_OBJECT_SCENE_CACHE: dict[str, Any] = {}


def _init_parallel_placement(terrain_path: str, world_size: tuple[float, float], tolerance_fraction: float) -> None:
    global _PLACEMENT_TERRAIN_SCENE, _PLACEMENT_TERRAIN_SAMPLE
    global _PLACEMENT_TOLERANCE_FRACTION, _PLACEMENT_OBJECT_SCENE_CACHE
    with np.load(terrain_path) as terrain:
        _PLACEMENT_TERRAIN_SCENE = raycast_scene(terrain["vertices"], terrain["triangles"])
        _PLACEMENT_TERRAIN_SAMPLE = terrain_sampler(terrain["height"], world_size)
    _PLACEMENT_TOLERANCE_FRACTION = float(tolerance_fraction)
    _PLACEMENT_OBJECT_SCENE_CACHE = {}


def _parallel_place_one(item: dict[str, Any]) -> tuple[str, list[list[float]], dict[str, Any]]:
    if _PLACEMENT_TERRAIN_SCENE is None or _PLACEMENT_TERRAIN_SAMPLE is None:
        raise RuntimeError("parallel placement worker was not initialized")
    _, transform, metrics = place_instance(
        item,
        _PLACEMENT_TERRAIN_SCENE,
        _PLACEMENT_TERRAIN_SAMPLE,
        _PLACEMENT_TOLERANCE_FRACTION,
        object_scene_cache=_PLACEMENT_OBJECT_SCENE_CACHE,
        return_mesh=False,
    )
    return item["id"], transform.tolist(), metrics


def raycast_scene(vertices: np.ndarray, triangles: np.ndarray):
    import open3d as o3d
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def cast(scene, origin: np.ndarray, direction: np.ndarray) -> tuple[float, np.ndarray] | None:
    import open3d as o3d
    ray = np.concatenate((origin, direction)).astype(np.float32)[None, :]
    result = scene.cast_rays(o3d.core.Tensor(ray))
    distance = float(result["t_hit"].numpy()[0])
    if not math.isfinite(distance):
        return None
    return distance, origin + direction * distance


def terrain_sampler(height: np.ndarray, world_size: tuple[float, float]):
    rows, columns = height.shape
    width, depth = world_size

    def sample(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """Bilinear height lookup for scalars or broadcastable NumPy arrays."""
        x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        gx = np.clip((x_values + width / 2) / width * (columns - 1), 0, columns - 1)
        gy = np.clip((y_values + depth / 2) / depth * (rows - 1), 0, rows - 1)
        x0, y0 = np.floor(gx).astype(np.intp), np.floor(gy).astype(np.intp)
        x1, y1 = np.minimum(columns - 1, x0 + 1), np.minimum(rows - 1, y0 + 1)
        tx, ty = gx - x0, gy - y0
        result = (
            height[y0, x0] * (1 - tx) * (1 - ty)
            + height[y0, x1] * tx * (1 - ty)
            + height[y1, x0] * (1 - tx) * ty
            + height[y1, x1] * tx * ty
        )
        return float(result) if result.ndim == 0 else result

    return sample


def yaw_rotation(source_direction: np.ndarray, target_direction: np.ndarray) -> np.ndarray:
    source_yaw = math.atan2(source_direction[1], source_direction[0])
    target_yaw = math.atan2(target_direction[1], target_direction[0])
    angle = target_yaw - source_yaw
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def project_bbox(vertices: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray) -> tuple[float, float, float, float]:
    world_to_camera = np.linalg.inv(camera_to_world)
    homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    valid = camera[:, 2] > 1e-6
    if not np.any(valid):
        raise RuntimeError("object mesh is behind its reconstruction camera")
    projected = (intrinsics @ camera[valid].T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return (
        float(projected[:, 0].min()), float(projected[:, 1].min()),
        float(projected[:, 0].max()), float(projected[:, 1].max()),
    )


def support_vertices(vertices: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
    """Return the lower support band once, before the placement search."""
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    return vertices[vertices[:, 2] <= zmin + max((zmax - zmin) * 0.1, tolerance)]


def contact_metrics(
    vertices: np.ndarray, sample_height, tolerance: float,
    support: np.ndarray | None = None,
) -> tuple[float, float, float]:
    bottom = support if support is not None else support_vertices(vertices, tolerance)
    if len(bottom) == 0:
        return 0.0, math.inf, math.inf
    signed = bottom[:, 2] - np.asarray(sample_height(bottom[:, 0], bottom[:, 1]), dtype=float)
    ratio = float(np.mean(np.abs(signed) <= tolerance)) if len(signed) else 0.0
    penetration = float(max(0.0, -signed.min())) if len(signed) else math.inf
    floating = float(max(0.0, signed.min())) if len(signed) else math.inf
    return ratio, penetration, floating


def canonical_object_camera(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    focal = 0.9 * max(width, height)
    intrinsics = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    camera_to_world = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -2.5],
        [0.0, -1.0, 0.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return intrinsics, camera_to_world


def load_mesh(path: Path):
    import trimesh
    value = trimesh.load(path, force="mesh", process=False)
    if isinstance(value, trimesh.Scene):
        value = trimesh.util.concatenate(tuple(value.geometry.values()))
    # Hunyuan and other imported GLBs are glTF Y-up. Normalize before all
    # contact, scale, and terrain-height calculations in the internal Z-up frame.
    value.vertices = gltf_vertices_to_z_up(value.vertices).astype(np.float32)
    return value


def label_at(labels: np.ndarray, world_size: tuple[float, float], x: float, y: float) -> int:
    width, depth = world_size
    column = int(np.clip(round((x + width / 2) / width * (labels.shape[1] - 1)), 0, labels.shape[1] - 1))
    row = int(np.clip(round((y + depth / 2) / depth * (labels.shape[0] - 1)), 0, labels.shape[0] - 1))
    return int(labels[row, column])


def terrain_slope(height: np.ndarray, world_size: tuple[float, float]) -> np.ndarray:
    width, depth = world_size
    dy, dx = np.gradient(height, depth / (height.shape[0] - 1), width / (height.shape[1] - 1))
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def centered_polygon(region: dict, world_size: tuple[float, float]) -> np.ndarray:
    """Convert planner polygon coordinates to the centered terrain frame."""
    points = np.asarray([[p["x"], p["y"]] for p in region.get("polygon", {}).get("points", [])], dtype=float)
    if len(points) < 3:
        center = np.asarray([region["center"]["x"], region["center"]["y"]], dtype=float)
        points = center[None, :] + np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    width, depth = world_size
    if points[:, 0].min() >= -1e-6 and points[:, 0].max() <= width + 1e-6:
        points[:, 0] -= width / 2
    if points[:, 1].min() >= -1e-6 and points[:, 1].max() <= depth + 1e-6:
        points[:, 1] -= depth / 2
    return points


def _distance_to_mask(mask: np.ndarray, world_size: tuple[float, float]) -> np.ndarray:
    """Approximate world-unit distance to a rasterized semantic region."""
    points = np.argwhere(mask)
    if len(points) == 0:
        return np.full(mask.shape, np.inf, dtype=np.float32)
    # Only boundary cells are needed for distance to a filled region. Keeping
    # the nearest-point set small prevents quadratic memory growth on large
    # layout grids.
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
        & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    boundary = np.argwhere(mask & ~interior)
    if len(boundary):
        points = boundary
    rows, cols = mask.shape
    spacing = np.asarray([
        world_size[1] / max(rows - 1, 1),
        world_size[0] / max(cols - 1, 1),
    ], dtype=np.float32)
    out = np.empty(mask.shape, dtype=np.float32)
    grid = np.argwhere(np.ones(mask.shape, dtype=bool))
    for start in range(0, len(grid), 4096):
        chunk = grid[start:start + 4096]
        delta = (chunk[:, None, :] - points[None, :, :]) * spacing
        out[chunk[:, 0], chunk[:, 1]] = np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)
    return out


def structure_mesh(region: dict, category: str, count_index: int, world_size: tuple[float, float], sample_height, output_path: Path):
    """Build a terrain-conforming strip for trails, roads, rivers, or streams."""
    import trimesh

    polygon = centered_polygon(region, world_size)
    center = polygon.mean(axis=0)
    covariance = np.cov((polygon - center).T) if len(polygon) > 2 else np.eye(2)
    direction = np.linalg.eigh(covariance)[1][:, -1]
    if direction[0] < 0:
        direction = -direction
    extent = float(np.max(np.abs((polygon - center) @ direction)))
    extent = max(extent, min(world_size) * 0.12)
    lateral = np.asarray([-direction[1], direction[0]])
    phase = count_index * 1.7
    samples = np.linspace(-extent, extent, 32)
    width_by_category = {
        "river": 10.0, "rivers": 10.0, "stream": 6.0, "streams": 6.0,
        "road": 7.0, "roads": 7.0, "trail": 4.0, "trails": 4.0,
        "winding_trail": 4.0, "winding_trails": 4.0,
    }
    strip_width = width_by_category.get(category.lower(), 4.0)
    points = []
    for distance in samples:
        meander = lateral * (0.11 * extent * math.sin(distance / max(extent, 1.0) * 3.0 + phase))
        point = center + direction * distance + meander
        points.append(point)
    verts = []
    for point in points:
        tangent = direction + lateral * (0.33 * math.cos(float((point - center) @ direction) / max(extent, 1.0) * 3.0 + phase))
        tangent /= max(np.linalg.norm(tangent), 1e-8)
        side = np.asarray([-tangent[1], tangent[0]]) * strip_width / 2
        for edge in (-side, side):
            x, y = point + edge
            verts.append([x, y, sample_height(float(x), float(y)) + 0.04])
    # Reverse the strip winding so generated surface normals point upward.
    faces = [[i, i + 2, i + 3, i + 1] for i in range(0, len(verts) - 2, 2)]
    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float32), faces=np.asarray(faces, dtype=np.int64), process=False)
    _ = mesh.vertex_normals
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path, file_type="glb")
    return mesh


def overlaps(candidate: tuple[float, float, float, float], occupied, threshold=0.05) -> bool:
    if hasattr(occupied, "overlaps"):
        return occupied.overlaps(candidate, threshold)
    ax0, ay0, ax1, ay1 = candidate
    for bx0, by0, bx1, by1 in occupied:
        intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
        area = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
        if area > 0 and intersection / area > threshold:
            return True
    return False


class OccupiedSpatialHash:
    """Broad-phase index for footprint overlap checks during scatter."""

    def __init__(self, cell_size: float):
        self.cell_size = max(float(cell_size), 1.0)
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.boxes: list[tuple[float, float, float, float]] = []

    def _keys(self, box):
        x0, y0, x1, y1 = box
        ix0, ix1 = math.floor(x0 / self.cell_size), math.floor(x1 / self.cell_size)
        iy0, iy1 = math.floor(y0 / self.cell_size), math.floor(y1 / self.cell_size)
        return ((ix, iy) for ix in range(ix0, ix1 + 1) for iy in range(iy0, iy1 + 1))

    def overlaps(self, candidate, threshold=0.05):
        seen = set()
        for key in self._keys(candidate):
            for index in self.cells.get(key, ()):
                if index in seen:
                    continue
                seen.add(index)
                if overlaps(candidate, [self.boxes[index]], threshold):
                    return True
        return False

    def add(self, box):
        index = len(self.boxes)
        self.boxes.append(box)
        for key in self._keys(box):
            self.cells.setdefault(key, []).append(index)


def scatter_environment(
    prototypes: list[dict], plan: dict, labels: np.ndarray, height: np.ndarray,
    sample_height, seed: int, tolerance_fraction: float, output_root: Path,
    exclusion_mask: np.ndarray | None = None,
    layout_weights: np.ndarray | None = None,
    structural_placement_weights: np.ndarray | None = None,
    trail_exclusion_mask: np.ndarray | None = None,
    surface_masks: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    world_size = tuple(plan["world_size_m"])
    width, depth = world_size
    slope = terrain_slope(height, world_size)
    if layout_weights is None:
        layout_weights = np.zeros((len(plan.get("regions", [])), *labels.shape), dtype=np.float32)
        for index in range(len(plan.get("regions", []))):
            layout_weights[index] = labels == index
    layout_weights = np.asarray(layout_weights, dtype=np.float32)
    if layout_weights.ndim != 3 or layout_weights.shape[1:] != labels.shape:
        raise ValueError("layout_weights shape must be [regions, rows, cols]")
    if structural_placement_weights is None:
        structural_placement_weights = np.ones_like(height, dtype=np.float32)
    structural_placement_weights = np.asarray(structural_placement_weights, dtype=np.float32)
    if structural_placement_weights.shape != labels.shape:
        raise ValueError("structural placement influence shape does not match layout")
    if trail_exclusion_mask is None:
        trail_exclusion_mask = np.zeros_like(labels, dtype=bool)
    surface_masks = prepare_surface_masks(dict(surface_masks or {}), labels.shape)
    surface_masks.setdefault("trail", np.asarray(trail_exclusion_mask, dtype=bool))
    if exclusion_mask is not None:
        surface_masks.setdefault("structural_exclusion", np.asarray(exclusion_mask, dtype=bool))
    distance_fields = {
        name: _distance_to_mask(np.asarray(mask, dtype=bool), world_size)
        for name, mask in surface_masks.items()
        if np.asarray(mask).shape == labels.shape and np.any(mask)
    }
    prototype_by_key = {(item.get("region_id"), item["category"]): item for item in prototypes}
    prototype_by_category = {item["category"]: item for item in prototypes}
    prototype_by_type = {(item.get("region_id"), item.get("asset_type")): item for item in prototypes}
    assets, diagnostics = [], []
    occupied = OccupiedSpatialHash(max(min(width, depth) / 64.0, 4.0))
    prototype_mesh_cache = {}
    prototype_support_cache = {}
    for region_index, region in enumerate(plan["regions"]):
        allowed = np.ones_like(labels, dtype=bool)
        region_weight = np.clip(layout_weights[region_index], 0.0, 1.0)
        base_allowed = (slope <= 32.0) & allowed & (region_weight >= 0.05)
        cursor = 0
        # Reserve scarce, non-overlapping solid-object locations before dense
        # vegetation. Tree canopies may overlap each other, but they must not
        # consume all candidates needed by cabins and rocks.
        specs = sorted(
            region.get("objects", []),
            key=lambda value: (
                1
                if str(value.get("density", "")).strip().lower() == "dense"
                else 0
            ),
        )
        for spec in specs:
            asset_type = effective_asset_type(
                spec["category"], spec.get("asset_role", spec.get("asset_type"))
            )
            if is_structural_feature(spec["category"], spec.get("asset_role", spec.get("asset_type"))):
                diagnostics.append({"asset_id": f"{region['id']}_{spec['category']}", "category": spec["category"], "placement": "structural_branch", "place_allowed": False})
                continue
            prototype = prototype_by_key.get((region["id"], spec["category"]))
            if prototype is None:
                prototype = prototype_by_category.get(spec["category"])
            # Only reusable semantic families may share a prototype. Solid
            # objects are category-specific: a cabin must never become the
            # fallback mesh for rocks or another building.
            if prototype is None and asset_type in {
                AssetType.REUSABLE_PROTOTYPE,
                AssetType.PROCEDURAL_NATIVE,
                AssetType.SCATTER_DETAIL,
            }:
                prototype = prototype_by_type.get((region["id"], asset_type.value))
            if prototype is None and asset_type in {
                AssetType.REUSABLE_PROTOTYPE,
                AssetType.PROCEDURAL_NATIVE,
                AssetType.SCATTER_DETAIL,
            }:
                prototype = next((item for item in prototypes if item.get("asset_type") == asset_type.value), None)
            if prototype is None:
                # Preflight/Stage 3 may intentionally reroute or drop a
                # category (for example a surface/terrain feature).  A
                # missing validated prototype is therefore a terminal skip,
                # not a placement-worker failure.  Keep the decision visible
                # in diagnostics so the manifest explains the lower count.
                diagnostics.append({
                    "asset_id": f"{region['id']}_{spec['category']}",
                    "category": spec["category"],
                    "placement": "skipped_no_validated_prototype",
                    "requested_count": int(spec.get("count", 0)),
                    "reason": "no validated environment prototype; category was dropped or rerouted before Place",
                    "place_allowed": False,
                })
                continue
            prototype_path = str(Path(prototype["mesh"]).resolve())
            mesh = prototype_mesh_cache.get(prototype_path)
            if mesh is None:
                mesh = load_mesh(Path(prototype_path))
                prototype_mesh_cache[prototype_path] = mesh
            vertices = np.asarray(mesh.vertices, dtype=float)
            support_template = prototype_support_cache.get(prototype_path)
            if support_template is None:
                support_template = support_vertices(vertices)
                prototype_support_cache[prototype_path] = support_template
            low, high = semantic_scale_range(spec["category"])
            category = str(spec["category"]).strip().lower()
            requirements = requirements_from_spec(spec)
            gate = apply_hard_gate(base_allowed, requirements, surface_masks)
            candidates = np.argwhere(gate).tolist()
            sampling_weight = region_weight * structural_placement_weights
            for surface_name, falloff in requirements.distance_preferences:
                distance = distance_fields.get(surface_name)
                if distance is not None:
                    sampling_weight *= np.exp(-distance / max(falloff, 1e-3))
            candidates = [item for item in candidates if sampling_weight[item[0], item[1]] > 1e-6]
            if not candidates:
                # Preserve a deterministic nearest-feasible fallback for
                # degenerate planner regions while retaining soft weighting.
                candidates = np.argwhere(gate).tolist()
                candidates.sort(key=lambda item: -float(sampling_weight[item[0], item[1]]))
            candidates.sort(key=lambda item: -math.log(max(rng.random(), 1e-12)) / max(float(sampling_weight[item[0], item[1]]), 1e-6))
            placed = 0
            attempts = 0
            while placed < int(spec["count"]) and cursor < len(candidates):
                row, column = candidates[cursor]
                cursor += 1
                attempts += 1
                x = -width / 2 + column * width / (labels.shape[1] - 1)
                y = -depth / 2 + row * depth / (labels.shape[0] - 1)
                scale = rng.uniform(low, high)
                angle = rng.uniform(-math.pi, math.pi)
                c, s = math.cos(angle), math.sin(angle)
                rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                rotated = (rotation @ (vertices * scale).T).T
                support = (rotation @ (support_template * scale).T).T
                terrain_support = sample_height(x + support[:, 0], y + support[:, 1]) - support[:, 2]
                z = float(np.max(terrain_support)) if len(terrain_support) else sample_height(x, y)
                world_vertices = rotated + [x, y, z]
                bounds = world_vertices[:, :2]
                footprint = (
                    float(bounds[:, 0].min()), float(bounds[:, 1].min()),
                    float(bounds[:, 0].max()), float(bounds[:, 1].max()),
                )
                forbidden = set(requirements.forbidden_support_surfaces)
                if requirements.requires_dry_support:
                    forbidden.add("water")
                if requirements.avoid_structural_exclusion:
                    forbidden.add("structural_exclusion")
                if any(
                    name in surface_masks and footprint_intersects_mask(footprint, surface_masks[name], world_size)
                    for name in forbidden
                ):
                    continue
                required = [surface_masks[name] for name in requirements.required_support_surfaces if name in surface_masks]
                if required and not any(footprint_intersects_mask(footprint, mask, world_size) for mask in required):
                    continue
                # Overlap tolerance is a placement requirement.  Density is a
                # generic distribution modifier and remains only the fallback
                # for archived plans that predate placement_profile.
                overlap_threshold = requirements.footprint_overlap_threshold
                if overlap_threshold is None:
                    overlap_threshold = 0.65 if str(spec.get("density", "")).strip().lower() == "dense" else 0.05
                if overlaps(footprint, occupied, threshold=overlap_threshold):
                    continue
                object_height = max(float(np.ptp(world_vertices[:, 2])), 0.1)
                ratio, penetration, floating = contact_metrics(
                    world_vertices, sample_height, max(0.01, object_height * tolerance_fraction),
                    support=world_vertices[world_vertices[:, 2] <= world_vertices[:, 2].min() + max(object_height * 0.1, 0.02)],
                )
                transform = np.eye(4)
                transform[:3, :3] = rotation * scale
                transform[:3, 3] = [x, y, z]
                asset_id = f"{region['id']}_{spec['category']}_{placed:04d}"
                asset = AssetInstance(
                    id=asset_id, category=spec["category"], asset_type=asset_type,
                    source_image=prototype["source_image"], mask=f"layout:{region['id']}",
                    mesh=prototype["mesh"], material=spec["category"],
                    transform_z_up=transform.tolist(), region_id=region["id"],
                    contact_ratio=ratio, source_model=prototype["model"]["model_id"], synthetic=False,
                )
                assets.append(asset.model_dump(mode="json"))
                diagnostics.append({
                    "asset_id": asset_id, "placement": "mask_scatter", "contact_ratio": ratio,
                    "penetration_m": penetration, "floating_m": floating,
                    "object_height_m": object_height, "slope_degrees": float(slope[row, column]),
                })
                occupied.add(footprint)
                placed += 1
            if placed < int(spec["count"]):
                raise RuntimeError(
                    f"could place only {placed}/{spec['count']} non-overlapping {spec['category']} "
                    f"instances in region {region['id']}"
                )
    return assets, diagnostics


def place_instance(
    item: dict[str, Any], terrain_scene, terrain_sample, tolerance_fraction: float,
    object_scene_cache: dict[str, Any] | None = None,
    return_mesh: bool = True,
):
    camera = item.get("terrain_camera")
    if not camera:
        raise RuntimeError(f"{item['id']}: terrain camera metadata is missing")
    terrain_k = np.asarray(camera["intrinsics"], dtype=float)
    terrain_c2w = np.asarray(camera["camera_to_world"], dtype=float)
    centroid = tuple(float(value) for value in item["centroid_xy"])
    terrain_origin, terrain_direction = pixel_ray(centroid, terrain_k, terrain_c2w)
    terrain_hit = cast(terrain_scene, terrain_origin, terrain_direction)
    terrain_ray_fallback = terrain_hit is None
    if terrain_hit is None:
        # Generated region images can place a centroid just outside the finite
        # terrain bounds. Keep the reconstructed object and use a valid central
        # terrain anchor instead of discarding the whole scene.
        terrain_anchor = np.asarray([0.0, 0.0, terrain_sample(0.0, 0.0)], dtype=float)
        terrain_depth = float(np.linalg.norm(terrain_anchor - terrain_origin))
    else:
        terrain_depth, terrain_anchor = terrain_hit

    mesh = load_mesh(Path(item["mesh"]))
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.faces, dtype=np.uint32)
    source_size = item.get("source_size") or [512, 512]
    object_k, object_c2w = canonical_object_camera(int(source_size[0]), int(source_size[1]))
    object_pixel = (float(source_size[0]) / 2, float(source_size[1]) / 2)
    object_origin, object_direction = pixel_ray(object_pixel, object_k, object_c2w)
    mesh_key = str(Path(item["mesh"]).resolve())
    object_scene = object_scene_cache.get(mesh_key) if object_scene_cache is not None else None
    if object_scene is None:
        object_scene = raycast_scene(vertices, triangles)
        if object_scene_cache is not None:
            object_scene_cache[mesh_key] = object_scene
    object_hit = cast(object_scene, object_origin, object_direction)
    object_anchor = object_hit[1] if object_hit is not None else np.asarray(mesh.centroid)

    target_bbox = tuple(float(value) for value in item["bbox_xyxy"])
    current_bbox = project_bbox(vertices, object_k, object_c2w)
    initial_scale = float(np.clip(projected_bbox_scale(target_bbox, current_bbox), 0.05, 100.0))
    rotation = yaw_rotation(object_direction, -terrain_direction)
    object_height = max(float(np.ptp(vertices[:, 2])) * initial_scale, 0.1)
    tolerance = max(object_height * tolerance_fraction, 0.01)
    local_support = support_vertices(vertices)
    best = None
    def evaluate(depth_offset_values, factor_values):
        nonlocal best
        for depth_offset in depth_offset_values:
            depth = terrain_depth + float(depth_offset)
            ray_anchor = terrain_origin + depth * terrain_direction
            for factor in factor_values:
                scale = initial_scale * float(factor)
                translation = ray_anchor - rotation @ (object_anchor * scale)
                support_world = (rotation @ (local_support * scale).T).T + translation
                ratio, penetration, floating = contact_metrics(
                    support_world, terrain_sample, tolerance, support=support_world
                )
                score = ratio - 4.0 * penetration / object_height - 2.0 * floating / object_height
                # Contact scoring only needs the support band. Defer the full
                # mesh transform until the winning candidate is known.
                candidate = (score, ratio, penetration, floating, scale, translation)
                if best is None or candidate[0] > best[0][0]:
                    best = (candidate, float(depth_offset), float(factor))

    def world_vertices_for(scale, translation):
        return (rotation @ (vertices * scale).T).T + translation

    coarse_offsets = np.linspace(-object_height, object_height, 5)
    coarse_factors = np.linspace(0.7, 1.3, 5)
    evaluate(coarse_offsets, coarse_factors)
    _, coarse_offset, coarse_factor = best
    offset_step = object_height / 4.0
    factor_step = 0.15
    evaluate(
        np.linspace(coarse_offset - offset_step, coarse_offset + offset_step, 3),
        np.linspace(max(0.7, coarse_factor - factor_step), min(1.3, coarse_factor + factor_step), 3),
    )
    assert best is not None
    candidate, _, _ = best
    _, ratio, penetration, floating, scale, translation = candidate
    world_vertices = world_vertices_for(scale, translation)
    transform = np.eye(4)
    transform[:3, :3] = rotation * scale
    transform[:3, 3] = translation
    return (mesh if return_mesh else None), transform, {
        "contact_ratio": ratio,
        "penetration_m": penetration,
        "floating_m": floating,
        "object_height_m": float(np.ptp(world_vertices[:, 2])),
        "terrain_anchor": terrain_anchor.tolist(),
        "terrain_ray_depth": terrain_depth,
        "terrain_ray_fallback": terrain_ray_fallback,
        "initial_projected_scale": initial_scale,
        "final_scale": scale,
        "object_camera": {
            "intrinsics": object_k.tolist(),
            "camera_to_world": object_c2w.tolist(),
            "source": "canonical Hunyuan3D preprocessing camera",
        },
    }


def _placement_worker_count(asset_count: int) -> int:
    """Resolve a bounded process count; tiny jobs stay serial."""
    if asset_count < 8:
        return 1
    raw = os.getenv("WORLDCLAW_PLACEMENT_WORKERS", "")
    if raw:
        try:
            requested = min(8, max(0, int(raw)))
        except ValueError as error:
            raise ValueError("WORLDCLAW_PLACEMENT_WORKERS must be an integer") from error
        if requested == 0:
            return 1
    else:
        # Four workers won the live benchmark; callers can opt into 5-8 when
        # the host has enough memory and the workload is substantially larger.
        requested = 4
    return min(requested, asset_count)


def _place_reconstruction_assets(
    items: list[dict[str, Any]], terrain_path: Path,
    terrain_scene, sample_height, world_size: tuple[float, float],
    tolerance_fraction: float,
) -> tuple[list[tuple[dict[str, Any], Any, np.ndarray, dict[str, Any]]], int, bool]:
    """Place reconstruction assets, reusing one object BVH per worker process."""
    if not items:
        return [], 1, False
    workers = _placement_worker_count(len(items))
    export_placed_meshes = os.getenv("WORLDCLAW_EXPORT_PLACED_MESHES", "0").lower() in {"1", "true", "yes"}

    def serial_results():
        cache: dict[str, Any] = {}
        local_scene = terrain_scene
        if local_scene is None:
            with np.load(terrain_path) as terrain:
                local_scene = raycast_scene(terrain["vertices"], terrain["triangles"])
        results = []
        for item in items:
            mesh, transform, metrics = place_instance(
                item, local_scene, sample_height, tolerance_fraction,
                object_scene_cache=cache, return_mesh=export_placed_meshes,
            )
            results.append((item, mesh, transform, metrics))
        return results

    if workers <= 1:
        return serial_results(), 1, False

    # The archive is loaded in each child once. This avoids pickling an
    # Open3D RaycastingScene and keeps the per-task payload small.
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_parallel_placement,
            initargs=(str(terrain_path), world_size, tolerance_fraction),
        ) as executor:
            chunksize = max(1, min(16, len(items) // (workers * 4) or 1))
            indexed = list(executor.map(_parallel_place_one, items, chunksize=chunksize))
    except Exception:
        # A deployment may have a restricted multiprocessing runtime. Keep a
        # correct serial fallback; callers still receive the worker count used.
        return serial_results(), 1, False

    results = []
    for item, (_asset_id, transform_data, metrics) in zip(items, indexed):
        results.append((item, None, np.asarray(transform_data, dtype=float), metrics))
    return results, workers, True


def main():
    args = worker_args()
    request = read_request(args.request)
    seed_everything(int(request["seed"]))
    work_dir = Path(request["work_dir"])
    reconstruction = load_validated_stage_response(work_dir, "reconstruction")
    environment = load_validated_stage_response(work_dir, "environment_assets")
    # Hard Place gate: only terminal pass/fallback_pass reports are eligible.
    # Never recover failed or dropped source assets from the raw upstream file.
    def placeable(response: dict) -> dict:
        reports = response.get("reports", [])
        if not isinstance(reports, list) or not reports:
            return response
        allowed = {
            str(item.get("asset_id")) for item in reports
            if item.get("final_status") in {"pass", "fallback_pass"}
            and item.get("place_allowed", True) is not False
        }
        response = dict(response)
        response["assets"] = [item for item in response.get("assets", []) if str(item.get("id")) in allowed]
        return response
    reconstruction = placeable(reconstruction)
    environment = placeable(environment)
    # Structural features are already integrated in world coordinates and are
    # never eligible for ordinary asset placement, regardless of an upstream
    # worker echoing a legacy route value.
    reconstruction["assets"] = [item for item in reconstruction.get("assets", []) if not is_structural_feature(item.get("category", ""), item.get("asset_role", item.get("asset_type")))]
    environment["assets"] = [item for item in environment.get("assets", []) if not is_structural_feature(item.get("category", ""), item.get("asset_role", item.get("asset_type")))]
    structural_terrain_path = work_dir / "terrain_structural.npz"
    terrain_path = structural_terrain_path if structural_terrain_path.exists() else work_dir / "terrain.npz"
    terrain = np.load(terrain_path)
    vertices = terrain["vertices"]
    triangles = terrain["triangles"]
    height = terrain["height"]
    labels = np.load(work_dir / "layout_labels.npy")
    plan = json.loads((work_dir / "scene_plan.json").read_text(encoding="utf-8"))
    sample_height = terrain_sampler(height, tuple(plan["world_size_m"]))
    output_root = work_dir / "placement"
    output_root.mkdir(parents=True, exist_ok=True)
    exclusion_path = work_dir / "structural" / "structural_exclusion_mask.npy"
    exclusion_mask = np.load(exclusion_path) if exclusion_path.exists() else None
    weights_path = work_dir / "layout_weights.npy"
    layout_weights = np.load(weights_path) if weights_path.exists() else None
    influence_path = work_dir / "structural" / "structural_placement_weights.npy"
    trail_path = work_dir / "structural" / "trail_exclusion_mask.npy"
    water_path = work_dir / "structural" / "water_surface_mask.npy"
    structural_placement_weights = np.load(influence_path) if influence_path.exists() else None
    trail_exclusion_mask = np.load(trail_path) if trail_path.exists() else None
    water_surface_mask = np.load(water_path) if water_path.exists() else None
    assets, diagnostics = scatter_environment(
        environment.get("assets", []), plan, labels, height, sample_height,
        int(request["seed"]), float(request.get("contact_tolerance_fraction", 0.05)), output_root,
        exclusion_mask=exclusion_mask,
        layout_weights=layout_weights,
        structural_placement_weights=structural_placement_weights,
        trail_exclusion_mask=trail_exclusion_mask,
        surface_masks=prepare_surface_masks({
            "structural_exclusion": exclusion_mask,
            "trail": trail_exclusion_mask,
            "water": water_surface_mask,
        }),
    )
    reconstruction_assets = reconstruction.get("assets", [])
    tolerance_fraction = float(request.get("contact_tolerance_fraction", 0.05))
    requested_workers = _placement_worker_count(len(reconstruction_assets))
    terrain_scene = raycast_scene(vertices, triangles) if reconstruction_assets and requested_workers <= 1 else None
    placed_results, workers_used, parallel = _place_reconstruction_assets(
        reconstruction_assets, terrain_path, terrain_scene, sample_height,
        tuple(plan["world_size_m"]), tolerance_fraction,
    )
    region_indices = {region["id"]: index for index, region in enumerate(plan["regions"])}
    export_placed_meshes = os.getenv("WORLDCLAW_EXPORT_PLACED_MESHES", "0").lower() in {"1", "true", "yes"}
    for item, mesh, transform, metrics in placed_results:
        placed_path = output_root / f"{item['id']}.glb"
        if export_placed_meshes:
            if mesh is None:
                mesh = load_mesh(Path(item["mesh"]))
            mesh.export(placed_path, file_type="glb")
            mesh_reference = str(placed_path)
        else:
            # The transform is already carried in `transform_z_up`; reusing the
            # source GLB avoids one full mesh serialization per reconstructed
            # asset. Export can be enabled for standalone placement debugging.
            mesh_reference = str(Path(item["mesh"]).resolve())
        region_id = item.get("region_id")
        if not region_id:
            raise RuntimeError(f"{item['id']}: target region is missing")
        if region_id not in region_indices:
            raise RuntimeError(f"{item['id']}: unknown target region {region_id}")
        anchor_x, anchor_y = metrics["terrain_anchor"][:2]
        anchor_label = label_at(labels, tuple(plan["world_size_m"]), anchor_x, anchor_y)
        if anchor_label != region_indices[region_id]:
            # Segmentation centroids can land on a generated region boundary.
            # Keep the camera-ray/contact solution and expose the mismatch for
            # refinement instead of discarding an otherwise valid reconstruction.
            diagnostics.append({
                "asset_id": item["id"], "placement": "region_label_mismatch",
                "target_region": region_id, "anchor_label": int(anchor_label),
                "target_label": int(region_indices[region_id]),
            })
        asset = AssetInstance(
            id=item["id"], category=item["category"],
            asset_type=effective_asset_type(item["category"], item.get("asset_type")),
            source_image=item["source_image"],
            mask=item.get("mask", ""), mesh=mesh_reference, material=item["category"],
            transform_z_up=transform.tolist(), region_id=region_id,
            contact_ratio=metrics["contact_ratio"], source_model=item["model"]["model_id"],
            synthetic=False,
        )
        assets.append(asset.model_dump(mode="json"))
        diagnostics.append({"asset_id": item["id"], "placement": "camera_ray", **metrics})
    structural_branch_path = work_dir / "structural_branch.json"
    structural_branch = json.loads(structural_branch_path.read_text(encoding="utf-8")) if structural_branch_path.exists() else {}
    write_response(args.response, {
        "status": "ok", "assets": assets, "diagnostics": diagnostics,
        "structural_constraints": {
            "barrier_satisfied": structural_branch_path.exists(),
            "exclusion_mask": structural_branch.get("exclusion_mask"),
            "occupied_mask": structural_branch.get("occupied_mask"),
            "terrain": structural_branch.get("terrain", str(terrain_path)),
        },
        "coordinate_system": {"internal": "Z-up right-handed", "glTF": "Y-up"},
        "placement_optimization": {
            "support_only_search": True,
            "search_candidates": 34,
            "vectorized_height_contact": True,
            "occupied_spatial_hash": True,
            "prototype_mesh_cache": True,
            "object_raycast_scene_cache": True,
            "reconstruction_process_pool": parallel,
            "reconstruction_workers": workers_used,
            "placed_mesh_export": os.getenv("WORLDCLAW_EXPORT_PLACED_MESHES", "0").lower() in {"1", "true", "yes"},
        },
    })


if __name__ == "__main__":
    main()
