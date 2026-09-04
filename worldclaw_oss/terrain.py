from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .layout import LayoutResult
from .schemas import ScenePlan, TerrainOperator


@dataclass(frozen=True)
class TerrainResult:
    height: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray


def validate_regional_detail_preserves_macro(
    macro_height: np.ndarray,
    final_height: np.ndarray,
    *,
    max_detail_ratio: float = 0.35,
    max_low_frequency_error_ratio: float = 0.25,
    coarse_cells: int = 16,
) -> dict[str, float | bool]:
    """Measure whether regional operators preserve the planner-owned field.

    The comparison is deliberately scene-independent: regional detail is
    measured as residual energy, while a coarse block average checks that the
    broad elevation arrangement was not displaced by that residual.
    """
    macro = np.asarray(macro_height, dtype=np.float64)
    final = np.asarray(final_height, dtype=np.float64)
    if macro.shape != final.shape or macro.ndim != 2:
        raise ValueError("macro_height and final_height must be matching 2D arrays")
    if not np.isfinite(macro).all() or not np.isfinite(final).all():
        raise ValueError("macro_height and final_height must be finite")
    if coarse_cells < 2 or max_detail_ratio < 0 or max_low_frequency_error_ratio < 0:
        raise ValueError("preservation thresholds must be non-negative")

    relief = float(np.percentile(macro, 95) - np.percentile(macro, 5))
    denominator = max(relief, 1.0)
    residual = final - macro
    residual_centered = residual - float(residual.mean())
    detail_rms = float(np.sqrt(np.mean(residual_centered * residual_centered)))

    # Sample matching coarse cells without adding a dependency on scipy. The
    # sampled field is sufficient for detecting broad topology drift.
    rows = np.linspace(0, macro.shape[0] - 1, min(coarse_cells, macro.shape[0])).round().astype(int)
    cols = np.linspace(0, macro.shape[1] - 1, min(coarse_cells, macro.shape[1])).round().astype(int)
    macro_coarse = macro[np.ix_(rows, cols)]
    # Regional operators may change the global datum through weighted base
    # heights.  Align that uniform offset before measuring spatial composition
    # drift; topology should be judged independently of absolute elevation.
    global_offset = float(np.mean(final - macro))
    final_coarse = (final - global_offset)[np.ix_(rows, cols)]
    low_frequency_error = float(np.sqrt(np.mean((final_coarse - macro_coarse) ** 2)))
    detail_ratio = detail_rms / denominator
    low_frequency_error_ratio = low_frequency_error / denominator
    preserved = bool(
        detail_ratio <= max_detail_ratio
        and low_frequency_error_ratio <= max_low_frequency_error_ratio
    )
    return {
        "macro_relief_m": relief,
        "regional_detail_rms_m": detail_rms,
        "regional_to_macro_rms_ratio": float(detail_ratio),
        "low_frequency_error_m": low_frequency_error,
        "low_frequency_error_ratio": float(low_frequency_error_ratio),
        "global_offset_m": global_offset,
        "max_detail_ratio": float(max_detail_ratio),
        "max_low_frequency_error_ratio": float(max_low_frequency_error_ratio),
        "macro_composition_preserved": preserved,
    }


def _smooth_noise(size: int, frequency: int, rng: np.random.Generator) -> np.ndarray:
    coarse = rng.normal(0.0, 1.0, (frequency + 1, frequency + 1))
    coords = np.linspace(0, frequency, size)
    lo = np.floor(coords).astype(int)
    hi = np.minimum(lo + 1, frequency)
    t = coords - lo
    t = t * t * (3 - 2 * t)
    rows = np.empty((size, frequency + 1))
    for i in range(size):
        rows[i] = coarse[lo[i]] * (1 - t[i]) + coarse[hi[i]] * t[i]
    out = np.empty((size, size))
    for j in range(size):
        out[:, j] = rows[:, lo[j]] * (1 - t[j]) + rows[:, hi[j]] * t[j]
    return out


def _operator(op: TerrainOperator, xx: np.ndarray, yy: np.ndarray, cx: float, cy: float) -> np.ndarray:
    r = np.hypot(xx - cx, yy - cy)
    if op.kind == "peak":
        return op.strength * np.exp(-((r / op.scale) ** 2))
    if op.kind == "dune":
        return op.strength * np.sin((xx + 0.45 * yy) * (2 * math.pi / op.scale))
    if op.kind == "terrace":
        raw = op.strength * np.exp(-r / op.scale)
        step = max(abs(op.strength) / 5, 0.1)
        return np.round(raw / step) * step
    # A shallow drainage basin as a deterministic approximation to erosion.
    return -abs(op.strength) * np.exp(-((r / op.scale) ** 2))


def _positive_plan_frame(plan: ScenePlan) -> bool:
    """Return whether authored plan coordinates use a lower-left origin."""
    width, depth = (float(value) for value in plan.world_size_m)
    points: list[tuple[float, float]] = []
    for region in plan.regions:
        points.append((float(region.center.x), float(region.center.y)))
        points.extend((float(point.x), float(point.y)) for point in region.polygon.points)
    return bool(
        points
        and all(x >= -1e-6 and y >= -1e-6 for x, y in points)
        and all(x <= width + 1e-6 and y <= depth + 1e-6 for x, y in points)
    )


def _terrain_xy_grid(plan: ScenePlan, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build Terrain coordinates in the canonical centered world frame."""
    width, depth = (float(value) for value in plan.world_size_m)
    return np.meshgrid(
        np.linspace(-width / 2.0, width / 2.0, size),
        np.linspace(-depth / 2.0, depth / 2.0, size),
    )


def _bound_regional_detail(
    macro_height: np.ndarray,
    regional_addition: np.ndarray,
    *,
    max_detail_ratio: float = 0.30,
    max_low_frequency_error_ratio: float = 0.20,
    coarse_cells: int = 16,
) -> np.ndarray:
    """Keep regional operators subordinate to a planner-owned macro field."""
    macro = np.asarray(macro_height, dtype=np.float64)
    addition = np.asarray(regional_addition, dtype=np.float64)
    relief = float(np.percentile(macro, 95) - np.percentile(macro, 5))
    denominator = max(relief, 1.0)
    centered = addition - float(np.mean(addition))
    detail_rms = float(np.sqrt(np.mean(centered * centered)))
    rows = np.linspace(0, macro.shape[0] - 1, min(coarse_cells, macro.shape[0])).round().astype(int)
    cols = np.linspace(0, macro.shape[1] - 1, min(coarse_cells, macro.shape[1])).round().astype(int)
    macro_coarse = macro[np.ix_(rows, cols)]
    addition_coarse = addition[np.ix_(rows, cols)]
    addition_offset = float(np.mean(addition_coarse))
    low_frequency_error = float(
        np.sqrt(np.mean((addition_coarse - addition_offset) ** 2))
    )
    factors = [1.0]
    if detail_rms > 0.0:
        factors.append(max_detail_ratio * denominator / detail_rms)
    if low_frequency_error > 0.0:
        factors.append(max_low_frequency_error_ratio * denominator / low_frequency_error)
    factor = float(np.clip(min(factors), 0.0, 1.0))
    return addition * factor


def generate_terrain(plan: ScenePlan, layout: LayoutResult, seed: int, macro_height: np.ndarray | None = None) -> TerrainResult:
    size = layout.labels.shape[0]
    width, depth = plan.world_size_m
    xx, yy = _terrain_xy_grid(plan, size)
    rng = np.random.default_rng(seed)
    height = np.asarray(macro_height, dtype=np.float64).copy() if macro_height is not None else np.zeros((size, size), dtype=np.float64)
    if height.shape != (size, size):
        raise ValueError("macro_height shape must match layout resolution")
    world_scale = max(width, depth)
    # Planner strengths are authored in meters and are intentionally small for
    # the 100m synthetic fixture.  Live worlds can be 1km across, so add a
    # broad, deterministic relief field and scale regional detail moderately
    # with world size instead of leaving the scene visually planar.
    regional_relief_scale = float(np.clip(world_scale / 250.0, 1.0, 4.0))
    # Noise is regional detail only once a planner-owned macro field exists.
    macro_relief_amplitude = 0.0 if macro_height is not None else (0.0 if world_scale <= 250.0 else float(np.clip(world_scale * 0.032, 8.0, 32.0)))
    height += macro_relief_amplitude * _smooth_noise(size, 3, rng)
    specs = {t.region_id: t for t in plan.terrain}
    positive_frame = _positive_plan_frame(plan)
    coordinate_offset = np.asarray([width / 2.0, depth / 2.0]) if positive_frame else np.zeros(2)
    regional_addition = np.zeros_like(height, dtype=np.float64)
    for index, region in enumerate(plan.regions):
        spec = specs[region.id]
        # Once a planner-owned macro field exists, regional base offsets are
        # detail too.  Scaling them with the regional operators prevents a
        # semantic region's datum from overwhelming low-frequency composition.
        base_scale = 0.25 if macro_height is not None else 1.0
        field = np.full_like(height, spec.base_height * base_scale)
        for octave, amplitude in enumerate(spec.noise_octaves):
            detail_scale = 0.25 if macro_height is not None else 1.0
            field += detail_scale * regional_relief_scale * float(amplitude) * _smooth_noise(size, 2 ** (octave + 1), rng)
        for op in spec.operators:
            detail_scale = 0.25 if macro_height is not None else 1.0
            field += detail_scale * regional_relief_scale * _operator(
                op, xx, yy,
                region.center.x - coordinate_offset[0],
                region.center.y - coordinate_offset[1],
            )
        regional_addition += layout.weights[index] * field
    if macro_height is not None:
        regional_addition = _bound_regional_detail(macro_height, regional_addition)
    height += regional_addition
    vertices = np.stack((xx, yy, height), axis=-1).reshape(-1, 3).astype(np.float32)
    tris = []
    for iy in range(size - 1):
        for ix in range(size - 1):
            a = iy * size + ix
            tris.extend(((a, a + 1, a + size + 1), (a, a + size + 1, a + size)))
    return TerrainResult(height.astype(np.float32), vertices, np.asarray(tris, dtype=np.uint32))


def boundary_discontinuity(height: np.ndarray, labels: np.ndarray) -> float:
    right = labels[:, 1:] != labels[:, :-1]
    down = labels[1:, :] != labels[:-1, :]
    diffs = []
    if right.any(): diffs.append(np.abs(height[:, 1:] - height[:, :-1])[right])
    if down.any(): diffs.append(np.abs(height[1:, :] - height[:-1, :])[down])
    return float(np.max(np.concatenate(diffs))) if diffs else 0.0
