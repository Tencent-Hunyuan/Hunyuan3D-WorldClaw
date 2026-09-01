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


def generate_terrain(plan: ScenePlan, layout: LayoutResult, seed: int) -> TerrainResult:
    size = layout.labels.shape[0]
    width, depth = plan.world_size_m
    x = np.linspace(-width / 2, width / 2, size)
    y = np.linspace(-depth / 2, depth / 2, size)
    xx, yy = np.meshgrid(x, y)
    rng = np.random.default_rng(seed)
    height = np.zeros((size, size), dtype=np.float64)
    world_scale = max(width, depth)
    # Planner strengths are authored in meters and are intentionally small for
    # the 100m synthetic fixture.  Live worlds can be 1km across, so add a
    # broad, deterministic relief field and scale regional detail moderately
    # with world size instead of leaving the scene visually planar.
    regional_relief_scale = float(np.clip(world_scale / 250.0, 1.0, 4.0))
    macro_relief_amplitude = 0.0 if world_scale <= 250.0 else float(np.clip(world_scale * 0.032, 8.0, 32.0))
    height += macro_relief_amplitude * _smooth_noise(size, 3, rng)
    specs = {t.region_id: t for t in plan.terrain}
    for index, region in enumerate(plan.regions):
        spec = specs[region.id]
        field = np.full_like(height, spec.base_height)
        for octave, amplitude in enumerate(spec.noise_octaves):
            field += regional_relief_scale * float(amplitude) * _smooth_noise(size, 2 ** (octave + 1), rng)
        for op in spec.operators:
            field += regional_relief_scale * _operator(op, xx, yy, region.center.x, region.center.y)
        height += layout.weights[index] * field
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
