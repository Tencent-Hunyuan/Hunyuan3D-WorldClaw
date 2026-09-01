"""Shared helpers for the auditable structural-side validation workflow.

The structural branch is deliberately split into data preparation, numerical
generation, view planning, rendering, and visual adjudication.  This module
contains only orchestration and file-format helpers; mesh construction and
deterministic checks remain in :mod:`worldclaw_oss.structural`.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .export import write_png
from .structural import (
    _centered_polygon,
    _polygon_mask,
    build_structural_geometry,
    structural_features_from_plan,
    validate_structural_branch,
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, target: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"required structural input is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"path": str(target), "sha256": _sha256(target), "size_bytes": target.stat().st_size}


def _feature_mask(plan: dict[str, Any], features: list[dict[str, Any]], shape: tuple[int, int], world_size: tuple[float, float]) -> np.ndarray:
    """Rasterize structural feature regions from the authored plan."""
    regions = {str(item.get("id")): item for item in plan.get("regions", [])}
    mask = np.zeros(shape, dtype=bool)
    for feature in features:
        region = regions.get(str(feature.get("region_id")))
        if region is None:
            continue
        mask |= _polygon_mask(_centered_polygon(region, world_size), world_size, shape)
    return mask


def _existing_structures(work_dir: Path, shape: tuple[int, int]) -> tuple[dict[str, Any], np.ndarray]:
    """Inventory prior structural outputs without inventing geometry."""
    branch_path = work_dir / "structural_branch.json"
    if branch_path.is_file():
        try:
            branch = json.loads(branch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing structural response is invalid: {branch_path}") from exc
        source_mask = work_dir / "structural" / "structural_exclusion_mask.npy"
        if source_mask.is_file():
            existing_mask = np.asarray(np.load(source_mask), dtype=bool)
            if existing_mask.shape != shape:
                raise ValueError("existing structural exclusion mask shape does not match layout")
        else:
            existing_mask = np.zeros(shape, dtype=bool)
        return {
            "status": "ok",
            "source": str(branch_path),
            "features": branch.get("structural_meshes", []),
            "feature_count": len(branch.get("structural_meshes", [])),
        }, existing_mask
    return {
        "status": "ok",
        "source": "shared_stage_inventory",
        "features": [],
        "feature_count": 0,
        "note": "No prior structural response exists before the first structural generation stage.",
    }, np.zeros(shape, dtype=bool)


def prepare_structural_agent_input(work_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Materialize the complete, auditable input tree requested by the agent."""
    work_dir = Path(work_dir)
    root = work_dir / "structural_agent_input"
    root.mkdir(parents=True, exist_ok=True)
    layout_dir = root / "layout"
    terrain_dir = root / "terrain"
    existing_dir = root / "existing_structures"
    masks_dir = existing_dir / "exclusion_masks"
    context_dir = root / "code_context"
    structural_context = context_dir / "structural_workers"
    common_context = context_dir / "common"
    masks_dir.mkdir(parents=True, exist_ok=True)

    labels_path = work_dir / "layout_labels.npy"
    weights_path = work_dir / "layout_weights.npy"
    terrain_path = work_dir / "terrain.npz"
    for path in (labels_path, weights_path, terrain_path):
        if not path.is_file():
            raise FileNotFoundError(f"shared stage did not produce required structural input: {path}")
    labels = np.asarray(np.load(labels_path))
    terrain = np.load(terrain_path)
    if "height" not in terrain:
        raise ValueError("terrain.npz must contain a height array")
    height = np.asarray(terrain["height"])
    features = structural_features_from_plan(plan)
    world_size = tuple(float(value) for value in plan["world_size_m"])

    _json(root / "scene_plan.json", plan)
    regions_by_id = {str(item.get("id")): item for item in plan.get("regions", [])}
    feature_entries = []
    for item in features:
        region = regions_by_id.get(str(item.get("region_id")), {})
        polygon = _centered_polygon(region, world_size) if region else np.zeros((0, 2), dtype=float)
        entry = dict(item)
        entry["world_bounds"] = {
            "min": [float(polygon[:, 0].min()), float(polygon[:, 1].min()), -float(item.get("geometry", {}).get("thickness_m", 0.0))] if len(polygon) else [],
            "max": [float(polygon[:, 0].max()), float(polygon[:, 1].max()), 0.0] if len(polygon) else [],
        }
        feature_entries.append(entry)
    feature_plan = {
        "schema": "worldclaw-oss-structural-feature-plan-v1",
        "source": "scene_plan.json",
        "features": feature_entries,
        "world_size_m": list(world_size),
    }
    _json(root / "feature_plan.json", feature_plan)
    _copy_file(labels_path, layout_dir / "layout_labels.npy")
    _copy_file(weights_path, layout_dir / "layout_weights.npy")
    feature_mask = _feature_mask(plan, features, height.shape, world_size)
    np.save(layout_dir / "feature_mask.npy", feature_mask)
    _copy_file(terrain_path, terrain_dir / "terrain.npz")

    existing, existing_mask = _existing_structures(work_dir, height.shape)
    _json(existing_dir / "structural_response.json", existing)
    np.save(masks_dir / "existing_exclusion_mask.npy", existing_mask)

    # Preserve real source context rather than creating a second implementation.
    source_files = {
        structural_context / "structural.py": Path(__file__).with_name("structural.py"),
        structural_context / "structural_worker.py": Path(__file__).resolve().parents[1] / "workers" / "structural_worker.py",
        structural_context / "structural_validator.py": Path(__file__).resolve().parents[1] / "workers" / "structural_validator.py",
        common_context / "schemas.py": Path(__file__).with_name("schemas.py"),
        common_context / "asset_semantics.py": Path(__file__).with_name("asset_semantics.py"),
        common_context / "pipeline.py": Path(__file__).with_name("pipeline.py"),
    }
    for target, source in source_files.items():
        _copy_file(source, target)
    task = {
        "schema": "worldclaw-oss-structural-task-v1",
        "objective": "Plan, generate, integrate, and validate all terrain-dependent features.",
        "required_inputs": [
            "scene_plan.json", "feature_plan.json", "layout/layout_labels.npy",
            "layout/layout_weights.npy", "layout/feature_mask.npy", "terrain/terrain.npz",
            "existing_structures/structural_response.json",
            "existing_structures/exclusion_masks/existing_exclusion_mask.npy",
            "code_context/structural_workers", "code_context/common", "code_context/common/schemas.py",
        ],
        "required_outputs": [
            "structural_plan.json", "structural_branch.json", "terrain_structural.npz",
            "structural_exclusion_mask.npy", "structural_occupied_mask.npy", "surface_type_mask.npy",
            "water_surface_mask.npy",
            "trail_clearance_field.npy", "trail_exclusion_mask.npy", "structural_placement_weights.npy",
            "structural_validation.json",
        ],
        "world_coordinates": "internal Z-up right-handed",
        "feature_ids": [item["feature_id"] for item in features],
        "constraints": {
            "preserve_region_relationships": True,
            "integrate_with_terrain": True,
            "deterministic_geometry_validation_before_visual": True,
            "no_hunyuan_for_structural_features": True,
        },
    }
    _json(root / "structural_task.json", task)
    return {
        "root": str(root),
        "feature_count": len(features),
        "feature_ids": [item["feature_id"] for item in features],
        "layout_shape": list(labels.shape),
        "terrain_shape": list(height.shape),
        "existing_feature_count": existing["feature_count"],
        "existing_exclusion_cells": int(existing_mask.sum()),
        "required_inputs_complete": True,
    }


def structural_feature_summary(branch: dict[str, Any]) -> dict[str, Any]:
    summaries = []
    for item in branch.get("structural_meshes", []):
        path = Path(item.get("mesh", ""))
        if not path.is_file():
            raise FileNotFoundError(f"structural mesh is missing: {path}")
        with np.load(path) as data:
            vertices = np.asarray(data["vertices"], dtype=float)
            triangles = np.asarray(data["triangles"])
        bounds = {
            "min": vertices.min(axis=0).tolist() if len(vertices) else [],
            "max": vertices.max(axis=0).tolist() if len(vertices) else [],
        }
        summaries.append({
            "feature_id": item.get("feature_id"),
            "category": item.get("category"),
            "region_id": item.get("region_id"),
            "representation": item.get("representation"),
            "vertex_count": int(len(vertices)),
            "triangle_count": int(len(triangles)),
            "world_bounds": bounds,
            "centerline_points": int(len(item.get("centerline", []))),
            "mesh": str(path),
        })
    return {"feature_count": len(summaries), "features": summaries}


def _bounds_from_summary(summary: dict[str, Any], world_size: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for item in summary.get("features", []):
        bounds = item.get("world_bounds", {})
        if bounds.get("min") and bounds.get("max"):
            points.extend([bounds["min"], bounds["max"]])
    if not points:
        half = np.asarray(world_size, dtype=float) / 2.0
        return np.asarray([-half[0], -half[1], 0.0]), np.asarray([half[0], half[1], 1.0])
    values = np.asarray(points, dtype=float)
    return values.min(axis=0), values.max(axis=0)


def _view_schema() -> dict[str, Any]:
    view = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "feature_ids": {"type": "array", "items": {"type": "string"}},
            "position": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
            "target": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
            "reason": {"type": "string"},
        },
        "required": ["id", "feature_ids", "position", "target", "reason"],
    }
    return {"type": "object", "additionalProperties": False, "properties": {"status": {"type": "string", "enum": ["ok"]}, "views": {"type": "array", "minItems": 1, "maxItems": 4, "items": view}}, "required": ["status", "views"]}


def _fixed_views(summary: dict[str, Any], world_size: tuple[float, float]) -> list[dict[str, Any]]:
    low, high = _bounds_from_summary(summary, world_size)
    center = (low + high) / 2.0
    horizontal = max(float(high[0] - low[0]), float(high[1] - low[1]), 4.0)
    vertical = max(float(high[2] - low[2]), 2.0)
    distance = max(horizontal * 1.8, 12.0)
    return [
        {"id": "top_down", "kind": "fixed", "feature_ids": [item["feature_id"] for item in summary.get("features", [])], "position": [float(center[0]), float(center[1]), float(high[2] + distance)], "target": [float(center[0]), float(center[1]), float(center[2])], "reason": "Stable overhead view for region containment and feature footprint."},
        {"id": "oblique_overview", "kind": "fixed", "feature_ids": [item["feature_id"] for item in summary.get("features", [])], "position": [float(center[0] - distance * 0.85), float(center[1] - distance * 0.85), float(high[2] + distance * 0.7 + vertical)], "target": [float(center[0]), float(center[1]), float(center[2])], "reason": "Stable oblique view for terrain contact and spatial relationships."},
    ]


def plan_validation_views(
    scene_plan: dict[str, Any], structural_plan: dict[str, Any], branch: dict[str, Any],
    deterministic: dict[str, Any], world_size: tuple[float, float],
    client=None, model=None, seed: int = 0, request_path: Path | None = None,
    response_path: Path | None = None,
) -> dict[str, Any]:
    """Use GPT to select feature-focused adaptive views after deterministic PASS."""
    if deterministic.get("status") != "pass":
        raise ValueError("view planning requires deterministic structural validation PASS")
    summary = structural_feature_summary(branch)
    fixed = _fixed_views(summary, world_size)
    user_payload = {
        "scene_plan": scene_plan,
        "structural_plan": structural_plan,
        "features": summary,
        "geometry_metrics": {"terrain_modified": branch.get("terrain_modified"), "feature_count": summary["feature_count"]},
        "deterministic_validation": deterministic,
        "structural_relationships": [region.get("spatial_relations", []) for region in scene_plan.get("regions", [])],
        "instruction": "Return 1-4 adaptive camera views that expose geometry-specific risks not fully covered by fixed views.",
    }
    if client is None or model is None:
        adaptive = []
        for index, item in enumerate(summary.get("features", []), start=1):
            lo = np.asarray(item["world_bounds"]["min"], dtype=float)
            hi = np.asarray(item["world_bounds"]["max"], dtype=float)
            center = (lo + hi) / 2.0
            span = max(float(np.ptp(np.asarray([lo[0], hi[0]]))), float(np.ptp(np.asarray([lo[1], hi[1]]))), 4.0)
            adaptive.append({"id": f"adaptive_{index:03d}", "kind": "adaptive", "feature_ids": [item["feature_id"]], "position": [float(center[0] - span), float(center[1] + span), float(hi[2] + span)], "target": center.tolist(), "reason": "Feature-local view for deterministic geometry and terrain contact."})
        raw = {"status": "ok", "views": adaptive[:4] or [{"id": "adaptive_overview", "kind": "adaptive", "feature_ids": [], "position": fixed[1]["position"], "target": fixed[1]["target"], "reason": "Fallback geometry-focused view for an empty feature set."}]}
        provider = "deterministic_fixture"
    else:
        user = json.dumps(user_payload, ensure_ascii=False)
        if request_path is not None:
            request_path.write_text(user + "\n", encoding="utf-8")
        raw = client.json_chat(model, "You are a GPT structural validation view planner. Use the supplied geometry and deterministic metrics. Do not alter geometry or invent features. Return only adaptive views with numeric camera position and target.", user, _view_schema(), seed)
        if response_path is not None:
            _json(response_path, raw)
        provider = "openai-compatible"
    if not isinstance(raw, dict) or raw.get("status") != "ok" or not isinstance(raw.get("views"), list):
        raise ValueError("view planner returned an invalid response")
    adaptive = []
    known = {item["feature_id"] for item in summary.get("features", [])}
    for item in raw["views"]:
        if not isinstance(item, dict) or item.get("id") in {view["id"] for view in adaptive}:
            raise ValueError("view planner returned duplicate or invalid view")
        feature_ids = [str(value) for value in item.get("feature_ids", [])]
        if any(value not in known for value in feature_ids):
            raise ValueError("view planner referenced an unknown feature")
        position = item.get("position")
        target = item.get("target")
        if not (isinstance(position, list) and len(position) == 3 and isinstance(target, list) and len(target) == 3):
            raise ValueError("view planner returned an invalid camera vector")
        if not all(math.isfinite(float(value)) for value in position + target):
            raise ValueError("view planner returned non-finite camera coordinates")
        adaptive.append({"id": str(item["id"]), "kind": "adaptive", "feature_ids": feature_ids, "position": [float(value) for value in position], "target": [float(value) for value in target], "reason": str(item.get("reason", "geometry-focused adaptive view"))})
    return {"schema": "worldclaw-oss-validation-views-v1", "status": "ok", "provider": provider, "fixed": fixed, "adaptive": adaptive, "feature_summary": summary, "deterministic_status": deterministic.get("status"), "world_size_m": list(world_size)}


def _draw_line(image: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int], width: int = 2) -> None:
    x0, y0 = a; x1, y1 = b
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for value in np.linspace(0.0, 1.0, steps + 1):
        x = int(round(x0 + (x1 - x0) * value)); y = int(round(y0 + (y1 - y0) * value))
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[max(0, y - width):min(image.shape[0], y + width + 1), max(0, x - width):min(image.shape[1], x + width + 1)] = color


def _software_render_views(work_dir: Path, views: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Render actual structural arrays when Blender is unavailable (synthetic mode)."""
    terrain_path = work_dir / "terrain_structural.npz"
    if not terrain_path.is_file():
        raise FileNotFoundError(terrain_path)
    terrain = np.load(terrain_path)
    height = np.asarray(terrain["height"], dtype=float)
    branch = json.loads((work_dir / "structural_branch.json").read_text(encoding="utf-8"))
    meshes = []
    for item in branch.get("structural_meshes", []):
        with np.load(item["mesh"]) as data:
            meshes.append((item, np.asarray(data["vertices"], dtype=float)))
    output_root.mkdir(parents=True, exist_ok=True)
    width, height_px = 640, 360
    world_size = tuple(float(value) for value in views.get("world_size_m", (100.0, 100.0)))
    lo, hi = -np.asarray(world_size) / 2.0, np.asarray(world_size) / 2.0
    rendered = []
    for group_name in ("fixed", "adaptive", "additional"):
        group = views.get(group_name, [])
        for view in group:
            image = np.zeros((height_px, width, 3), dtype=np.uint8)
            ys = np.linspace(0, height.shape[0] - 1, height_px).astype(int)
            xs = np.linspace(0, height.shape[1] - 1, width).astype(int)
            sample = height[np.ix_(ys, xs)]
            normalized = (sample - float(height.min())) / max(float(height.max() - height.min()), 1e-8)
            image[:] = np.stack([50 + normalized * 80, 78 + normalized * 80, 48 + normalized * 48], axis=-1).astype(np.uint8)
            def project(points):
                values = np.asarray(points)
                px = ((values[:, 0] - lo[0]) / max(world_size[0], 1e-8) * (width - 1)).astype(int)
                py = ((hi[1] - values[:, 1]) / max(world_size[1], 1e-8) * (height_px - 1)).astype(int)
                return list(zip(px.tolist(), py.tolist()))
            for item, vertices in meshes:
                points = project(vertices)
                if item.get("category") == "lake" and points:
                    p0, p1 = points[0], points[2] if len(points) > 2 else points[-1]
                    xa, xb = sorted((p0[0], p1[0])); ya, yb = sorted((p0[1], p1[1]))
                    image[max(0, ya):min(height_px, yb + 1), max(0, xa):min(width, xb + 1)] = (45, 130, 170)
                elif len(points) > 1:
                    for first, second in zip(points, points[1:]):
                        _draw_line(image, first, second, (115, 72, 38), 2)
            relative = Path("renders") / group_name / f"{view['id']}.png"
            path = output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write_png(path, image)
            rendered.append({"id": view["id"], "kind": group_name, "path": str(path), "renderer": "numpy_structural_renderer"})
    return {"status": "ok", "renderer": "numpy_structural_renderer", "renders": rendered}


def build_validation_bundle(
    work_dir: Path, views: dict[str, Any], deterministic: dict[str, Any], structural_plan: dict[str, Any], scene_plan: dict[str, Any],
    render_result: dict[str, Any], additional: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    root = Path(work_dir) / "validation_bundle"
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "context").mkdir(parents=True, exist_ok=True)
    branch = json.loads((Path(work_dir) / "structural_branch.json").read_text(encoding="utf-8"))
    summary = structural_feature_summary(branch)
    # Add follow-up views before persisting the metadata.  Keeping this order
    # matters because the final GPT validation request must see exactly the
    # same view set that is present on disk.
    existing_ids = {item.get("id") for item in views.get("additional", [])}
    for item in additional:
        if item.get("id") not in existing_ids:
            views.setdefault("additional", []).append(item)
            existing_ids.add(item.get("id"))
    _json(root / "metadata" / "feature_summary.json", summary)
    _json(root / "metadata" / "validation_views.json", views)
    _json(root / "metrics" / "deterministic_validation.json", deterministic)
    _json(root / "context" / "structural_plan.json", structural_plan)
    _json(root / "context" / "scene_context.json", {"scene_plan": scene_plan, "structural_branch_metrics": branch.get("metrics", {}), "render": render_result})
    return {"root": str(root), "feature_count": summary["feature_count"], "render_count": len(render_result.get("renders", [])), "complete": True}


def render_structural_views(
    work_dir: Path, views: dict[str, Any], *, blender: Path | None = None,
    run_dir: Path | None = None, views_path: Path | None = None,
) -> dict[str, Any]:
    """Render with Blender for live runs, or real array-based renderer for synthetic runs."""
    output_root = Path(work_dir) / "validation_bundle"
    if blender is None or run_dir is None:
        return _software_render_views(Path(work_dir), views, output_root)
    script = Path(__file__).resolve().parents[1] / "scripts" / "blender_structural_validation.py"
    request = Path(work_dir) / "structural_render_request.json"
    selected_views_path = Path(views_path or (Path(work_dir) / "validation_views.json"))
    _json(request, {"run": str(run_dir), "views": views, "output": str(output_root)})
    import subprocess
    completed = subprocess.run([str(blender), "--background", "--python", str(script), "--", "--run", str(run_dir), "--views", str(selected_views_path), "--output", str(output_root)], capture_output=True, text=True, timeout=14400)
    if completed.returncode != 0:
        raise RuntimeError(f"structural Blender rendering failed: {completed.stderr[-4000:]}")
    response = output_root / "render_response.json"
    if not response.is_file():
        raise RuntimeError("structural Blender renderer did not produce render_response.json")
    return json.loads(response.read_text(encoding="utf-8"))


def visual_validation_schema() -> dict[str, Any]:
    additional = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"}, "feature_ids": {"type": "array", "items": {"type": "string"}},
            "position": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
            "target": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
            "reason": {"type": "string"},
        }, "required": ["id", "feature_ids", "position", "target", "reason"],
    }
    return {"type": "object", "additionalProperties": False, "properties": {"status": {"type": "string", "enum": ["pass", "fail"]}, "confidence": {"type": "string", "enum": ["high", "low"]}, "diagnosis": {"type": "string"}, "findings": {"type": "array", "items": {"type": "string"}}, "additional_views": {"type": "array", "maxItems": 2, "items": additional}}, "required": ["status", "confidence", "diagnosis", "findings", "additional_views"]}


def _bundle_images(bundle: Path, include_additional: bool = False) -> list[Path]:
    paths = sorted((bundle / "renders" / "fixed").glob("*.png")) + sorted((bundle / "renders" / "adaptive").glob("*.png"))
    if include_additional:
        paths += sorted((bundle / "renders" / "additional").glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"validation bundle has no renders: {bundle}")
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ValueError("validation bundle contains an empty render")
    return paths


def visual_validate_bundle(bundle: Path, deterministic: dict[str, Any], *, client=None, model=None, seed: int = 0, include_additional: bool = False, request_path: Path | None = None, response_path: Path | None = None) -> dict[str, Any]:
    if deterministic.get("status") != "pass":
        return {"status": "fail", "confidence": "high", "diagnosis": "Deterministic geometry validation failed; visual validation was not used to override it.", "findings": list(deterministic.get("defects", [])), "additional_views": [], "provider": "deterministic_gate"}
    images = _bundle_images(Path(bundle), include_additional)
    context = {
        "deterministic_validation": deterministic,
        "feature_summary": json.loads((Path(bundle) / "metadata" / "feature_summary.json").read_text(encoding="utf-8")),
        "validation_views": json.loads((Path(bundle) / "metadata" / "validation_views.json").read_text(encoding="utf-8")),
        "scene_context": json.loads((Path(bundle) / "context" / "scene_context.json").read_text(encoding="utf-8")),
        "instruction": "Judge terrain contact, representation, region relationships, and visible structural defects. Do not use images alone; use all supplied metrics and context.",
    }
    if client is None or model is None:
        result = {"status": "pass", "confidence": "high", "diagnosis": "Synthetic renderer produced non-empty fixed and adaptive structural views; deterministic checks passed.", "findings": [], "additional_views": [], "provider": "deterministic_fixture"}
    else:
        user = json.dumps(context, ensure_ascii=False)
        if request_path is not None:
            request_path.write_text(user + "\n", encoding="utf-8")
        result = client.vision_json(system="You are the GPT Structural Visual Validator. Combine structural context, deterministic metrics, geometry summaries, and all supplied renders. Return PASS or FAIL, confidence, diagnosis, findings, and at most two targeted additional views when visibility is insufficient.", user=user, image_paths=images, schema=visual_validation_schema(), model=model.model_id)
        if response_path is not None:
            _json(response_path, result)
        result = dict(result)
        result["provider"] = "openai-compatible"
    if result.get("status") not in {"pass", "fail"} or result.get("confidence") not in {"high", "low"} or not isinstance(result.get("diagnosis"), str) or not isinstance(result.get("additional_views"), list):
        raise ValueError("visual validator returned an invalid response")
    if len(result["additional_views"]) > 2:
        raise ValueError("visual validator requested more than two additional views")
    return result


def render_additional_views(work_dir: Path, views: dict[str, Any], requests: list[dict[str, Any]], *, blender: Path | None = None, run_dir: Path | None = None) -> dict[str, Any]:
    requested = []
    known = {item["id"] for item in views.get("fixed", []) + views.get("adaptive", [])}
    for item in requests[:2]:
        if item.get("id") in known:
            continue
        requested.append({"id": str(item["id"]), "kind": "additional", "feature_ids": list(item.get("feature_ids", [])), "position": list(item["position"]), "target": list(item["target"]), "reason": str(item.get("reason", "targeted follow-up view"))})
    if not requested:
        return {"status": "ok", "needed": False, "renders": []}
    # Render only the requested follow-up views.  Passing the original view
    # file here would silently repeat the first-round adaptive cameras and
    # produce no ``renders/additional`` artifacts.
    updated = {
        "schema": views.get("schema", "worldclaw-oss-validation-views-v1"),
        "status": "ok",
        "world_size_m": views.get("world_size_m", [100.0, 100.0]),
        "fixed": [],
        "adaptive": [],
        "additional": requested,
    }
    if blender is None or run_dir is None:
        result = _software_render_views(Path(work_dir), updated, Path(work_dir) / "validation_bundle")
    else:
        temporary_views = Path(work_dir) / "structural_additional_views.json"
        _json(temporary_views, updated)
        result = render_structural_views(
            work_dir, updated, blender=blender, run_dir=run_dir,
            views_path=temporary_views,
        )
    return {"status": "ok", "needed": True, "requested": requested, "renders": result.get("renders", [])}


__all__ = [
    "prepare_structural_agent_input", "structural_feature_summary", "plan_validation_views",
    "render_structural_views", "build_validation_bundle", "visual_validate_bundle",
    "render_additional_views",
]
