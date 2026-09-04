#!/usr/bin/env python3
"""Three-round geometry and vision-model render-feedback refinement worker."""
from __future__ import annotations

import base64
import copy
import json
import math
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import requests

from workers.common import load_stage_response, parse_defect_report, read_request, worker_args, write_response
from workers.export_worker import build_scene, load_mesh
from workers.placement_worker import contact_metrics, terrain_sampler
from worldclaw_oss.asset_semantics import effective_asset_type
from worldclaw_oss.asset_types import AssetType
from worldclaw_oss.hunyuan_pool import HunyuanDaemonPool
from worldclaw_oss.openai_api import OpenAIClient
from worldclaw_oss.prompts import VLM_INSPECT_SYSTEM
from worldclaw_oss.schemas import Defect, DefectReport


def world_vertices(asset: dict, vertex_cache: dict[str, np.ndarray] | None = None) -> np.ndarray:
    mesh_path = str(Path(asset["mesh"]))
    if vertex_cache is not None and mesh_path in vertex_cache:
        vertices = vertex_cache[mesh_path]
    else:
        mesh = load_mesh(Path(mesh_path))
        vertices = np.asarray(mesh.vertices, dtype=float)
        if vertex_cache is not None:
            vertex_cache[mesh_path] = vertices
    transform = np.asarray(asset["transform_z_up"], dtype=float)
    return (transform[:3, :3] @ vertices.T).T + transform[:3, 3]


def geometry_defects(assets: list[dict], sample_height) -> tuple[list[dict], dict]:
    defects = []
    bounds = {}
    metrics = {}
    vertex_cache: dict[str, np.ndarray] = {}
    for asset in assets:
        vertices = world_vertices(asset, vertex_cache)
        height = max(float(np.ptp(vertices[:, 2])), 0.1)
        ratio, penetration, floating = contact_metrics(vertices, sample_height, height * 0.05)
        bounds[asset["id"]] = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
        metrics[asset["id"]] = {
            "contact_ratio": ratio, "penetration_m": penetration,
            "floating_m": floating, "height_m": height,
        }
        if floating > height * 0.05:
            defects.append(Defect(
                asset_id=asset["id"], kind="floating",
                severity=min(1.0, floating / height), recommendation="lower object vertically",
            ).model_dump())
        if penetration > height * 0.05:
            defects.append(Defect(
                asset_id=asset["id"], kind="penetration",
                severity=min(1.0, penetration / height), recommendation="raise object vertically",
            ).model_dump())
    ids = [asset["id"] for asset in assets]
    population_overlap = {}
    for asset in assets:
        route = effective_asset_type(asset.get("category", ""), asset.get("asset_type"))
        population_overlap[asset["id"]] = (
            route == AssetType.PROCEDURAL_NATIVE
            or (
                str(asset.get("category", "")).strip().lower() == "tree"
            )
        )
    for index, left_id in enumerate(ids):
        left = bounds[left_id]
        for right_id in ids[index + 1:]:
            # Canopy/native-vegetation overlap is the intended population
            # representation. Placement already reserves isolated solids
            # first and limits high-ratio vegetation collisions, so refinement
            # only tests solid-vs-solid pairs for collision defects.
            if population_overlap[left_id] or population_overlap[right_id]:
                continue
            right = bounds[right_id]
            intersection = np.maximum(0.0, np.minimum(left[1], right[1]) - np.maximum(left[0], right[0]))
            overlap = float(np.prod(intersection))
            left_volume = float(np.prod(np.maximum(left[1] - left[0], 1e-8)))
            right_volume = float(np.prod(np.maximum(right[1] - right[0], 1e-8)))
            rate = overlap / min(left_volume, right_volume)
            if rate > 0.1:
                defects.append(Defect(
                    asset_id=right_id, kind="overlap", severity=min(1.0, rate),
                    recommendation=f"move away from {left_id}",
                ).model_dump())
    return defects, metrics


def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def vlm_report(request: dict, views: list[Path], geometry: list[dict]) -> DefectReport:
    # Overlap analysis can produce thousands of pairwise defects. Send a
    # bounded, representative summary to the VLM so the visual request stays
    # within the model context window while preserving every defect locally.
    by_kind: dict[str, int] = {}
    for item in geometry:
        kind = str(item.get("kind", "other"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    geometry_summary = {
        "defect_counts": by_kind,
        "representative_defects": geometry[:32],
        "total_local_defects": len(geometry),
    }
    validation_provider = os.getenv("WORLDCLAW_VALIDATION_PROVIDER", "openai").lower()
    if validation_provider == "openai":
        client = OpenAIClient()
        value = client.vision_json(
            system=VLM_INSPECT_SYSTEM,
            user=(
                "Inspect these diagnostic views. Geometry checks are supplied below. "
                "Report only visible or supplied geometric/texture defects.\n"
                + json.dumps(geometry_summary, ensure_ascii=False)
            ),
            image_paths=views,
            schema=DefectReport.model_json_schema(),
            model=client.config.validation_model,
        )
        return parse_defect_report(value)
    url = os.getenv("WORLDCLAW_VLM_URL", os.getenv("WORLDCLAW_VLLM_URL", "")).rstrip("/")
    if not url:
        raise RuntimeError("WORLDCLAW_VLM_URL is not configured")
    content = [{
        "type": "text",
        "text": (
            "Inspect these diagnostic views. Geometry checks are supplied below. "
            "Do not request unrelated content.\n" + json.dumps(geometry_summary, ensure_ascii=False)
        ),
    }]
    content.extend({"type": "image_url", "image_url": {"url": image_data_url(path)}} for path in views)
    schema = DefectReport.model_json_schema()
    payload = {
        "model": request["models"]["vlm"]["model_id"],
        "messages": [
            {"role": "system", "content": VLM_INSPECT_SYSTEM},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "seed": int(request["seed"]),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "defect_report", "strict": True, "schema": schema},
        },
    }
    headers = {"Content-Type": "application/json"}
    token = os.getenv("VLLM_API_KEY", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.post(url + "/v1/chat/completions", json=payload, headers=headers, timeout=900)
    response.raise_for_status()
    content_value = response.json()["choices"][0]["message"]["content"]
    return parse_defect_report(json.loads(content_value))


def apply_transform_repairs(assets: list[dict], defects: list[dict], metrics: dict) -> None:
    by_id = {asset["id"]: asset for asset in assets}
    overlap_index = 0
    for defect in defects:
        asset = by_id.get(defect.get("asset_id"))
        if asset is None:
            continue
        transform = np.asarray(asset["transform_z_up"], dtype=float)
        item_metrics = metrics.get(asset["id"], {})
        if defect["kind"] == "floating":
            transform[2, 3] -= float(item_metrics.get("floating_m", 0.0))
        elif defect["kind"] == "penetration":
            transform[2, 3] += float(item_metrics.get("penetration_m", 0.0))
        elif defect["kind"] == "scale":
            transform[:3, :3] *= 0.9 if defect["severity"] > 0.5 else 1.05
        elif defect["kind"] == "overlap":
            angle = overlap_index * 2.399963229728653
            distance = max(float(item_metrics.get("height_m", 1.0)) * 0.35, 0.25)
            transform[0, 3] += math.cos(angle) * distance
            transform[1, 3] += math.sin(angle) * distance
            overlap_index += 1
        asset["transform_z_up"] = transform.tolist()


def regenerate_bad_assets(request: dict, work_dir: Path, assets: list[dict], defects: list[dict], round_index: int):
    target_ids = {
        item.get("asset_id") for item in defects
        if item.get("kind") in ("mesh", "texture") and item.get("asset_id")
    }
    if not target_ids:
        return
    by_id = {asset["id"]: asset for asset in assets}
    refinement_assets = []
    for asset_id in sorted(target_ids):
        asset = by_id.get(asset_id)
        if asset is None or effective_asset_type(asset.get("category", ""), asset.get("asset_type")) in {
            AssetType.LINEAR_STRUCTURE, AssetType.SURFACE_REGION_FEATURE,
        }:
            continue
        refinement_assets.append({
            "id": asset_id, "category": asset["category"], "asset_type": asset.get("asset_type"),
            "path": asset["source_image"],
            "source_image": asset["source_image"], "region_id": asset["region_id"],
        })
    if not refinement_assets:
        return
    command = os.getenv("WORLDCLAW_IMAGE3D_WORKER")
    if not command:
        raise RuntimeError("mesh/texture refinement requested but WORLDCLAW_IMAGE3D_WORKER is unset")
    request_path = work_dir / f"refinement_reconstruction_{round_index}_request.json"
    response_path = work_dir / f"refinement_reconstruction_{round_index}_response.json"
    payload = dict(request)
    payload.update({
        "stage": "refinement_reconstruction",
        "seed": int(request["seed"]) + round_index * 1000,
        "refinement_assets": refinement_assets,
    })
    request_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pool = HunyuanDaemonPool.from_environment()
    if pool is not None:
        response = pool.run(payload, request_path, response_path)
    else:
        completed = subprocess.run(
            shlex.split(command, posix=os.name != "nt")
            + ["--request", str(request_path), "--response", str(response_path)],
            capture_output=True, text=True, check=False, timeout=7200,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"asset regeneration failed: {completed.stderr[-4000:]}")
        response = json.loads(response_path.read_text(encoding="utf-8"))
    if response.get("status") != "ok":
        raise RuntimeError(f"asset regeneration returned failure: {response.get('error')}")
    for generated in response.get("assets", []):
        if generated["id"] in by_id:
            by_id[generated["id"]]["mesh"] = generated["mesh"]
            by_id[generated["id"]]["source_model"] = generated["model"]["model_id"]


def repair_local_terrain(height: np.ndarray, vertices: np.ndarray, assets: list[dict], metrics: dict, world_size):
    rows, columns = height.shape
    width, depth = world_size
    changed = False
    vertex_cache: dict[str, np.ndarray] = {}
    for asset in assets:
        item = metrics.get(asset["id"], {})
        if item.get("contact_ratio", 1.0) >= 0.5:
            continue
        points = world_vertices(asset, vertex_cache)
        x0, y0 = points[:, 0].min(), points[:, 1].min()
        x1, y1 = points[:, 0].max(), points[:, 1].max()
        c0 = int(np.clip((x0 + width / 2) / width * (columns - 1), 0, columns - 1))
        c1 = int(np.clip((x1 + width / 2) / width * (columns - 1), 0, columns - 1))
        r0 = int(np.clip((y0 + depth / 2) / depth * (rows - 1), 0, rows - 1))
        r1 = int(np.clip((y1 + depth / 2) / depth * (rows - 1), 0, rows - 1))
        if c1 <= c0 or r1 <= r0:
            continue
        target = float(points[:, 2].min())
        patch = height[r0:r1 + 1, c0:c1 + 1]
        patch[:] = patch * 0.25 + target * 0.75
        changed = True
    if changed:
        vertices[:, 2] = height.reshape(-1)
    return changed


def main():
    args = worker_args()
    request = read_request(args.request)
    work_dir = Path(request["work_dir"])
    run_dir = Path(request["run_dir"])
    placement = load_stage_response(work_dir, "placement")
    assets = copy.deepcopy(placement["assets"])
    terrain_path = work_dir / "terrain.npz"
    terrain_data = np.load(terrain_path)
    height = terrain_data["height"].copy()
    vertices = terrain_data["vertices"].copy()
    triangles = terrain_data["triangles"]
    plan = json.loads((work_dir / "scene_plan.json").read_text(encoding="utf-8"))
    world_size = tuple(plan["world_size_m"])
    blender = os.getenv("BLENDER_BIN", str(Path.home() / "apps" / "blender-4.2.0-linux-x64" / "blender"))
    diagnostic_script = Path(__file__).resolve().parents[1] / "scripts" / "blender_diagnostic.py"
    rounds = []
    best = None
    diagnostic_root = run_dir / "diagnostic_views"
    diagnostic_root.mkdir(parents=True, exist_ok=True)

    for round_index in range(1, 4):
        sample_height = terrain_sampler(height, world_size)
        geometry, metrics = geometry_defects(assets, sample_height)
        temporary_glb = work_dir / f"refinement_round_{round_index}.glb"
        build_scene(work_dir, assets, temporary_glb)
        round_dir = diagnostic_root / f"round_{round_index}"
        subprocess.run(
            [blender, "--background", "--python", str(diagnostic_script), "--",
             "--glb", str(temporary_glb), "--output", str(round_dir)],
            check=True, timeout=3600,
        )
        views = sorted(round_dir.glob("view_*.png"))
        if len(views) != 4 or any(path.stat().st_size == 0 for path in views):
            raise RuntimeError(f"round {round_index}: Blender did not produce four diagnostic views")
        visual = vlm_report(request, views, geometry)
        combined = geometry + [item.model_dump() for item in visual.defects]
        score = sum(float(item["severity"]) for item in combined)
        accepted = not geometry and visual.acceptable
        record = {
            "round": round_index, "accepted": accepted, "score": score,
            "geometry_defects": geometry, "visual_report": visual.model_dump(),
            "diagnostic_views": [str(path) for path in views],
        }
        rounds.append(record)
        (work_dir / f"refinement_{round_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if best is None or score < best["score"]:
            best = {"score": score, "assets": copy.deepcopy(assets), "height": height.copy(), "vertices": vertices.copy()}
        if accepted:
            break
        regenerate_bad_assets(request, work_dir, assets, combined, round_index)
        apply_transform_repairs(assets, combined, metrics)
        if repair_local_terrain(height, vertices, assets, metrics, world_size):
            np.savez_compressed(terrain_path, height=height, vertices=vertices, triangles=triangles)

    assert best is not None
    final_assets = assets if rounds[-1]["accepted"] else best["assets"]
    if not rounds[-1]["accepted"]:
        np.savez_compressed(
            terrain_path, height=best["height"], vertices=best["vertices"], triangles=triangles
        )
    write_response(args.response, {
        "status": "ok", "assets": final_assets, "rounds": rounds,
        "accepted": rounds[-1]["accepted"], "best_candidate_retained": not rounds[-1]["accepted"],
    })


if __name__ == "__main__":
    main()
