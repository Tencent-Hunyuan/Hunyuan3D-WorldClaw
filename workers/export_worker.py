#!/usr/bin/env python3
"""Assemble terrain and independently editable placed asset nodes into GLB."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from workers.common import load_stage_response, read_request, worker_args, write_response
from worldclaw_oss.export import orient_terrain_triangles_upward
from worldclaw_oss.geometry import gltf_vertices_to_z_up, z_up_to_gltf_matrix


def load_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        return trimesh.Trimesh(vertices=np.asarray(data["vertices"], dtype=np.float32), faces=np.asarray(data["triangles"], dtype=np.uint32), process=False)
    value = trimesh.load(path, force="mesh", process=False)
    if isinstance(value, trimesh.Scene):
        geometries = tuple(value.geometry.values())
        if not geometries:
            raise RuntimeError(f"mesh scene is empty: {path}")
        value = trimesh.util.concatenate(geometries)
    # Placement GLBs are serialized through glTF's Y-up basis. Convert them
    # back to the internal Z-up frame before composing the final scene.
    value.vertices = gltf_vertices_to_z_up(value.vertices).astype(np.float32)
    return value


def terrain_asset_side_stats(
    vertices: np.ndarray,
    triangles: np.ndarray,
    height: np.ndarray,
    assets: list[dict],
) -> dict[str, float | int]:
    """Check that placed asset anchors lie on Terrain's positive-normal side."""
    rows, columns = height.shape
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    dots: list[float] = []
    for asset in assets:
        transform = np.asarray(asset.get("transform_z_up"), dtype=float)
        if transform.shape != (4, 4):
            continue
        point = transform[:3, 3]
        if not (minimum[0] <= point[0] <= maximum[0] and minimum[1] <= point[1] <= maximum[1]):
            continue
        fx = (point[0] - minimum[0]) / max(maximum[0] - minimum[0], 1e-9) * (columns - 1)
        fy = (point[1] - minimum[1]) / max(maximum[1] - minimum[1], 1e-9) * (rows - 1)
        ix = int(np.clip(np.floor(fx), 0, columns - 2))
        iy = int(np.clip(np.floor(fy), 0, rows - 2))
        vertex_index = iy * columns + ix
        tri_index = min(vertex_index * 2, len(triangles) - 1)
        face = vertices[triangles[tri_index]]
        normal = np.cross(face[1] - face[0], face[2] - face[0]).astype(float)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        terrain_point = np.array([point[0], point[1], float(height[iy, ix])], dtype=float)
        dots.append(float(np.dot(point - terrain_point, normal)))
    mean_dot = float(np.mean(dots)) if dots else 0.0
    positive_ratio = float(np.mean(np.asarray(dots) > 0.0)) if dots else 1.0
    return {
        "asset_samples": len(dots),
        "asset_side_dot_mean": mean_dot,
        "asset_positive_side_ratio": positive_ratio,
        "asset_side_check_passed": bool(not dots or mean_dot > 0.0),
    }


def build_scene(work_dir: Path, assets: list[dict], output_path: Path, structural_meshes: list[dict] | None = None) -> dict:
    terrain_path = work_dir / "terrain_structural.npz"
    terrain = np.load(terrain_path if terrain_path.exists() else work_dir / "terrain.npz")
    terrain_vertices = np.asarray(terrain["vertices"], dtype=np.float32)
    terrain_triangles, terrain_orientation = orient_terrain_triangles_upward(
        terrain_vertices, terrain["triangles"]
    )
    terrain_side = terrain_asset_side_stats(
        terrain_vertices, terrain_triangles, np.asarray(terrain["height"]), assets
    )
    # Trimesh exports coordinates verbatim; encode the pipeline's internal
    # right-handed Z-up frame in glTF's Y-up frame explicitly.  This is a
    # proper rotation (det=+1), so the validated terrain winding remains
    # unchanged while Blender receives the intended upward normals.
    basis = z_up_to_gltf_matrix()
    terrain_gltf_vertices = (basis[:3, :3] @ terrain_vertices.T).T
    terrain_mesh = trimesh.Trimesh(
        vertices=terrain_gltf_vertices, faces=terrain_triangles, process=False
    )
    # Force trimesh to materialize vertex normals before GLB export. Without
    # this access the exporter emits POSITION/COLOR only, leaving Blender to
    # infer flat or weak normals for the large terrain surface.
    _ = terrain_mesh.vertex_normals
    terrain_mesh.visual = trimesh.visual.ColorVisuals(
        terrain_mesh, face_colors=[74, 119, 63, 255]
    )
    scene = trimesh.Scene()
    scene.add_geometry(
        terrain_mesh, node_name="Terrain", geom_name="Terrain",
        transform=np.eye(4), metadata={"kind": "terrain"},
    )
    mesh_cache: dict[str, tuple[trimesh.Trimesh, str]] = {}
    for asset in assets:
        mesh_path = str(Path(asset["mesh"]).resolve())
        cached = mesh_cache.get(mesh_path)
        if cached is None:
            mesh_value = load_mesh(Path(mesh_path))
            _ = mesh_value.vertex_normals
            # Asset meshes are loaded in internal Z-up above; convert their
            # geometry and instance transform to glTF Y-up at the export
            # boundary, keeping scene.json's persisted transform_z_up intact.
            mesh_value.vertices = (basis[:3, :3] @ np.asarray(mesh_value.vertices).T).T.astype(np.float32)
            cached = (mesh_value, f"AssetMesh_{len(mesh_cache):04d}")
            mesh_cache[mesh_path] = cached
        mesh, geometry_name = cached
        transform = np.asarray(asset["transform_z_up"], dtype=float)
        transform = basis @ transform @ np.linalg.inv(basis)
        metadata = {
            "asset_id": asset["id"], "category": asset["category"],
            "region_id": asset["region_id"], "source_model": asset["source_model"],
        }
        if geometry_name in scene.geometry:
            # Scene.add_geometry intentionally uniquifies an existing geometry
            # name, which would duplicate hundreds of prototype meshes. Add a
            # transform node that references the cached geometry instead.
            scene.graph.update(
                frame_to=asset["id"], frame_from=None, matrix=transform,
                geometry=geometry_name, geometry_flags={"visible": True}, metadata=metadata,
            )
        else:
            scene.add_geometry(
                mesh, node_name=asset["id"], geom_name=geometry_name,
                transform=transform, metadata=metadata,
            )
    for feature in structural_meshes or []:
        mesh_path = Path(feature["mesh"])
        mesh_value = load_mesh(mesh_path)
        _ = mesh_value.vertex_normals
        structural_color = {
            "lake": [45, 130, 170, 235],
            "water": [45, 130, 170, 235],
            "river": [45, 130, 170, 235],
            "stream": [45, 130, 170, 235],
            "trail": [115, 72, 38, 255],
            "road": [92, 82, 64, 255],
        }.get(str(feature.get("category", "")).lower(), [128, 128, 128, 255])
        # Structural meshes are generated locally rather than through a
        # textured asset pipeline.  Persist an explicit face color so Blender
        # can distinguish water and route surfaces in validation renders.
        mesh_value.visual = trimesh.visual.ColorVisuals(
            mesh_value,
            face_colors=np.tile(np.asarray(structural_color, dtype=np.uint8), (len(mesh_value.faces), 1)),
        )
        mesh_value.vertices = (basis[:3, :3] @ np.asarray(mesh_value.vertices).T).T.astype(np.float32)
        transform = basis @ np.asarray(feature.get("transform_z_up", np.eye(4)), dtype=float) @ np.linalg.inv(basis)
        scene.add_geometry(mesh_value, node_name=feature["feature_id"], geom_name=f"Structural_{feature['feature_id']}", transform=transform, metadata={"structural_feature": True, "category": feature.get("category", "")})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))
    return {
        "mesh_nodes": 1 + len(assets),
        "asset_nodes": len(assets),
        "structural_nodes": len(structural_meshes or []),
        "independent_nodes": True,
        "terrain_winding": "validated_internal_z_up",
        "terrain_orientation": terrain_orientation,
        "terrain_asset_side": terrain_side,
        "terrain_normal_convention": "internal_Z_up_to_gltf_Y_up_to_blender_Z_up",
    }


def main():
    args = worker_args()
    request = read_request(args.request)
    work_dir = Path(request["work_dir"])
    run_dir = Path(request["run_dir"])
    refinement_path = work_dir / "refinement_response.json"
    if refinement_path.is_file():
        source = json.loads(refinement_path.read_text(encoding="utf-8"))
    else:
        source = load_stage_response(work_dir, "placement")
    assets = source.get("assets", [])
    if not assets:
        raise RuntimeError("no placed assets are available for export")
    branch_path = work_dir / "structural_branch.json"
    branch = json.loads(branch_path.read_text(encoding="utf-8")) if branch_path.is_file() else {}
    structural_meshes = branch.get("structural_meshes", [])
    scene_glb = run_dir / "scene.glb"
    stats = build_scene(work_dir, assets, scene_glb, structural_meshes=structural_meshes)
    plan = json.loads((work_dir / "scene_plan.json").read_text(encoding="utf-8"))
    scene_json = {
        "schema": "worldclaw-oss-scene-v1",
        "official_implementation": False,
        "synthetic": False,
        "prompt": request["prompt"],
        "seed": request["seed"],
        "coordinate_system": {"internal": "Z-up right-handed", "glTF": "Y-up"},
        "plan": plan,
        "assets": assets,
        "structural_features": structural_meshes,
        "models": request["models"],
        "refinement": source.get("rounds", []),
    }
    (run_dir / "scene.json").write_text(
        json.dumps(scene_json, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "diagnostic_views").mkdir(parents=True, exist_ok=True)
    (run_dir / "errors.log").touch(exist_ok=True)
    write_response(args.response, {
        "status": "ok", "scene_glb": str(scene_glb),
        "scene_json": str(run_dir / "scene.json"), **stats,
    })


if __name__ == "__main__":
    main()
