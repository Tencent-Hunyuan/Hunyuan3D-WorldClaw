#!/usr/bin/env python3
"""Hunyuan3D-2.1 image-to-mesh worker with an explicit failure-only fallback."""
from __future__ import annotations

import os
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from workers.common import artifact_key, image_entries, load_stage_response, read_request, seed_everything, worker_args, write_response
from worldclaw_oss.asset_types import AssetType


_PIPELINE_CACHE: dict[tuple[str, str, int, int], tuple[object | None, object]] = {}


def worker_asset_type(category: str, value: str | None = None) -> str:
    if value is not None:
        try:
            return AssetType(value).value
        except (TypeError, ValueError):
            pass
    key = category.strip().lower()
    if key in {"trail", "trails", "winding_trail", "winding_trails", "road", "roads", "river", "rivers", "stream", "streams"}:
        return "linear_structure"
    if key in {"lake", "lakes", "water", "shore", "shoreline", "mud", "mudflat", "forest_ground", "ground", "terrain"}:
        return "surface_region_feature"
    if key in {"reeds", "water_reeds", "grass", "grasses", "fern", "ferns", "shoreline_vegetation", "vegetation", "aquatic_vegetation"}:
        return "procedural_native"
    if key in {"pebble", "pebbles", "small_stone", "small_stones", "stones", "leaf", "leaves", "debris", "trail_rocks"}:
        return "scatter_detail"
    if key in {"tree", "trees", "dense_trees", "forest_dense_trees", "pine", "pines", "birch", "broadleaf", "broadleaf_tree", "palm", "palms"}:
        return "reusable_prototype"
    return "solid_object"


def worker_uses_hunyuan(asset_type: str) -> bool:
    return asset_type in {"solid_object", "reusable_prototype", "scatter_detail"}


def portable_remesh_mesh(input_path: str, output_path: str) -> None:
    """Remesh adapter for PyMeshLab builds without OBJ import/export plugins.

    Hunyuan's paint pipeline expects this step to reduce the shape mesh before
    UV baking.  The Hunyuan environment does not include Open3D, so use the
    orchestrator interpreter when it is available and retain the plain Trimesh
    conversion as a compatibility fallback.
    """
    import trimesh

    target_count = int(os.environ.get("WORLDCLAW_TEXTURE_TARGET_FACES", "40000"))
    helper_python = os.environ.get("WORLDCLAW_OPEN3D_PYTHON")
    if not helper_python:
        model_root = os.environ.get("WORLDCLAW_MODEL_ROOT")
        if model_root:
            candidate = Path(model_root) / "envs" / "orchestrator" / "bin" / "python"
            if candidate.is_file():
                helper_python = str(candidate)
    helper = Path(__file__).with_name("open3d_decimate.py")
    if helper_python and helper.is_file():
        completed = subprocess.run(
            [helper_python, str(helper), "--input", input_path, "--output", output_path,
             "--target-faces", str(target_count)],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode == 0 and Path(output_path).is_file() and Path(output_path).stat().st_size:
            return

    loaded = trimesh.load(input_path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("Hunyuan3D shape mesh is empty")
        loaded = trimesh.util.concatenate(meshes)
    loaded.export(output_path, file_type="obj")


def clean_and_normalize(source_path: Path, output_path: Path, max_extent: float = 1.0) -> dict:
    import trimesh
    loaded = trimesh.load(source_path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("Hunyuan3D returned an empty scene")
        loaded = trimesh.util.concatenate(meshes)
    mesh = loaded.copy()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("Hunyuan3D returned an empty mesh")
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    extent = float(np.max(bounds[1] - bounds[0]))
    if extent <= 1e-8:
        raise RuntimeError("Hunyuan3D returned a zero-size mesh")
    mesh.apply_translation(-center)
    mesh.apply_scale(max_extent / extent)
    # Normalize every prototype to the same authored extent and put its
    # support plane exactly on local z=0 for deterministic ground alignment.
    mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0, 2])])
    _ = mesh.vertex_normals
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path, file_type="glb")
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "normalized_extent": max_extent,
        "bounds": mesh.bounds.tolist(),
        "ground_aligned": bool(abs(float(mesh.bounds[0, 2])) <= 1e-6),
        "normals": int(len(mesh.vertex_normals)),
    }


def hunyuan3d_pipeline(record: dict, load_shape: bool = True):
    max_num_view = int(os.environ.get("HUNYUAN_MAX_NUM_VIEW", "6"))
    resolution = int(os.environ.get("HUNYUAN_PAINT_RESOLUTION", "512"))
    cache_key = (record["model_id"], record["revision"], max_num_view, resolution)
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None and (not load_shape or cached[0] is not None):
        return cached
    source_root = os.environ.get("HUNYUAN3D_SOURCE")
    if not source_root:
        raise RuntimeError("HUNYUAN3D_SOURCE is not configured")
    source = Path(source_root).resolve()
    os.chdir(source)
    # BasicSR versions used by Hunyuan still import the torchvision module
    # removed in torchvision 0.22; expose its unchanged implementation under
    # the legacy name before importing the paint pipeline.
    try:
        import torchvision.transforms.functional_tensor  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        if error.name != "torchvision.transforms.functional_tensor":
            raise
        import sys
        import types
        from torchvision.transforms.functional import rgb_to_grayscale

        compatibility = types.ModuleType("torchvision.transforms.functional_tensor")
        compatibility.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[compatibility.__name__] = compatibility
    for subdirectory in (
        source / "hy3dshape",
        source / "hy3dpaint",
        source / "hy3dpaint" / "custom_rasterizer",
    ):
        if str(subdirectory) not in sys.path:
            sys.path.insert(0, str(subdirectory))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    import textureGenPipeline as texture_module
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    # The bundled simplify_mesh_utils delegates OBJ I/O to PyMeshLab. Some
    # wheels omit that plugin, while Trimesh provides the same intermediate
    # mesh contract needed by the paint pipeline.
    texture_module.remesh_mesh = portable_remesh_mesh
    shape = None
    if load_shape:
        shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            record["model_id"], subfolder="hunyuan3d-dit-v2-1", revision=record["revision"]
        )
    # Keep the paper-style defaults, while allowing a documented lower-memory
    # retry on shared GPUs without changing the Hunyuan3D model boundary.
    paint = Hunyuan3DPaintPipeline(
        Hunyuan3DPaintConfig(max_num_view=max_num_view, resolution=resolution)
    )
    _PIPELINE_CACHE[cache_key] = (shape, paint)
    return _PIPELINE_CACHE[cache_key]


def generate_mesh(
    shape,
    paint,
    image_path: Path,
    shape_glb_path: Path,
    textured_obj_path: Path,
    seed: int,
) -> Path:
    if not shape_glb_path.is_file() or shape_glb_path.stat().st_size == 0:
        image = Image.open(image_path).convert("RGB")
        result = shape(image=image, seed=seed)
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        # Hunyuan's PyMeshLab remesher has an explicit GLB loading path.
        # Passing the shape mesh as OBJ is not portable across plugin sets.
        try:
            mesh.export(shape_glb_path, file_type="glb")
        except TypeError:
            # Lightweight test doubles and older trimesh-compatible meshes may
            # not accept the optional file_type keyword.
            mesh.export(shape_glb_path)
    try:
        paint(
            mesh_path=str(shape_glb_path), image_path=str(image_path),
            output_mesh_path=str(textured_obj_path), save_glb=True,
        )
    except Exception as error:
        raise RuntimeError(f"texture refinement failed: {error}") from error
    textured_glb_path = textured_obj_path.with_suffix(".glb")
    if not textured_glb_path.is_file() or textured_glb_path.stat().st_size == 0:
        raise RuntimeError(
            "texture refinement did not produce a non-empty GLB at "
            f"{textured_glb_path}"
        )
    return textured_glb_path


def generate_vegetation_card(image_path: Path, output_path: Path) -> dict:
    """Create a small textured radial crossed-card cluster for vegetation."""
    import trimesh
    from trimesh.visual.texture import TextureVisuals

    image = Image.open(image_path).convert("RGBA")
    aspect = float(np.clip(image.width / max(image.height, 1), 0.35, 2.5))
    width, height = 0.78 * aspect, 1.0
    vertices, faces, uvs = [], [], []
    for angle in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0):
        tangent = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        base = len(vertices)
        for z in (0.0, height):
            vertices.extend((
                (-tangent * width / 2 + [0.0, 0.0, z]).tolist(),
                (tangent * width / 2 + [0.0, 0.0, z]).tolist(),
            ))
        faces.extend((
            [base, base + 1, base + 3], [base, base + 3, base + 2],
            [base + 3, base + 1, base], [base + 2, base + 3, base],
        ))
        uvs.extend(([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]))
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64), process=False,
    )
    mesh.visual = TextureVisuals(uv=np.asarray(uvs, dtype=np.float32), image=image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path, file_type="glb")
    _ = mesh.vertex_normals
    return {
        "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
        "normalized_extent": 1.0, "bounds": mesh.bounds.tolist(),
        "ground_aligned": True, "normals": int(len(mesh.vertex_normals)),
        "generator": "radial-crossed-card-v1",
    }


def run_once(request_path: Path, response_path: Path):
    request = read_request(request_path)
    seed = int(request["seed"])
    seed_everything(seed)
    work_dir = Path(request["work_dir"])
    record = request["models"]["hunyuan3d"]
    stage = request["stage"]
    if stage == "environment_assets":
        # Retry requests may carry a freshly regenerated reference image. Keep
        # the canonical reference response immutable and prefer the explicit
        # override for this attempt.
        override = request.get("reference_override")
        if override:
            sources = image_entries({"images": override})
        else:
            upstream = load_stage_response(work_dir, "environment_references")
            sources = image_entries(upstream)
    elif stage == "reconstruction":
        upstream = load_stage_response(work_dir, "segmentation")
        sources = [
            {"path": item["crop"], "id": item["id"], "category": item["category"],
             "asset_type": item.get("asset_type"),
             "bbox_xyxy": item["bbox_xyxy"], "centroid_xy": item["centroid_xy"],
             "source_image": item["source_image"], "region_id": item.get("region_id"),
             "terrain_camera": item.get("terrain_camera"), "source_size": item.get("source_size"),
             "mask": item.get("mask")}
            for item in upstream.get("instances", [])
        ]
    elif stage == "refinement_reconstruction":
        sources = request.get("refinement_assets", [])
    else:
        sources = []
    source_order = {
        str(source.get("id")): index for index, source in enumerate(sources)
    }
    selected_ids = request.get("source_ids")
    if selected_ids is not None:
        allowed = {str(item) for item in selected_ids}
        sources = [source for source in sources if str(source.get("id")) in allowed]
    if not sources:
        raise RuntimeError(f"{stage} produced no image inputs")

    output_root = work_dir / "reconstruction" / stage
    output_root.mkdir(parents=True, exist_ok=True)
    procedural_sources = [source for source in sources if not worker_uses_hunyuan(worker_asset_type(source.get("category", ""), source.get("asset_type")))]
    hunyuan_sources = [source for source in sources if source not in procedural_sources]
    shape = paint = None
    error = None
    if hunyuan_sources:
        # A timed-out parent may leave every final mesh intact.  Rebuild the
        # response from those artifacts without reloading the multi-GB paint
        # pipeline or repeating expensive generation.
        mesh_cache_ready = all(
            (output_root / f"{artifact_key(Path(source['path']), record['model_id'], record['revision'], {'stage': stage, 'seed': seed, 'category': source.get('category', ''), 'asset_type': worker_asset_type(source.get('category', ''), source.get('asset_type'))})}.glb").is_file()
            for source in hunyuan_sources
        )
        shape_cache_ready = all(
            (output_root / f"{artifact_key(Path(source['path']), record['model_id'], record['revision'], {'stage': stage, 'seed': seed, 'category': source.get('category', '')})}.shape.glb").is_file()
            for source in hunyuan_sources
        )
        if not mesh_cache_ready:
            try:
                shape, paint = hunyuan3d_pipeline(record, load_shape=not shape_cache_ready)
            except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as caught:
                error = caught
    if error is not None:
        # The orchestrator may launch a separately installed TRELLIS worker.
        fallback_command = os.getenv("WORLDCLAW_TRELLIS_WORKER")
        if not fallback_command:
            raise RuntimeError(f"[FALLBACK_ELIGIBLE] Hunyuan3D unavailable: {error}") from error
        fallback_request = work_dir / f"{stage}_trellis_request.json"
        fallback_response = work_dir / f"{stage}_trellis_response.json"
        fallback_request.write_text(request_path.read_text(encoding="utf-8"), encoding="utf-8")
        command = fallback_command.split() + ["--request", str(fallback_request), "--response", str(fallback_response)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"[FALLBACK_FAILED] TRELLIS: {completed.stderr[-4000:]}") from error
        value = __import__("json").loads(fallback_response.read_text(encoding="utf-8"))
        value["fallback"] = {"from": "hunyuan3d", "to": "trellis", "reason": str(error)}
        write_response(response_path, value)
        return

    outputs = []
    for index, source in enumerate(sources):
        stable_index = source_order.get(str(source.get("id")), index)
        image_path = Path(source["path"])
        asset_type = worker_asset_type(source.get("category", ""), source.get("asset_type"))
        parameters = {"stage": stage, "seed": seed, "category": source.get("category", ""), "asset_type": asset_type}
        key = artifact_key(image_path, record["model_id"], record["revision"], parameters)
        shape_glb_path = output_root / f"{key}.shape.glb"
        textured_obj_path = output_root / f"{key}.textured.obj"
        textured_glb_path = textured_obj_path.with_suffix(".glb")
        mesh_path = output_root / f"{key}.glb"
        if not mesh_path.exists():
            if asset_type == "procedural_native":
                mesh_stats = generate_vegetation_card(image_path, mesh_path)
            else:
                if not textured_glb_path.exists():
                    textured_glb_path = generate_mesh(
                        shape,
                        paint,
                        image_path,
                        shape_glb_path,
                        textured_obj_path,
                        seed + stable_index,
                    )
                if textured_glb_path.stat().st_size == 0:
                    raise RuntimeError(f"cached textured GLB is empty: {textured_glb_path}")
                mesh_stats = clean_and_normalize(textured_glb_path, mesh_path)
        else:
            mesh_stats = {"cached": True}
        outputs.append({
            "id": source.get("id", f"asset_{index:04d}"),
            "category": source.get("category", "asset"),
            "asset_type": asset_type,
            "source_image": str(image_path),
            "mesh": str(mesh_path),
            "region_id": source.get("region_id"),
            "count": source.get("count"),
            "scene_role": source.get("scene_role"),
            "importance": source.get("importance"),
            "bbox_xyxy": source.get("bbox_xyxy"),
            "centroid_xy": source.get("centroid_xy"),
            "source_image_path": source.get("source_image"),
            "terrain_camera": source.get("terrain_camera"),
            "source_size": source.get("source_size"),
            "mask": source.get("mask"),
            "model": record,
            "mesh_stats": mesh_stats,
        })
    write_response(response_path, {"status": "ok", "assets": outputs, "model": record, "fallback": None})


def _forward_request(socket_path: Path, request_path: Path, response_path: Path) -> None:
    payload = {"request": read_request(request_path), "response": str(response_path)}
    last_error = None
    for _ in range(60):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(7200)
                connection.connect(str(socket_path))
                connection.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
                data = b""
                while not data.endswith(b"\n"):
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    data += chunk
            break
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout) as error:
            last_error = error
            time.sleep(1)
    else:
        raise RuntimeError(f"Hunyuan daemon is unavailable at {socket_path}: {last_error}")
    result = json.loads(data.decode("utf-8")) if data else {"status": "error", "error": "daemon closed connection"}
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "Hunyuan daemon request failed"))


def serve(socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(4)
        while True:
            connection, _ = server.accept()
            with connection:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                try:
                    payload = json.loads(data.decode("utf-8"))
                    if payload.get("ping"):
                        connection.sendall(b'{"status":"ok","ping":true}\n')
                        continue
                    request_path = Path(payload["response"]).with_suffix(".daemon_request.json")
                    request_path.write_text(json.dumps(payload["request"], ensure_ascii=False), encoding="utf-8")
                    run_once(request_path, Path(payload["response"]))
                    result = {"status": "ok"}
                except Exception as error:
                    result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
                connection.sendall(json.dumps(result, ensure_ascii=False).encode("utf-8") + b"\n")


def main():
    args = worker_args()
    if args.server:
        serve(args.socket or Path(os.environ.get("HUNYUAN_WORKER_SOCKET", "/tmp/hunyuan3d.sock")))
        return
    if not args.request or not args.response:
        raise SystemExit("--request and --response are required outside --server mode")
    socket_name = os.environ.get("HUNYUAN_WORKER_SOCKET")
    if socket_name:
        _forward_request(Path(socket_name), args.request, args.response)
    else:
        run_once(args.request, args.response)


if __name__ == "__main__":
    main()
