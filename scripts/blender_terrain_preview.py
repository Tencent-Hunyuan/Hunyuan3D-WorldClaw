"""Build a Blender-only preview from a WorldClaw terrain.npz artifact.

The script intentionally has no dependency on the downstream asset pipeline.
It reads NumPy .npy members from the compressed archive with the Python
standard library so it also works with Blender's bundled Python.
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
import zipfile
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def read_npy(blob: bytes):
    if blob[:6] != b"\x93NUMPY":
        raise ValueError("unsupported .npy header")
    major, minor = blob[6], blob[7]
    header_size_bytes = 2 if major == 1 else 4
    offset = 8
    header_size = int.from_bytes(blob[offset : offset + header_size_bytes], "little")
    offset += header_size_bytes
    header = ast.literal_eval(blob[offset : offset + header_size].decode("latin1"))
    offset += header_size
    descr = header["descr"]
    if header.get("fortran_order"):
        raise ValueError("Fortran-order arrays are not supported")
    shape = tuple(int(value) for value in header["shape"])
    if descr not in {"<f4", "<f8", "<u4", "<i4"}:
        raise ValueError(f"unsupported terrain dtype: {descr}")
    fmt = {"<f4": "<f", "<f8": "<d", "<u4": "<I", "<i4": "<i"}[descr]
    item_size = struct.calcsize(fmt)
    count = 1
    for value in shape:
        count *= value
    expected = offset + count * item_size
    if expected > len(blob):
        raise ValueError("truncated .npy payload")
    values = [item[0] for item in struct.iter_unpack(fmt, blob[offset:expected])]
    if len(shape) == 1:
        return values
    width = shape[-1]
    return [tuple(values[index : index + width]) for index in range(0, len(values), width)]


def load_terrain(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {"vertices.npy", "triangles.npy"}
        missing = required - names
        if missing:
            raise ValueError(f"terrain archive missing: {sorted(missing)}")
        vertices = read_npy(archive.read("vertices.npy"))
        triangles = read_npy(archive.read("triangles.npy"))
    if not vertices or not triangles:
        raise ValueError("terrain archive contains an empty mesh")
    return vertices, triangles


def material(name: str, color: tuple[float, float, float]):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1.0)
    value.use_nodes = True
    shader = value.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = 0.92
    return value


def point_camera(camera, target, position):
    camera.location = position
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def main():
    args = parse_args()
    vertices, triangles = load_terrain(args.terrain)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    available = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    render_engine = "BLENDER_EEVEE_NEXT"
    if render_engine not in available:
        render_engine = "BLENDER_EEVEE"
    if render_engine not in available:
        raise RuntimeError(f"no Eevee render engine available: {sorted(available)}")
    scene.render.engine = render_engine
    scene.render.resolution_x, scene.render.resolution_y = 800, 600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(args.preview)
    scene.view_settings.look = "AgX - Medium High Contrast"

    mesh = bpy.data.meshes.new("TerrainMesh")
    mesh.from_pydata(vertices, [], triangles)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new("Terrain", mesh)
    scene.collection.objects.link(obj)
    obj.data.materials.append(material("TerrainPreview", (0.16, 0.29, 0.12)))
    obj["scene_role"] = "terrain_only_preview"
    obj["source_npz"] = str(args.terrain)
    obj["vertex_count"] = len(vertices)
    obj["triangle_count"] = len(triangles)

    minimum = Vector((min(v[0] for v in vertices), min(v[1] for v in vertices), min(v[2] for v in vertices)))
    maximum = Vector((max(v[0] for v in vertices), max(v[1] for v in vertices), max(v[2] for v in vertices)))
    center = (minimum + maximum) * 0.5
    extent = max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z, 1.0)

    camera_data = bpy.data.cameras.new("TerrainPreviewCamera")
    camera = bpy.data.objects.new("TerrainPreviewCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 45
    point_camera(camera, center, center + Vector((extent * 0.95, -extent * 1.05, extent * 0.85)))

    light_data = bpy.data.lights.new("TerrainPreviewSun", type="SUN")
    light_data.energy = 2.2
    light = bpy.data.objects.new("TerrainPreviewSun", light_data)
    scene.collection.objects.link(light)
    light.rotation_euler = (0.55, -0.45, -0.65)
    area_data = bpy.data.lights.new("TerrainPreviewFill", type="AREA")
    area_data.energy = 900.0
    area_data.shape = "DISK"
    area_data.size = extent * 0.8
    area = bpy.data.objects.new("TerrainPreviewFill", area_data)
    scene.collection.objects.link(area)
    area.location = center + Vector((-extent * 0.25, -extent * 0.25, extent * 1.3))
    point_camera(area, center, area.location)

    scene.world = bpy.data.worlds.new("TerrainPreviewWorld")
    scene.world.color = (0.035, 0.05, 0.07)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    bpy.ops.render.render(write_still=True)
    metadata = {
        "status": "ok",
        "source_npz": str(args.terrain),
        "blend": str(args.output),
        "preview": str(args.preview),
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "bounds": {"min": list(minimum), "max": list(maximum)},
        "blender_version": bpy.app.version_string,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
