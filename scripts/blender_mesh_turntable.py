"""Render eight deterministic turntable views for one mesh asset."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("mesh validation GLB has no mesh objects")
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (lo + hi) / 2
    radius = max((hi - lo).length * 1.35, 1.0)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 320
    scene.render.resolution_y = 320
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    world = scene.world or bpy.data.worlds.new("MeshValidationWorld")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.06, 0.07, 0.08, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
    light_data = bpy.data.lights.new("MeshValidationKey", "AREA")
    light_data.energy = 900
    light_data.shape = "DISK"
    light_data.size = radius
    light = bpy.data.objects.new("MeshValidationKey", light_data)
    scene.collection.objects.link(light)
    light.location = center + Vector((radius, -radius, radius * 1.4))
    light.rotation_euler = (Vector(center) - light.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("MeshValidationCamera")
    camera_data.lens = 52
    camera = bpy.data.objects.new("MeshValidationCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for index, degrees in enumerate(range(0, 360, 45)):
        angle = math.radians(degrees)
        camera.location = center + Vector((math.cos(angle) * radius, math.sin(angle) * radius, radius * 0.42))
        camera.rotation_euler = (Vector(center) - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str((args.output / f"view_{degrees:03d}.png").resolve())
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
