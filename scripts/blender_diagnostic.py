"""Render four deterministic diagnostic views of a GLB with Blender 4.2."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))
    meshes = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    if not meshes:
        raise RuntimeError("diagnostic GLB has no mesh objects")
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    # Keep diagnostic images compact enough for multi-image VLM inspection.
    scene.render.resolution_x = 384
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.look = "AgX - Medium High Contrast"
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.055, 0.07, 0.09, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    sun_data = bpy.data.lights.new("DiagnosticSun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("DiagnosticSun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(30), math.radians(-20), math.radians(-35))
    camera_data = bpy.data.cameras.new("DiagnosticCamera")
    camera_data.lens = 48
    camera = bpy.data.objects.new("DiagnosticCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    target = bpy.data.objects.new("DiagnosticTarget", None)
    scene.collection.objects.link(target)
    target.location = (0, 0, 3)
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    # Camera local up is Y; world height remains Z-up.
    constraint.up_axis = "UP_Y"
    positions = [(75, -90, 60), (-75, -70, 55), (-70, 75, 58), (70, 75, 52)]
    for index, position in enumerate(positions):
        camera.location = position
        scene.render.filepath = str((args.output / f"view_{index:02d}.png").resolve())
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
