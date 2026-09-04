"""Render structural validation views from validation_views.json in Blender."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _set_feature_focus(scene, feature_ids):
    """Temporarily isolate GPT-selected structural features for close views."""
    requested = {str(value) for value in (feature_ids or [])}
    if not requested:
        return []
    hidden = []
    for obj in scene.objects:
        if obj.type != "MESH" or "terrain" in obj.name.lower() or obj.name in requested:
            continue
        if not obj.hide_render:
            obj.hide_render = True
            hidden.append(obj)
    return hidden


def render_group(scene, camera, views, group_name, output, key_light=None):
    rendered = []
    for view in views:
        hidden = _set_feature_focus(scene, view.get("feature_ids", [])) if group_name != "fixed" else []
        position = Vector(view["position"])
        target = Vector(view["target"])
        camera.location = position
        camera.rotation_euler = (target - position).to_track_quat("-Z", "Y").to_euler()
        if key_light is not None:
            aim_key_light(key_light, position, target)
        scene.render.filepath = str((output / "renders" / group_name / f"{view['id']}.png").resolve())
        Path(scene.render.filepath).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.render.render(write_still=True)
        rendered.append({"id": view["id"], "kind": group_name, "path": scene.render.filepath, "renderer": "blender"})
        for obj in hidden:
            obj.hide_render = False
    return rendered


def ensure_key_light(scene):
    """Add a validation-only neutral light; the Blend is never saved here."""
    light_object = bpy.data.objects.get("StructuralValidationKeyLight")
    if light_object is None:
        light_data = bpy.data.lights.new("StructuralValidationKeyLight", type="AREA")
        light_object = bpy.data.objects.new("StructuralValidationKeyLight", light_data)
        scene.collection.objects.link(light_object)
    light_object.data.energy = 2200.0
    light_object.data.shape = "DISK"
    light_object.data.size = 55.0
    return light_object


def aim_key_light(light_object, position, target):
    light_object.location = position
    light_object.rotation_euler = (target - position).to_track_quat("-Z", "Y").to_euler()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str((args.run / "scene.blend").resolve()))
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    camera = bpy.data.objects.get("StructuralValidationCamera")
    if camera is None:
        data = bpy.data.cameras.new("StructuralValidationCamera")
        camera = bpy.data.objects.new("StructuralValidationCamera", data)
        scene.collection.objects.link(camera)
    camera.data.lens = 48
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100000.0
    scene.camera = camera
    key_light = ensure_key_light(scene)
    views = json.loads(args.views.read_text(encoding="utf-8"))
    rendered = []
    # Keep the camera/light synchronized for each view so close-up follow-up
    # renders remain readable even when the imported scene has dark canopy.
    rendered += render_group(scene, camera, views.get("fixed", []), "fixed", args.output, key_light)
    rendered += render_group(scene, camera, views.get("adaptive", []), "adaptive", args.output, key_light)
    rendered += render_group(scene, camera, views.get("additional", []), "additional", args.output, key_light)
    response = {"status": "ok", "renderer": "blender", "renders": rendered}
    (args.output / "render_response.json").write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
