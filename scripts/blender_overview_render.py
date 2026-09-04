"""Render an auto-framed overview preview and walkthrough from a live run."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def scene_bounds(scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        points.extend((evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box))
    if not points:
        raise RuntimeError("scene contains no renderable mesh objects")
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return minimum, maximum


def point_camera(camera, target, position):
    camera.location = position
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def make_material(name, color, roughness=0.82):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    if shader.inputs.get("Emission Color"):
        shader.inputs["Emission Color"].default_value = (*color, 1.0)
    if shader.inputs.get("Emission Strength"):
        shader.inputs["Emission Strength"].default_value = 0.22
    return material


def _scene_has_tokens(scene, tokens):
    """Inspect imported semantic object metadata instead of run-directory names."""
    wanted = tuple(str(token).lower() for token in tokens)
    for obj in scene.objects:
        values = [obj.name.lower()]
        for key in ("category", "asset_role", "scene_role"):
            if key in obj:
                values.append(str(obj[key]).lower())
        if any(token in value for token in wanted for value in values):
            return True
    return False


def prepare_materials(scene, run_name):
    desert = _scene_has_tokens(scene, ("desert", "sand", "dune"))
    palette = {
        "terrain": (0.62, 0.42, 0.18) if desert else (0.08, 0.22, 0.06),
        "tree": (0.02, 0.20, 0.03),
        "rock": (0.34, 0.36, 0.38),
        "castle": (0.58, 0.28, 0.10),
        "building": (0.58, 0.31, 0.14) if desert else (0.42, 0.20, 0.06),
        "water": (0.04, 0.28, 0.62),
        "road": (0.22, 0.16, 0.10),
        "farmland": (0.48, 0.38, 0.10),
        "sand": (0.74, 0.55, 0.24),
        "default": (0.52, 0.43, 0.30),
    }
    materials = {key: make_material(f"Overview_{key}", value) for key, value in palette.items()}
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if obj.name.lower().startswith(("presentation_", "forestwatersurface")):
            continue
        lowered = obj.name.lower()
        if "terrain" in lowered:
            key = "terrain"
        elif any(token in lowered for token in ("tree", "palm", "forest")):
            key = "tree"
        elif any(token in lowered for token in ("rock", "stone", "mountain")):
            key = "rock"
        elif "castle" in lowered:
            key = "castle"
        elif any(token in lowered for token in ("building", "house", "cabin", "wall")):
            key = "building"
        elif any(token in lowered for token in ("water", "lake", "river", "pool")):
            key = "water"
        elif any(token in lowered for token in ("road", "trail")):
            key = "road"
        elif "farm" in lowered:
            key = "farmland"
        elif any(token in lowered for token in ("dune", "sand")):
            key = "sand"
        else:
            key = "default"
        obj.data.materials.clear()
        obj.data.materials.append(materials[key])

    # The imported GLBs have no dependable lighting setup.  Give the review
    # render a readable, neutral presentation without changing source assets.
    scene.view_settings.exposure = 0.8
    for light in (item for item in scene.objects if item.type == "LIGHT"):
        if hasattr(light.data, "energy"):
            light.data.energy = max(float(light.data.energy), 5.0)


def _ground_z(scene, x, y):
    terrain = next((obj for obj in scene.objects if obj.type == "MESH" and "terrain" in obj.name.lower()), None)
    if terrain is None:
        return 0.0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(terrain, depsgraph)
    inverse = terrain.matrix_world.inverted()
    origin = inverse @ Vector((x, y, 10000.0))
    direction = (inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    hit, _, _, _ = bvh.ray_cast(origin, direction, 20000.0)
    return float((terrain.matrix_world @ hit).z) if hit is not None else 0.0


def _clamp_anchor_to_terrain(scene, anchor, margin=500.0):
    terrain = next((obj for obj in scene.objects if obj.type == "MESH" and "terrain" in obj.name.lower()), None)
    if terrain is None:
        return anchor
    points = [terrain.matrix_world @ Vector(corner) for corner in terrain.bound_box]
    min_x, max_x = min(p.x for p in points) + margin, max(p.x for p in points) - margin
    min_y, max_y = min(p.y for p in points) + margin, max(p.y for p in points) - margin
    return Vector((min(max(anchor.x, min_x), max_x), min(max(anchor.y, min_y), max_y), anchor.z))


def _proxy_cube(name, location, dimensions, material):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def _proxy_cone(name, location, radius, depth, material, vertices=4):
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=radius, radius2=0.0, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def _forest_anchor(scene):
    candidates = [obj for obj in scene.objects if obj.type == "MESH" and obj.name.lower() != "terrain"]
    subject = next(
        (obj for obj in candidates if any(token in obj.name.lower() for token in ("lake", "water", "cabin", "house"))),
        max(candidates, key=lambda obj: obj.dimensions.length),
    )
    terrain = next((obj for obj in scene.objects if obj.type == "MESH" and "terrain" in obj.name.lower()), None)
    if terrain is not None:
        points = [terrain.matrix_world @ Vector(corner) for corner in terrain.bound_box]
        center_x = (min(point.x for point in points) + max(point.x for point in points)) * 0.5
        center_y = (min(point.y for point in points) + max(point.y for point in points)) * 0.5
        anchor = Vector((center_x, center_y, subject.matrix_world.translation.z))
    else:
        anchor = subject.matrix_world.translation.copy()
    anchor = _clamp_anchor_to_terrain(scene, anchor, margin=80.0)
    return anchor, _ground_z(scene, anchor.x, anchor.y)


def add_forest_water_surface(scene):
    """Create a shallow, wavy water surface instead of a cuboid lake proxy."""
    anchor, ground = _forest_anchor(scene)
    material = bpy.data.materials.get("ForestWaterSurface") or bpy.data.materials.new("ForestWaterSurface")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = 0.26
    if shader.inputs.get("Metallic"):
        shader.inputs["Metallic"].default_value = 0.12
    if shader.inputs.get("Transmission Weight"):
        shader.inputs["Transmission Weight"].default_value = 0.0
    texcoord = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.2
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.55
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.008, 0.08, 0.22, 1.0)
    ramp.color_ramp.elements[1].color = (0.04, 0.32, 0.68, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.35
    links.new(texcoord.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    width, depth = 52.0, 34.0
    radial_steps, angular_steps = 12, 64
    # Keep the presentation water just above the sampled terrain surface so
    # it reads as a lake rather than a floating blue overlay.
    water_level = ground + 0.25
    verts = [(anchor.x, anchor.y, water_level)]
    for ring in range(1, radial_steps + 1):
        radius = ring / radial_steps
        for index in range(angular_steps):
            angle = 2.0 * math.pi * index / angular_steps
            x = (width * 0.5) * radius * math.cos(angle)
            y = (depth * 0.5) * radius * math.sin(angle)
            ripple = 0.12 * math.sin(x * 0.45) * math.cos(y * 0.38)
            verts.append((anchor.x + x, anchor.y + y, water_level + ripple))
    faces = []
    for index in range(angular_steps):
        faces.append((0, 1 + index, 1 + (index + 1) % angular_steps))
    for ring in range(1, radial_steps):
        start = 1 + (ring - 1) * angular_steps
        next_start = start + angular_steps
        for index in range(angular_steps):
            a = start + index
            b = start + (index + 1) % angular_steps
            c = next_start + (index + 1) % angular_steps
            d = next_start + index
            faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new("ForestWaterSurfaceMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    water = bpy.data.objects.new("ForestWaterSurface", mesh)
    scene.collection.objects.link(water)
    water.data.materials.append(material)
    for poly in water.data.polygons:
        poly.use_smooth = True
    return {"enabled": True, "type": "procedural_wavy_water_surface", "anchor": [float(anchor.x), float(anchor.y), float(ground)], "dimensions": [width, depth]}


def prepare_forest_terrain(scene):
    terrain = next((obj for obj in scene.objects if obj.type == "MESH" and "terrain" in obj.name.lower()), None)
    if terrain is None:
        return {"smooth": False}
    for poly in terrain.data.polygons:
        poly.use_smooth = True
    material = bpy.data.materials.get("ForestTerrainMaterial") or bpy.data.materials.new("ForestTerrainMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.012
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.7
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.025, 0.075, 0.018, 1.0)
    ramp.color_ramp.elements[1].color = (0.30, 0.52, 0.14, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    height_map = nodes.new("ShaderNodeMapRange")
    height_ramp = nodes.new("ShaderNodeValToRGB")
    heights = [float(vertex.co.z) for vertex in terrain.data.vertices]
    height_map.inputs["From Min"].default_value = min(heights)
    height_map.inputs["From Max"].default_value = max(heights)
    height_map.inputs["To Min"].default_value = 0.0
    height_map.inputs["To Max"].default_value = 1.0
    height_map.clamp = True
    height_ramp.color_ramp.elements[0].position = 0.0
    height_ramp.color_ramp.elements[0].color = (0.012, 0.028, 0.018, 1.0)
    height_ramp.color_ramp.elements[1].position = 1.0
    height_ramp.color_ramp.elements[1].color = (0.58, 0.30, 0.07, 1.0)
    height_ramp.color_ramp.elements.new(0.30).color = (0.025, 0.16, 0.035, 1.0)
    height_ramp.color_ramp.elements.new(0.62).color = (0.32, 0.40, 0.06, 1.0)
    links.new(geometry.outputs["Position"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], height_map.inputs["Value"])
    links.new(height_map.outputs["Result"], height_ramp.inputs["Fac"])
    links.new(height_ramp.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = 0.96
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.32
    bump.inputs["Distance"].default_value = 2.5
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 24.0
    ao_mix = nodes.new("ShaderNodeMixRGB")
    ao_mix.blend_type = "MULTIPLY"
    ao_mix.inputs["Fac"].default_value = 0.35
    links.new(height_ramp.outputs["Color"], ao_mix.inputs[1])
    links.new(ao.outputs["Color"], ao_mix.inputs[2])
    links.new(ao_mix.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    terrain.data.materials.clear()
    terrain.data.materials.append(material)
    return {"smooth": True, "subdivision_levels": 0, "material": "ForestTerrainMaterial"}


def configure_directional_lighting(scene):
    """Set explicit key/fill lights and enable available shadow controls."""
    for name in ("OverviewSun", "OverviewFill"):
        old = bpy.data.objects.get(name)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)

    sun_data = bpy.data.lights.new("OverviewSun", "SUN")
    sun_data.energy = 4.0
    sun_data.angle = math.radians(7.0)
    if hasattr(sun_data, "use_shadow"):
        sun_data.use_shadow = True
    if hasattr(sun_data, "use_contact_shadow"):
        sun_data.use_contact_shadow = True
    sun = bpy.data.objects.new("OverviewSun", sun_data)
    scene.collection.objects.link(sun)
    # A low, lateral key creates readable slope shading across the terrain.
    sun.rotation_euler = (math.radians(38), math.radians(-28), math.radians(-42))

    fill_data = bpy.data.lights.new("OverviewFill", "AREA")
    fill_data.energy = 80.0
    fill_data.shape = "DISK"
    fill_data.size = 500.0
    fill = bpy.data.objects.new("OverviewFill", fill_data)
    scene.collection.objects.link(fill)
    fill.location = (0.0, 0.0, 260.0)
    fill.rotation_euler = (Vector((0.0, 0.0, 0.0)) - fill.location).to_track_quat("-Z", "Y").to_euler()

    for obj in scene.objects:
        if hasattr(obj, "visible_shadow"):
            obj.visible_shadow = True
    light_settings = getattr(scene.world, "light_settings", None) if scene.world else None
    if light_settings is not None and hasattr(light_settings, "use_ambient_occlusion"):
        light_settings.use_ambient_occlusion = True


def add_presentation_proxy(scene, run_name=None):
    """Add a clearly marked low-poly subject when the generated mesh is degenerate."""
    candidates = [obj for obj in scene.objects if obj.type == "MESH" and obj.name.lower() != "terrain"]
    if not candidates:
        return {"enabled": False}
    if _scene_has_tokens(scene, ("castle",)):
        subject = next((o for o in candidates if "castle_castle" in o.name.lower()), max(candidates, key=lambda o: o.dimensions.length))
        anchor = _clamp_anchor_to_terrain(scene, subject.matrix_world.translation.copy())
        subject.hide_render = True
        ground = _ground_z(scene, anchor.x, anchor.y)
        stone = make_material("Presentation_CastleStone", (0.42, 0.18, 0.06), 0.7)
        roof = make_material("Presentation_CastleRoof", (0.16, 0.025, 0.015), 0.75)
        path = make_material("Presentation_CastlePath", (0.28, 0.16, 0.06), 0.9)
        _proxy_cube("Presentation_Castle_Courtyard", (anchor.x, anchor.y, ground + 0.8), (34.0, 28.0, 1.6), path)
        _proxy_cube("Presentation_Castle_Keep", (anchor.x, anchor.y, ground + 9.0), (12.0, 12.0, 16.0), stone)
        _proxy_cone("Presentation_Castle_KeepRoof", (anchor.x, anchor.y, ground + 19.0), 9.0, 7.0, roof)
        for dx, dy in ((-12, -9), (12, -9), (-12, 9), (12, 9)):
            _proxy_cube("Presentation_Castle_Tower", (anchor.x + dx, anchor.y + dy, ground + 8.0), (5.0, 5.0, 15.0), stone)
            _proxy_cone("Presentation_Castle_TowerRoof", (anchor.x + dx, anchor.y + dy, ground + 18.0), 4.0, 5.0, roof)
        return {"enabled": True, "type": "castle_low_poly_proxy", "anchor": [float(anchor.x), float(anchor.y), float(ground)]}
    if _scene_has_tokens(scene, ("desert", "oasis", "dune")):
        subject = max(candidates, key=lambda o: o.dimensions.length)
        anchor = _clamp_anchor_to_terrain(scene, subject.matrix_world.translation.copy())
        subject.hide_render = True
        ground = _ground_z(scene, anchor.x, anchor.y)
        wall = make_material("Presentation_DesertWall", (0.78, 0.38, 0.08), 0.85)
        roof = make_material("Presentation_DesertRoof", (0.24, 0.045, 0.012), 0.9)
        _proxy_cube("Presentation_Desert_Palace", (anchor.x, anchor.y, ground + 8.0), (40.0, 30.0, 14.0), wall)
        _proxy_cone("Presentation_Desert_Dome", (anchor.x, anchor.y, ground + 18.0), 14.0, 10.0, roof, vertices=16)
        for dx, dy in ((-26, -18), (26, -18), (-26, 18), (26, 18)):
            _proxy_cube("Presentation_Desert_Block", (anchor.x + dx, anchor.y + dy, ground + 5.0), (16.0, 14.0, 10.0), wall)
        return {"enabled": True, "type": "desert_city_low_poly_proxy", "anchor": [float(anchor.x), float(anchor.y), float(ground)]}
    # Forest has no castle region in the live placement response.  Keep its
    # real cabin/tree assets and replace only the old cuboid lake proxy with a
    # dedicated water surface created below.
    return {"enabled": False, "reason": "forest_scene_has_no_castle_proxy"}


def settle_assets_on_terrain(scene):
    terrain = next((obj for obj in scene.objects if obj.type == "MESH" and "terrain" in obj.name.lower()), None)
    if terrain is None:
        return 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    terrain_bvh = BVHTree.FromObject(terrain, depsgraph)
    moved = 0
    for obj in scene.objects:
        if obj.type != "MESH" or obj == terrain or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
        min_z = min(point.z for point in points)
        center = sum(points, Vector()) / len(points)
        hit, _, _, _ = terrain_bvh.ray_cast(Vector((center.x, center.y, 10000.0)), Vector((0.0, 0.0, -1.0)), 20000.0)
        if hit is None:
            continue
        obj.location.z += float(hit.z - min_z + 0.5)
        moved += 1
    return moved


def asset_focus_views(scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    centers = []
    all_points = []
    objects = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render or obj.name.lower() == "terrain":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
        center = sum(points, Vector()) / len(points)
        centers.append(center)
        all_points.extend(points)
        objects.append((obj, center, points))
    if not centers:
        raise RuntimeError("scene contains no non-terrain asset meshes")
    # The world can span kilometres while the generated hero objects are only
    # metres apart.  Spatially largest clusters are often empty terrain/trees,
    # so prioritize the scene's semantic subject for the first camera target.
    priority = ("castle", "palace", "building", "house", "cabin", "village", "oasis", "lake", "water")
    anchors = [item for item in objects if any(token in item[0].name.lower() for token in priority)]
    anchor = max(anchors or objects, key=lambda item: max((p - item[1]).length for p in item[2]))
    anchor_center = anchor[1]
    # Keep the hero frame local.  A 70 m radius still includes the generated
    # castle/buildings and their surrounding vegetation while excluding the
    # far-away region rows that would flatten the composition.
    nearby = [item for item in objects if (item[1] - anchor_center).length <= 70.0]
    if not nearby:
        nearby = [anchor]
    # Keep the subject and its surrounding trees/buildings together in every
    # view.  The four views form a short orbit, rather than jumping between
    # unrelated world corners.
    local_points = [p for _, _, points in nearby for p in points]
    local_lo = Vector((min(p.x for p in local_points), min(p.y for p in local_points), min(p.z for p in local_points)))
    local_hi = Vector((max(p.x for p in local_points), max(p.y for p in local_points), max(p.z for p in local_points)))
    local_center = (local_lo + local_hi) * 0.5
    horizontal_radius = max((local_hi.x - local_lo.x), (local_hi.y - local_lo.y), 12.0) * 0.5
    vertical_radius = max(local_hi.z - local_lo.z, 8.0)
    views = []
    for offset in (Vector((0.0, 0.0, 0.0)), Vector((10.0, 8.0, 0.0)), Vector((-8.0, 10.0, 0.0)), Vector((6.0, -10.0, 0.0))):
        target = local_center + offset
        target.z -= 10.0
        views.append((target, max(95.0, horizontal_radius * 2.0 + vertical_radius * 1.3)))
    return views


def configure_scene(scene, minimum, maximum, run_name=None):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.render.fps_base = 1.0
    scene.frame_start, scene.frame_end = 1, 288
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.8

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.12, 0.16, 0.22, 1.0)
    background.inputs["Strength"].default_value = 0.28
    configure_directional_lighting(scene)

    prepare_materials(scene, run_name)
    forest = _scene_has_tokens(scene, ("forest", "tree", "vegetation"))
    terrain_presentation = prepare_forest_terrain(scene) if forest else None
    if forest:
        scene.view_settings.exposure = 1.1
    views = asset_focus_views(scene)
    camera_data = bpy.data.cameras.get("OverviewCamera") or bpy.data.cameras.new("OverviewCamera")
    camera = bpy.data.objects.get("OverviewCamera") or bpy.data.objects.new("OverviewCamera", camera_data)
    if camera.name not in scene.objects:
        scene.collection.objects.link(camera)
    # Overview framing should remain wide on kilometre-scale terrain.
    camera.data.lens = 32
    max_depth = max(distance for _, distance in views)
    camera.data.clip_start = max(0.01, max_depth * 0.001)
    camera.data.clip_end = 10000.0
    scene.camera = camera

    # Blender is natively Z-up. Forest uses an orbit in the X-Y ground plane.
    orbit = None
    if forest:
        target, depth = views[0]
        radius = max(145.0, depth * 0.90)
        orbit_height = max(72.0, radius * 0.48)
        orbit_end_frame = 144
        orbit_end_angle = 1.55 * math.pi
        frames = list(range(1, 289, 12))
        if frames[-1] != 288:
            frames.append(288)
        camera.rotation_mode = "QUATERNION"
        camera.animation_data_clear()
        orbit_end_position = None
        for frame in frames:
            scene.frame_set(frame)
            if frame <= orbit_end_frame:
                progress = (frame - 1) / float(orbit_end_frame - 1)
                angle = orbit_end_angle * progress
                # Elevated oblique orbit with a slow altitude change and a
                # small target drift that keeps salient regions in view.
                height = orbit_height - 20.0 * progress
                target_path = target + Vector((8.0 * progress, -6.0 * progress, -2.0 * progress))
                position = Vector((target_path.x + radius * math.cos(angle), target_path.y + radius * math.sin(angle), target_path.z + height))
                orbit_end_position = position.copy()
            else:
                progress = (frame - orbit_end_frame) / float(288 - orbit_end_frame)
                start = orbit_end_position
                forward = Vector((target.x - start.x, target.y - start.y, 0.0)).normalized()
                # Forward fly-through: approach and pass the salient region
                # while lowering the camera and easing the look target.
                end = target + forward * 22.0 + Vector((0.0, 0.0, 28.0))
                position = start.lerp(end, progress)
                target_path = target + Vector((8.0, -6.0, -2.0)) * progress
            camera.location = position
            camera.rotation_quaternion = (Vector(target_path) - camera.location).to_track_quat("-Z", "Y")
            camera.keyframe_insert("location", frame=frame)
            camera.keyframe_insert("rotation_quaternion", frame=frame)
        if camera.animation_data and camera.animation_data.action:
            for curve in camera.animation_data.action.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = "LINEAR"
        orbit = {
            "type": "elevated_oblique_orbit_then_forward_flythrough",
            "target": list(target),
            "orbit_frames": [1, orbit_end_frame],
            "orbit_degrees": math.degrees(orbit_end_angle),
            "orbit_radius": radius,
            "orbit_height_start": orbit_height,
            "orbit_height_end": orbit_height - 20.0,
            "flythrough_frames": [orbit_end_frame, 288],
            "flythrough_end_height": 28.0,
            "plane": "X-Y",
            "start_direction": "+X",
            "degrees": 360,
        }
        return camera, views, minimum, maximum, orbit, terrain_presentation

    # Non-forest runs retain their existing framing path.
    elevation = 0.48 if _scene_has_tokens(scene, ("desert", "forest", "tree")) else 0.08
    directions = [
        Vector((0.85, -0.95, elevation)),
        Vector((-0.95, -0.75, elevation)),
        Vector((-0.8, 0.95, elevation)),
        Vector((0.9, 0.8, elevation)),
        Vector((0.85, -0.95, elevation)),
    ]
    frames = [1, 72, 144, 216, 288]
    for frame, direction, (target, depth) in zip(frames, directions, views + [views[0]]):
        scene.frame_set(frame)
        point_camera(camera, target, target + direction.normalized() * depth)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
    return camera, views, minimum, maximum, orbit, terrain_presentation


def main():
    args = parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str((run / "scene.blend").resolve()))
    scene = bpy.context.scene
    minimum, maximum = scene_bounds(scene)
    settled_assets = settle_assets_on_terrain(scene)
    presentation_proxy = add_presentation_proxy(scene)
    water_surface = add_forest_water_surface(scene) if _scene_has_tokens(scene, ("lake", "water", "river")) else None
    # Bounds and focus views must be computed after the coordinate correction.
    minimum, maximum = scene_bounds(scene)
    camera, views, minimum, maximum, orbit, terrain_presentation = configure_scene(scene, minimum, maximum)

    scene.frame_set(1)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str((output / "preview.png").resolve())
    bpy.ops.render.render(write_still=True)

    diagnostics = output / "diagnostic_views"
    diagnostics.mkdir(exist_ok=True)
    for index, frame in enumerate((1, 72, 144, 216)):
        scene.frame_set(frame)
        scene.render.filepath = str((diagnostics / f"view_{index:02d}.png").resolve())
        bpy.ops.render.render(write_still=True)

    if args.video:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.ffmpeg.audio_codec = "NONE"
        scene.render.filepath = str((output / "walkthrough.mp4").resolve())
        bpy.ops.render.render(animation=True)

    metadata = {
        "camera": "OverviewCamera",
        "coordinate_system": "Blender native Z-up after glTF import",
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "focus_views": [{"target": list(target), "distance": distance} for target, distance in views],
        "orbit": orbit,
        "settled_assets": settled_assets,
        "presentation_proxy": presentation_proxy,
        "water_surface": water_surface,
        "terrain_presentation": terrain_presentation,
        "resolution": [1280, 720],
        "fps": 24,
        "frames": 288,
        "duration_seconds": 12,
    }
    (output / "overview_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str((output / "scene.blend").resolve()))


if __name__ == "__main__":
    main()
