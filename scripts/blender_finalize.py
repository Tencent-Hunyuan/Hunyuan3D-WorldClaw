"""Blender 4.2 finalizer for worldclaw_oss live runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import struct
import sys
from pathlib import Path

import bpy
import mathutils
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("diagnostic", "full"),
        default=os.getenv("WORLDCLAW_RENDER_PROFILE", "diagnostic"),
        help="diagnostic renders four small QA frames; full also renders the walkthrough",
    )
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def deduplicate_mesh_data(scene):
    """Link objects with identical mesh/material data to one Blender Mesh."""
    signatures = {}
    reused = 0
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        mesh = obj.data
        digest = hashlib.blake2b(digest_size=16)
        for vertex in mesh.vertices:
            digest.update(struct.pack("<3f", *vertex.co))
        for polygon in mesh.polygons:
            digest.update(struct.pack("<I", polygon.material_index))
            digest.update(struct.pack("<I", len(polygon.vertices)))
            for index in polygon.vertices:
                digest.update(struct.pack("<I", index))
        for material in mesh.materials:
            digest.update((material.name if material else "").encode("utf-8"))
        key = digest.digest()
        canonical = signatures.get(key)
        if canonical is None:
            signatures[key] = obj
        elif canonical.data.as_pointer() != mesh.as_pointer():
            obj.data = canonical.data
            reused += 1
    return {"unique_mesh_datablocks": len(signatures), "reused_mesh_objects": reused}


def add_lod_modifiers(scene):
    """Attach conservative render-only LOD modifiers to expensive props."""
    added = 0
    source_polygons = 0
    estimated_polygons = 0
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render or "terrain" in obj.name.lower():
            continue
        polygons = len(obj.data.polygons)
        source_polygons += polygons
        if polygons < int(os.getenv("WORLDCLAW_LOD_MIN_POLYGONS", "12000")):
            estimated_polygons += polygons
            continue
        lowered = obj.name.lower()
        if any(token in lowered for token in ("tree", "pine", "forest", "fern", "reed", "grass", "vegetation")):
            ratio = 0.22
        elif any(token in lowered for token in ("rock", "stone", "pebble")):
            ratio = 0.35
        else:
            ratio = 0.55
        if obj.modifiers.get("WorldClawLOD") is None:
            modifier = obj.modifiers.new("WorldClawLOD", "DECIMATE")
            modifier.ratio = ratio
            added += 1
        estimated_polygons += max(4, int(polygons * ratio))
    return {
        "lod_modifiers": added,
        "source_polygons": source_polygons,
        "estimated_render_polygons": estimated_polygons,
    }


def ensure_terrain_material(terrain):
    """Attach Terrain's glTF COLOR_0 attribute to a Principled shader."""
    if terrain is None or terrain.type != "MESH":
        return {"available": False, "reason": "terrain_missing"}
    material = bpy.data.materials.get("WorldClawTerrainMaterial")
    if material is None:
        material = bpy.data.materials.new("WorldClawTerrainMaterial")
    material.use_nodes = True
    # Terrain is authored as a single-sided exterior surface. Keep backface
    # culling enabled so an orientation error cannot be hidden by double-sided
    # shading during normal validation.
    material.use_backface_culling = True
    material.diffuse_color = (0.24, 0.42, 0.14, 1.0)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.name = "Principled BSDF"
    shader.inputs["Roughness"].default_value = 0.92
    shader.inputs["Metallic"].default_value = 0.0
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    attributes = getattr(terrain.data, "color_attributes", None)
    color_layer = None
    if attributes:
        color_layer = next(
            (item for item in attributes if item.domain in {"CORNER", "POINT"}
             and item.data_type in {"BYTE_COLOR", "FLOAT_COLOR"}),
            None,
        )
    node = None
    source = "fallback_rgb"
    layer_name = None
    if color_layer is not None:
        layer_name = color_layer.name
        try:
            node = nodes.new("ShaderNodeVertexColor")
            node.layer_name = layer_name
            source = "vertex_color"
        except (RuntimeError, TypeError):
            # Blender builds without ShaderNodeVertexColor can still resolve
            # the imported attribute through the generic Attribute node.
            node = nodes.new("ShaderNodeAttribute")
            node.attribute_name = layer_name
            source = "attribute_color"
        node.name = "WorldClawTerrainColorAttribute"
        links.new(node.outputs.get("Color"), shader.inputs["Base Color"])
    else:
        node = nodes.new("ShaderNodeRGB")
        node.name = "WorldClawTerrainFallbackColor"
        node.outputs[0].default_value = (0.24, 0.42, 0.14, 1.0)
        links.new(node.outputs[0], shader.inputs["Base Color"])

    if not terrain.data.materials:
        terrain.data.materials.append(material)
    else:
        terrain.data.materials[0] = material
        while len(terrain.data.materials) > 1:
            terrain.data.materials.pop(index=len(terrain.data.materials) - 1)
    material["WorldClawTerrainMaterial"] = True
    material["WorldClawTerrainColorSource"] = source
    material["WorldClawTerrainColorLayer"] = layer_name or ""
    return {
        "available": True,
        "material": material.name,
        "color_source": source,
        "color_layer": layer_name,
        "base_color_connected": True,
    }


def unify_material_look(scene):
    """Make imported GLB PBR inputs visible while preserving authored links."""
    changed = 0
    linked_preserved = 0
    categories = {}
    base_color_links = {"texture": 0, "vertex_color": 0, "other": 0, "unlinked": 0}
    pbr_links = {"normal": 0, "metallic": 0, "roughness": 0}
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        lowered = " ".join(
            [obj.name.lower()]
            + [material.name.lower() for material in obj.data.materials if material is not None]
        )
        if any(token in lowered for token in ("water", "lake", "river")):
            category, roughness, metallic = "water", 0.24, 0.08
        elif any(token in lowered for token in ("tree", "fern", "reed", "grass", "vegetation")):
            category, roughness, metallic = "foliage", 0.86, 0.0
        elif any(token in lowered for token in ("rock", "stone")):
            category, roughness, metallic = "rock", 0.72, 0.0
        elif any(token in lowered for token in ("wood", "cabin", "log")):
            category, roughness, metallic = "wood", 0.78, 0.0
        elif any(token in lowered for token in ("trail", "road", "soil", "path")):
            category, roughness, metallic = "trail_soil", 0.9, 0.0
        elif "terrain" in lowered:
            category, roughness, metallic = "terrain", 0.92, 0.0
        else:
            category, roughness, metallic = "default", 0.68, 0.0
        categories[category] = categories.get(category, 0) + 1
        for material in obj.data.materials:
            if material is None or not material.use_nodes:
                continue
            shader = material.node_tree.nodes.get("Principled BSDF")
            if shader is None:
                continue
            base_socket = shader.inputs.get("Base Color")
            if base_socket is not None and base_socket.is_linked:
                source_socket = base_socket.links[0].from_socket
                source_node = source_socket.node
                source_type = source_node.bl_idname.lower()
                if "teximage" in source_type:
                    base_color_links["texture"] += 1
                elif "vertexcolor" in source_type or "attribute" in source_type:
                    base_color_links["vertex_color"] += 1
                else:
                    base_color_links["other"] += 1
            elif base_socket is not None:
                base_color_links["unlinked"] += 1
            if base_socket is not None and base_socket.is_linked and not material.get("WorldClawTinted") and not material.get("WorldClawTerrainMaterial"):
                # Preserve the authored texture connection while applying a
                # restrained category tint for otherwise-white GLB materials.
                # This is a reversible lookdev node, not a texture replacement.
                tint_values = {
                    "terrain": (0.34, 0.48, 0.24, 1.0),
                    "water": (0.12, 0.42, 0.58, 1.0),
                    "foliage": (0.16, 0.38, 0.12, 1.0),
                    "rock": (0.48, 0.50, 0.48, 1.0),
                    "wood": (0.52, 0.25, 0.10, 1.0),
                    "trail_soil": (0.42, 0.24, 0.10, 1.0),
                    "default": (0.62, 0.62, 0.62, 1.0),
                }
                links = material.node_tree.links
                tint = material.node_tree.nodes.new("ShaderNodeRGB")
                tint.name = "WorldClawCategoryTint"
                tint.outputs[0].default_value = tint_values[category]
                mix = material.node_tree.nodes.new("ShaderNodeMixRGB")
                mix.name = "WorldClawTextureTint"
                mix.blend_type = "MULTIPLY"
                # A mostly-white generated texture otherwise renders as a
                # white clay proxy. Keep texture variation, but guarantee a
                # category signal in the final look.
                mix.inputs[0].default_value = 0.82
                links.remove(base_socket.links[0])
                links.new(source_socket, mix.inputs[1])
                links.new(tint.outputs[0], mix.inputs[2])
                links.new(mix.outputs[0], base_socket)
                material["WorldClawTinted"] = True
                changed += 1
            elif base_socket is not None and not base_socket.is_linked:
                tint_values = {
                    "terrain": (0.24, 0.42, 0.14, 1.0),
                    "water": (0.04, 0.28, 0.52, 1.0),
                    "foliage": (0.08, 0.30, 0.07, 1.0),
                    "rock": (0.38, 0.39, 0.36, 1.0),
                    "wood": (0.42, 0.16, 0.045, 1.0),
                    "trail_soil": (0.30, 0.16, 0.055, 1.0),
                    "default": (0.50, 0.50, 0.50, 1.0),
                }
                current = tuple(float(value) for value in base_socket.default_value)
                # Preserve authored colors unless they are effectively white;
                # generated GLBs commonly use white factors with no texture.
                if max(current[:3]) > 0.82 and min(current[:3]) > 0.72:
                    base_socket.default_value = tint_values[category]
                    material.diffuse_color = tint_values[category]
                    changed += 1
            roughness_socket = shader.inputs.get("Roughness")
            metallic_socket = shader.inputs.get("Metallic")
            if roughness_socket is not None and not roughness_socket.is_linked:
                shader.inputs["Roughness"].default_value = roughness
            elif roughness_socket is not None:
                linked_preserved += 1
                pbr_links["roughness"] += 1
            if metallic_socket is not None and not metallic_socket.is_linked:
                shader.inputs["Metallic"].default_value = metallic
            elif metallic_socket is not None:
                linked_preserved += 1
                pbr_links["metallic"] += 1
            normal_socket = shader.inputs.get("Normal")
            if normal_socket is not None and normal_socket.is_linked:
                pbr_links["normal"] += 1
            changed += 1
    return {
        "materials_tuned": changed,
        "linked_pbr_preserved": linked_preserved,
        "categories": categories,
        "base_color_links": base_color_links,
        "pbr_links": pbr_links,
    }


def scene_bounds(scene):
    """Return world-space bounds for all renderable mesh objects."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    if not points:
        raise RuntimeError("scene contains no renderable mesh objects")
    return (
        Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points))),
        Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points))),
    )


def quarantine_outlier_meshes(scene, terrain):
    """Hide render-only mesh outliers whose footprint is outside Terrain.

    Keep these objects in the blend for debugging, but prevent malformed
    linear assets from dominating the cinematic camera and global bounds.
    """
    terrain_bounds = object_world_bounds(terrain)
    if terrain_bounds is None:
        return []
    minimum, maximum = terrain_bounds
    margin_x = max((maximum.x - minimum.x) * 0.08, 10.0)
    margin_y = max((maximum.y - minimum.y) * 0.08, 10.0)
    hidden = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj == terrain:
            continue
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        if not corners:
            continue
        center = sum(corners, Vector()) / len(corners)
        if (
            center.x < minimum.x - margin_x or center.x > maximum.x + margin_x
            or center.y < minimum.y - margin_y or center.y > maximum.y + margin_y
        ):
            obj.hide_render = True
            hidden.append({"name": obj.name, "center": list(center), "reason": "outside_terrain_footprint"})
    return hidden


def object_world_bounds(obj, depsgraph=None):
    """Compute evaluated world-space bounds from mesh vertices, not a local box."""
    if obj is None or obj.type != "MESH":
        return None
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()
    if not points:
        return None
    return (
        Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points))),
        Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points))),
    )


def bounds_metadata(bounds):
    minimum, maximum = bounds
    return {
        "x_min": float(minimum.x), "x_max": float(maximum.x),
        "y_min": float(minimum.y), "y_max": float(maximum.y),
        "z_min": float(minimum.z), "z_max": float(maximum.z),
    }


def terrain_bvh(scene):
    terrain = next(
        (obj for obj in scene.objects if obj.type == "MESH" and "terrain" in obj.name.lower()),
        None,
    )
    if terrain is None:
        return None, None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return terrain, BVHTree.FromObject(terrain, depsgraph)


def normalize_imported_scene_z_up(scene, imported):
    """Keep Blender's glTF-imported scene in its native Z-up coordinates.

    The glTF exporter declares Y-up, but Blender's glTF importer converts that
    basis to Blender's native Z-up during import. Applying another +90 degree X
    rotation here would turn a horizontal Terrain into a vertical wall and
    invalidate all grounding, camera, and structural placement queries.
    """
    root = bpy.data.objects.get("WorldClaw_ZUpRoot")
    if root is None:
        root = bpy.data.objects.new("WorldClaw_ZUpRoot", None)
        scene.collection.objects.link(root)
    root.rotation_euler = (0.0, 0.0, 0.0)
    root.rotation_mode = "XYZ"
    for obj in imported:
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = root.matrix_world @ world
    return {
        "applied": False,
        "source_up_axis": "Y (glTF; importer-converted)",
        "blender_up_axis": "Z",
        "root_object": root.name,
        "rotation_x_degrees": 0.0,
    }


def terrain_height(terrain, bvh, x, y, fallback, terrain_bounds=None):
    """Sample Blender native Z-up terrain height at world X/Y coordinates.

    BVHTree.FromObject uses the terrain object's local coordinates.  Transforming
    the ray into that space avoids incorrect samples when imported nodes carry a
    transform, then the hit point is transformed back to world space.
    """
    if terrain is None or bvh is None:
        return fallback
    terrain_bounds = terrain_bounds or object_world_bounds(terrain)
    if terrain_bounds is None:
        return fallback
    minimum, maximum = terrain_bounds
    # Keep samples on the finite terrain footprint.  Orbit points may be beyond
    # an edge, but the edge height is still a better clearance reference than a
    # hard-coded scene minimum.
    sample_x = min(max(float(x), minimum.x), maximum.x)
    sample_y = min(max(float(y), minimum.y), maximum.y)
    inverse = terrain.matrix_world.inverted()
    world_origin = Vector((sample_x, sample_y, maximum.z + max(10000.0, maximum.z - minimum.z + 1.0)))
    local_origin = inverse @ world_origin
    local_direction = (inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    hit, _, _, _ = bvh.ray_cast(local_origin, local_direction, 20000.0)
    if hit is None:
        return fallback
    return float((terrain.matrix_world @ hit).z)


def settle_assets_on_terrain(scene, terrain, bvh, clearance=0.08, protected_names=None):
    """Translate visible assets upward until their footprint clears Terrain.

    External GLBs can carry inconsistent origins or authoring axes.  This
    final world-space pass is intentionally translation-only and leaves hidden
    outliers untouched, so camera planning and rendering cannot be blocked by
    a buried asset while the source placement metadata remains inspectable.
    """
    protected_names = {str(name) for name in (protected_names or ())}
    if terrain is None or bvh is None:
        return {"available": False, "moved": 0, "penetrating_before": 0, "max_shift_m": 0.0}
    terrain_bounds = object_world_bounds(terrain)
    if terrain_bounds is None:
        return {"available": False, "moved": 0, "penetrating_before": 0, "max_shift_m": 0.0}
    minimum, maximum = terrain_bounds
    moved = 0
    penetrating_before = 0
    shifts = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj == terrain or obj.hide_render:
            continue
        # Structural meshes already contain their terrain-integrated height
        # samples.  Grounding their bounding boxes as ordinary assets lifts
        # lake/trail surfaces above the surrounding terrain.
        if obj.name in protected_names:
            continue
        # Procedural strips are already sampled against the source heightfield.
        # Their intentionally large footprint spans multiple terrain heights;
        # corner-based grounding would incorrectly lift them hundreds of metres.
        if any(token in obj.name.lower() for token in ("winding_trail", "winding_trails")):
            continue
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        if not corners:
            continue
        obj_minimum = Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners)))
        obj_maximum = Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners)))
        center = (obj_minimum + obj_maximum) * 0.5
        # Objects fully outside the Terrain footprint are handled by the
        # quarantine pass; do not clamp them onto the edge here.
        if (obj_maximum.x < minimum.x or obj_minimum.x > maximum.x
                or obj_maximum.y < minimum.y or obj_minimum.y > maximum.y):
            continue
        samples = (
            (center.x, center.y),
            (obj_minimum.x, obj_minimum.y), (obj_minimum.x, obj_maximum.y),
            (obj_maximum.x, obj_minimum.y), (obj_maximum.x, obj_maximum.y),
        )
        ground_values = [
            terrain_height(terrain, bvh, x, y, None, terrain_bounds=terrain_bounds)
            for x, y in samples
        ]
        ground_values = [value for value in ground_values if value is not None]
        if not ground_values:
            continue
        required_bottom = max(ground_values) + float(clearance)
        shift = required_bottom - obj_minimum.z
        if shift <= 0.0:
            continue
        penetrating_before += 1
        world_matrix = obj.matrix_world.copy()
        world_matrix.translation.z += shift
        obj.matrix_world = world_matrix
        moved += 1
        shifts.append(float(shift))
    return {
        "available": True,
        "moved": moved,
        "penetrating_before": penetrating_before,
        "max_shift_m": max(shifts, default=0.0),
        "mean_shift_m": (sum(shifts) / len(shifts)) if shifts else 0.0,
    }


def ensure_structural_materials(scene, structural_features):
    """Assign explicit readable materials to route-native structural meshes."""
    assigned = 0
    categories = {}
    palette = {
        "lake": ((0.025, 0.28, 0.46, 1.0), 0.20, 0.05),
        "water": ((0.025, 0.28, 0.46, 1.0), 0.20, 0.05),
        "river": ((0.025, 0.28, 0.46, 1.0), 0.20, 0.05),
        "stream": ((0.025, 0.28, 0.46, 1.0), 0.20, 0.05),
        "trail": ((0.30, 0.14, 0.045, 1.0), 0.88, 0.0),
        "road": ((0.26, 0.22, 0.13, 1.0), 0.90, 0.0),
    }
    for feature in structural_features or []:
        feature_id = str(feature.get("feature_id", ""))
        category = str(feature.get("category", "")).lower()
        obj = scene.objects.get(feature_id)
        if obj is None or obj.type != "MESH" or category not in palette:
            continue
        color, roughness, metallic = palette[category]
        material_name = f"WorldClawStructural{category.title()}"
        material = bpy.data.materials.get(material_name)
        if material is None:
            material = bpy.data.materials.new(material_name)
        material.use_nodes = True
        material.diffuse_color = color
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.name = "Principled BSDF"
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
        transmission = shader.inputs.get("Transmission Weight")
        if transmission is not None:
            transmission.default_value = 0.08 if category in {"lake", "water", "river", "stream"} else 0.0
        links.new(shader.outputs["BSDF"], output.inputs["Surface"])
        material["WorldClawStructuralMaterial"] = True
        material["WorldClawStructuralCategory"] = category
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.material_index = 0
        assigned += 1
        categories[category] = categories.get(category, 0) + 1
    return {"structural_materials_assigned": assigned, "structural_material_categories": categories}


def _terrain_normal_stats(terrain):
    mesh = terrain.data
    mesh.update()
    normal_matrix = terrain.matrix_world.to_3x3().inverted().transposed()
    upward = downward = side = 0
    alignments = []
    for polygon in mesh.polygons:
        normal = (normal_matrix @ polygon.normal).normalized()
        alignment = normal.dot(Vector((0.0, 0.0, 1.0)))
        alignments.append(float(alignment))
        if alignment > 0.1:
            upward += 1
        elif alignment < -0.1:
            downward += 1
        else:
            side += 1
    count = upward + downward + side
    return {
        "face_count": count,
        "upward_faces": upward,
        "downward_faces": downward,
        "side_faces": side,
        "upward_ratio": float(upward / count) if count else 0.0,
        "downward_ratio": float(downward / count) if count else 0.0,
        "mean_normal_z": float(sum(alignments) / len(alignments)) if alignments else 0.0,
    }


def diagnose_terrain_normals(terrain):
    """Validate Terrain Face Orientation in Blender's native Z-up frame.

    Terrain winding is fixed at export time.  Finalization deliberately fails
    on an inverted or mixed mesh instead of silently flipping it or rendering
    it double-sided, so an invalid GLB cannot pass visual QA by masking the
    winding defect.
    """
    if terrain is None:
        return {"available": False}
    before = _terrain_normal_stats(terrain)
    after = _terrain_normal_stats(terrain)
    materials = []
    for material in terrain.data.materials:
        if material is None:
            continue
        was_culled = bool(getattr(material, "use_backface_culling", False))
        # Never disable culling to conceal a winding error.  Preserve the
        # authored material setting while recording it for the audit.
        materials.append({"name": material.name, "backface_culling_before": was_culled, "backface_culling_after": bool(getattr(material, "use_backface_culling", False))})
    return {
        "available": True,
        "object": terrain.name,
        "up_axis": "Z",
        "before": before,
        "after": after,
        "repaired": False,
        "passed": after["mean_normal_z"] > 0.0 and after["upward_ratio"] > 0.99 and after["downward_ratio"] == 0.0,
        "double_sided_diagnostic": False,
        "materials": materials,
    }


def smooth_keyframes(obj):
    action = obj.animation_data.action if obj.animation_data else None
    if action:
        for curve in action.fcurves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "BEZIER"
        for curve in action.fcurves:
            for keyframe in curve.keyframe_points:
                # AUTO_CLAMPED was added after Blender 4.2.  Keep the
                # clamped mode when available, with AUTO as the 4.2 fallback.
                for attribute in ("handle_left_type", "handle_right_type"):
                    try:
                        setattr(keyframe, attribute, "AUTO_CLAMPED")
                    except (TypeError, ValueError):
                        setattr(keyframe, attribute, "AUTO")


def catmull_rom(a, b, c, d, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2.0 * b) + (-a + c) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 + (-a + 3.0 * b - 3.0 * c + d) * t3)


def create_camera_spline(scene, positions, frames):
    """Create an inspectable Bezier spline matching the sampled camera flight."""
    curve_data = bpy.data.curves.get("WorldClawCameraSpline") or bpy.data.curves.new("WorldClawCameraSpline", "CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 16
    while curve_data.splines:
        curve_data.splines.remove(curve_data.splines[0])
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(positions) - 1)
    for point, location in zip(spline.bezier_points, positions):
        point.co = location
        # Clamped handles keep the dense terrain-safe samples from overshooting.
        # One sample per frame keeps the compatible vector fallback smooth.
        for attribute in ("handle_left_type", "handle_right_type"):
            try:
                setattr(point, attribute, "AUTO_CLAMPED")
            except (TypeError, ValueError):
                # Blender 4.2 has no AUTO_CLAMPED. VECTOR handles prevent the
                # fallback spline from overshooting the terrain-safe samples.
                setattr(point, attribute, "VECTOR")
    path = bpy.data.objects.get("WorldClawCameraSpline")
    if path is None:
        path = bpy.data.objects.new("WorldClawCameraSpline", curve_data)
        scene.collection.objects.link(path)
    else:
        path.data = curve_data
    path.hide_render = True
    path.hide_viewport = True
    return path


def load_semantic_pois(run):
    """Collect flexible POI hints from plan/assets/region metadata files."""
    roots = [run, run / "work"]
    names = ("scene_plan.json", "assets.json", "region_composition.json", "layout.json")
    found = []
    for root in roots:
        for name in names:
            path = root / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            # Scene plans store major region centers in plan space (typically
            # [0, world_size]) rather than as literal `position` lists.
            if isinstance(payload, dict) and isinstance(payload.get("regions"), list):
                size = payload.get("world_size_m") or [1000.0, 1000.0]
                try:
                    width, depth = float(size[0]), float(size[1])
                except (TypeError, ValueError, IndexError):
                    width, depth = 1000.0, 1000.0
                centers = [region.get("center") for region in payload["regions"] if isinstance(region, dict) and isinstance(region.get("center"), dict)]
                # The schema permits both centered coordinates and image-like
                # [0, world_size] coordinates. Do not subtract half the world
                # size from an already-centered plan.
                centered = any(
                    float(center.get(axis, 0.0)) < -1e-6
                    for center in centers for axis in ("x", "y")
                    if isinstance(center.get(axis), (int, float))
                )
                offset_x, offset_y = (0.0, 0.0) if centered else (-width * 0.5, -depth * 0.5)
                for region in payload["regions"]:
                    if not isinstance(region, dict) or not isinstance(region.get("center"), dict):
                        continue
                    center = region["center"]
                    try:
                        found.append({
                            "name": str(region.get("id") or region.get("function") or "region"),
                            "category": str(region.get("function") or region.get("id") or "region"),
                            "region_id": str(region.get("id") or ""),
                            "importance": float(region.get("importance", region.get("coverage", 0.5))),
                            "position": [float(center["x"]) + offset_x, float(center["y"]) + offset_y, 0.0],
                            "source": "scene_plan_region",
                            "camera_hint": str(region.get("camera_hint", "wide")),
                        })
                    except (TypeError, ValueError, KeyError):
                        continue
            stack = [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    point = item.get("position") or item.get("location") or item.get("center")
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        try:
                            found.append({
                                "name": str(item.get("name") or item.get("kind") or item.get("category") or "poi"),
                                "category": str(item.get("category") or item.get("type") or item.get("kind") or ""),
                                "region_id": str(item.get("region_id") or item.get("region") or ""),
                                "importance": float(item.get("importance", item.get("weight", 0.5))),
                                "position": [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0],
                            })
                        except (TypeError, ValueError):
                            pass
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
    # Recursive metadata often repeats the same region/object. Keeping the
    # highest-importance record makes POI selection deterministic and avoids
    # silently selecting a later fallback duplicate.
    unique = {}
    for item in found:
        key = (str(item.get("name", "")).lower(), str(item.get("region_id", "")).lower(), tuple(round(float(v), 3) for v in item.get("position", ())))
        previous = unique.get(key)
        if previous is None or float(item.get("importance", 0.5)) > float(previous.get("importance", 0.5)):
            unique[key] = item
    return list(unique.values())


def clamp_poi_to_terrain(point, terrain_minimum, terrain_maximum):
    return Vector((
        min(max(point.x, terrain_minimum.x), terrain_maximum.x),
        min(max(point.y, terrain_minimum.y), terrain_maximum.y),
        point.z,
    ))


def inspect_diagnostic_views(run, paths, camera_metadata, lighting_metadata):
    """Run inexpensive local VLM-style QA, with an optional external inspector."""
    defects = []
    views = []
    for path in paths:
        try:
            image = bpy.data.images.load(str(path), check_existing=False)
            pixels = image.pixels[:]
            bpy.data.images.remove(image)
            values = [0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2] for i in range(0, len(pixels), 4)]
            mean = sum(values) / max(len(values), 1)
            # A dark World background often has a nonzero linear value around
            # 0.03-0.09; count that as dark so mostly-empty shots fail QA.
            dark_fraction = sum(value < 0.18 for value in values) / max(len(values), 1)
            views.append({"path": str(path), "mean_luminance": mean, "dark_fraction": dark_fraction})
            if mean < 0.12 or dark_fraction > 0.75:
                defects.append({"type": "scene_too_dark", "path": str(path), "severity": 0.8})
            elif mean > 0.96:
                defects.append({"type": "scene_overexposed", "path": str(path), "severity": 0.6})
        except (RuntimeError, OSError):
            defects.append({"type": "diagnostic_image_unreadable", "path": str(path), "severity": 1.0})
    if camera_metadata.get("clearance_min_m", 0.0) < camera_metadata.get("clearance_m", 0.0):
        defects.append({"type": "camera_below_clearance", "severity": 1.0})
    positions = camera_metadata.get("positions", [])
    targets = camera_metadata.get("targets", [])
    if positions and targets and len(positions) == len(targets):
        top_down = 0
        for position, target in zip(positions, targets):
            horizontal = math.hypot(float(target[0]) - float(position[0]), float(target[1]) - float(position[1]))
            vertical = abs(float(target[2]) - float(position[2]))
            if vertical > max(horizontal * 2.5, 1.0):
                top_down += 1
        top_down_fraction = top_down / len(positions)
        if top_down_fraction > 0.55:
            defects.append({"type": "camera_too_top_down", "severity": 0.65, "fraction": top_down_fraction})
    lens_values = [float(value) for value in camera_metadata.get("lens_by_shot", [])]
    # The cinematic sequence intentionally spans wide aerials through a
    # compressed detail lens; keep the QA gate broad enough for both.
    if lens_values and (min(lens_values) < 20.0 or max(lens_values) > 75.0):
        defects.append({"type": "lens_out_of_cinematic_range", "severity": 0.45})
    external = None
    suggested_changes = {}
    command = os.getenv("WORLDCLAW_VLM_INSPECT_CMD")
    if command:
        payload = run / "diagnostic_vlm_input.json"
        payload.write_text(json.dumps({"views": views, "camera": camera_metadata, "lighting": lighting_metadata}, indent=2) + "\n", encoding="utf-8")
        try:
            result = subprocess.run(command + " " + str(payload), shell=True, capture_output=True, text=True, timeout=180, check=False)
            if result.returncode == 0 and result.stdout.strip():
                external = json.loads(result.stdout.strip().splitlines()[-1])
                defects.extend(external.get("defects", []))
                suggested_changes = external.get("suggested_changes", {}) or {}
                if str(external.get("status", "pass")).lower() == "fail" and not external.get("defects"):
                    defects.append({"type": "external_vlm_rejected", "severity": float(external.get("severity", 0.7))})
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            external = {"status": "unavailable"}
    severity = max([float(item.get("severity", 0.0)) for item in defects] or [0.0])
    return {
        "status": "pass" if not defects else "fail",
        "defects": defects,
        "severity": severity,
        "suggested_changes": suggested_changes or ({} if not defects else {"lighting": {"exposure_delta": 0.5}, "camera": {"aperture_fstop": 8.0}}),
        "views": views,
        "inspector": "external_vlm+heuristic" if external is not None else "heuristic",
    }


def adjust_after_qa(scene, camera, lighting_metadata, qa):
    """Bounded QA adjustment for dark/flat diagnostic renders."""
    scene.view_settings.exposure = min(float(scene.view_settings.exposure) + 0.45, 2.0)
    sun = bpy.data.objects.get("WorldClaw_Sun")
    if sun is not None:
        sun.data.energy = min(float(sun.data.energy) * 1.2, 5.0)
    camera.data.dof.aperture_fstop = max(float(camera.data.dof.aperture_fstop), 8.0)
    return {"exposure": scene.view_settings.exposure, "sun_energy": float(sun.data.energy) if sun else None, "aperture_fstop": camera.data.dof.aperture_fstop}


def configure_cinematic_lighting(scene, center, extent, local_target, profile="full"):
    """Configure stable global Sun/world light plus a fixed local accent."""
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.035, 0.06, 0.09, 1.0)
        background.inputs["Strength"].default_value = 0.32

    sun = bpy.data.objects.get("WorldClaw_Sun")
    if sun is None:
        sun_data = bpy.data.lights.new("WorldClaw_Sun", "SUN")
        sun = bpy.data.objects.new("WorldClaw_Sun", sun_data)
        scene.collection.objects.link(sun)
    sun.data.energy = 2.5
    sun.data.angle = math.radians(18.0)
    sun.rotation_euler = (math.radians(28.0), math.radians(-22.0), math.radians(-32.0))

    fixed_target = bpy.data.objects.get("WorldClawLightingTarget")
    if fixed_target is None:
        fixed_target = bpy.data.objects.new("WorldClawLightingTarget", None)
        scene.collection.objects.link(fixed_target)
    fixed_target.location = local_target

    def area(name, location, energy, size):
        obj = bpy.data.objects.get(name)
        if obj is None:
            data = bpy.data.lights.new(name, "AREA")
            obj = bpy.data.objects.new(name, data)
            scene.collection.objects.link(obj)
        obj.location = location
        obj.data.energy = energy
        obj.data.shape = "DISK"
        obj.data.size = size
        constraint = obj.constraints.get("AimAtLocalPOI") or obj.constraints.new(type="TRACK_TO")
        constraint.name = "AimAtLocalPOI"
        constraint.target = fixed_target
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        # UP_Y is the camera/light local roll axis; UP_Z conflicts with the
        # TRACK_NEGATIVE_Z axis and leaves Blender 4.2 rotation at identity.
        constraint.up_axis = "UP_Y"
        return obj

    scale = max(extent, 40.0)
    area("WorldClaw_Key", center + Vector((-scale * 0.55, scale * 0.95, scale * 0.95)), 1100.0, scale * 0.45)
    area("WorldClaw_Fill", center + Vector((scale * 0.75, scale * 0.45, scale * 0.35)), 350.0, scale * 0.60)
    area("WorldClaw_Rim", center + Vector((scale * 0.15, scale * 0.70, scale * 1.10)), 700.0, scale * 0.35)
    atmosphere_enabled = profile == "full" and os.getenv("WORLDCLAW_ATMOSPHERE", "0") != "0"
    # Keep aerial perspective subtle: at 1 km this is a light veil, not a
    # foreground-obscuring volume. Scale inversely with world footprint.
    density = min(0.0004, max(0.00002, 0.00012 * (1000.0 / max(extent, 250.0))))
    if profile == "diagnostic":
        density *= 0.25
    if atmosphere_enabled:
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        output = nodes.get("World Output")
        volume = next((node for node in nodes if node.bl_idname == "ShaderNodeVolumePrincipled"), None)
        if volume is None:
            volume = nodes.new("ShaderNodeVolumePrincipled")
        volume.inputs["Density"].default_value = density
        volume.inputs["Color"].default_value = (0.46, 0.58, 0.68, 1.0)
        if output is not None and output.inputs.get("Volume") and not output.inputs["Volume"].is_linked:
            links.new(volume.outputs["Volume"], output.inputs["Volume"])
    return {
        "rig": ["WorldClaw_Sun", "WorldClaw_Key", "WorldClaw_Fill", "WorldClaw_Rim"],
        "global_sun_fixed": True,
        "local_target": list(local_target),
        "atmosphere": atmosphere_enabled,
        "atmosphere_density": density if atmosphere_enabled else 0.0,
    }


def configure_camera_path(scene, camera, target, bounds, terrain, bvh, semantic_pois=None):
    minimum, maximum = bounds
    terrain_bounds = object_world_bounds(terrain) if terrain is not None else bounds
    terrain_minimum, terrain_maximum = terrain_bounds or bounds
    center = (terrain_minimum + terrain_maximum) * 0.5
    horizontal_extent = max(terrain_maximum.x - terrain_minimum.x, terrain_maximum.y - terrain_minimum.y)
    vertical_extent = max(terrain_maximum.z - terrain_minimum.z, 1.0)
    # Keep the establishing orbit proportional to the actual Terrain
    # footprint. Near shots use explicit fractions below so the sequence has
    # a readable wide -> approach -> detail -> fly-through -> pull-back arc.
    radius = max(horizontal_extent * 0.78, 120.0)
    lens = max(24.0, min(32.0, 30000.0 / max(horizontal_extent, 1000.0)))
    camera.data.lens = lens
    camera.data.clip_start = 0.5
    camera.data.clip_end = max(5000.0, horizontal_extent * 4.0)

    clearance = max(24.0, vertical_extent * 0.16, horizontal_extent * 0.022)
    # Altitudes and radii are relative to each sampled ground point, never
    # absolute world coordinates. This remains stable when a scene is
    # translated in Z while making the shot language visibly distinct.
    shot_specs = [
        {"name": "establishing", "tokens": ("lake", "water"), "radius": 0.56, "altitude": 0.78, "lens": 24.0, "look_height": 0.22, "angle": 28.0},
        {"name": "hero_approach", "tokens": ("cabin", "cabins", "house", "building", "castle", "village"), "radius": 0.38, "altitude": 0.34, "lens": 42.0, "look_height": 0.16, "angle": 118.0},
        {"name": "detail", "tokens": ("rock", "tree", "fern", "vegetation", "forest"), "radius": 0.16, "altitude": 0.16, "lens": 68.0, "look_height": 0.10, "angle": 205.0},
        {"name": "flythrough", "tokens": ("trail", "trails", "path", "road", "shoreline"), "radius": 0.25, "altitude": 0.20, "lens": 32.0, "look_height": 0.13, "angle": 292.0},
        {"name": "reestablish", "tokens": ("lake", "water"), "radius": 0.65, "altitude": 0.64, "lens": 22.0, "look_height": 0.20, "angle": 382.0},
    ]
    lens_by_shot = [spec["lens"] for spec in shot_specs]
    candidates = [obj for obj in scene.objects if obj.type == "MESH" and obj != terrain and not obj.hide_render]
    semantic_pois = semantic_pois or []
    def valid(point):
        point = clamp_poi_to_terrain(point, terrain_minimum, terrain_maximum)
        return point

    def semantic_match(tokens, fallback, prefer_objects=False):
        def semantic_equal(token, value):
            token, value = str(token).lower(), str(value).lower()
            return token == value or token.rstrip("s") == value.rstrip("s")

        def match_score(item):
            name = str(item.get("name", "")).lower()
            region_id = str(item.get("region_id", "")).lower()
            category = str(item.get("category", "")).lower()
            exact = max((semantic_equal(token, name) or semantic_equal(token, region_id)) for token in tokens)
            contains = max((str(token).lower() in (name + " " + region_id + " " + category)) for token in tokens)
            return (int(exact), int(contains), float(item.get("importance", 0.5)))

        name_matches = [obj for obj in candidates if any(str(token).lower() in obj.name.lower() for token in tokens)]
        in_bounds = [obj for obj in name_matches if (obj.matrix_world.translation - center).length <= horizontal_extent * 1.5]
        if prefer_objects and in_bounds:
            selected = max(in_bounds, key=lambda item: min(item.dimensions.length, horizontal_extent * 0.25))
            p = valid(selected.matrix_world.translation.copy())
            return p, {"source": "validated_object_name", "name": selected.name, "category": "object"}
        matches = [item for item in semantic_pois if match_score(item)[1]]
        if matches:
            # Exact region ids (lake/cabins/trails) outrank incidental words in
            # prose descriptions such as "world" or "forest".
            selected = max(matches, key=match_score)
            p = Vector(selected["position"])
            p.z = terrain_height(terrain, bvh, p.x, p.y, terrain_minimum.z, terrain_bounds=terrain_bounds)
            return valid(p), {"source": "semantic_metadata", **selected}
        if in_bounds:
            selected = max(in_bounds, key=lambda item: min(item.dimensions.length, horizontal_extent * 0.25))
            p = valid(selected.matrix_world.translation.copy())
            return p, {"source": "validated_object_name", "name": selected.name, "category": "object"}
        return valid(fallback.copy()), {"source": "terrain_center", "name": "terrain_center", "category": "terrain"}

    shot_targets, poi_metadata = [], []
    for spec in shot_specs:
        fallback = shot_targets[-1] if shot_targets else center
        point, metadata = semantic_match(spec["tokens"], fallback, prefer_objects=spec["name"] == "detail")
        metadata = dict(metadata)
        metadata["shot"] = spec["name"]
        shot_targets.append(point)
        poi_metadata.append(metadata)
    shot_radii = [max(40.0, horizontal_extent * spec["radius"]) for spec in shot_specs]
    beat_frames = [1, 72, 144, 216, 288]
    control_positions, control_targets, target_visibility = [], [], []

    def line_visible(camera_position, target_point, footprint):
        if bvh is None:
            return True
        direction = target_point - camera_position
        if direction.length <= 1e-6:
            return False
        hit, _normal, _index, hit_distance = bvh.ray_cast(camera_position, direction.normalized(), direction.length)
        if hit is None or hit_distance >= direction.length - 2.0:
            return True
        # A target sitting on the terrain is expected to have the terrain hit
        # at its own XY footprint; only a displaced ridge counts as occlusion.
        return math.hypot(hit.x - target_point.x, hit.y - target_point.y) <= footprint
    for index, frame in enumerate(beat_frames):
        spec = shot_specs[index]
        angle = math.radians(spec["angle"])
        shot_center = shot_targets[index]
        shot_radius = shot_radii[index]
        x = shot_center.x + shot_radius * math.cos(angle)
        y = shot_center.y + shot_radius * math.sin(angle)
        # Keep every control point on the finite Terrain footprint. Sampling
        # outside the mesh can provide a fallback height, but it produces an
        # empty/black view and is not a valid world showcase shot.
        x = min(max(x, terrain_minimum.x), terrain_maximum.x)
        y = min(max(y, terrain_minimum.y), terrain_maximum.y)
        ground = terrain_height(terrain, bvh, x, y, terrain_minimum.z, terrain_bounds=terrain_bounds)
        # Aim at the semantic POI itself. The previous tiny offset toward the
        # Terrain center made every shot read as an orbit around one hill.
        look_x, look_y = shot_center.x, shot_center.y
        look_x = min(max(look_x, terrain_minimum.x), terrain_maximum.x)
        look_y = min(max(look_y, terrain_minimum.y), terrain_maximum.y)
        look_ground = terrain_height(terrain, bvh, look_x, look_y, terrain_minimum.z, terrain_bounds=terrain_bounds)
        shot_altitude = max(clearance, horizontal_extent * spec["altitude"])
        camera_z = ground + shot_altitude
        look_height = max(8.0, vertical_extent * spec["look_height"])
        target_z = look_ground + look_height
        # If a ridge lies between the camera and the POI, lift the target
        # enough to clear it before falling back to a generic terrain target.
        target_point = Vector((look_x, look_y, target_z))
        direction = target_point - Vector((x, y, camera_z))
        target_footprint = max(10.0, horizontal_extent * 0.012)
        visibility = False
        visibility_retries = 0
        for visibility_retries in range(4):
            if line_visible(Vector((x, y, camera_z)), target_point, target_footprint):
                visibility = True
                break
            target_z += max(10.0, vertical_extent * 0.10)
            target_point.z = target_z
            direction = target_point - Vector((x, y, camera_z))
        # High aerial shots can still look through a ridge at one azimuth.
        # Try the opposite and quarter-turn positions before accepting a
        # hidden semantic target.
        if not visibility and index in (0, 4):
            for angle_offset in (math.pi * 0.5, math.pi, math.pi * 1.5):
                candidate_angle = angle + angle_offset
                candidate_x = min(max(shot_center.x + shot_radius * math.cos(candidate_angle), terrain_minimum.x), terrain_maximum.x)
                candidate_y = min(max(shot_center.y + shot_radius * math.sin(candidate_angle), terrain_minimum.y), terrain_maximum.y)
                candidate_ground = terrain_height(terrain, bvh, candidate_x, candidate_y, terrain_minimum.z, terrain_bounds=terrain_bounds)
                candidate_position = Vector((candidate_x, candidate_y, candidate_ground + shot_altitude))
                if line_visible(candidate_position, target_point, target_footprint):
                    x, y, camera_z = candidate_x, candidate_y, candidate_position.z
                    visibility = True
                    break
        control_positions.append(Vector((x, y, camera_z)))
        control_targets.append(target_point)
        target_visibility.append({"shot": spec["name"], "visible": visibility, "retries": visibility_retries, "target": list(target_point)})

    # Dense Catmull-Rom samples provide continuous position and target motion;
    # Bezier keyframes preserve smooth velocity without linear corner jumps.
    positions, targets, clearance_checks = [], [], []
    for frame in range(1, 289):
        segment = min((frame - 1) // 72, 3)
        t = ((frame - 1) % 72) / 72.0
        p0 = control_positions[max(0, segment - 1)]
        p1 = control_positions[segment]
        p2 = control_positions[segment + 1]
        p3 = control_positions[min(4, segment + 2)]
        q0 = control_targets[max(0, segment - 1)]
        q1 = control_targets[segment]
        q2 = control_targets[segment + 1]
        q3 = control_targets[min(4, segment + 2)]
        position = catmull_rom(p0, p1, p2, p3, t)
        look_target = catmull_rom(q0, q1, q2, q3, t)
        ground = terrain_height(terrain, bvh, position.x, position.y, terrain_minimum.z, terrain_bounds=terrain_bounds)
        position.z = max(position.z, ground + clearance)
        target_ground = terrain_height(terrain, bvh, look_target.x, look_target.y, terrain_minimum.z, terrain_bounds=terrain_bounds)
        look_target.z = max(look_target.z, target_ground + 8.0)
        positions.append(tuple(position))
        targets.append(tuple(look_target))
        clearance_checks.append({
            "frame": frame,
            "camera_ground_z": float(ground),
            "camera_z": float(position.z),
            "camera_clearance": float(position.z - ground),
            "target_ground_z": float(target_ground),
            "target_z": float(look_target.z),
            "target_look_height": float(look_target.z - target_ground),
        })
        # Camera translation is driven exclusively by FOLLOW_PATH below.
        # Keeping location keyframes here would make Blender add the sampled
        # position twice (constraint offset + object transform).
        target.location = look_target
        target.keyframe_insert("location", frame=frame)
    # Catmull-Rom interpolation can dip below the terrain between control
    # points even when every control point satisfies the clearance constraint.
    # Clamp the authored world-space path once more before building the spline.
    for index, position in enumerate(positions):
        ground = terrain_height(
            terrain, bvh, position[0], position[1], terrain_minimum.z,
            terrain_bounds=terrain_bounds,
        )
        minimum_z = ground + clearance
        if position[2] < minimum_z:
            position = (position[0], position[1], minimum_z)
            positions[index] = position
        clearance_checks[index]["camera_ground_z"] = float(ground)
        clearance_checks[index]["camera_z"] = float(position[2])
        clearance_checks[index]["camera_clearance"] = float(position[2] - ground)
    smooth_keyframes(target)
    spline = create_camera_spline(scene, positions, list(range(1, 289)))
    follow = camera.constraints.get("WorldClawFollowSpline") or camera.constraints.new(type="FOLLOW_PATH")
    follow.name = "WorldClawFollowSpline"
    follow.target = spline
    follow.use_fixed_location = True
    follow.use_curve_follow = False
    follow.forward_axis = "FORWARD_X"
    # Evaluate path position before the Track To orientation constraint.
    try:
        camera.constraints.move(camera.constraints.find(follow.name), 0)
    except (AttributeError, TypeError, RuntimeError):
        pass
    follow.offset_factor = 0.0
    follow.keyframe_insert("offset_factor", frame=1)
    follow.offset_factor = 1.0
    follow.keyframe_insert("offset_factor", frame=288)
    for frame, lens_value in zip(beat_frames, lens_by_shot):
        camera.data.lens = lens_value
        camera.data.keyframe_insert("lens", frame=frame)
    smooth_keyframes(camera.data)
    if any(item["camera_clearance"] < clearance for item in clearance_checks):
        raise RuntimeError("camera path violates Terrain clearance")
    if any(item["target_look_height"] <= 0.0 for item in clearance_checks):
        raise RuntimeError("camera target is not above Terrain")
    return {
        "bounds_min": list(terrain_minimum), "bounds_max": list(terrain_maximum),
        "scene_bounds": {"min": list(minimum), "max": list(maximum)},
        "terrain_bounds": bounds_metadata((terrain_minimum, terrain_maximum)),
        "horizontal_extent": horizontal_extent, "vertical_extent": vertical_extent,
        "radius": radius, "orbit_radius_m": radius, "lens_mm": lens_by_shot[0], "lens_by_shot": lens_by_shot, "shot_radii_m": shot_radii, "clearance_m": clearance,
        "up_axis": "Z (Blender native Z-up after glTF import)",
        "positions": positions, "targets": targets,
        "clearance_checks": clearance_checks,
        "clearance_min_m": min(item["camera_clearance"] for item in clearance_checks),
        "spline": {"object": spline.name, "type": "BEZIER", "samples": len(positions), "frames": [1, 288], "follow_path": True, "constraint": follow.name},
        "poi": poi_metadata,
        "target_visibility": target_visibility,
        "shots": [
            {"name": spec["name"], "frame": frame, "target": list(targets[frame - 1]), "lens_mm": spec["lens"], "radius_m": shot_radii[index], "altitude_m": shot_radii[index] * spec["altitude"], "poi": poi_metadata[index].get("name"), "poi_source": poi_metadata[index].get("source")}
            for index, (spec, frame) in enumerate(zip(shot_specs, beat_frames))
        ],
    }


def main():
    options = parse_args()
    run = options.run.resolve()
    profile = options.profile
    glb = run / "scene.glb"
    scene_json = json.loads((run / "scene.json").read_text(encoding="utf-8"))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    imported = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    if not imported:
        raise RuntimeError("GLB contained no mesh nodes")

    scene = bpy.context.scene
    coordinate_metadata = normalize_imported_scene_z_up(scene, imported)
    terrain_for_filter = next((obj for obj in imported if "terrain" in obj.name.lower()), None)
    # Validate immediately after import/coordinate normalization.  Evaluated
    # mesh queries used by outlier filtering can cause Blender to rebuild
    # polygon normals on very large terrains; the import-time result is the
    # authoritative winding evidence for the export gate.
    terrain_normal_metadata = diagnose_terrain_normals(terrain_for_filter)
    if terrain_for_filter is not None and not terrain_normal_metadata.get("passed", False):
        raise RuntimeError(
            "Terrain face orientation is invalid after GLB import; "
            "fix triangle winding in Export before finalization"
        )
    outlier_meshes = quarantine_outlier_meshes(scene, terrain_for_filter) if terrain_for_filter else []
    terrain_material_metadata = ensure_terrain_material(terrain_for_filter)
    structural_features = scene_json.get("structural_features", [])
    structural_material_metadata = ensure_structural_materials(scene, structural_features)
    semantic_pois = load_semantic_pois(run)
    material_metadata = terrain_material_metadata | structural_material_metadata | unify_material_look(scene)
    if os.getenv("WORLDCLAW_SKIP_MESH_DEDUP", "0").lower() in {"1", "true", "yes"}:
        instance_metadata = {"unique_mesh_datablocks": len({obj.data.as_pointer() for obj in scene.objects if obj.type == "MESH"}), "reused_mesh_objects": 0, "skipped": True}
    else:
        instance_metadata = deduplicate_mesh_data(scene)
    if os.getenv("WORLDCLAW_SKIP_LOD", "0").lower() in {"1", "true", "yes"}:
        lod_metadata = {"lod_modifiers": 0, "source_polygons": 0, "estimated_render_polygons": 0, "skipped": True}
    else:
        lod_metadata = add_lod_modifiers(scene)
    requested_render_engine = os.getenv("BLENDER_RENDER_ENGINE", "BLENDER_EEVEE_NEXT")
    available_render_engines = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    }
    # Blender 4.0 names Eevee ``BLENDER_EEVEE`` while 4.2 renamed it to
    # ``BLENDER_EEVEE_NEXT``. Keep the requested profile portable across the
    # user-local package used for audit reruns.
    render_engine = requested_render_engine
    if render_engine not in available_render_engines:
        if render_engine == "BLENDER_EEVEE_NEXT" and "BLENDER_EEVEE" in available_render_engines:
            render_engine = "BLENDER_EEVEE"
        else:
            raise RuntimeError(
                f"requested render engine {render_engine!r} is unavailable; "
                f"available={sorted(available_render_engines)}"
            )
    scene.render.engine = render_engine
    if render_engine == "CYCLES":
        # Unlike Eevee's OpenGL backend, Cycles honors CUDA_VISIBLE_DEVICES;
        # this lets a render be isolated to a temporarily released GPU.
        scene.cycles.device = "GPU"
        scene.cycles.samples = int(os.getenv("BLENDER_CYCLES_SAMPLES", "16"))
        scene.cycles.use_denoising = True
        cycles = bpy.context.preferences.addons.get("cycles")
        if cycles is not None:
            preferences = cycles.preferences
            preferences.compute_device_type = "CUDA"
            preferences.get_devices()
            for device in getattr(preferences, "devices", ()):
                device.use = True
    resolution = (640, 360) if profile == "diagnostic" else (1280, 720)
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.render.fps_base = 1.0
    scene.frame_start, scene.frame_end = 1, 288
    scene.render.use_file_extension = True
    scene.render.image_settings.color_mode = "RGB"
    if render_engine == "BLENDER_EEVEE_NEXT":
        # Property names differ between Blender 3.x and 4.x; use guarded
        # assignments so the finalizer remains portable across the remote host.
        eevee = getattr(scene, "eevee", None)
        if eevee is not None:
            for name, value in (("taa_render_samples", 32), ("taa_samples", 32)):
                if hasattr(eevee, name):
                    setattr(eevee, name, value)
    scene.view_settings.look = "AgX - Medium High Contrast"

    camera_data = bpy.data.cameras.new("WalkthroughCamera")
    camera_data.lens, camera_data.clip_end = 28, 5000
    camera = bpy.data.objects.new("WalkthroughCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    target = bpy.data.objects.new("CameraTarget", None)
    scene.collection.objects.link(target)
    track = camera.constraints.new(type="TRACK_TO")
    # Camera local up is Y even though the world vertical axis is Z.
    track.target, track.track_axis, track.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    terrain, bvh = terrain_bvh(scene)
    structural_names = {str(item.get("feature_id", "")) for item in structural_features}
    grounding_metadata = settle_assets_on_terrain(scene, terrain, bvh, protected_names=structural_names)
    bounds = scene_bounds(scene)
    camera_metadata = configure_camera_path(scene, camera, target, bounds, terrain, bvh, semantic_pois)
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = target
    camera.data.dof.aperture_fstop = 8.0
    terrain_bounds = object_world_bounds(terrain) if terrain is not None else bounds
    center = (terrain_bounds[0] + terrain_bounds[1]) * 0.5
    lighting_target = Vector(camera_metadata["targets"][71])
    lighting_metadata = configure_cinematic_lighting(
        scene, center, max(camera_metadata["horizontal_extent"], camera_metadata["vertical_extent"]), lighting_target, profile
    )

    diagnostic = run / "diagnostic_views"
    diagnostic.mkdir(exist_ok=True)
    retry_history = []
    max_render_retry = max(0, int(os.getenv("WORLDCLAW_MAX_RENDER_RETRY", "2")))
    qa = {"status": "fail", "defects": [{"type": "not_run"}]}
    for attempt in range(max_render_retry + 1):
        paths = []
        scene.render.image_settings.file_format = "PNG"
        for index, frame in enumerate((1, 72, 144, 216)):
            scene.frame_set(frame)
            path = diagnostic / f"view_{index:02d}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            paths.append(path)
        qa = inspect_diagnostic_views(run, paths, camera_metadata, lighting_metadata)
        retry_history.append({"attempt": attempt, "status": qa["status"], "defects": qa["defects"], "adjustment": None})
        if qa["status"] == "pass":
            break
        if attempt < max_render_retry:
            retry_history[-1]["adjustment"] = adjust_after_qa(scene, camera, lighting_metadata, qa)
    if profile == "full" and qa["status"] != "pass":
        # Do not publish a walkthrough when the diagnostic gate still reports
        # a severe composition/lighting defect.
        raise RuntimeError(f"diagnostic render QA failed after {max_render_retry + 1} attempts: {qa['defects']}")
    scene.frame_set(1)
    scene.render.filepath = str(run / "preview.png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(run / "scene.blend"))

    if profile == "full":
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.ffmpeg.audio_codec = "NONE"
        scene.render.filepath = str(run / "walkthrough.mp4")
        bpy.ops.render.render(animation=True)
        candidates = sorted(run.glob("walkthrough*.mp4"))
        if not (run / "walkthrough.mp4").exists() and candidates:
            candidates[-1].replace(run / "walkthrough.mp4")
    metadata = {
        "official_implementation": False,
        "blender_version": bpy.app.version_string,
        "mesh_nodes": len(imported),
        "resolution": list(resolution), "fps": 24, "frames": 288, "duration_seconds": 12,
        "render_profile": profile, "video_generated": profile == "full",
        "mesh_optimization": material_metadata | instance_metadata | lod_metadata,
        "outlier_meshes_quarantined": outlier_meshes,
        "asset_grounding": grounding_metadata,
        "terrain": {
            "world_space_bounds": camera_metadata.get("terrain_bounds", bounds_metadata(bounds)),
            "normal_diagnostics": terrain_normal_metadata,
        },
        "lighting": lighting_metadata,
        "poi": {"semantic_candidates": len(semantic_pois), "selected": camera_metadata.get("poi", [])},
        "render_qa": qa,
        "render_retry_history": retry_history,
        "coordinate_system": scene_json["coordinate_system"],
        "coordinate_normalization": coordinate_metadata,
        "camera_path": camera_metadata,
    }
    (run / "blender_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
