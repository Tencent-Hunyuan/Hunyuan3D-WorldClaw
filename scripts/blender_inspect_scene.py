"""Print mesh bounds and material statistics from a Blender scene."""
from __future__ import annotations

import bpy
import sys
from mathutils import Vector


def main():
    path = sys.argv[sys.argv.index("--") + 1]
    bpy.ops.wm.open_mainfile(filepath=path)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rows = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
        lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
        hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
        dim = hi - lo
        rows.append((max(dim), obj.name, tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi), len(obj.data.materials)))
    for row in sorted(rows, reverse=True)[:30]:
        print(row)
    print("MESH_COUNT", len(rows))


if __name__ == "__main__":
    main()
