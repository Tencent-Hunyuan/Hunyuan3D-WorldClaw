"""Open a Blend or import a GLB and emit machine-readable structural checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if options.path.suffix.lower() == ".glb":
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.import_scene.gltf(filepath=str(options.path.resolve()))
    elif options.path.suffix.lower() == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(options.path.resolve()))
    meshes = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    actions = [action for action in bpy.data.actions if len(action.fcurves) > 0]
    result = {
        "path": str(options.path),
        "mesh_nodes": len(meshes),
        "unique_mesh_node_names": len({item.name for item in meshes}),
        "animated_actions": len(actions),
        "frame_end": bpy.context.scene.frame_end,
        "valid": bool(meshes) and len(meshes) == len({item.name for item in meshes}),
    }
    print("WORLDCLAW_VALIDATE " + json.dumps(result, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
