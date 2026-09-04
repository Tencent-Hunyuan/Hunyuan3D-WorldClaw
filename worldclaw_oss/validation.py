from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


CORE_REQUIRED=("scene.glb","scene.json","preview.png","run_manifest.json","metrics.json","errors.log","state.sqlite3","events.jsonl")
FULL_REQUIRED=CORE_REQUIRED+("scene.blend","walkthrough.mp4","diagnostic_views")


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def validate_run(run: Path,full: bool=False) -> dict[str,Any]:
    run=Path(run); missing=[]
    for name in FULL_REQUIRED if full else CORE_REQUIRED:
        if not (run/name).exists(): missing.append(name)
    errors=[]
    if (run/"scene.glb").is_file():
        data=(run/"scene.glb").read_bytes()[:12]
        if len(data)!=12 or data[:4]!=b"glTF" or struct.unpack("<I",data[4:8])[0]!=2: errors.append("scene.glb is not glTF 2.0 GLB")
    if (run/"scene.json").is_file():
        scene=json.loads((run/"scene.json").read_text(encoding="utf-8"))
        if scene.get("official_implementation") is not False: errors.append("replacement disclosure missing")
        ids={r["id"] for r in scene["plan"]["regions"]}
        for asset in scene.get("assets",[]):
            if asset["region_id"] not in ids: errors.append(f"asset outside known region: {asset['id']}")
    if full:
        for name in ("scene.blend", "walkthrough.mp4", "preview.png"):
            path = run / name
            if path.is_file() and path.stat().st_size == 0:
                errors.append(f"{name} is empty")
        mp4 = run / "walkthrough.mp4"
        if mp4.is_file() and b"ftyp" not in mp4.read_bytes()[:64]:
            errors.append("walkthrough.mp4 has no ISO base media signature")
        diagnostics = list((run / "diagnostic_views").glob("view_*.png"))
        if len([path for path in diagnostics if path.stat().st_size > 0]) < 4:
            errors.append("fewer than four non-empty Blender diagnostic views")
        metadata_path = run / "blender_metadata.json"
        if not metadata_path.is_file():
            errors.append("blender_metadata.json is missing")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("resolution") != [1280, 720]:
                errors.append("video resolution is not 1280x720")
            if metadata.get("fps") != 24 or metadata.get("frames", 0) < 288:
                errors.append("video is shorter than 12 seconds at 24 fps")
            if metadata.get("mesh_nodes", 0) < 1:
                errors.append("Blend/GLB mesh node validation is empty")
    else:
        # Diagnostic Blender runs are intentionally video-free, but still
        # need enough views for a meaningful visual QA pass.
        metadata_path = run / "blender_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("render_profile") == "diagnostic":
                diagnostics = list((run / "diagnostic_views").glob("view_*.png"))
                if len([path for path in diagnostics if path.stat().st_size > 0]) < 4:
                    errors.append("fewer than four non-empty Blender diagnostic views")
                if metadata.get("video_generated"):
                    errors.append("diagnostic render unexpectedly generated a video")
    return {"valid":not missing and not errors,"profile":"full" if full else "core","missing":missing,"errors":errors}
