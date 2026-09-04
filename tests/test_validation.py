import json

from worldclaw_oss.validation import validate_run


def test_full_validation_rejects_empty_diagnostics(tmp_path):
    for name in (
        "scene.glb", "scene.json", "preview.png", "run_manifest.json", "metrics.json",
        "errors.log", "state.sqlite3", "scene.blend", "walkthrough.mp4",
    ):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "scene.glb").write_bytes(b"glTF" + (2).to_bytes(4, "little") + b"\x00" * 4)
    (tmp_path / "walkthrough.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom")
    (tmp_path / "scene.json").write_text(json.dumps({
        "official_implementation": False, "plan": {"regions": []}, "assets": [],
    }), encoding="utf-8")
    (tmp_path / "blender_metadata.json").write_text(json.dumps({
        "resolution": [1280, 720], "fps": 24, "frames": 288, "mesh_nodes": 1,
    }), encoding="utf-8")
    (tmp_path / "diagnostic_views").mkdir()
    report = validate_run(tmp_path, full=True)
    assert not report["valid"]
    assert "fewer than four non-empty Blender diagnostic views" in report["errors"]


def test_diagnostic_validation_accepts_views_without_video(tmp_path):
    for name in (
        "scene.glb", "scene.json", "preview.png", "run_manifest.json", "metrics.json",
        "errors.log", "state.sqlite3", "events.jsonl",
    ):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "scene.glb").write_bytes(b"glTF" + (2).to_bytes(4, "little") + b"\x00" * 4)
    (tmp_path / "scene.json").write_text(json.dumps({
        "official_implementation": False, "plan": {"regions": []}, "assets": [],
    }), encoding="utf-8")
    (tmp_path / "blender_metadata.json").write_text(json.dumps({
        "render_profile": "diagnostic", "video_generated": False,
    }), encoding="utf-8")
    diagnostics = tmp_path / "diagnostic_views"
    diagnostics.mkdir()
    for index in range(4):
        (diagnostics / f"view_{index:02d}.png").write_bytes(b"png")
    report = validate_run(tmp_path)
    assert report["valid"]
