import json
import struct
from pathlib import Path

from worldclaw_oss.pipeline import Pipeline
from worldclaw_oss.validation import validate_run


LOCK = Path(__file__).resolve().parents[1] / "models.lock.json"


def run_synthetic(path, prompt):
    return Pipeline(path, LOCK, "synthetic", 42, prompt, ["test"]).run()


def test_synthetic_pipeline_and_resume(tmp_path):
    run = run_synthetic(tmp_path / "castle", "A medieval castle on a hill")
    report = validate_run(run)
    assert report["valid"]
    scene = json.loads((run / "scene.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert scene["official_implementation"] is False
    assert scene["synthetic"] is True
    assert metrics["region_count"] == 3
    assert metrics["asset_count"] >= 10
    layout_bundle = run / "work" / "terrain_planner_input" / "layout"
    assert (layout_bundle / "layout_labels.npy").is_file()
    assert (layout_bundle / "layout_weights.npy").is_file()
    assert len(list((layout_bundle / "masks").glob("*.npy"))) == metrics["region_count"]
    before = (run / "scene.glb").read_bytes()
    run_synthetic(run, "A medieval castle on a hill")
    assert (run / "scene.glb").read_bytes() == before
    assert struct.unpack("<4sI", before[:8]) == (b"glTF", 2)


def test_three_fixed_prompts_have_distinct_plans(tmp_path):
    themes = []
    for name, prompt in [
        ("castle", "medieval castle and village"),
        ("desert", "desert city and oasis"),
        ("forest", "forest lake and cabins"),
    ]:
        run = run_synthetic(tmp_path / name, prompt)
        themes.append(json.loads((run / "scene.json").read_text(encoding="utf-8"))["plan"]["theme"])
    assert len(set(themes)) == 3


def test_mesh_reuse_manifest_records_live_meshes(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    mesh = tmp_path / "prototype.glb"
    mesh.write_bytes(b"mesh")
    (work / "placement_response.json").write_text(json.dumps({
        "assets": [
            {"id": "tree_0000", "category": "tree", "mesh": str(mesh), "source_model": "model-a"},
            {"id": "tree_0001", "category": "tree", "mesh": str(mesh), "source_model": "model-a"},
        ]
    }), encoding="utf-8")
    (work / "scene_plan.json").write_text("{}", encoding="utf-8")
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.work = work
    pipeline.run_dir = tmp_path
    pipeline._write_mesh_reuse_manifest()
    manifest = json.loads((tmp_path / "mesh_reuse_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reason"]["regenerated_count"] == 1
    assert manifest["regenerated"][0]["asset_ids"] == ["tree_0000", "tree_0001"]
    assert manifest["regenerated"][0]["status"] == "regenerated"


def test_mesh_reuse_manifest_requires_complete_dependency_fingerprint(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    (baseline / "work").mkdir(parents=True)
    baseline_mesh = baseline / "tree.glb"
    baseline_mesh.write_bytes(b"same-mesh")
    baseline_item = {
        "id": "tree_0000", "category": "tree", "asset_type": "reusable_prototype",
        "prompt": "isolated tree", "source_model": "model-a",
        "mesh": str(baseline_mesh), "validation_status": "pass",
    }
    (baseline / "work" / "placement_response.json").write_text(json.dumps({"assets": [baseline_item]}), encoding="utf-8")
    current = tmp_path / "current"
    work = current / "work"
    work.mkdir(parents=True)
    current_mesh = current / "tree.glb"
    current_mesh.write_bytes(b"same-mesh")
    (work / "placement_response.json").write_text(json.dumps({"assets": [baseline_item | {"mesh": str(current_mesh)}]}), encoding="utf-8")
    monkeypatch.setenv("WORLDCLAW_BASELINE_RUN", str(baseline))
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.work = work
    pipeline.run_dir = current
    pipeline._write_mesh_reuse_manifest()
    manifest = json.loads((current / "mesh_reuse_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["reuse_eligible"]) == 1
    assert manifest["reason"]["reuse_eligible_count"] == 1
    assert manifest["reused"] == []


def test_mesh_reuse_manifest_marks_only_actual_verified_path_as_reused(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    (baseline / "work").mkdir(parents=True)
    mesh = baseline / "tree.glb"
    mesh.write_bytes(b"same-mesh")
    item = {
        "id": "tree_0000", "category": "tree", "asset_type": "reusable_prototype",
        "prompt": "isolated tree", "source_model": "model-a",
        "mesh": str(mesh), "validation_status": "pass",
    }
    (baseline / "work" / "placement_response.json").write_text(
        json.dumps({"assets": [item]}), encoding="utf-8"
    )
    current = tmp_path / "current"
    (current / "work").mkdir(parents=True)
    (current / "work" / "placement_response.json").write_text(
        json.dumps({"assets": [item]}), encoding="utf-8"
    )
    monkeypatch.setenv("WORLDCLAW_BASELINE_RUN", str(baseline))
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.work = current / "work"
    pipeline.run_dir = current
    pipeline._write_mesh_reuse_manifest()
    manifest = json.loads((current / "mesh_reuse_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reason"]["reused_count"] == 1
    assert len(manifest["reused"]) == 1
    assert manifest["reused"][0]["baseline_match"]["dependency_match"] is True
