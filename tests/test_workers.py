import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workers.hunyuan3d_worker import generate_mesh
import workers.mesh_validation_worker as mesh_validation_worker
from workers.mesh_validation_worker import (
    _coerce_report,
    _make_fallback_asset,
    _mesh_metrics,
    _mesh_sha256,
    _reference_sha256,
    _validation_cache_key,
)
from workers.placement_worker import (
    contact_metrics,
    label_at,
    overlaps,
    terrain_sampler,
    yaw_rotation,
    OccupiedSpatialHash,
    _placement_worker_count,
    scatter_environment,
)
from workers.segmentation_worker import deduplicate, windows
from worldclaw_oss.models import ModelLock
from worldclaw_oss.geometry import gltf_vertices_to_z_up


def test_sliding_windows_cover_edges():
    result = list(windows(2050, 1300, 1024, 0.25))
    assert result[0] == (0, 0, 1024, 1024)
    assert max(item[2] for item in result) == 2050
    assert max(item[3] for item in result) == 1300
    assert all(item[0] >= 0 and item[1] >= 0 for item in result)


def test_mask_deduplication_is_category_aware():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    candidates = [
        {"category": "tree", "score": 0.9, "mask": mask},
        {"category": "tree", "score": 0.7, "mask": mask.copy()},
        {"category": "rock", "score": 0.8, "mask": mask.copy()},
    ]
    kept = deduplicate(candidates, 0.8)
    assert [(item["category"], item["score"]) for item in kept] == [
        ("tree", 0.9), ("rock", 0.8)
    ]


def test_terrain_sampler_and_contact():
    height = np.array([[0.0, 1.0], [1.0, 2.0]])
    sample = terrain_sampler(height, (2.0, 2.0))
    assert sample(0.0, 0.0) == 1.0
    vertices = np.array([
        [-0.1, -0.1, sample(-0.1, -0.1)],
        [0.1, 0.1, sample(0.1, 0.1)],
        [0.0, 0.0, 2.0],
    ])
    ratio, penetration, floating = contact_metrics(vertices, sample, 0.02)
    assert ratio == 1.0
    assert penetration == 0.0
    assert floating == 0.0
    assert np.allclose(sample(np.array([-0.1, 0.1]), np.array([-0.1, 0.1])), [0.9, 1.1])


def test_region_lookup_overlap_and_yaw():
    labels = np.array([[0, 0], [1, 1]])
    assert label_at(labels, (2.0, 2.0), 0.0, 0.8) == 1
    assert overlaps((0, 0, 2, 2), [(1, 1, 3, 3)], threshold=0.2)
    rotation = yaw_rotation(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    index = OccupiedSpatialHash(2.0)
    index.add((0.0, 0.0, 2.0, 2.0))
    assert index.overlaps((1.0, 1.0, 3.0, 3.0), threshold=0.2)
    assert not index.overlaps((4.0, 4.0, 5.0, 5.0), threshold=0.2)


def test_placement_worker_count_is_bounded_and_configurable(monkeypatch):
    assert _placement_worker_count(1) == 1
    monkeypatch.setenv("WORLDCLAW_PLACEMENT_WORKERS", "0")
    assert _placement_worker_count(20) == 1
    monkeypatch.setenv("WORLDCLAW_PLACEMENT_WORKERS", "6")
    assert _placement_worker_count(20) == 6
    monkeypatch.setenv("WORLDCLAW_PLACEMENT_WORKERS", "99")
    assert _placement_worker_count(20) == 8
    assert _placement_worker_count(3) == 1


def test_scatter_does_not_reuse_cabin_mesh_for_other_solid_categories(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    cabin_mesh = tmp_path / "cabin.glb"
    trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(cabin_mesh, file_type="glb")
    plan = {
        "world_size_m": [10.0, 10.0],
        "regions": [{
            "id": "forest", "center": {"x": 0.0, "y": 0.0},
            "objects": [
                {"category": "cabins", "asset_type": "solid_object", "count": 1},
                {"category": "forest_rocks", "asset_type": "solid_object", "count": 1},
            ],
        }],
    }
    labels = np.zeros((32, 32), dtype=np.int16)
    height = np.zeros((32, 32), dtype=np.float32)
    sample = terrain_sampler(height, (10.0, 10.0))
    prototypes = [{
        "id": "cabins", "category": "cabins", "asset_type": "solid_object",
        "mesh": str(cabin_mesh), "source_image": "synthetic://cabin",
        "model": {"model_id": "test"},
    }]
    assets, diagnostics = scatter_environment(
        prototypes, plan, labels, height, sample, 42, 0.05, tmp_path / "placement"
    )
    assert [asset["category"] for asset in assets] == ["cabins"]
    skipped = [item for item in diagnostics if item.get("category") == "forest_rocks"]
    assert skipped and skipped[0]["placement"] == "skipped_no_validated_prototype"


def test_gltf_vertices_convert_y_up_to_internal_z_up():
    vertices = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert np.allclose(gltf_vertices_to_z_up(vertices), [[1.0, -3.0, 2.0], [-4.0, 6.0, 5.0]])


def test_model_lock_is_resolved_and_requires_headroom():
    lock = ModelLock(__import__("pathlib").Path(__file__).resolve().parents[1] / "models.lock.json")
    lock.require_resolved(list(lock.records))
    assert lock.total_size(list(lock.records)) == 167_644_432_977
    assert lock.records["planner"].model_id == "gpt-5.6-sol"


class _FakeMesh:
    def export(self, path):
        Path(path).write_text("shape", encoding="utf-8")


def test_hunyuan_texture_pipeline_uses_obj_and_generated_glb(tmp_path):
    image_path = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    calls = {}

    def shape(*, image, seed):
        calls["seed"] = seed
        calls["image_size"] = image.size
        return [_FakeMesh()]

    def paint(**kwargs):
        calls["paint"] = kwargs
        Path(kwargs["output_mesh_path"]).with_suffix(".glb").write_bytes(b"glb")

    shape_obj = tmp_path / "asset.shape.obj"
    textured_obj = tmp_path / "asset.textured.obj"
    result = generate_mesh(shape, paint, image_path, shape_obj, textured_obj, 42)

    assert result == tmp_path / "asset.textured.glb"
    assert result.read_bytes() == b"glb"
    assert calls["seed"] == 42
    assert calls["image_size"] == (8, 8)
    assert calls["paint"]["mesh_path"] == str(shape_obj)
    assert calls["paint"]["output_mesh_path"] == str(textured_obj)


def test_hunyuan_texture_pipeline_rejects_missing_glb(tmp_path):
    image_path = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    with pytest.raises(RuntimeError, match="did not produce a non-empty GLB"):
        generate_mesh(
            lambda **_kwargs: [_FakeMesh()],
            lambda **_kwargs: None,
            image_path,
            tmp_path / "asset.shape.obj",
            tmp_path / "asset.textured.obj",
            42,
        )


@pytest.mark.parametrize(
    ("category", "asset_type"),
    [("shoreline_vegetation", "procedural_native"),
     ("trail", "linear_structure"),
     ("lake", "surface_region_feature")],
)
def test_mesh_validation_builds_route_native_fallback(tmp_path, category, asset_type):
    pytest.importorskip("trimesh")
    asset = {
        "id": f"asset_{category}", "category": category,
        "asset_type": asset_type, "mesh": str(tmp_path / "bad.glb"),
    }
    fallback = _make_fallback_asset(asset, tmp_path / "fallbacks")
    assert fallback is not None
    assert fallback["fallback"].startswith("procedural-")
    metrics, defects = _mesh_metrics(Path(fallback["mesh"]), category, asset_type)
    assert metrics["faces"] >= 4
    assert "missing_volume" not in defects


def test_mesh_validation_report_keeps_failure_and_final_status_separate():
    report = _coerce_report({"decision": "reject", "issues": ["holes"], "summary": "broken"})
    assert report["status"] == "fail"
    report["final_status"] = "dropped"
    assert report["final_status"] == "dropped"


def test_mesh_validation_coerces_text_severity():
    report = _coerce_report({"status": "fail", "severity": "major", "defects": ["holes"]})
    assert report["severity"] == 0.75


def test_mesh_validation_coerces_null_optional_fields():
    report = _coerce_report({"status": "pass", "severity": 0, "defects": None, "retry_action": None})
    assert report["defects"] == []
    assert report["retry_action"] == ""


def test_mesh_validation_cache_key_contains_required_identity(tmp_path):
    mesh = tmp_path / "prototype.glb"
    mesh.write_bytes(b"prototype")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    mesh_hash = _mesh_sha256(mesh)
    reference_hash = _reference_sha256(reference)
    key = _validation_cache_key(mesh_hash, reference_hash, "validator-v1")
    assert key == _validation_cache_key(mesh_hash, reference_hash, "validator-v1")
    assert key != _validation_cache_key(mesh_hash, reference_hash, "validator-v2")
    assert key != _validation_cache_key(mesh_hash, "other", "validator-v1")


def test_mesh_validation_checks_duplicate_prototype_once(tmp_path, monkeypatch):
    mesh = tmp_path / "prototype.glb"
    mesh.write_bytes(b"prototype")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    reference_other = tmp_path / "reference_other.png"
    reference_other.write_bytes(b"different reference")
    response_path = tmp_path / "validation_response.json"
    request_path = tmp_path / "request.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = {
        "work_dir": str(tmp_path), "run_dir": str(run_dir), "source_stage": "environment_assets",
        "seed": 42, "mesh_retry_max": 0, "input_regen_max": 0,
    }
    upstream = {"status": "ok", "assets": [
        {"id": "tree_0001", "category": "tree", "asset_type": "reusable_prototype",
         "mesh": str(mesh), "source_image": str(reference)},
        {"id": "tree_0002", "category": "tree", "asset_type": "reusable_prototype",
         "mesh": str(mesh), "source_image": str(reference)},
        {"id": "tree_0003", "category": "tree", "asset_type": "reusable_prototype",
         "mesh": str(mesh), "source_image": str(reference_other)},
    ]}
    calls = {"metrics": 0, "render": 0, "inspect": 0}

    def fake_metrics(*_args):
        calls["metrics"] += 1
        return ({"vertices": 8, "faces": 4, "extents": [1, 1, 1], "volume_ratio": 1,
                 "watertight": True, "non_manifold_edges": 0, "topology_warnings": []}, [])

    def fake_render(_blender, _mesh, output):
        calls["render"] += 1
        output.mkdir(parents=True, exist_ok=True)
        views = []
        for index in range(8):
            view = output / f"view_{index:02d}.png"
            view.write_bytes(b"view")
            views.append(view)
        return views

    def fake_inspect(*_args):
        calls["inspect"] += 1
        return {"status": "pass", "severity": 0, "defects": [], "reason": "ok", "retry_action": ""}

    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(mesh_validation_worker, "worker_args", lambda: SimpleNamespace(request=request_path, response=response_path))
    monkeypatch.setattr(mesh_validation_worker, "load_stage_response", lambda *_args: upstream)
    monkeypatch.setattr(mesh_validation_worker, "_mesh_metrics", fake_metrics)
    monkeypatch.setattr(mesh_validation_worker, "_render", fake_render)
    monkeypatch.setattr(mesh_validation_worker, "_inspect", fake_inspect)
    mesh_validation_worker.main()

    result = json.loads(response_path.read_text(encoding="utf-8"))
    assert calls == {"metrics": 1, "render": 1, "inspect": 1}
    assert result["validation_cache"]["validation_calls"] == 1
    assert result["validation_cache"]["cache_hits"] == 1
    assert result["validation_cache"]["mesh_cache_hits"] == 1
    assert len(result["assets"]) == 3
    assert {item["asset_id"] for item in result["reports"]} == {"tree_0001", "tree_0002", "tree_0003"}


def test_mesh_validation_retry_reuses_identical_mesh_failure(tmp_path, monkeypatch):
    mesh = tmp_path / "failed.glb"
    mesh.write_bytes(b"failed prototype")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    response_path = tmp_path / "validation_response.json"
    request_path = tmp_path / "request.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    asset = {
        "id": "rock_0001", "category": "rock", "asset_type": "solid_object",
        "mesh": str(mesh), "source_image": str(reference),
    }
    request = {
        "work_dir": str(tmp_path), "run_dir": str(run_dir), "source_stage": "environment_assets",
        "seed": 42, "mesh_retry_max": 2, "input_regen_max": 0,
    }
    calls = {"inspect": 0, "retry": 0}

    def fake_metrics(*_args):
        return ({"vertices": 8, "faces": 4, "extents": [1, 1, 1], "volume_ratio": 1,
                 "watertight": True, "non_manifold_edges": 0, "topology_warnings": []}, [])

    def fake_render(_blender, _mesh, output):
        output.mkdir(parents=True, exist_ok=True)
        views = []
        for index in range(8):
            view = output / f"view_{index:02d}.png"
            view.write_bytes(b"view")
            views.append(view)
        return views

    def fake_inspect(*_args):
        calls["inspect"] += 1
        return {"status": "fail", "severity": 1, "defects": ["broken"],
                "reason": "broken", "retry_action": "reconstruct"}

    def fake_retry(*_args):
        calls["retry"] += 1
        return {"status": "ok", "assets": [asset]}

    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setenv("WORLDCLAW_IMAGE3D_WORKER", "dummy-worker")
    monkeypatch.setattr(mesh_validation_worker, "worker_args", lambda: SimpleNamespace(request=request_path, response=response_path))
    monkeypatch.setattr(mesh_validation_worker, "load_stage_response", lambda *_args: {"status": "ok", "assets": [asset]})
    monkeypatch.setattr(mesh_validation_worker, "_mesh_metrics", fake_metrics)
    monkeypatch.setattr(mesh_validation_worker, "_render", fake_render)
    monkeypatch.setattr(mesh_validation_worker, "_inspect", fake_inspect)
    monkeypatch.setattr(mesh_validation_worker, "_run_worker", fake_retry)
    monkeypatch.setattr(mesh_validation_worker, "_make_fallback_asset", lambda *_args: None)
    mesh_validation_worker.main()

    result = json.loads(response_path.read_text(encoding="utf-8"))
    assert calls == {"inspect": 1, "retry": 1}
    assert result["validation_cache"]["mesh_cache_hits"] == 1
    assert result["validation_cache"]["validation_calls"] == 1
    assert result["reports"][0]["final_status"] == "dropped"
