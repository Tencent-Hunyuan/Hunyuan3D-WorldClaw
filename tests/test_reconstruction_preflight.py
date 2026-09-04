import json
from types import SimpleNamespace

from PIL import Image

from workers import reconstruction_preflight_worker as preflight


def _run(monkeypatch, request_path, response_path):
    monkeypatch.setenv("WORLDCLAW_RECON_PREFLIGHT_LOCAL_ONLY", "1")
    monkeypatch.setattr(
        preflight,
        "worker_args",
        lambda: SimpleNamespace(request=request_path, response=response_path),
    )
    preflight.main()
    return json.loads(response_path.read_text(encoding="utf-8"))


def test_environment_preflight_reroutes_non_object_inputs(tmp_path, monkeypatch):
    image = tmp_path / "tree.png"
    Image.new("RGB", (64, 64), (30, 90, 40)).save(image)
    references = tmp_path / "environment_references_response.json"
    references.write_text(json.dumps({
        "status": "ok",
        "images": [
            {"id": "tree", "category": "dense_trees", "asset_type": "reusable_prototype", "path": str(image)},
            {"id": "trail", "category": "winding_trails", "asset_type": "linear_structure", "path": str(image)},
        ],
    }), encoding="utf-8")
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(json.dumps({
        "work_dir": str(tmp_path), "source_stage": "environment_assets",
        "models": {"vlm": {"model_id": "gpt-5.6-sol"}},
    }), encoding="utf-8")

    result = _run(monkeypatch, request, response)

    assert [item["id"] for item in result["images"]] == ["tree"]
    assert result["dropped_ids"] == ["trail"]
    assert result["vlm"]["status"] == "skipped"


def test_tree_route_cannot_be_changed_to_vegetation_card(tmp_path, monkeypatch):
    image = tmp_path / "tree.png"
    Image.new("RGB", (64, 64), (30, 90, 40)).save(image)
    references = tmp_path / "environment_references_response.json"
    references.write_text(json.dumps({
        "status": "ok",
        "images": [{
            "id": "tree", "category": "dense_trees",
            "asset_type": "reusable_prototype", "path": str(image),
        }],
    }), encoding="utf-8")
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(json.dumps({
        "work_dir": str(tmp_path), "source_stage": "environment_assets",
        "models": {"vlm": {"model_id": "gpt-5.6-sol"}},
    }), encoding="utf-8")
    monkeypatch.setenv("WORLDCLAW_RECON_PREFLIGHT_LOCAL_ONLY", "0")
    monkeypatch.setattr(
        preflight,
        "_vlm_review",
        lambda *_args: ({"tree": {
            "action": "reroute", "representation": "vegetation_card",
            "reason": "use a card", "confidence": 0.99,
        }}, {"provider": "openai", "status": "ok"}),
    )
    monkeypatch.setattr(
        preflight,
        "worker_args",
        lambda: SimpleNamespace(request=request, response=response),
    )
    preflight.main()
    result = json.loads(response.read_text(encoding="utf-8"))
    assert result["images"][0]["asset_type"] == "reusable_prototype"
    assert result["decisions"][0]["action"] == "reconstruct"


def test_reconstruction_preflight_recrops_invalid_bbox(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    crop = tmp_path / "crop.png"
    Image.new("RGBA", (100, 80), (80, 120, 40, 255)).save(source)
    mask_image = Image.new("L", (100, 80), 0)
    for x in range(20, 70):
        for y in range(10, 70):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask)
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(crop)
    segmentation = tmp_path / "segmentation_response.json"
    segmentation.write_text(json.dumps({
        "status": "ok",
        "instances": [{
            "id": "instance_1", "category": "cabin", "asset_type": "solid_object",
            "source_image": str(source), "mask": str(mask), "crop": str(crop),
            "bbox_xyxy": [0, 0, 1, 1],
        }],
    }), encoding="utf-8")
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(json.dumps({
        "work_dir": str(tmp_path), "source_stage": "reconstruction",
        "models": {"vlm": {"model_id": "gpt-5.6-sol"}},
    }), encoding="utf-8")

    result = _run(monkeypatch, request, response)

    item = result["instances"][0]
    assert result["decisions"][0]["action"] == "recrop"
    assert result["resegment_required"] is False
    assert item["bbox_xyxy"] == [20, 10, 70, 70]
    assert Image.open(crop).size == (50, 60)
