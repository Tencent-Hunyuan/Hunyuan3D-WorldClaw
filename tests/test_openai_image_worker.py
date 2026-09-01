import json
import sys
from pathlib import Path
from types import SimpleNamespace

from worldclaw_oss.openai_api import OpenAIConfig
from workers import openai_image_worker


def test_reference_worker_forces_gpt_image_2(monkeypatch, tmp_path):
    base_config = OpenAIConfig(
        api_key="fixture",
        base_url="https://gateway.example/v1",
        text_model="gpt-5.6-sol",
        vision_model="gpt-5.6-sol",
        image_model="configured-but-ignored",
        image_via_responses=True,
    )
    captured = {}

    monkeypatch.setattr(
        openai_image_worker,
        "OpenAIConfig",
        SimpleNamespace(from_environment=lambda: base_config),
    )

    class FakeClient:
        def __init__(self, config):
            captured["config"] = config
            self.config = config

        def generate_image(self, *, prompt, output_path, conditioning_image=None):
            del prompt, conditioning_image
            output_path.write_bytes(b"PNG")
            return {"path": str(output_path), "model": self.config.image_model}

    monkeypatch.setattr(openai_image_worker, "OpenAIClient", FakeClient)
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text(json.dumps({
        "stage": "environment_references",
        "work_dir": str(tmp_path),
        "image_requests": [{"id": "reference_a", "prompt": "one tree"}],
    }), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["openai_image_worker", "--request", str(request_path), "--response", str(response_path)]
    )

    openai_image_worker.main()

    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert captured["config"].image_model == "gpt-image-2"
    assert captured["config"].image_via_responses is False
    assert response["model"] == "gpt-image-2"
    assert response["images"][0]["model"] == "gpt-image-2"
    assert Path(response["images"][0]["path"]).read_bytes() == b"PNG"

