import json
from pathlib import Path

from worldclaw_oss.models import ModelLock
from worldclaw_oss.openai_api import OpenAIConfig


def test_openai_config_reads_codex_files_without_exposing_key(tmp_path, monkeypatch):
    auth = tmp_path / "auth.json"
    config = tmp_path / "config.toml"
    auth.write_text(json.dumps({"OPENAI_API_KEY": "fixture-key"}), encoding="utf-8")
    config.write_text(
        "model_provider = 'gateway'\n"
        "model = 'gpt-test'\n"
        "[model_providers.gateway]\n"
        "base_url = 'https://gateway.example'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("WORLDCLAW_OPENAI_AUTH_FILE", str(auth))
    monkeypatch.setenv("WORLDCLAW_OPENAI_CONFIG_FILE", str(config))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    value = OpenAIConfig.from_environment()
    assert value.base_url == "https://gateway.example/v1"
    assert value.text_model == "gpt-test"
    assert value.validation_model == "gpt-5.6-sol"
    assert value.image_via_responses is True
    assert "fixture-key" not in json.dumps(value.public_dict())


def test_openai_model_overlay_keeps_hunyuan_lock():
    root = Path(__file__).resolve().parents[1]
    lock = ModelLock(root / "models.lock.json")
    config = OpenAIConfig(
        api_key="secret",
        base_url="https://gateway.example/v1",
        text_model="gpt-test",
        vision_model="gpt-vision-test",
        image_model="gpt-image-test",
        validation_model="gpt-validation-test",
    )
    records = lock.openai_records(config)
    assert records["planner"].model_id == "gpt-5.6-sol"
    assert records["reference_image"].model_id == "gpt-image-2"
    assert records["vlm"].model_id == "gpt-validation-test"
    assert records["hunyuan3d"].model_id == lock.records["hunyuan3d"].model_id
