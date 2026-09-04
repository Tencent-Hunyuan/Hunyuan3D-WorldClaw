import json

from worldclaw_oss import hunyuan_pool
from worldclaw_oss.hunyuan_pool import _command_parts, _source_ids


def test_hunyuan_command_parts_extracts_inline_environment():
    args, env = _command_parts(
        "env LD_LIBRARY_PATH=/opt/cuda CUDA_VISIBLE_DEVICES=4 python /srv/hunyuan3d_worker.py"
    )
    assert args == ["python", "/srv/hunyuan3d_worker.py"]
    assert env == {"LD_LIBRARY_PATH": "/opt/cuda", "CUDA_VISIBLE_DEVICES": "4"}


def test_hunyuan_pool_reads_source_ids_for_sharding(tmp_path):
    (tmp_path / "environment_references_response.json").write_text(
        json.dumps({"images": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}),
        encoding="utf-8",
    )
    request = {"stage": "environment_assets", "work_dir": str(tmp_path)}
    assert _source_ids(request) == ["a", "b", "c"]
    assert _source_ids(request | {"reference_override": [{"id": "retry_a"}]}) == ["retry_a"]


def test_hunyuan_pool_reads_refinement_assets_without_model_import(tmp_path):
    request = {
        "stage": "refinement_reconstruction", "work_dir": str(tmp_path),
        "refinement_assets": [{"id": "asset_1"}, {"id": "asset_2"}],
    }
    assert _source_ids(request) == ["asset_1", "asset_2"]


def test_explicit_hunyuan_gpu_list_is_not_auto_expanded(monkeypatch):
    monkeypatch.setattr(hunyuan_pool.os, "name", "posix")
    monkeypatch.setenv("WORLDCLAW_IMAGE3D_WORKER", "python /srv/hunyuan3d_worker.py")
    monkeypatch.setenv("WORLDCLAW_HUNYUAN_GPUS", "7")
    monkeypatch.delenv("WORLDCLAW_HUNYUAN_AUTO_EXPAND", raising=False)
    monkeypatch.setattr(hunyuan_pool, "_worker_script_exists", lambda command: True)
    monkeypatch.setattr(hunyuan_pool, "_discover_free_gpus", lambda base: ["7", "1", "2"])
    monkeypatch.setattr(hunyuan_pool.HunyuanDaemonPool, "ensure_started", lambda self: None)
    pool = hunyuan_pool.HunyuanDaemonPool.from_environment()
    assert pool is not None
    assert list(pool.gpu_ids) == ["7"]
    hunyuan_pool.HunyuanDaemonPool._instances.clear()
