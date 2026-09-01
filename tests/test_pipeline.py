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

