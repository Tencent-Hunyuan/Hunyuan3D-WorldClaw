import pytest

from worldclaw_oss.layout import synthetic_plan
from worldclaw_oss.schemas import ScenePlan


def test_scene_plan_round_trip():
    plan = synthetic_plan("castle")
    assert ScenePlan.model_validate_json(plan.model_dump_json()) == plan
    assert len(plan.regions) == 3


def test_scene_plan_rejects_unknown_neighbor():
    raw = synthetic_plan("castle").model_dump()
    raw["regions"][0]["neighbors"] = ["missing"]
    with pytest.raises(ValueError, match="invalid neighbors"):
        ScenePlan.model_validate(raw)


def test_scene_plan_rejects_bad_coverage():
    raw = synthetic_plan("forest lake").model_dump()
    raw["regions"][0]["coverage"] = 0.1
    with pytest.raises(ValueError, match="coverage"):
        ScenePlan.model_validate(raw)

