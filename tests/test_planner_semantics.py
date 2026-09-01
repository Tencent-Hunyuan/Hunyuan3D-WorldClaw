from worldclaw_oss.asset_semantics import normalize_object_semantics
from worldclaw_oss.layout import normalize_scene_plan, synthetic_plan
from worldclaw_oss.models import Planner, scene_plan_api_schema
from worldclaw_oss.prompts import INTENT_SYSTEM, SCENE_PLANNER_SYSTEM
from worldclaw_oss.schemas import ModelRecord


def test_compound_labels_split_into_entity_distribution_and_strategy():
    assert normalize_object_semantics("dense_trees") == {
        "category": "tree",
        "asset_role": "reusable_prototype",
        "instance_strategy": "scatter",
        "density": "dense",
    }
    assert normalize_object_semantics("rock_cluster") == {
        "category": "rock",
        "asset_role": "solid_object",
        "instance_strategy": "cluster",
        "density": None,
    }
    assert normalize_object_semantics("rock", asset_role="reusable_prototype")["asset_role"] == "solid_object"
    assert normalize_object_semantics("lush reeds") == {
        "category": "reed",
        "asset_role": "procedural_native",
        "instance_strategy": "cluster",
        "density": "dense",
    }
    lake = normalize_object_semantics("lake")
    assert lake["instance_strategy"] == "region"
    assert lake["density"] is None


def test_planner_contract_requests_normalized_semantics_from_model():
    calls = []

    class FakePlannerClient:
        def json_chat(self, _model, system, user, schema, _seed):
            calls.append((system, user, schema))
            if system == INTENT_SYSTEM:
                return {"constraints": ["forest lake"], "verbatim_prompt": "forest lake"}
            candidate = synthetic_plan("forest lake").model_dump(mode="json")
            candidate["regions"][1]["objects"] = [{
                # Simulate a model that followed the old compound vocabulary;
                # Planner's semantic normalization must repair it.
                "category": "dense_trees",
                "asset_role": "reusable_prototype",
                "count": 1400,
                "instance_strategy": "scatter",
                "density": "dense",
                "appearance": "mixed conifer forest",
            }]
            return candidate

    model = ModelRecord(
        model_id="gpt-test", revision="api", license="provider-api",
        size_bytes=0, purpose="test",
    )
    _intent, plan, repairs = Planner(FakePlannerClient(), model).plan("forest lake", 42)
    obj = plan.regions[1].objects[0]
    assert repairs == 0
    assert (obj.category, obj.asset_role.value, obj.instance_strategy, obj.density) == (
        "tree", "reusable_prototype", "scatter", "dense",
    )
    assert "category, count, instance_strategy, and asset_role are mandatory" in SCENE_PLANNER_SYSTEM
    assert "count is the operational number" in SCENE_PLANNER_SYSTEM
    assert "dense trees=1400" in SCENE_PLANNER_SYSTEM
    assert "A lake, shoreline region, or trail network" in SCENE_PLANNER_SYSTEM
    assert "one structural\n  feature" in SCENE_PLANNER_SYSTEM
    object_schema = scene_plan_api_schema()["properties"]["regions"]["items"]["properties"]["objects"]["items"]
    assert set(["category", "count", "asset_role", "instance_strategy"]).issubset(object_schema["required"])
    assert "density" not in object_schema["required"]


def test_live_plan_normalization_canonicalizes_model_aliases():
    plan = synthetic_plan("forest lake")
    region = plan.regions[1].model_copy(update={
        "objects": [plan.regions[1].objects[0].model_copy(update={
            "category": "dense_trees",
            "density": "dense",
        })],
    })
    normalized = normalize_scene_plan(plan.model_copy(update={"regions": [plan.regions[0], region, plan.regions[2]]}))
    obj = normalized.regions[1].objects[0]
    assert obj.category == "tree"
    assert obj.asset_role.value == "reusable_prototype"
    assert obj.instance_strategy == "scatter"
    assert obj.density == "dense"
