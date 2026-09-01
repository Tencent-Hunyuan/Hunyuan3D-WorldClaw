from worldclaw_oss.asset_semantics import (
    effective_asset_type,
    infer_asset_type,
    is_spatial_structure,
    reference_subject,
    semantic_scale_range,
)
from worldclaw_oss.layout import synthetic_plan
from worldclaw_oss.models import AssetTypeClassifier
from worldclaw_oss.schemas import ModelRecord
from worldclaw_oss.pipeline import (
    REGION_COMPOSITION_WORKER_ENV,
    REFERENCE_IMAGE_WORKER_ENV,
    environment_image_requests,
)
from worldclaw_oss.schemas import ObjectSpec
from worldclaw_oss.asset_types import AssetType


def test_reference_prompts_describe_single_reusable_instances():
    assert "single mature tree" in reference_subject("dense_trees")
    assert "single natural rock" in reference_subject("rocks")
    assert "single reed" in reference_subject("water_reeds")
    assert is_spatial_structure("winding_trails")
    assert not is_spatial_structure("dense_trees")


def test_environment_references_preserve_region_appearance_and_skip_paths():
    plan = synthetic_plan("forest lake")
    first = plan.regions[0].model_copy(update={
        "objects": [ObjectSpec(category="dense_trees", count=100, density=1.0, appearance="wet conifers")],
        "appearance": "lake edge",
    })
    second = plan.regions[1].model_copy(update={
        "objects": [ObjectSpec(category="dense_trees", count=100, density=1.0, appearance="dry conifers"),
                    ObjectSpec(category="winding_trails", count=1, density=0.1, appearance="dirt")],
        "appearance": "forest interior",
    })
    third = plan.regions[2]
    plan = plan.model_copy(update={"regions": [first, second, third]})
    requests = environment_image_requests(plan)
    assert len(requests) == 3
    assert {item["region_id"] for item in requests} >= {first.id, second.id}
    tree_requests = [item for item in requests if item["category"] == "dense_trees"]
    assert len(tree_requests) == 2
    assert all("single mature tree" in item["prompt"] for item in tree_requests)
    assert all("no cluster" in item["prompt"] for item in requests)
    assert not any(item["category"] == "winding_trails" for item in requests)


def test_semantic_scales_restore_physical_category_size():
    assert semantic_scale_range("dense_trees") == (3.0, 8.0)
    assert semantic_scale_range("cabins") == (4.0, 8.0)
    assert semantic_scale_range("shoreline_rocks") == (0.6, 2.0)


def test_asset_type_routes_match_recommended_workflow():
    assert infer_asset_type("cabin") == AssetType.SOLID_OBJECT
    assert infer_asset_type("pine") == AssetType.REUSABLE_PROTOTYPE
    assert infer_asset_type("shoreline_vegetation") == AssetType.PROCEDURAL_NATIVE
    assert infer_asset_type("pebbles") == AssetType.SCATTER_DETAIL
    assert infer_asset_type("river") == AssetType.LINEAR_STRUCTURE
    assert infer_asset_type("lake") == AssetType.SURFACE_REGION_FEATURE
    assert effective_asset_type("unknown", "bad-value") == AssetType.SOLID_OBJECT


def test_asset_type_reads_legacy_route_values_as_canonical():
    assert AssetType("tree_prototype") is AssetType.REUSABLE_PROTOTYPE
    assert AssetType("vegetation_cluster") is AssetType.PROCEDURAL_NATIVE
    assert ObjectSpec(category="tree", count=1, asset_role="tree_prototype").asset_role is AssetType.REUSABLE_PROTOTYPE
    assert ObjectSpec(category="reed", count=1, asset_role="vegetation_cluster").asset_role is AssetType.PROCEDURAL_NATIVE


def test_reference_requests_include_asset_route_and_skip_non_objects():
    plan = synthetic_plan("forest lake")
    region = plan.regions[0].model_copy(update={
        "objects": [
            ObjectSpec(category="shoreline_vegetation", count=2, density=1.0),
            ObjectSpec(category="lake", count=1, density=1.0),
        ],
    })
    requests = environment_image_requests(plan.model_copy(update={"regions": [region, *plan.regions[1:]]}))
    vegetation = [item for item in requests if item["category"] == "shoreline_vegetation"]
    assert len(vegetation) == 1
    assert vegetation[0]["asset_type"] == AssetType.PROCEDURAL_NATIVE.value
    assert not any(item["category"] == "lake" for item in requests)


def test_reference_and_regional_image_workers_are_separate():
    assert REFERENCE_IMAGE_WORKER_ENV == "WORLDCLAW_REFERENCE_IMAGE_WORKER"
    assert REGION_COMPOSITION_WORKER_ENV == "WORLDCLAW_FLUX_WORKER"


def test_asset_classifier_preserves_tree_semantics_on_bad_model_route():
    class FakeClient:
        def json_chat(self, _model, _system, _user, _schema, _seed):
            return {
                "decisions": [
                    {"object_id": "forest:0", "asset_type": "procedural_native"},
                    {"object_id": "cabins:0", "asset_type": "solid_object"},
                ]
            }

    plan = synthetic_plan("forest lake")
    plan = plan.model_copy(update={
        "regions": [
            plan.regions[0].model_copy(update={"objects": []}),
            plan.regions[1].model_copy(update={"objects": [plan.regions[1].objects[0]]}),
            plan.regions[2].model_copy(update={"objects": [plan.regions[2].objects[0]]}),
        ]
    })
    model = ModelRecord(
        model_id="gpt-test", revision="api", license="provider-api",
        size_bytes=0, purpose="test",
    )
    routed, _ = AssetTypeClassifier(FakeClient(), model).classify(plan, 42)
    assert routed.regions[1].objects[0].asset_type == AssetType.REUSABLE_PROTOTYPE
