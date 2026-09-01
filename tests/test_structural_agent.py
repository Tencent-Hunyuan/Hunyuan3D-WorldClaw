import json

import numpy as np

from worldclaw_oss.structural import StructuralFeatureAgent


class FakeClient:
    def __init__(self):
        self.calls = []

    def json_chat(self, model, system, user, schema, seed):
        self.calls.append({"model": model, "system": system, "user": json.loads(user), "schema": schema, "seed": seed})
        return {
            "status": "ok",
            "features": [
                {
                    "feature_id": "forest_trail_000",
                    "region_id": "forest",
                    "semantic_category": "trail",
                    "generation_class": "structural_feature",
                    "structural_subtype": "linear",
                    "representation": "terrain_conforming_swept_strip",
                    "instance_strategy": "world_integrated",
                    "geometry": {
                        "width_m": 3.0, "thickness_m": None, "depth_m": None,
                        "bank_width_m": None, "terrain_following": True,
                        "meander": 0.2, "water_surface": None,
                        "terrain_operation": None, "shape": None, "region_mask": None,
                    },
                    "region_relation": {"region_id": "forest", "function": "woods"},
                    "importance": "normal",
                },
            ],
        }


def test_structural_agent_calls_api_and_preserves_defaults(tmp_path):
    plan = {
        "world_size_m": [100.0, 100.0],
        "regions": [{
            "id": "forest", "function": "woods", "center": {"x": 50, "y": 50},
            "polygon": {"points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}]},
            "coverage": 1.0, "neighbors": [], "appearance": "woods",
            "camera_hint": "wide", "spatial_relations": [],
            "objects": [{"category": "trail", "count": 1, "asset_role": "structural_feature", "instance_strategy": "path"}],
        }],
    }
    client = FakeClient()
    result = StructuralFeatureAgent(client, object()).plan(
        plan, layout=np.zeros((8, 8), dtype=np.int16), terrain=np.ones((8, 8)),
        world_size=(100.0, 100.0), request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json", seed=42,
    )
    assert len(client.calls) == 1
    assert client.calls[0]["seed"] == 42
    assert result["provider"] == "openai-compatible"
    assert result["features"][0]["geometry"]["thickness_m"] == 0.08
    assert (tmp_path / "request.json").is_file()
    assert (tmp_path / "response.json").is_file()
