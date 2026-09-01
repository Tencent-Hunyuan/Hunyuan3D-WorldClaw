from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .prompts import ASSET_TYPE_CLASSIFIER_SYSTEM, INTENT_SYSTEM, REPAIR_SYSTEM, SCENE_PLANNER_SYSTEM
from .asset_semantics import infer_asset_type, is_structural_feature
from .asset_types import AssetType
from .schemas import ModelRecord, ScenePlan


FORCED_PLANNER_MODEL = "gpt-5.6-sol"


class ModelLock:
    def __init__(self,path: Path):
        raw=json.loads(Path(path).read_text(encoding="utf-8"))
        self.records={name:ModelRecord.model_validate(value) for name,value in raw["models"].items()}
        self.fallbacks=raw.get("fallbacks",{})
    def require_resolved(self,names: list[str]):
        bad = []
        for name in names:
            record = self.records.get(name)
            if record is None or not record.resolved:
                bad.append(name)
                continue
            if record.license == "provider-api" and record.revision == "api":
                continue
            if len(record.revision) != 40:
                bad.append(name)
        if bad: raise RuntimeError("unresolved model lock entries: "+", ".join(bad)+"; run scripts/resolve_model_lock.py on a host with Hugging Face access")
    def total_size(self,names: list[str]) -> int: return sum(self.records[n].size_bytes for n in names)

    def openai_records(self, config) -> dict[str, ModelRecord]:
        """Overlay API model names while retaining Hunyuan's pinned HF record."""
        records = dict(self.records)
        api_models = {
            # Planner and AssetTypeClassifier are deliberately pinned to the
            # configured GPT-5.6 Sol model; OPENAI_TEXT_MODEL cannot switch
            # them back to the old Qwen planner.
            "planner": (FORCED_PLANNER_MODEL, "intent extraction, scene planning, and asset routing"),
            "vlm": (config.validation_model, "mesh and render validation"),
            # A live run may intentionally route reference generation through
            # the local FLUX worker. In that mode preserve the pinned HF
            # records; otherwise the OpenAI overlay would turn them into a
            # gpt-* API alias that diffusers tries to fetch from Hugging Face.
            **({} if os.getenv("WORLDCLAW_LOCAL_FLUX", "0").lower() in {"1", "true", "yes"} else {
                "flux_dev": (config.image_model, "regional image composition compatibility alias"),
                "flux_schnell": (config.image_model, "image generation compatibility alias"),
            }),
            "reference_image": ("gpt-image-2", "environment asset reference images"),
            # Grounding DINO and SAM2 are local image-segmentation models in
            # the configured worker. Keep their pinned HF records even when
            # planner/VLM calls use the OpenAI-compatible API.
            "trellis": ("disabled", "unused because Hunyuan3D is retained"),
        }
        for name, (model_id, purpose) in api_models.items():
            if name not in records:
                continue
            records[name] = records[name].model_copy(update={
                "model_id": model_id,
                "revision": "api",
                "license": "provider-api",
                "size_bytes": 0,
                "purpose": purpose,
                "resolved": True,
                "requested_model_id": None,
                "resolution_reason": "OpenAI-compatible API configured at runtime",
            })
        return records


class VLLMClient:
    def __init__(self,base_url: str,api_key_env: str="VLLM_API_KEY",timeout: int=300):
        self.url=base_url.rstrip("/")+"/v1/chat/completions"; self.api_key=os.getenv(api_key_env,""); self.timeout=timeout
    def json_chat(self,model: ModelRecord,system: str,user: str,schema: dict[str,Any],seed: int) -> dict[str,Any]:
        payload={"model":model.model_id,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":0.0,"seed":seed,"response_format":{"type":"json_schema","json_schema":{"name":"response","strict":True,"schema":schema}}}
        request=urllib.request.Request(self.url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Authorization":f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request,timeout=self.timeout) as response: data=json.load(response)
        except urllib.error.URLError as exc: raise RuntimeError(f"vLLM request failed: {exc}") from exc
        return json.loads(data["choices"][0]["message"]["content"])


class OpenAIJSONClient:
    """Planner adapter with the same method contract as ``VLLMClient``."""

    def __init__(self, client=None):
        if client is None:
            from .openai_api import OpenAIClient
            client = OpenAIClient()
        self.client = client

    def json_chat(self, model, system: str, user: str, schema: dict[str, Any], seed: int):
        del seed  # API generation is recorded but not guaranteed deterministic.
        return self.client.json_response(
            model=model.model_id,
            system=system,
            user=user,
            schema=schema,
        )


class IntentPayload(BaseModel):
    constraints: list[str]
    verbatim_prompt: str


def scene_plan_api_schema() -> dict[str, Any]:
    """A flat strict-schema subset accepted by OpenAI-compatible gateways.

    Pydantic remains the authoritative validator for bounds and cross-field
    invariants after the response is decoded.
    """
    vec2 = {
        "type": "object", "additionalProperties": False,
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["x", "y"],
    }
    object_spec = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "category": {"type": "string"}, "count": {"type": "integer"},
            "asset_role": {"type": "string", "enum": [item.value for item in AssetType]},
            "instance_strategy": {
                "type": "string", "enum": ["single", "scatter", "cluster", "path", "region"],
            },
            "density": {"type": "string", "enum": ["sparse", "medium", "dense"]},
            "appearance": {"type": "string"},
            "placement_profile": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "requires_dry_support": {"type": "boolean"},
                    "avoid_structural_exclusion": {"type": "boolean"},
                    "required_support_surfaces": {"type": "array", "items": {"type": "string"}},
                    "forbidden_support_surfaces": {"type": "array", "items": {"type": "string"}},
                    "allowed_support_surfaces": {"type": "array", "items": {"type": "string"}},
                    "distance_preferences": {"type": "object", "additionalProperties": {"type": "number"}},
                    "footprint_overlap_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.1},
                },
            },
        },
        "required": [
            "category", "count", "asset_role", "instance_strategy",
        ],
    }
    region = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"}, "function": {"type": "string"},
            "center": vec2,
            "polygon": {
                "type": "object", "additionalProperties": False,
                "properties": {"points": {"type": "array", "items": vec2}},
                "required": ["points"],
            },
            "coverage": {"type": "number"},
            "neighbors": {"type": "array", "items": {"type": "string"}},
            "objects": {"type": "array", "items": object_spec},
            "spatial_relations": {"type": "array", "items": {"type": "string"}},
            "appearance": {"type": "string"}, "camera_hint": {"type": "string"},
        },
        "required": [
            "id", "function", "center", "polygon", "coverage", "neighbors",
            "objects", "spatial_relations", "appearance", "camera_hint",
        ],
    }
    terrain_operator = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": ["peak", "dune", "terrace", "erosion"]},
            "strength": {"type": "number"}, "scale": {"type": "number"},
        },
        "required": ["kind", "strength", "scale"],
    }
    terrain = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "region_id": {"type": "string"}, "mask_key": {"type": "string"},
            "base_height": {"type": "number"},
            "noise_octaves": {"type": "array", "items": {"type": "number"}},
            "operators": {"type": "array", "items": terrain_operator},
            "boundary_blend": {"type": "number"},
        },
        "required": [
            "region_id", "mask_key", "base_height", "noise_octaves", "operators",
            "boundary_blend",
        ],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "theme": {"type": "string"},
            "world_size_m": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
            "regions": {"type": "array", "items": region},
            "terrain": {"type": "array", "items": terrain},
            "materials": {"type": "object", "additionalProperties": {"type": "string"}},
            "explicit_constraints": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["theme", "world_size_m", "regions", "terrain", "materials", "explicit_constraints"],
    }


def asset_type_api_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "object_id": {"type": "string"},
                        "asset_type": {"type": "string", "enum": [item.value for item in AssetType]},
                    },
                    "required": ["object_id", "asset_type"],
                },
            },
        },
        "required": ["decisions"],
    }


class AssetTypeClassifier:
    """Use the configured GPT planner model to route each planned object."""

    def __init__(self, client: VLLMClient | OpenAIJSONClient, model: ModelRecord):
        self.client = client
        self.model = model

    def classify(self, plan: ScenePlan, seed: int) -> tuple[ScenePlan, list[dict[str, str]]]:
        objects: list[dict[str, str]] = []
        for region in plan.regions:
            for index, obj in enumerate(region.objects):
                objects.append({
                    "object_id": f"{region.id}:{index}",
                    "region_id": region.id,
                    "category": obj.category,
                    "asset_role": obj.asset_role.value if obj.asset_role else None,
                    "instance_strategy": obj.instance_strategy,
                    "density": obj.density,
                    "appearance": obj.appearance or region.appearance,
                })
        if not objects:
            return plan, []
        request = json.dumps({"objects": objects}, ensure_ascii=False)
        response = self.client.json_chat(
            self.model, ASSET_TYPE_CLASSIFIER_SYSTEM, request,
            asset_type_api_schema(), seed,
        )
        decisions = response.get("decisions") if isinstance(response, dict) else None
        if not isinstance(decisions, list):
            raise ValueError("asset type classifier returned no decisions")
        expected = {item["object_id"] for item in objects}
        by_id: dict[str, AssetType] = {}
        for item in decisions:
            if not isinstance(item, dict) or item.get("object_id") not in expected:
                raise ValueError("asset type classifier returned an unknown object_id")
            object_id = item["object_id"]
            if object_id in by_id:
                raise ValueError(f"asset type classifier returned duplicate {object_id}")
            by_id[object_id] = AssetType(item["asset_type"])
        missing = expected - set(by_id)
        if missing:
            raise ValueError(f"asset type classifier omitted objects: {sorted(missing)}")
        regions = []
        normalized: list[dict[str, str]] = []
        for region in plan.regions:
            region_objects = []
            for index, obj in enumerate(region.objects):
                object_id = f"{region.id}:{index}"
                asset_type = by_id[object_id]
                # Category semantics are authoritative for reusable trees. A
                # classifier mistake would route them to card geometry and
                # permit unrelated vegetation meshes to be shared later.
                if infer_asset_type(obj.category) == AssetType.REUSABLE_PROTOTYPE:
                    asset_type = AssetType.REUSABLE_PROTOTYPE
                if is_structural_feature(obj.category, obj.asset_role):
                    asset_type = AssetType.STRUCTURAL_FEATURE
                region_objects.append(obj.model_copy(update={"asset_role": asset_type}))
                normalized.append({"object_id": object_id, "category": obj.category, "asset_type": asset_type.value})
            regions.append(region.model_copy(update={"objects": region_objects}))
        return plan.model_copy(update={"regions": regions}), normalized


class Planner:
    def __init__(self,client: VLLMClient,model: ModelRecord): self.client=client; self.model=model
    def plan(self,prompt: str,seed: int) -> tuple[IntentPayload,ScenePlan,int]:
        intent=IntentPayload.model_validate(self.client.json_chat(self.model,INTENT_SYSTEM,prompt,IntentPayload.model_json_schema(),seed))
        api_schema = scene_plan_api_schema()
        request=json.dumps({"intent":intent.model_dump(),"schema":api_schema},ensure_ascii=False)
        candidate=self.client.json_chat(self.model,SCENE_PLANNER_SYSTEM,request,api_schema,seed)
        for repair in range(4):
            try:
                # Normalize only object semantics here. World-size scaling is
                # owned by the pipeline so authored dimensions remain auditable.
                from .layout import normalize_plan_semantics
                return intent, normalize_plan_semantics(ScenePlan.model_validate(candidate)), repair
            except ValidationError as exc:
                if repair == 3: raise RuntimeError(f"ScenePlan failed validation after 3 repairs: {exc}") from exc
                # Cross-field coverage errors are deterministic and the model
                # occasionally repeats the same unnormalised values on every
                # repair attempt.  Normalise only this specific, recoverable
                # case before asking the model for the next repair so the
                # candidate still goes through the strict Pydantic checks.
                coverage_error = any(
                    "region coverage must sum" in str(error.get("msg", ""))
                    for error in exc.errors(include_url=False)
                )
                regions = candidate.get("regions") if isinstance(candidate, dict) else None
                if coverage_error and isinstance(regions, list) and regions:
                    values = [item.get("coverage") for item in regions if isinstance(item, dict)]
                    total = sum(value for value in values if isinstance(value, (int, float)))
                    if len(values) == len(regions) and total > 0 and all(value > 0 for value in values):
                        for item in regions:
                            item["coverage"] = round(float(item["coverage"]) / total, 8)
                # Pydantic may place a native ValueError in ``ctx`` for model-level
                # validators.  Normalize those details before sending the repair
                # request to vLLM so one malformed candidate still gets all three
                # documented repair attempts.
                repair_request=json.dumps(
                    {"candidate": candidate, "errors": exc.errors(include_url=False),
                     "schema": api_schema},
                    ensure_ascii=False, default=str,
                )
                candidate=self.client.json_chat(self.model,REPAIR_SYSTEM,repair_request,api_schema,seed)
        raise AssertionError("unreachable")


class CommandWorker:
    """Strict file-contract boundary for isolated model environments."""
    def __init__(self,env_name: str): self.env_name=env_name
    def run(self,request: dict[str,Any],request_path: Path,response_path: Path,timeout: int=7200) -> dict[str,Any]:
        started = time.monotonic()
        command=os.getenv(self.env_name)
        metadata = {
            "environment": self.env_name,
            "gpu": os.getenv("CUDA_VISIBLE_DEVICES", "") or self.command_gpu(command or ""),
        }
        if not command:
            error = f"{self.env_name} is not configured"
            self._write_error(response_path, error, metadata, started)
            raise RuntimeError(error)
        request_path.write_text(json.dumps(request,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        args=shlex.split(command,posix=os.name!="nt")+["--request",str(request_path),"--response",str(response_path)]
        try:
            completed=subprocess.run(args,check=False,timeout=timeout,text=True,capture_output=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._write_error(response_path, error, metadata, started)
            raise RuntimeError(error) from exc
        if completed.returncode != 0:
            error = f"worker failed ({completed.returncode}): {completed.stderr[-4000:]}"
            self._write_error(response_path, error, metadata, started)
            raise RuntimeError(error)
        if not response_path.is_file():
            error = "worker did not create response JSON"
            self._write_error(response_path, error, metadata, started)
            raise RuntimeError(error)
        response=json.loads(response_path.read_text(encoding="utf-8"))
        response["worker"] = metadata | {
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_hashes": self._output_hashes(response),
        }
        response_path.write_text(json.dumps(response, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        if response.get("status") != "ok": raise RuntimeError(response.get("error","worker returned non-ok status"))
        return response

    @staticmethod
    def command_gpu(command: str) -> str:
        """Extract an inline CUDA_VISIBLE_DEVICES assignment from a worker command."""
        for token in shlex.split(command, posix=os.name != "nt"):
            if token.startswith("CUDA_VISIBLE_DEVICES="):
                return token.split("=", 1)[1]
        return ""

    @staticmethod
    def _write_error(path: Path, error: str, metadata: dict[str, str], started: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "status": "error", "error": error,
            "worker": metadata | {"duration_seconds": round(time.monotonic() - started, 3), "output_hashes": {}},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _output_hashes(value: Any) -> dict[str, str]:
        hashes: dict[str, str] = {}
        def visit(item: Any) -> None:
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)
            elif isinstance(item, str):
                # Worker responses also contain prompts and other arbitrary
                # strings.  Treat only valid, existing paths as artifacts;
                # long or malformed strings must never make a successful
                # worker fail during metadata collection.
                try:
                    candidate = Path(item)
                    is_file = candidate.is_file()
                except (OSError, ValueError):
                    return
                if is_file:
                    digest = hashlib.sha256()
                    with candidate.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    hashes[str(candidate)] = digest.hexdigest()
        visit(value)
        return hashes


def image_data_url(path: Path) -> str:
    mime="image/png" if path.suffix.lower()==".png" else "image/jpeg"
    return f"data:{mime};base64,"+base64.b64encode(path.read_bytes()).decode()
