from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cache import ContentCache, cache_key
from .asset_semantics import (
    appearance_key, effective_asset_type, infer_asset_type, is_spatial_structure,
    is_structural_feature, needs_reference_asset, reference_subject,
)
from .export import box_mesh, layout_preview, write_glb
from .hunyuan_pool import HunyuanDaemonPool
from .layout import (
    deterministic_layout,
    normalize_scene_plan,
    sample_layout,
    stable_seed,
    synthetic_plan,
    target_world_size_from_environment,
)
from .models import AssetTypeClassifier, CommandWorker, ModelLock, OpenAIJSONClient, Planner, VLLMClient
from .placement_constraints import apply_hard_gate, prepare_surface_masks, requirements_from_spec
from .schemas import AssetInstance, RunManifest, ScenePlan, Stage, TerrainMacroPlan
from .state import StateDB
from .structural import (
    StructuralFeatureAgent, build_structural_geometry, structural_features_from_plan, validate_structural_branch,
)
from .structural_workflow import (
    build_validation_bundle, plan_validation_views, prepare_structural_agent_input,
    render_additional_views, render_structural_views, visual_validate_bundle,
)
from .terrain import boundary_discontinuity, generate_terrain, validate_regional_detail_preserves_macro
from .terrain_macro import derive_macro_plan, generate_macro_height, macro_vertices, validate_macro_height, write_validation_bundle, TerrainMacroPlanner, visual_validate_terrain, replan_macro_plan, summarize_layout
from .validation import sha256, validate_run


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _terrain_planner_context(work: Path, scene: ScenePlan, layout_summary: dict[str, Any]) -> dict[str, Any]:
    """Load the auditable inputs used by the first Terrain Planner request."""
    root = work / "terrain_planner_input"
    context: dict[str, Any] = {"layout_summary": layout_summary}
    for name in ("intent.json", "scene_plan.json", "world_spec.json", "terrain_constraints.json", "structural_intents.json"):
        path = root / name
        if path.is_file():
            try:
                context[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                context[name.removesuffix(".json")] = {"path": str(path), "invalid_json": True}
    # Preserve the exact first planner exchange when this helper is called
    # during REPLAN.  The input snapshots above are the source materials, but
    # these records also capture the serialized request and provider response
    # that produced the current Macro Plan.
    for name in ("terrain_macro_planner_request.json", "terrain_macro_planner_response.json"):
        path = work / name
        if path.is_file():
            try:
                context[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                context[name.removesuffix(".json")] = {"path": str(path), "invalid_json": True}
    # A standalone Terrain validation response is an explicit contract for the
    # next planner call. Keep the payload and relative path so REPLAN can mark
    # every reported failure/recommendation as a hard constraint.
    response_constraints: list[dict[str, Any]] = []
    for path in sorted(work.rglob("*response.json")):
        relative = path.relative_to(work)
        if (
            "terrain" not in str(relative).lower()
            or path.name not in {"response.json", "terrain_visual_validation_response.json"}
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"path": str(path), "invalid_json": True}
        response_constraints.append({"path": str(relative), "payload": payload})
    if response_constraints:
        context["response_json_hard_constraints"] = response_constraints
    arrays: dict[str, Any] = {}
    for path in sorted((root / "layout").glob("*.npy")):
        try:
            array = np.load(path, mmap_mode="r")
            arrays[path.name] = {"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype)}
        except (OSError, ValueError):
            arrays[path.name] = {"path": str(path), "unreadable": True}
    context["layout_arrays"] = arrays
    masks: dict[str, Any] = {}
    for path in sorted((root / "layout" / "masks").glob("*.npy")):
        try:
            array = np.load(path, mmap_mode="r")
            masks[path.stem] = {"path": str(path), "shape": list(array.shape), "cells": int(np.asarray(array).sum())}
        except (OSError, ValueError):
            masks[path.stem] = {"path": str(path), "unreadable": True}
    context["region_masks"] = masks
    return context


def _write_terrain_blend(work: Path) -> dict[str, str]:
    """Persist an inspectable Terrain-only Blend beside terrain.npz.

    Final ``scene.blend`` is still produced by the export/finalization stage.
    This lightweight artifact is useful when a run stops before assets exist;
    missing Blender is recorded as a skip so synthetic control-host runs stay
    executable.
    """
    terrain_root = work / "terrain"
    terrain_root.mkdir(exist_ok=True)
    source = terrain_root / "terrain.npz"
    metadata_path = terrain_root / "terrain_blend_metadata.json"
    output = terrain_root / "terrain.blend"
    preview = terrain_root / "terrain_preview.png"
    blender = Path(os.getenv(
        "BLENDER_BIN",
        str(Path.home() / "apps" / "blender-4.2.0-linux-x64" / "blender"),
    ))
    if not source.is_file():
        raise FileNotFoundError(f"terrain artifact is missing: {source}")
    if not blender.is_file():
        result = {
            "status": "skipped",
            "reason": "Blender executable is unavailable",
            "source_npz": str(source),
            "blend": str(output),
            "preview": str(preview),
        }
        _json(metadata_path, result)
        return result
    script = Path(__file__).resolve().parents[1] / "scripts" / "blender_terrain_preview.py"
    subprocess.run(
        [
            str(blender), "--background", "--python", str(script), "--",
            "--terrain", str(source), "--output", str(output),
            "--metadata", str(metadata_path), "--preview", str(preview),
        ],
        check=True,
        timeout=900,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _replan_macro_artifacts(
    work: Path,
    labels: np.ndarray,
    weights: np.ndarray,
    region_ids: tuple[str, ...],
    resolution: int,
    feedback: dict,
    replanned_plan: TerrainMacroPlan | None = None,
) -> tuple[TerrainMacroPlan, np.ndarray, dict]:
    """Apply a targeted Macro replan and regenerate all dependent fields."""
    plan_path = work / "terrain_macro_plan.json"
    current = TerrainMacroPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    replanned = replanned_plan if replanned_plan is not None else replan_macro_plan(current, feedback)
    _json(plan_path, replanned.model_dump(mode="json", exclude_none=True))
    terrain_root = work / "terrain"
    terrain_root.mkdir(exist_ok=True)
    shutil.copy2(plan_path, terrain_root / "terrain_macro_plan.json")
    height = generate_macro_height(
        replanned, resolution, layout_weights=weights, region_ids=region_ids,
    )
    np.save(work / "macro_height.npy", height)
    shutil.copy2(work / "macro_height.npy", terrain_root / "macro_height.npy")
    metrics = validate_macro_height(height, replanned, labels, weights, region_ids)
    _json(work / "terrain_macro_validation.json", metrics)
    write_validation_bundle(work / "terrain_validation", height, labels, metrics)
    shutil.copytree(work / "terrain_validation", terrain_root / "validation", dirs_exist_ok=True)
    return replanned, height, metrics


def _fallback_macro_artifacts(
    work: Path,
    scene: ScenePlan,
    labels: np.ndarray,
    weights: np.ndarray,
    region_ids: tuple[str, ...],
    resolution: int,
) -> tuple[TerrainMacroPlan, np.ndarray, dict]:
    """Rebuild a malformed provider plan from generic ScenePlan semantics."""
    plan = derive_macro_plan(scene)
    plan_path = work / "terrain_macro_plan.json"
    _json(plan_path, plan.model_dump(mode="json", exclude_none=True))
    terrain_root = work / "terrain"
    terrain_root.mkdir(exist_ok=True)
    shutil.copy2(plan_path, terrain_root / "terrain_macro_plan.json")
    height = generate_macro_height(plan, resolution, layout_weights=weights, region_ids=region_ids)
    np.save(work / "macro_height.npy", height)
    shutil.copy2(work / "macro_height.npy", terrain_root / "macro_height.npy")
    metrics = validate_macro_height(height, plan, labels, weights, region_ids)
    _json(work / "terrain_macro_validation.json", metrics)
    write_validation_bundle(work / "terrain_validation", height, labels, metrics)
    shutil.copytree(work / "terrain_validation", terrain_root / "validation", dirs_exist_ok=True)
    return plan, height, metrics


def _record_terrain_retry(work: Path, feedback: dict) -> None:
    """Persist targeted retry feedback without changing the random seed."""
    path = work / "terrain" / "logs" / "terrain_retry_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        history = {"strategy": "macro_plan_replan", "seed_changes": 0, "attempts": []}
    history.setdefault("strategy", "macro_plan_replan")
    history.setdefault("seed_changes", 0)
    history.setdefault("attempts", [])
    history["attempts"].append({
        "attempt": len(history["attempts"]) + 1,
        "reason": feedback.get("failures", ["validation"]),
        "recommendations": feedback.get("recommendations", []),
        "metrics": feedback.get("metrics", {}),
    })
    history["attempt"] = len(history["attempts"])
    _json(path, history)


def _recover_regional_detail(
    work: Path,
    scene: ScenePlan,
    layout,
    labels: np.ndarray,
    weights: np.ndarray,
    region_ids: tuple[str, ...],
    seed: int,
    macro_height: np.ndarray,
) -> tuple[object, np.ndarray, dict]:
    """Regenerate regional terrain after a preservation failure.

    The random seed and layout remain immutable.  Recovery first adjusts the
    planner-owned Macro Plan, then falls back to the generic ScenePlan
    derivation if the targeted plan still cannot absorb regional detail.
    """
    value = generate_terrain(scene, layout, seed, macro_height=macro_height)
    preservation = validate_regional_detail_preserves_macro(macro_height, value.height)
    if preservation["macro_composition_preserved"]:
        return value, macro_height, preservation

    feedback = {
        "status": "REPLAN",
        "failures": ["regional_detail_preservation"],
        "metrics": preservation,
    }
    _record_terrain_retry(work, feedback)
    resolution = int(macro_height.shape[0])
    _, replanned_height, _ = _replan_macro_artifacts(
        work, labels, weights, region_ids, resolution, feedback,
    )
    value = generate_terrain(scene, layout, seed, macro_height=replanned_height)
    preservation = validate_regional_detail_preserves_macro(replanned_height, value.height)
    if preservation["macro_composition_preserved"]:
        return value, replanned_height, preservation

    fallback_feedback = {
        "status": "FALLBACK",
        "failures": ["regional_detail_replan_invalid", "generic_scene_plan_fallback"],
        "metrics": preservation,
    }
    _record_terrain_retry(work, fallback_feedback)
    _, fallback_height, _ = _fallback_macro_artifacts(
        work, scene, labels, weights, region_ids, resolution,
    )
    value = generate_terrain(scene, layout, seed, macro_height=fallback_height)
    preservation = validate_regional_detail_preserves_macro(fallback_height, value.height)
    return value, fallback_height, preservation


def _materialize_structural_agent_outputs(work: Path) -> Path:
    """Create the executable Structural Agent output snapshot for this run.

    The repository workers are the maintained implementation, while this
    per-run directory is the artifact consumed by the subprocesses. Keeping
    both scripts and the selected plan together makes the executed contract
    reproducible after the mutable work directory changes.
    """
    output = Path(work) / "structural_agent_output"
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1] / "workers"
    for name in ("structural_worker.py", "structural_validator.py"):
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / name)
    plan = Path(work) / "structural_plan.json"
    if not plan.is_file():
        raise FileNotFoundError(plan)
    shutil.copy2(plan, output / "structural_plan.json")
    _json(output / "agent_outputs.json", {
        "schema": "worldclaw-oss-structural-agent-outputs-v1",
        "worker": str(output / "structural_worker.py"),
        "validator": str(output / "structural_validator.py"),
        "plan": str(output / "structural_plan.json"),
        "execution_contract": ["planning", "generation", "terrain_integration", "deterministic_geometry_validation"],
    })
    return output


def _merge_structural_validation_metrics(run_dir: Path, report: dict) -> dict:
    """Expose Structural deterministic/final decisions in run metrics."""
    work = Path(run_dir) / "work"
    for name, key in (
        ("structural_validation.json", "structural_validation"),
        ("structural_final_validation.json", "structural_final_validation"),
    ):
        path = work / name
        if path.is_file():
            report[key] = json.loads(path.read_text(encoding="utf-8"))
    final = report.get("structural_final_validation")
    if isinstance(final, dict) and final.get("status") != "pass":
        report["valid"] = False
        diagnosis = str(final.get("diagnosis", "structural visual validation failed"))
        report.setdefault("errors", []).append(f"structural_final_validation: {diagnosis}")
    return report


REFERENCE_IMAGE_WORKER_ENV = "WORLDCLAW_REFERENCE_IMAGE_WORKER"
REGION_COMPOSITION_WORKER_ENV = "WORLDCLAW_FLUX_WORKER"
RECON_PREFLIGHT_WORKER_ENV = "WORLDCLAW_RECON_PREFLIGHT_WORKER"


def environment_image_requests(plan: ScenePlan) -> list[dict]:
    """Create one single-instance reference request per regional appearance."""
    requests = []
    seen = set()
    for region in plan.regions:
        for obj in region.objects:
            asset_type = effective_asset_type(obj.category, obj.asset_role)
            if obj.count <= 0 or is_structural_feature(obj.category, obj.asset_role) or not needs_reference_asset(asset_type):
                continue
            appearance = obj.appearance or region.appearance
            key = (region.id, obj.category, appearance_key(region.id, obj.category, appearance))
            if key in seen:
                continue
            seen.add(key)
            requests.append({
                "id": f"reference_{region.id}_{obj.category}_{key[2]}",
                "category": obj.category,
                "asset_role": obj.asset_role.value if obj.asset_role else None,
                "asset_type": asset_type.value,
                "region_id": region.id,
                "count": int(obj.count),
                "instance_strategy": obj.instance_strategy,
                "density": obj.density,
                "scene_role": region.function,
                "importance": (
                    "critical" if obj.category.strip().lower() in {"cabin", "cabins", "house", "houses", "building", "castle"}
                    else ("important" if obj.count > 20 else "decorative")
                ),
                "prompt": (
                    f"Isolated {reference_subject(obj.category, asset_type)}, {appearance}, "
                    "exactly one reusable instance, centered, full body, neutral lighting, "
                    "transparent-friendly background, no pile, no cluster, no group; "
                    f"asset route: {asset_type.value}"
                ),
            })
    return requests


class Pipeline:
    def __init__(self, run_dir: Path, lock_path: Path, mode: str, seed: int, prompt: str,
                 command: list[str], cache_root: Path | None = None,
                 allow_retry_exhausted: bool = False):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = ModelLock(lock_path)
        self.mode, self.seed, self.prompt, self.command = mode, seed, prompt.strip(), command
        self.allow_retry_exhausted = allow_retry_exhausted
        self.run_id = self.run_dir.name
        self.db = StateDB(self.run_dir / "state.sqlite3")
        self.db.initialize(self.run_id)
        self.work = self.run_dir / "work"
        self.work.mkdir(exist_ok=True)
        # Every stage attempt gets an immutable, inspectable snapshot.  The
        # live work directory is intentionally still reused for resume; these
        # snapshots are the audit trail that prevents retries from erasing the
        # previous request, response, or generated output.
        self.stage_artifacts = self.run_dir / "stage_artifacts"
        self.stage_artifacts.mkdir(exist_ok=True)
        self.cache = ContentCache(cache_root or self.run_dir / "cache")
        self.started = time.time()
        self.manifest = self._load_or_create_manifest()
        prompt_path = self.run_dir / "prompt.txt"
        if not prompt_path.exists():
            prompt_path.write_text(self.prompt + "\n", encoding="utf-8")

    def _load_or_create_manifest(self) -> RunManifest:
        path = self.run_dir / "run_manifest.json"
        if path.exists():
            return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        manifest = RunManifest(
            run_id=self.run_id,
            prompt_sha256=hashlib.sha256(self.prompt.encode()).hexdigest(),
            seed=self.seed,
            mode=self.mode,
            command=self.command,
            models=self._runtime_records(),
            synthetic_outputs=self.mode == "synthetic",
        )
        _json(path, manifest.model_dump(mode="json"))
        return manifest

    def _api_enabled(self) -> bool:
        # Planner and AssetTypeClassifier are always served by the explicit
        # GPT-5.6 Sol API model. The legacy local Qwen planner is disabled.
        return self.mode == "live"

    def _validation_api_enabled(self) -> bool:
        """Whether mesh/render quality checks use the OpenAI validation model."""
        return (
            self.mode == "live"
            and os.getenv("WORLDCLAW_VALIDATION_PROVIDER", "openai").lower() == "openai"
        )

    def _openai_config(self):
        if not self._api_enabled():
            return None
        from .openai_api import OpenAIConfig
        return OpenAIConfig.from_environment()

    def _validation_config(self):
        if not self._validation_api_enabled():
            return None
        from .openai_api import OpenAIConfig
        return OpenAIConfig.from_environment()

    @staticmethod
    def _render_profile() -> str:
        profile = os.getenv("WORLDCLAW_RENDER_PROFILE", "diagnostic").lower()
        if profile not in {"diagnostic", "full"}:
            raise ValueError("WORLDCLAW_RENDER_PROFILE must be diagnostic or full")
        return profile

    def _runtime_records(self):
        config = self._openai_config()
        records = self.lock.openai_records(config) if config is not None else dict(self.lock.records)
        if config is not None and not self._validation_api_enabled() and "vlm" in records:
            # An API planner does not imply an API validation backend.
            records["vlm"] = self.lock.records["vlm"]
        # Keep the manifest honest when only validation uses OpenAI. The
        # planner and generation workers can still use their locked models.
        validation = self._validation_config()
        if validation is not None and config is None and "vlm" in records:
            records["vlm"] = records["vlm"].model_copy(update={
                "model_id": validation.validation_model,
                "revision": "api",
                "license": "provider-api",
                "size_bytes": 0,
                "purpose": "mesh and render validation",
                "resolved": True,
                "requested_model_id": None,
                "resolution_reason": "OpenAI validation model configured at runtime",
            })
        return records

    def _save_manifest(self):
        _json(self.run_dir / "run_manifest.json", self.manifest.model_dump(mode="json"))

    def _write_mesh_reuse_manifest(self) -> None:
        """Persist dependency-auditable mesh reuse decisions for this run.

        Terrain/layout/structural outputs are always regenerated.  Unless a
        caller supplies a verified cache record, no baseline mesh is claimed
        as reused; this prevents directory-copy provenance from masquerading
        as a cache hit.
        """
        baseline = os.getenv(
            "WORLDCLAW_BASELINE_RUN",
            "/mnt/data/v-huguangyu/worldclaw-oss/runs/live_forest_boundary_20260901_gate_v2",
        )
        def dependency_fingerprint(item: dict) -> str | None:
            """Hash only explicit upstream inputs; missing inputs are unsafe."""
            fields = (
                "id", "category", "asset_type", "asset_role", "prompt", "appearance",
                "reference_sha256", "reference_hash", "crop_sha256", "crop_hash",
                "mask_sha256", "mask_hash", "bbox", "source_model", "model_id",
                "model_revision", "revision", "inference_config", "validation_status",
            )
            selected = {key: item[key] for key in fields if key in item and item[key] not in (None, "")}
            required = ("category", "asset_type", "prompt", "source_model")
            if any(key not in selected for key in required):
                return None
            return hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

        def missing_dependency_fields(item: dict) -> list[str]:
            required = ("category", "asset_type", "prompt", "source_model")
            return [field for field in required if item.get(field) in (None, "")]

        def load_candidates(root: Path) -> list[dict]:
            values: list[dict] = []
            for filename in (
                "asset_mesh_index.json", "assets.json", "placement_response.json",
                "reconstruction_response.json", "environment_assets_response.json",
            ):
                path = root / "work" / filename
                if not path.is_file():
                    continue
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                items = value if isinstance(value, list) else value.get("assets", []) if isinstance(value, dict) else []
                values.extend(item for item in items if isinstance(item, dict))
            return values

        # Build entries from the actual stage responses.  A live run does not
        # necessarily produce the synthetic ``asset_mesh_index.json`` file,
        # so relying on that one optional index silently under-reports meshes.
        candidates = load_candidates(self.run_dir)
        baseline_candidates = load_candidates(Path(baseline))
        baseline_by_fingerprint: dict[str, dict] = {}
        for item in baseline_candidates:
            fingerprint = dependency_fingerprint(item)
            mesh = item.get("mesh")
            if fingerprint and mesh and Path(str(mesh)).is_file() and str(item.get("validation_status", "pass")).lower() in {"pass", "validated", "ok"}:
                baseline_by_fingerprint[fingerprint] = item
        grouped: dict[tuple[str, str], dict] = {}
        for item in candidates:
            mesh = item.get("mesh")
            if not mesh:
                continue
            mesh_path = Path(str(mesh))
            if not mesh_path.is_file():
                continue
            digest = sha256(mesh_path)
            key = (str(mesh_path), digest)
            entry = grouped.setdefault(key, {
                "mesh": str(mesh_path), "mesh_sha256": digest,
                "asset_ids": [], "categories": [], "source_models": [],
            })
            fingerprint = dependency_fingerprint(item)
            if fingerprint:
                entry["dependency_fingerprint"] = fingerprint
                baseline_item = baseline_by_fingerprint.get(fingerprint)
                if baseline_item:
                    baseline_mesh = Path(str(baseline_item.get("mesh", "")))
                    if baseline_mesh.is_file() and sha256(baseline_mesh) == digest:
                        entry["baseline_match"] = {
                            "mesh": str(baseline_mesh),
                            "mesh_sha256": digest,
                            "dependency_match": True,
                        }
            else:
                missing = missing_dependency_fields(item)
                entry.setdefault("missing_dependency_fields", [])
                entry["missing_dependency_fields"] = sorted(set(entry["missing_dependency_fields"]) | set(missing))
            for field, target in (("id", "asset_ids"), ("category", "categories"), ("source_model", "source_models")):
                value = item.get(field)
                if value is not None and str(value) not in entry[target]:
                    entry[target].append(str(value))
        regenerated = [
            entry | {
                "status": "regenerated",
                "reason": (
                    "missing dependency fields: " + ", ".join(entry.get("missing_dependency_fields", []))
                    if entry.get("missing_dependency_fields")
                    else "no verified baseline dependency-complete cache hit"
                ),
            }
            for entry in sorted(grouped.values(), key=lambda value: value["mesh"])
        ]
        dependency_names = (
            "scene_plan.json", "terrain_macro_plan.json", "layout_labels.npy",
            "layout_weights.npy", "terrain_structural.npz", "structural_plan.json",
        )
        input_hashes = {
            name: sha256(self.work / name) if (self.work / name).is_file() else None
            for name in dependency_names
        }
        baseline_manifest = Path(baseline) / "mesh_reuse_manifest.json"
        baseline_available = baseline_manifest.is_file()
        eligible = [entry for entry in grouped.values() if entry.get("baseline_match")]
        _json(self.run_dir / "mesh_reuse_manifest.json", {
            "schema_version": "worldclaw-oss-mesh-reuse-v1",
            "baseline_run": baseline,
            "reused": [entry for entry in eligible if Path(entry["mesh"]).resolve() == Path(entry["baseline_match"]["mesh"]).resolve()],
            "reuse_eligible": eligible,
            "regenerated": regenerated,
            "reason": {
                "baseline_manifest_available": baseline_available,
                "reused_count": len([entry for entry in eligible if Path(entry["mesh"]).resolve() == Path(entry["baseline_match"]["mesh"]).resolve()]),
                "reuse_eligible_count": len(eligible),
                "regenerated_count": len(regenerated),
                "baseline_candidate_count": len(baseline_candidates),
                "unverified_meshes": "entries without complete dependency fingerprints remain regenerated",
            },
            "input_hashes": input_hashes,
        })

    def _work_file_state(self) -> dict[str, dict[str, object]]:
        """Return hashes and sizes for files currently in the stage work area."""
        state: dict[str, dict[str, object]] = {}
        for path in sorted(self.work.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.work).as_posix()
            state[relative] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        return state

    def _snapshot_stage(
        self,
        stage: Stage,
        attempt: int,
        started: float,
        started_at: datetime,
        status: str,
        before: dict[str, dict[str, object]],
        payload: dict | None = None,
        error: str | None = None,
    ) -> str:
        """Persist one stage attempt without overwriting prior attempts."""
        destination = self.stage_artifacts / stage.value / f"attempt_{attempt:02d}"
        if destination.exists():
            destination = self.stage_artifacts / stage.value / (
                f"attempt_{attempt:02d}_recovery_{uuid.uuid4().hex[:8]}"
            )
        destination.mkdir(parents=True, exist_ok=False)
        after = self._work_file_state()
        force = {
            name for name in after
            if name.endswith("_request.json") or name.endswith("_response.json")
        }
        changed = sorted(
            name for name, record in after.items()
            if name in force or before.get(name) != record
        )
        copied: list[dict[str, object]] = []
        for name in changed:
            source = self.work / Path(name)
            target = destination / "files" / Path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({
                "path": name,
                "snapshot": target.relative_to(destination).as_posix(),
                "sha256": sha256(source),
                "size_bytes": source.stat().st_size,
            })
        deleted = sorted(set(before) - set(after))
        record = {
            "schema": "worldclaw-oss-stage-attempt-v1",
            "stage": stage.value,
            "attempt": attempt,
            "status": status,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "payload": payload,
            "error": error,
            "files": copied,
            "deleted_files": deleted,
            "unchanged_files": sorted(set(after) - set(changed)),
        }
        _json(destination / "attempt.json", record)
        return destination.relative_to(self.run_dir).as_posix()

    def _run_stage(self, stage: Stage, fn: Callable[[], dict | None]):
        if self.db.status(self.run_id, stage) == "complete":
            return
        if self.db.status(self.run_id, stage) == "failed" and self.db.attempts(self.run_id, stage) >= 3:
            message = f"{stage.value}: retry limit exhausted; resume after fixing the recorded blocker"
            if message not in self.manifest.errors:
                self.manifest.errors.append(message)
            previous = self.manifest.stage_results.get(stage.value, {})
            self.manifest.stage_results[stage.value] = {
                **previous, "status": "failed", "attempt": self.db.attempts(self.run_id, stage),
                "resume_blocked": True, "error": previous.get("error", message),
            }
            self._save_manifest()
            raise RuntimeError(message)
        while self.db.attempts(self.run_id, stage) < 3:
            self.db.start(self.run_id, stage)
            self.db.export_events(self.run_id, self.run_dir / "events.jsonl")
            started = time.monotonic()
            started_at = datetime.now(timezone.utc)
            before = self._work_file_state()
            attempt = self.db.attempts(self.run_id, stage)
            try:
                payload = fn() or {}
                snapshot = self._snapshot_stage(
                    stage, attempt, started, started_at, "complete", before, payload=payload
                )
                self.db.complete(self.run_id, stage, payload)
                self.db.export_events(self.run_id, self.run_dir / "events.jsonl")
                self.manifest.stage_results[stage.value] = {
                    "status": "complete", "attempt": self.db.attempts(self.run_id, stage),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "payload": payload, "snapshot": snapshot,
                }
                self._save_manifest()
                return
            except Exception as exc:
                message = f"{stage.value}: {type(exc).__name__}: {exc}"
                snapshot = None
                try:
                    snapshot = self._snapshot_stage(
                        stage, attempt, started, started_at, "failed", before, error=message
                    )
                except Exception as snapshot_exc:
                    # The original stage error is authoritative, but retain a
                    # diagnostic when a filesystem copy itself is unavailable.
                    message += f"; snapshot failed: {type(snapshot_exc).__name__}: {snapshot_exc}"
                self.db.fail(self.run_id, stage, message)
                self.db.export_events(self.run_id, self.run_dir / "events.jsonl")
                self.manifest.errors.append(message)
                self.manifest.retries[stage.value] = self.db.attempts(self.run_id, stage) - 1
                self.manifest.stage_results[stage.value] = {
                    "status": "failed", "attempt": self.db.attempts(self.run_id, stage),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "error": message, "snapshot": snapshot,
                }
                self._save_manifest()
                if self.db.attempts(self.run_id, stage) >= 3:
                    raise

    def run(self) -> Path:
        if self.mode == "live":
            if self._api_enabled():
                required = ["hunyuan3d"]
                if not self._validation_api_enabled():
                    required.append("vlm")
                self.lock.require_resolved(required)
            else:
                required = [
                    "planner", "vlm", "flux_dev", "flux_schnell", "grounding_dino",
                    "sam2", "hunyuan3d", "trellis",
                ]
                if self._validation_api_enabled():
                    required.remove("vlm")
                self.lock.require_resolved(required)
        handlers = self._synthetic_handlers() if self.mode == "synthetic" else self._live_handlers()
        try:
            for stage in Stage:
                if (
                    self.allow_retry_exhausted
                    and self.db.status(self.run_id, stage) in {"failed", "running"}
                    and self.db.attempts(self.run_id, stage) >= 3
                ):
                    self.db.reopen(self.run_id, stage)
                    self.db.export_events(self.run_id, self.run_dir / "events.jsonl")
                self._run_stage(stage, handlers[stage])
        except Exception:
            (self.run_dir / "errors.log").write_text("\n".join(self.manifest.errors) + "\n", encoding="utf-8")
            self.db.export_events(self.run_id, self.run_dir / "events.jsonl")
            self._save_manifest()
            raise
        self.manifest.finished_at = datetime.now(timezone.utc).isoformat()
        self.manifest.elapsed_seconds = round(time.time() - self.started, 3)
        self.manifest.outputs = {
            p.relative_to(self.run_dir).as_posix(): sha256(p)
            for p in sorted(self.run_dir.rglob("*"))
            if p.is_file() and p.name != "run_manifest.json"
        }
        (self.run_dir / "errors.log").write_text(
            "\n".join(self.manifest.errors) + ("\n" if self.manifest.errors else ""), encoding="utf-8"
        )
        self.db.export_events(self.run_id, self.run_dir / "events.jsonl")
        self._save_manifest()
        return self.run_dir

    def _plan(self) -> ScenePlan:
        return ScenePlan.model_validate_json((self.work / "scene_plan.json").read_text(encoding="utf-8"))

    def _synthetic_handlers(self):
        def intent():
            _json(self.work / "intent.json", {
                "constraints": [self.prompt], "verbatim_prompt": self.prompt, "synthetic": True,
            })
            return {"intent": "intent.json"}

        def plan():
            value = synthetic_plan(self.prompt)
            _json(self.work / "scene_plan.json", value.model_dump(mode="json", exclude_none=True))
            return {"regions": len(value.regions)}

        def layout():
            value = sample_layout(self._plan(), 128, seed=self.seed)
            np.save(self.work / "layout_labels.npy", value.labels)
            np.save(self.work / "layout_weights.npy", value.weights)
            _json(self.work / "layout_generation.json", value.generation)
            _json(self.work / "layout.json", {
                "region_ids": value.region_ids,
                "deterministic": True,
                "seed": self.seed,
                "selected_candidate": value.generation["selected_candidate"],
            })
            return {
                "resolution": 128,
                "seed": self.seed,
                "candidate_count": value.generation["candidate_count"],
                "selected_candidate": value.generation["selected_candidate"],
            }

        def terrain_macro_plan():
            layout_labels = np.load(self.work / "layout_labels.npy")
            layout_weights = np.load(self.work / "layout_weights.npy")
            layout_summary = summarize_layout(layout_labels, layout_weights, tuple(r.id for r in self._plan().regions))
            planner_input = self.work / "terrain_planner_input"
            planner_input.mkdir(exist_ok=True)
            shutil.copy2(self.work / "intent.json", planner_input / "intent.json")
            shutil.copy2(self.work / "scene_plan.json", planner_input / "scene_plan.json")
            layout_dir = planner_input / "layout"
            masks_dir = layout_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            np.save(layout_dir / "layout_labels.npy", layout_labels)
            np.save(layout_dir / "layout_weights.npy", layout_weights)
            for index, region in enumerate(self._plan().regions):
                safe_id = str(region.id).replace("/", "_").replace("\\", "_")
                np.save(masks_dir / f"{safe_id}.npy", layout_labels == index)
            _json(planner_input / "world_spec.json", {"world_size_m": list(self._plan().world_size_m)})
            _json(planner_input / "terrain_constraints.json", {"explicit_constraints": self._plan().explicit_constraints})
            _json(planner_input / "structural_intents.json", {"status": "pending", "source": "structural branch follows terrain"})
            plan = derive_macro_plan(self._plan())
            _json(self.work / "terrain_macro_plan.json", plan.model_dump(mode="json", exclude_none=True))
            terrain_root = self.work / "terrain"
            terrain_root.mkdir(exist_ok=True)
            shutil.copy2(self.work / "terrain_macro_plan.json", terrain_root / "terrain_macro_plan.json")
            return {"landforms": len(plan.landforms), "functional_zones": len(plan.functional_zones), "planner": "semantic_composition"}

        def terrain_macro_generate():
            macro = TerrainMacroPlan.model_validate_json((self.work / "terrain_macro_plan.json").read_text(encoding="utf-8"))
            height = generate_macro_height(
                macro, 128, layout_weights=np.load(self.work / "layout_weights.npy"),
                region_ids=tuple(r.id for r in self._plan().regions),
            )
            vertices, triangles = macro_vertices(height, macro.world_size_m, macro)
            np.save(self.work / "macro_height.npy", height)
            terrain_root = self.work / "terrain"
            terrain_root.mkdir(exist_ok=True)
            shutil.copy2(self.work / "macro_height.npy", terrain_root / "macro_height.npy")
            metrics = validate_macro_height(
                height, macro, np.load(self.work / "layout_labels.npy"),
                np.load(self.work / "layout_weights.npy"),
                tuple(r.id for r in self._plan().regions),
            )
            _json(self.work / "terrain_macro_validation.json", metrics)
            _json(self.work / "terrain_frequency.json", {"macro": {"scale_m": [50, 150], "source": "terrain_macro_plan.json", "share": "70-90%"}, "regional": {"scale_m": [5, 30], "source": "terrain.py regional operators", "share": "10-30%"}, "micro": {"scale_m": [0.2, 5], "source": "material/displacement operators", "share": "surface detail"}})
            write_validation_bundle(self.work / "terrain_validation", height, np.load(self.work / "layout_labels.npy"), metrics)
            shutil.copytree(self.work / "terrain_validation", terrain_root / "validation", dirs_exist_ok=True)
            if not metrics.get("valid"):
                feedback = {
                    "status": "REPLAN",
                    "failures": [key for key in (
                        "lake_basin_containment", "trail_traversability",
                        "macro_composition_preserved", "terrain_spikes",
                    ) if not metrics.get(key, True)],
                    "metrics": metrics,
                }
                _record_terrain_retry(self.work, feedback)
                macro, height, metrics = _replan_macro_artifacts(
                    self.work, np.load(self.work / "layout_labels.npy"),
                    np.load(self.work / "layout_weights.npy"),
                    tuple(r.id for r in self._plan().regions), 128, feedback,
                )
                if not metrics.get("valid"):
                    feedback["metrics"] = metrics
                    feedback["failures"].append("targeted_replan_invalid")
                    _record_terrain_retry(self.work, feedback)
                    macro, height, metrics = _fallback_macro_artifacts(
                        self.work, self._plan(),
                        np.load(self.work / "layout_labels.npy"),
                        np.load(self.work / "layout_weights.npy"),
                        tuple(r.id for r in self._plan().regions), 128,
                    )
                if not metrics.get("valid"):
                    raise ValueError(f"macro terrain validation failed after bounded Macro Replan: {metrics}")
            return {"resolution": 128, "global_relief": float(height.max() - height.min())}

        def terrain_visual_validate():
            metrics = json.loads((self.work / "terrain_macro_validation.json").read_text(encoding="utf-8"))
            result = visual_validate_terrain(self.work / "terrain_validation", metrics)
            if result["status"] == "REPLAN":
                macro = TerrainMacroPlan.model_validate_json((self.work / "terrain_macro_plan.json").read_text(encoding="utf-8"))
                macro = replan_macro_plan(macro, result)
                _json(self.work / "terrain_macro_plan.json", macro.model_dump(mode="json", exclude_none=True))
                height = generate_macro_height(
                    macro, 128, layout_weights=np.load(self.work / "layout_weights.npy"),
                    region_ids=tuple(r.id for r in self._plan().regions),
                )
                np.save(self.work / "macro_height.npy", height)
                metrics = validate_macro_height(
                    height, macro, np.load(self.work / "layout_labels.npy"),
                    np.load(self.work / "layout_weights.npy"),
                    tuple(r.id for r in self._plan().regions),
                )
                _json(self.work / "terrain_macro_validation.json", metrics)
                write_validation_bundle(self.work / "terrain_validation", height, np.load(self.work / "layout_labels.npy"), metrics)
            result = visual_validate_terrain(self.work / "terrain_validation", metrics)
            _json(self.work / "terrain_visual_validation.json", result)
            if result["status"] != "PASS":
                raise ValueError(result)
            return result

        def terrain():
            from .layout import LayoutResult
            plan_value = self._plan()
            labels = np.load(self.work / "layout_labels.npy")
            weights = np.load(self.work / "layout_weights.npy")
            region_ids = tuple(r.id for r in plan_value.regions)
            macro_height = np.load(self.work / "macro_height.npy")
            value, macro_height, macro_preservation = _recover_regional_detail(
                self.work, plan_value, LayoutResult(labels, weights, region_ids),
                labels, weights, region_ids, stable_seed(self.prompt, self.seed),
                macro_height,
            )
            np.savez_compressed(
                self.work / "terrain.npz",
                height=value.height, vertices=value.vertices, triangles=value.triangles,
            )
            _json(self.work / "terrain_regional_validation.json", macro_preservation)
            if not macro_preservation["macro_composition_preserved"]:
                raise ValueError(f"regional detail damaged macro composition: {macro_preservation}")
            terrain_root = self.work / "terrain"
            terrain_root.mkdir(exist_ok=True)
            shutil.copy2(self.work / "terrain.npz", terrain_root / "terrain.npz")
            _write_terrain_blend(self.work)
            (terrain_root / "logs").mkdir(exist_ok=True)
            history_path = terrain_root / "logs" / "terrain_retry_history.json"
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                history = {"strategy": "macro_plan_replan", "seed_changes": 0, "attempts": []}
            history.setdefault("strategy", "macro_plan_replan")
            history.setdefault("seed_changes", 0)
            history.setdefault("attempts", [])
            history["attempt"] = len(history["attempts"])
            _json(history_path, history)
            return {
                "vertices": len(value.vertices),
                "boundary_max_delta": boundary_discontinuity(value.height, labels),
                "macro_composition_preserved": macro_preservation["macro_composition_preserved"],
            }

        def structural_input():
            plan = self._plan().model_dump(mode="json", exclude_none=True)
            return prepare_structural_agent_input(self.work, plan)

        def structural_replan():
            plan = self._plan().model_dump(mode="json", exclude_none=True)
            result = StructuralFeatureAgent().plan(
                plan, layout=np.load(self.work / "layout_labels.npy"),
                terrain=np.load(self.work / "terrain.npz")["height"],
                world_size=tuple(plan["world_size_m"]),
                agent_input={"root": str(self.work / "structural_agent_input"), "task": str(self.work / "structural_agent_input" / "structural_task.json")},
            )
            features = result["features"]
            _json(self.work / "structural_plan.json", result | {"deterministic": True})
            _json(self.work / "structural_agent_input" / "structural_plan.json", result | {"deterministic": True})
            output = _materialize_structural_agent_outputs(self.work)
            return {"structural_feature_count": len(features), "structural_agent_calls": result.get("agent_calls", 1), "agent_output": str(output)}

        def structural_generate():
            started = time.monotonic()
            output = self.work / "structural_agent_output"
            if not (output / "structural_worker.py").is_file():
                output = _materialize_structural_agent_outputs(self.work)
            worker = output / "structural_worker.py"
            completed = subprocess.run([sys.executable, str(worker), "--input", str(self.work / "structural_agent_input"), "--output", str(self.work / "structural"), "--seed", str(self.seed), "--structural-plan", str(output / "structural_plan.json")], capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"structural_worker.py failed: {completed.stderr[-4000:]}")
            output = json.loads(completed.stdout)
            return {"structural_feature_count": len(output.get("features", [])), "hunyuan_calls_avoided": output.get("metrics", {}).get("hunyuan_calls_avoided", 0), "structural_generation_seconds": round(time.monotonic() - started, 3)}

        def structural_integrate():
            started = time.monotonic()
            structural_dir = self.work / "structural"
            source = np.load(self.work / "terrain.npz")
            structural = np.load(structural_dir / "terrain_structural.npz")
            modified = np.asarray(structural["height"], dtype=np.float32)
            vertices = np.asarray(structural["vertices"] if "vertices" in structural else source["vertices"], dtype=np.float32).copy()
            if len(vertices) != modified.size:
                raise ValueError("integrated terrain vertex count does not match height grid")
            vertices[:, 2] = modified.reshape(-1)
            np.savez_compressed(self.work / "terrain_structural.npz", height=modified, vertices=vertices, triangles=source["triangles"])
            branch = json.loads((structural_dir / "structural_branch.json").read_text(encoding="utf-8"))
            _json(self.work / "structural_branch.json", branch | {"status": "ok", "integration": "world_coordinates", "exclusion_mask": str(structural_dir / "structural_exclusion_mask.npy"), "occupied_mask": str(structural_dir / "structural_occupied_mask.npy"), "surface_type_mask": str(structural_dir / "surface_type_mask.npy"), "water_surface_mask": str(structural_dir / "water_surface_mask.npy"), "trail_clearance_field": str(structural_dir / "trail_clearance_field.npy"), "trail_exclusion_mask": str(structural_dir / "trail_exclusion_mask.npy"), "structural_placement_weights": str(structural_dir / "structural_placement_weights.npy"), "placement_constraints": {"avoid_exclusion": True, "modified_terrain_required": True, "base_layout_weights_immutable": True}, "terrain": str(self.work / "terrain_structural.npz")})
            return {"terrain_modified": bool(branch.get("terrain_modified")), "structural_meshes": len(branch.get("structural_meshes", [])), "terrain_modification_seconds": round(time.monotonic() - started, 3)}

        def structural_validate():
            started = time.monotonic()
            output = self.work / "structural_agent_output"
            if not (output / "structural_validator.py").is_file():
                output = _materialize_structural_agent_outputs(self.work)
            validator = output / "structural_validator.py"
            completed = subprocess.run([sys.executable, str(validator), "--input", str(self.work / "structural"), "--terrain", str(self.work / "terrain_structural.npz"), "--output", str(self.work / "structural_validation.json")], capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"structural_validator.py failed: {completed.stderr[-4000:]}")
            report = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
            if report["status"] != "pass":
                raise RuntimeError(str(report))
            report["structural_validation_seconds"] = round(time.monotonic() - started, 3)
            _json(self.work / "structural_validation.json", report)
            return report

        def assets():
            plan_value = self._plan()
            labels = np.load(self.work / "layout_labels.npy")
            layout_weights = np.load(self.work / "layout_weights.npy")
            terrain_path = self.work / "terrain_structural.npz"
            terrain = np.load(terrain_path if terrain_path.exists() else self.work / "terrain.npz")
            height = terrain["height"]
            exclusion = np.load(self.work / "structural" / "structural_exclusion_mask.npy") if (self.work / "structural" / "structural_exclusion_mask.npy").exists() else np.zeros_like(height, dtype=bool)
            influence = np.load(self.work / "structural" / "structural_placement_weights.npy") if (self.work / "structural" / "structural_placement_weights.npy").exists() else np.ones_like(height, dtype=np.float32)
            trail_exclusion = np.load(self.work / "structural" / "trail_exclusion_mask.npy") if (self.work / "structural" / "trail_exclusion_mask.npy").exists() else np.zeros_like(labels, dtype=bool)
            water_path = self.work / "structural" / "water_surface_mask.npy"
            water_surface = np.load(water_path) if water_path.exists() else None
            surface_masks = prepare_surface_masks({
                "structural_exclusion": exclusion,
                "trail": trail_exclusion,
                "water": water_surface,
            }, labels.shape)
            rng = random.Random(stable_seed(self.prompt, self.seed))
            records, meshes = [], []
            width, depth = plan_value.world_size_m
            base_v, base_t = box_mesh()
            for region_index, region in enumerate(plan_value.regions):
                for spec in region.objects:
                    if is_structural_feature(spec.category, spec.asset_role):
                        continue
                    category = spec.category.lower()
                    requirements = requirements_from_spec(spec.model_dump(mode="json"))
                    gate = apply_hard_gate(
                        layout_weights[region_index] >= 0.05,
                        requirements,
                        surface_masks,
                    )
                    probabilities = np.clip(layout_weights[region_index] * influence, 0.0, None)
                    # Distance preferences are optional environmental factors;
                    # hard eligibility remains independent of probability.
                    for surface_name, falloff in requirements.distance_preferences:
                        mask = surface_masks.get(surface_name)
                        if mask is not None and np.any(mask):
                            points = np.argwhere(mask)
                            spacing = np.asarray([
                                plan_value.world_size_m[1] / max(labels.shape[0] - 1, 1),
                                plan_value.world_size_m[0] / max(labels.shape[1] - 1, 1),
                            ])
                            grid = np.argwhere(np.ones_like(labels, dtype=bool))
                            distances = np.empty(labels.shape, dtype=np.float32)
                            for start in range(0, len(grid), 4096):
                                chunk = grid[start:start + 4096]
                                delta = (chunk[:, None, :] - points[None, :, :]) * spacing
                                distances[chunk[:, 0], chunk[:, 1]] = np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)
                            probabilities *= np.exp(-distances / max(falloff, 1e-3))
                    candidates = [item for item in np.argwhere(gate).tolist() if probabilities[item[0], item[1]] > 1e-6]
                    candidates.sort(key=lambda item: -math.log(max(rng.random(), 1e-12)) / max(float(probabilities[item[0], item[1]]), 1e-6))
                    cursor = 0
                    for index in range(spec.count):
                        if cursor >= len(candidates):
                            break
                        iy, ix = candidates[cursor]
                        cursor += 1
                        x = -width / 2 + ix * width / (labels.shape[1] - 1)
                        y = -depth / 2 + iy * depth / (labels.shape[0] - 1)
                        z = float(height[iy, ix])
                        sx = 2.0 if spec.category in ("house", "cabin", "building") else 1.0
                        sy = sx
                        sz = 4.0 if spec.category in ("tree", "palm") else (5.0 if spec.category == "castle" else 2.0)
                        asset_id = f"{region.id}_{spec.category}_{index:03d}"
                        asset_type = effective_asset_type(spec.category, spec.asset_type)
                        # Keep prototype vertices local and put placement in the
                        # node transform.  The exporter can now share one box
                        # mesh across thousands of trees/rocks/buildings.
                        vertices = base_v.copy()
                        key = cache_key(
                            f"synthetic:{spec.category}", "none", "synthetic-box", "v1",
                            {"size": [sx, sy, sz]},
                        )
                        mesh_path = self.cache.put_bytes(
                            key, "json", json.dumps({"synthetic": True, "category": spec.category}).encode()
                        )
                        matrix = np.eye(4)
                        matrix[:3, :3] = np.diag([sx, sy, sz])
                        matrix[:3, 3] = [x, y, z + sz / 2]
                        records.append(AssetInstance(
                            id=asset_id, category=spec.category, asset_type=asset_type,
                            source_image="synthetic://none",
                            mask=f"layout:{region.id}", mesh=str(mesh_path), material=spec.category,
                            transform_z_up=matrix.tolist(), region_id=region.id, contact_ratio=1.0,
                            source_model="synthetic-box-v1", synthetic=True,
                        ).model_dump(mode="json"))
                        meshes.append({
                            "name": asset_id, "category": spec.category, "vertices": vertices,
                            "triangles": base_t, "transform_z_up": matrix.tolist(),
                            "extras": {"region_id": region.id, "synthetic": True, "scale": [sx, sy, sz]},
                        })
            _json(self.work / "assets.json", records)
            np.savez_compressed(
                self.work / "asset_meshes.npz", **{f"v{i}": m["vertices"] for i, m in enumerate(meshes)}
            )
            _json(self.work / "asset_mesh_index.json", [
                {"name": m["name"], "category": m["category"], "transform_z_up": m["transform_z_up"],
                 "extras": m["extras"]} for m in meshes
            ])
            return {"assets": len(records)}

        def marker(name):
            def run():
                _json(self.work / f"{name}.json", {"mode": "synthetic", "completed": True})
                return {"synthetic": True}
            return run

        def place():
            records = json.loads((self.work / "assets.json").read_text(encoding="utf-8"))
            return {
                "placed": len(records),
                "contact_min": min((a["contact_ratio"] for a in records), default=1.0),
            }

        def refine():
            report = {
                "round": 1, "floating": 0, "penetration": 0,
                "severe_overlap": 0, "accepted": True, "synthetic": True,
            }
            _json(self.work / "refinement_1.json", report)
            return {"rounds": 1, "accepted": True}

        def export():
            plan_value = self._plan()
            terrain_path = self.work / "terrain_structural.npz"
            terrain_data = np.load(terrain_path if terrain_path.exists() else self.work / "terrain.npz")
            meshes = [{
                "name": "Terrain", "category": "terrain", "vertices": terrain_data["vertices"],
                "triangles": terrain_data["triangles"], "extras": {"kind": "terrain"},
            }]
            index = json.loads((self.work / "asset_mesh_index.json").read_text(encoding="utf-8"))
            data = np.load(self.work / "asset_meshes.npz")
            _, box_triangles = box_mesh()
            for i, item in enumerate(index):
                meshes.append(item | {"vertices": data[f"v{i}"], "triangles": box_triangles})
            branch_path = self.work / "structural_branch.json"
            structural_count = 0
            if branch_path.exists():
                branch = json.loads(branch_path.read_text(encoding="utf-8"))
                for item in branch.get("structural_meshes", []):
                    with np.load(item["mesh"]) as structural_mesh:
                        meshes.append({"name": item["feature_id"], "category": item["category"], "vertices": structural_mesh["vertices"], "triangles": structural_mesh["triangles"], "transform_z_up": item.get("transform_z_up", np.eye(4).tolist()), "extras": {"structural_feature": True, "representation": item.get("representation", "")}})
                    structural_count += 1
            write_glb(self.run_dir / "scene.glb", meshes)
            labels = np.load(self.work / "layout_labels.npy")
            layout_preview(labels, self.run_dir / "preview.png")
            scene = {
                "schema": "worldclaw-oss-scene-v1",
                "official_implementation": False,
                "synthetic": True,
                "prompt": self.prompt,
                "seed": self.seed,
                "coordinate_system": {"internal": "Z-up right-handed", "glTF": "Y-up"},
                "plan": plan_value.model_dump(mode="json"),
                "assets": json.loads((self.work / "assets.json").read_text(encoding="utf-8")),
            }
            _json(self.run_dir / "scene.json", scene)
            (self.run_dir / "diagnostic_views").mkdir(exist_ok=True)
            shutil.copy2(self.run_dir / "preview.png", self.run_dir / "diagnostic_views" / "round_0.png")
            return {"nodes": len(meshes), "structural_meshes": structural_count, "blender_deferred": True}

        def structural_view_plan():
            deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
            if deterministic.get("status") != "pass":
                raise RuntimeError("structural view planning is gated by deterministic validation")
            branch = json.loads((self.work / "structural_branch.json").read_text(encoding="utf-8"))
            structural_plan = json.loads((self.work / "structural_plan.json").read_text(encoding="utf-8"))
            plan = self._plan().model_dump(mode="json", exclude_none=True)
            views = plan_validation_views(plan, structural_plan, branch, deterministic, tuple(plan["world_size_m"]), seed=self.seed, request_path=self.work / "structural_view_plan_request.json", response_path=self.work / "structural_view_plan_response.json")
            _json(self.work / "validation_views.json", views)
            return {"provider": views["provider"], "fixed_views": len(views["fixed"]), "adaptive_views": len(views["adaptive"])}

        def structural_render():
            views = json.loads((self.work / "validation_views.json").read_text(encoding="utf-8"))
            result = render_structural_views(self.work, views)
            plan = self._plan().model_dump(mode="json", exclude_none=True)
            structural_plan = json.loads((self.work / "structural_plan.json").read_text(encoding="utf-8"))
            deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
            bundle = build_validation_bundle(self.work, views, deterministic, structural_plan, plan, result)
            return result | bundle

        def structural_visual_validate():
            deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
            result = visual_validate_bundle(self.work / "validation_bundle", deterministic)
            _json(self.work / "structural_visual_validation.json", result)
            return result

        def structural_adaptive_render():
            visual = json.loads((self.work / "structural_visual_validation.json").read_text(encoding="utf-8"))
            views = json.loads((self.work / "validation_views.json").read_text(encoding="utf-8"))
            result = render_additional_views(self.work, views, visual.get("additional_views", [])) if visual.get("confidence") == "low" else {"status": "ok", "needed": False, "renders": []}
            if result.get("needed"):
                views["additional"] = result.get("requested", [])
                _json(self.work / "validation_views.json", views)
                plan = self._plan().model_dump(mode="json", exclude_none=True)
                structural_plan = json.loads((self.work / "structural_plan.json").read_text(encoding="utf-8"))
                deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
                build_validation_bundle(self.work, views, deterministic, structural_plan, plan, result, result.get("requested", []))
            _json(self.work / "structural_adaptive_render.json", result)
            return result

        def structural_final_validate():
            first = json.loads((self.work / "structural_visual_validation.json").read_text(encoding="utf-8"))
            if first.get("confidence") != "low":
                result = first | {"phase": "final", "follow_up": False}
            else:
                deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
                result = visual_validate_bundle(self.work / "validation_bundle", deterministic, include_additional=True) | {"phase": "final", "follow_up": True}
            _json(self.work / "structural_final_validation.json", result)
            return result

        def validate():
            self._write_mesh_reuse_manifest()
            labels = np.load(self.work / "layout_labels.npy")
            terrain_path = self.work / "terrain_structural.npz"
            height = np.load(terrain_path if terrain_path.exists() else self.work / "terrain.npz")["height"]
            assets_value = json.loads((self.work / "assets.json").read_text(encoding="utf-8"))
            metrics = {
                "synthetic": True,
                "asset_count": len(assets_value),
                "region_count": len(self._plan().regions),
                "boundary_max_delta": boundary_discontinuity(height, labels),
                "min_contact_ratio": min((a["contact_ratio"] for a in assets_value), default=1.0),
                "non_intentional_overlap_rate": 0.0,
                "full_model_acceptance": False,
            }
            structural_path = self.work / "structural_validation.json"
            if structural_path.exists():
                metrics["structural_validation"] = json.loads(structural_path.read_text(encoding="utf-8"))
            final_path = self.work / "structural_final_validation.json"
            if final_path.exists():
                metrics["structural_final_validation"] = json.loads(final_path.read_text(encoding="utf-8"))
            _json(self.run_dir / "metrics.json", metrics)
            (self.run_dir / "errors.log").touch(exist_ok=True)
            report = validate_run(self.run_dir)
            final = metrics.get("structural_final_validation")
            if isinstance(final, dict) and final.get("status") != "pass":
                report["valid"] = False
                report.setdefault("errors", []).append(
                    f"structural_final_validation: {final.get('diagnosis', 'failed')}"
                )
            _json(self.work / "validation.json", report)
            _json(self.run_dir / "metrics.json", metrics | {"run_validation": report})
            if not report["valid"]:
                raise RuntimeError(str(report))
            return report

        return {
            Stage.INTENT: intent, Stage.PLAN: plan, Stage.LAYOUT: layout,
            Stage.TERRAIN_MACRO_PLAN: terrain_macro_plan, Stage.TERRAIN_MACRO_GENERATE: terrain_macro_generate,
            Stage.TERRAIN: terrain, Stage.TERRAIN_VISUAL_VALIDATE: terrain_visual_validate,
            Stage.STRUCTURAL_INPUT: structural_input,
            Stage.STRUCTURAL_REPLAN: structural_replan,
            Stage.STRUCTURAL_GENERATE: structural_generate,
            Stage.STRUCTURAL_INTEGRATE: structural_integrate,
            Stage.STRUCTURAL_VALIDATE: structural_validate,
            Stage.ENV_ASSETS: assets,
            Stage.MESH_VALIDATE_ENV_ASSETS: marker("mesh_validation_environment_assets"),
            Stage.REGION_COMPOSE: marker("region_composition"),
            Stage.SEGMENT: marker("segmentation"), Stage.RECONSTRUCT: marker("reconstruction"),
            Stage.MESH_VALIDATE_RECONSTRUCT: marker("mesh_validation_reconstruct"),
            Stage.PLACE: place, Stage.REFINE: refine, Stage.EXPORT: export,
            Stage.STRUCTURAL_VIEW_PLAN: structural_view_plan,
            Stage.STRUCTURAL_RENDER: structural_render,
            Stage.STRUCTURAL_VISUAL_VALIDATE: structural_visual_validate,
            Stage.STRUCTURAL_ADAPTIVE_RENDER: structural_adaptive_render,
            Stage.STRUCTURAL_FINAL_VALIDATE: structural_final_validate,
            Stage.VALIDATE: validate,
        }

    def _live_handlers(self):
        def planner_adapter():
            return OpenAIJSONClient()

        def classify_plan(plan_value, client=None, force=False):
            if not force and all(
                obj.asset_role is not None
                for region in plan_value.regions for obj in region.objects
            ):
                return plan_value, []
            client = client or planner_adapter()
            classified, decisions = AssetTypeClassifier(
                client, self._runtime_records()["planner"]
            ).classify(plan_value, self.seed)
            _json(self.work / "asset_type_classification.json", {
                "status": "ok", "decisions": decisions, "model": self._runtime_records()["planner"].model_dump(),
            })
            return classified, decisions

        def context():
            return {
                "run_dir": str(self.run_dir), "work_dir": str(self.work), "seed": self.seed,
                "prompt": self.prompt,
                "cache_root": str(self.cache.root),
                "models": {k: v.model_dump() for k, v in self._runtime_records().items()},
            }

        def planning():
            client = planner_adapter()
            planner = Planner(client, self._runtime_records()["planner"])
            intent_value, plan_value, repairs = planner.plan(self.prompt, self.seed)
            authored_world_size = plan_value.world_size_m
            target_world_size = target_world_size_from_environment()
            plan_value = normalize_scene_plan(plan_value, target_world_size)
            _json(self.work / "intent.json", intent_value.model_dump())
            _json(self.work / "scene_plan.json", plan_value.model_dump(mode="json", exclude_none=True))
            return {
                "repairs": repairs,
                "asset_type_decisions": 0,
                "authored_world_size_m": list(authored_world_size),
                "target_world_size_m": list(plan_value.world_size_m),
                "world_size_normalized": authored_world_size != plan_value.world_size_m,
            }

        def plan():
            if not (self.work / "scene_plan.json").exists():
                return planning()
            self._plan()
            return {"validated": True}

        def layout():
            value = sample_layout(self._plan(), 512, seed=self.seed)
            np.save(self.work / "layout_labels.npy", value.labels)
            np.save(self.work / "layout_weights.npy", value.weights)
            _json(self.work / "layout_generation.json", value.generation)
            _json(self.work / "layout.json", {
                "region_ids": value.region_ids,
                "deterministic": True,
                "seed": self.seed,
                "selected_candidate": value.generation["selected_candidate"],
            })
            return {
                "resolution": 512,
                "seed": self.seed,
                "candidate_count": value.generation["candidate_count"],
                "selected_candidate": value.generation["selected_candidate"],
            }

        def terrain_macro_plan():
            # The live GPT planner can provide this same schema; the generic
            # semantic derivation is a deterministic fallback when the
            # optional terrain-planner endpoint is unavailable.
            layout_labels = np.load(self.work / "layout_labels.npy")
            layout_weights = np.load(self.work / "layout_weights.npy")
            layout_summary = summarize_layout(layout_labels, layout_weights, tuple(r.id for r in self._plan().regions))
            planner_input = self.work / "terrain_planner_input"
            planner_input.mkdir(exist_ok=True)
            shutil.copy2(self.work / "intent.json", planner_input / "intent.json")
            shutil.copy2(self.work / "scene_plan.json", planner_input / "scene_plan.json")
            layout_dir = planner_input / "layout"
            masks_dir = layout_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            np.save(layout_dir / "layout_labels.npy", layout_labels)
            np.save(layout_dir / "layout_weights.npy", layout_weights)
            for index, region in enumerate(self._plan().regions):
                safe_id = str(region.id).replace("/", "_").replace("\\", "_")
                np.save(masks_dir / f"{safe_id}.npy", layout_labels == index)
            _json(planner_input / "world_spec.json", {"world_size_m": list(self._plan().world_size_m)})
            _json(planner_input / "terrain_constraints.json", {"explicit_constraints": self._plan().explicit_constraints})
            _json(planner_input / "structural_intents.json", {"status": "pending", "source": "structural branch follows terrain"})
            plan, planner_meta = TerrainMacroPlanner(planner_adapter(), self._runtime_records()["planner"]).plan(
                self._plan(), self.seed, layout_summary=layout_summary,
                planner_context=_terrain_planner_context(self.work, self._plan(), layout_summary),
                request_path=self.work / "terrain_macro_planner_request.json",
                response_path=self.work / "terrain_macro_planner_response.json",
            )
            _json(self.work / "terrain_macro_plan.json", plan.model_dump(mode="json", exclude_none=True))
            terrain_root = self.work / "terrain"
            terrain_root.mkdir(exist_ok=True)
            shutil.copy2(self.work / "terrain_macro_plan.json", terrain_root / "terrain_macro_plan.json")
            return {"landforms": len(plan.landforms), "functional_zones": len(plan.functional_zones), "planner": planner_meta}

        def terrain_macro_generate():
            macro = TerrainMacroPlan.model_validate_json((self.work / "terrain_macro_plan.json").read_text(encoding="utf-8"))
            height = generate_macro_height(
                macro, 512, layout_weights=np.load(self.work / "layout_weights.npy"),
                region_ids=tuple(r.id for r in self._plan().regions),
            )
            np.save(self.work / "macro_height.npy", height)
            terrain_root = self.work / "terrain"
            terrain_root.mkdir(exist_ok=True)
            shutil.copy2(self.work / "macro_height.npy", terrain_root / "macro_height.npy")
            metrics = validate_macro_height(
                height, macro, np.load(self.work / "layout_labels.npy"),
                np.load(self.work / "layout_weights.npy"),
                tuple(r.id for r in self._plan().regions),
            )
            _json(self.work / "terrain_macro_validation.json", metrics)
            _json(self.work / "terrain_frequency.json", {"macro": {"scale_m": [50, 150], "source": "terrain_macro_plan.json", "share": "70-90%"}, "regional": {"scale_m": [5, 30], "source": "terrain.py regional operators", "share": "10-30%"}, "micro": {"scale_m": [0.2, 5], "source": "material/displacement operators", "share": "surface detail"}})
            write_validation_bundle(self.work / "terrain_validation", height, np.load(self.work / "layout_labels.npy"), metrics)
            shutil.copytree(self.work / "terrain_validation", terrain_root / "validation", dirs_exist_ok=True)
            if not metrics.get("valid"):
                feedback = {
                    "status": "REPLAN",
                    "failures": [key for key in (
                        "lake_basin_containment", "trail_traversability",
                        "macro_composition_preserved", "terrain_spikes",
                    ) if not metrics.get(key, True)],
                    "metrics": metrics,
                }
                _record_terrain_retry(self.work, feedback)
                macro, height, metrics = _replan_macro_artifacts(
                    self.work, np.load(self.work / "layout_labels.npy"),
                    np.load(self.work / "layout_weights.npy"),
                    tuple(r.id for r in self._plan().regions), 512, feedback,
                )
                if not metrics.get("valid"):
                    feedback["metrics"] = metrics
                    feedback["failures"].append("targeted_replan_invalid")
                    _record_terrain_retry(self.work, feedback)
                    macro, height, metrics = _fallback_macro_artifacts(
                        self.work, self._plan(),
                        np.load(self.work / "layout_labels.npy"),
                        np.load(self.work / "layout_weights.npy"),
                        tuple(r.id for r in self._plan().regions), 512,
                    )
                if not metrics.get("valid"):
                    raise ValueError(f"macro terrain validation failed after bounded Macro Replan: {metrics}")
            return {"resolution": 512, "global_relief": float(height.max() - height.min())}

        def terrain_visual_validate():
            metrics = json.loads((self.work / "terrain_macro_validation.json").read_text(encoding="utf-8"))
            labels = np.load(self.work / "layout_labels.npy")
            weights = np.load(self.work / "layout_weights.npy")
            layout_summary = summarize_layout(labels, weights, tuple(r.id for r in self._plan().regions))
            result = visual_validate_terrain(
                self.work / "terrain_validation", metrics,
                client=planner_adapter(), model=self._runtime_records()["planner"], seed=self.seed,
                scene_plan=self._plan().model_dump(mode="json", exclude_none=True),
                layout_summary=layout_summary,
                request_path=self.work / "terrain_visual_validation_request.json",
                response_path=self.work / "terrain_visual_validation_response.json",
            )
            if result["status"] == "REPLAN":
                labels = np.load(self.work / "layout_labels.npy")
                weights = np.load(self.work / "layout_weights.npy")
                region_ids = tuple(r.id for r in self._plan().regions)
                feedback = {
                    "status": "REPLAN",
                    "failures": result.get("failures", ["terrain_visual_validation"]),
                    "recommendations": result.get("recommendations", []),
                    "metrics": metrics,
                }
                _record_terrain_retry(self.work, feedback)
                current_macro = TerrainMacroPlan.model_validate_json(
                    (self.work / "terrain_macro_plan.json").read_text(encoding="utf-8")
                )
                replanner = TerrainMacroPlanner(
                    planner_adapter(), self._runtime_records()["planner"]
                )
                replanned, replan_meta = replanner.replan(
                    self._plan(), current_macro, feedback, seed=self.seed,
                    layout_summary=layout_summary,
                    planner_context=_terrain_planner_context(self.work, self._plan(), layout_summary),
                    request_path=self.work / "terrain_macro_replan_request.json",
                    response_path=self.work / "terrain_macro_replan_response.json",
                )
                _json(self.work / "terrain_macro_replan.json", replan_meta)
                _, height, metrics = _replan_macro_artifacts(
                    self.work, labels, weights, region_ids,
                    int(np.load(self.work / "macro_height.npy").shape[0]), feedback,
                    replanned_plan=replanned,
                )
                result = visual_validate_terrain(
                    self.work / "terrain_validation", metrics,
                    client=planner_adapter(), model=self._runtime_records()["planner"], seed=self.seed,
                    scene_plan=self._plan().model_dump(mode="json", exclude_none=True),
                    layout_summary=layout_summary,
                    request_path=self.work / "terrain_visual_validation_request.json",
                    response_path=self.work / "terrain_visual_validation_response.json",
                )
            _json(self.work / "terrain_visual_validation.json", result)
            if result["status"] != "PASS":
                raise ValueError(result)
            return result

        def terrain():
            from .layout import LayoutResult
            plan_value = self._plan()
            labels = np.load(self.work / "layout_labels.npy")
            weights = np.load(self.work / "layout_weights.npy")
            region_ids = tuple(r.id for r in plan_value.regions)
            macro_height = np.load(self.work / "macro_height.npy")
            value, macro_height, macro_preservation = _recover_regional_detail(
                self.work, plan_value, LayoutResult(labels, weights, region_ids),
                labels, weights, region_ids, stable_seed(self.prompt, self.seed),
                macro_height,
            )
            np.savez_compressed(
                self.work / "terrain.npz",
                height=value.height, vertices=value.vertices, triangles=value.triangles,
            )
            _json(self.work / "terrain_regional_validation.json", macro_preservation)
            if not macro_preservation["macro_composition_preserved"]:
                raise ValueError(f"regional detail damaged macro composition: {macro_preservation}")
            terrain_root = self.work / "terrain"
            terrain_root.mkdir(exist_ok=True)
            shutil.copy2(self.work / "terrain.npz", terrain_root / "terrain.npz")
            _write_terrain_blend(self.work)
            (terrain_root / "logs").mkdir(exist_ok=True)
            history_path = terrain_root / "logs" / "terrain_retry_history.json"
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                history = {"strategy": "macro_plan_replan", "seed_changes": 0, "attempts": []}
            history.setdefault("strategy", "macro_plan_replan")
            history.setdefault("seed_changes", 0)
            history.setdefault("attempts", [])
            history["attempt"] = len(history["attempts"])
            _json(history_path, history)
            return {
                "vertices": len(value.vertices),
                "macro_composition_preserved": macro_preservation["macro_composition_preserved"],
            }

        def structural_input():
            plan = self._plan().model_dump(mode="json", exclude_none=True)
            return prepare_structural_agent_input(self.work, plan)

        def structural_replan():
            plan = self._plan().model_dump(mode="json", exclude_none=True)
            agent_root = self.work / "structural_agent_input"
            result = StructuralFeatureAgent(
                planner_adapter(), self._runtime_records()["planner"]
            ).plan(
                plan, layout=np.load(self.work / "layout_labels.npy"),
                terrain=np.load(self.work / "terrain.npz")["height"],
                world_size=tuple(plan["world_size_m"]),
                agent_input={"root": str(agent_root), "task": str(agent_root / "structural_task.json")},
                request_path=self.work / "structural_replan_request.json",
                response_path=self.work / "structural_replan_response.json",
                seed=self.seed,
            )
            features = result["features"]
            _json(self.work / "structural_plan.json", result | {"deterministic": False})
            _json(agent_root / "structural_plan.json", result | {"deterministic": False})
            output = _materialize_structural_agent_outputs(self.work)
            return {"structural_feature_count": len(features), "structural_agent_calls": result.get("agent_calls", 1), "agent_output": str(output)}

        def structural_generate():
            started = time.monotonic()
            output = self.work / "structural_agent_output"
            if not (output / "structural_worker.py").is_file():
                output = _materialize_structural_agent_outputs(self.work)
            worker = output / "structural_worker.py"
            completed = subprocess.run([sys.executable, str(worker), "--input", str(self.work / "structural_agent_input"), "--output", str(self.work / "structural"), "--seed", str(self.seed), "--structural-plan", str(output / "structural_plan.json")], capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"structural_worker.py failed: {completed.stderr[-4000:]}")
            output = json.loads(completed.stdout)
            return {"structural_feature_count": len(output.get("features", [])), "hunyuan_calls_avoided": output.get("metrics", {}).get("hunyuan_calls_avoided", 0), "structural_generation_seconds": round(time.monotonic() - started, 3)}

        def structural_integrate():
            started = time.monotonic()
            structural_dir = self.work / "structural"
            source = np.load(self.work / "terrain.npz")
            structural = np.load(structural_dir / "terrain_structural.npz")
            modified = np.asarray(structural["height"], dtype=np.float32)
            vertices = np.asarray(structural["vertices"] if "vertices" in structural else source["vertices"], dtype=np.float32).copy()
            if len(vertices) != modified.size:
                raise ValueError("integrated terrain vertex count does not match height grid")
            vertices[:, 2] = modified.reshape(-1)
            np.savez_compressed(self.work / "terrain_structural.npz", height=modified, vertices=vertices, triangles=source["triangles"])
            branch = json.loads((structural_dir / "structural_branch.json").read_text(encoding="utf-8"))
            _json(self.work / "structural_branch.json", branch | {"status": "ok", "integration": "world_coordinates", "exclusion_mask": str(structural_dir / "structural_exclusion_mask.npy"), "occupied_mask": str(structural_dir / "structural_occupied_mask.npy"), "surface_type_mask": str(structural_dir / "surface_type_mask.npy"), "water_surface_mask": str(structural_dir / "water_surface_mask.npy"), "trail_clearance_field": str(structural_dir / "trail_clearance_field.npy"), "trail_exclusion_mask": str(structural_dir / "trail_exclusion_mask.npy"), "structural_placement_weights": str(structural_dir / "structural_placement_weights.npy"), "placement_constraints": {"avoid_exclusion": True, "modified_terrain_required": True, "base_layout_weights_immutable": True}, "terrain": str(self.work / "terrain_structural.npz")})
            return {"terrain_modified": bool(branch.get("terrain_modified")), "structural_meshes": len(branch.get("structural_meshes", [])), "terrain_modification_seconds": round(time.monotonic() - started, 3)}

        def structural_validate():
            started = time.monotonic()
            output = self.work / "structural_agent_output"
            if not (output / "structural_validator.py").is_file():
                output = _materialize_structural_agent_outputs(self.work)
            validator = output / "structural_validator.py"
            completed = subprocess.run([sys.executable, str(validator), "--input", str(self.work / "structural"), "--terrain", str(self.work / "terrain_structural.npz"), "--output", str(self.work / "structural_validation.json")], capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"structural_validator.py failed: {completed.stderr[-4000:]}")
            report = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
            if report["status"] != "pass":
                raise RuntimeError(str(report))
            report["structural_validation_seconds"] = round(time.monotonic() - started, 3)
            _json(self.work / "structural_validation.json", report)
            return report

        def run_worker(stage_name, env_name, extra=None):
            request = self.work / f"{stage_name}_request.json"
            response = self.work / f"{stage_name}_response.json"
            payload = context() | {"stage": stage_name}
            if extra:
                payload.update(extra)
            if env_name == "WORLDCLAW_IMAGE3D_WORKER" and stage_name in {
                "environment_assets", "reconstruction", "refinement_reconstruction",
            }:
                pool = HunyuanDaemonPool.from_environment()
                if pool is not None:
                    return pool.run(payload, request, response)
            return CommandWorker(env_name).run(payload, request, response)

        def worker(stage_name, env_name, extra=None):
            return lambda: run_worker(stage_name, env_name, extra)

        def mesh_validation(stage_name, source_stage):
            return lambda: run_worker(
                stage_name, "WORLDCLAW_MESH_VALIDATION_WORKER",
                {"source_stage": source_stage, "mesh_retry_max": int(os.getenv("WORLDCLAW_MESH_RETRY_MAX", "1")),
                 "input_regen_max": int(os.getenv("WORLDCLAW_MESH_INPUT_REGEN_MAX", "1"))},
            )

        def reconstruction_preflight(source_stage: str, attempt: int = 0):
            return run_worker(
                f"reconstruction_preflight_{source_stage}_{attempt}", RECON_PREFLIGHT_WORKER_ENV,
                {"source_stage": source_stage, "preflight_attempt": attempt},
            )

        def environment_assets():
            # Routing runs here, after the semantic plan is persisted.  Its
            # route annotations are passed to reference generation locally;
            # it does not rewrite the planner's semantic categories.
            # The planner's required asset_role is the semantic routing
            # contract. Reclassify only legacy/incomplete plans; forcing a
            # second model decision can overwrite a valid role and adds an
            # unnecessary API dependency before reference generation.
            plan_value, decisions = classify_plan(self._plan(), force=False)
            image_requests = environment_image_requests(plan_value)
            reference_response = self.work / "environment_references_response.json"
            reuse_references = False
            if reference_response.is_file():
                try:
                    cached = json.loads(reference_response.read_text(encoding="utf-8"))
                    cached_ids = {item.get("id") for item in cached.get("images", [])}
                    expected_ids = {item["id"] for item in image_requests}
                    reuse_references = (
                        cached.get("status") == "ok"
                        and "flux" in str(cached.get("model", "")).lower()
                        and cached_ids == expected_ids
                    )
                except (OSError, json.JSONDecodeError):
                    reuse_references = False
            if not reuse_references:
                run_worker(
                    "environment_references", REFERENCE_IMAGE_WORKER_ENV,
                    {"image_requests": image_requests},
                )
            rewrite_max = max(0, min(int(os.getenv("WORLDCLAW_PREFLIGHT_REWRITE_MAX", "1")), 1))
            preflight = {}
            for attempt in range(rewrite_max + 1):
                preflight = reconstruction_preflight("environment_assets", attempt)
                rewritten = preflight.get("rewritten_requests", [])
                if rewritten and attempt < rewrite_max:
                    run_worker(
                        "environment_references", REFERENCE_IMAGE_WORKER_ENV,
                        {"image_requests": rewritten},
                    )
                    continue
                current = json.loads(reference_response.read_text(encoding="utf-8"))
                current["images"] = preflight.get("images", [])
                current["preflight"] = {
                    "version": preflight.get("preflight_version"),
                    "decisions": preflight.get("decisions", []),
                    "vlm": preflight.get("vlm", {}),
                }
                _json(reference_response, current)
                break
            _json(self.work / "reconstruction_preflight_environment_assets_summary.json", preflight)
            if not preflight.get("images"):
                response = {"status": "ok", "assets": [], "preflight": preflight}
                _json(self.work / "environment_assets_response.json", response)
                return response | {"asset_type_decisions": len(decisions), "reference_requests": len(image_requests)}
            response = run_worker("environment_assets", "WORLDCLAW_IMAGE3D_WORKER")
            # Persistent Hunyuan daemons may outlive a source checkout and
            # return legacy route labels (for example ``tree_prototype`` or
            # a vegetation item as ``solid_object``). The reference response
            # is the stage-local routing contract, so restore its canonical
            # route by stable source id before validation and placement.
            reference_by_id = {
                str(item.get("id")): item
                for item in json.loads(reference_response.read_text(encoding="utf-8")).get("images", [])
                if isinstance(item, dict) and item.get("id") is not None
            }
            canonical_assets = []
            for asset in response.get("assets", []):
                item = dict(asset)
                reference = reference_by_id.get(str(item.get("id")))
                if reference is not None:
                    route = effective_asset_type(
                        str(reference.get("category", item.get("category", ""))),
                        reference.get("asset_role", reference.get("asset_type")),
                    ).value
                    item["asset_type"] = route
                    item["asset_role"] = route
                canonical_assets.append(item)
            response = response | {"assets": canonical_assets}
            _json(self.work / "environment_assets_response.json", response)
            return response | {"asset_type_decisions": len(decisions), "reference_requests": len(image_requests), "preflight": preflight}

        def reconstruction():
            resegment_max = max(0, min(int(os.getenv("WORLDCLAW_PREFLIGHT_RESEGMENT_MAX", "1")), 1))
            preflight = {}
            for attempt in range(resegment_max + 1):
                preflight = reconstruction_preflight("reconstruction", attempt)
                if preflight.get("resegment_required") and attempt < resegment_max:
                    segmentation_value = run_worker(
                        "segmentation_preflight_resegment", "WORLDCLAW_SEGMENT_WORKER",
                        {"seed": self.seed + 11001, "preflight_resegment": True},
                    )
                    # The bounded resegment uses an audit-specific response
                    # filename; make it the canonical input for the next
                    # preflight pass and for Hunyuan's reconstruction contract.
                    _json(self.work / "segmentation_response.json", segmentation_value)
                    continue
                segmentation_response = self.work / "segmentation_response.json"
                current = json.loads(segmentation_response.read_text(encoding="utf-8"))
                current["instances"] = preflight.get("instances", [])
                current["preflight"] = {
                    "version": preflight.get("preflight_version"),
                    "decisions": preflight.get("decisions", []),
                    "vlm": preflight.get("vlm", {}),
                }
                _json(segmentation_response, current)
                break
            _json(self.work / "reconstruction_preflight_summary.json", preflight)
            if not preflight.get("instances"):
                response = {"status": "ok", "assets": [], "preflight": preflight}
                _json(self.work / "reconstruction_response.json", response)
                return response
            response = run_worker("reconstruction", "WORLDCLAW_IMAGE3D_WORKER")
            return response | {"preflight": preflight}

        def region_composition():
            from .export import write_png
            plan_value = self._plan()
            terrain_data = np.load(self.work / "terrain.npz")
            labels = np.load(self.work / "layout_labels.npy")
            height = terrain_data["height"]
            normalized = (height - height.min()) / max(float(height.max() - height.min()), 1e-8)
            palette = np.asarray([
                [75, 116, 66], [177, 143, 80], [67, 112, 154],
                [125, 104, 82], [92, 124, 91],
            ], dtype=np.float32)
            shade = (0.55 + normalized[..., None] * 0.45)
            condition = np.clip(palette[labels % len(palette)] * shade, 0, 255).astype(np.uint8)
            condition_path = self.work / "terrain_condition.png"
            write_png(condition_path, condition)
            image_height, image_width = condition.shape[:2]
            camera_height = max(plan_value.world_size_m) * 1.2
            focal = camera_height * image_width / plan_value.world_size_m[0]
            camera = {
                "intrinsics": [[focal, 0.0, image_width / 2], [0.0, focal, image_height / 2], [0.0, 0.0, 1.0]],
                "camera_to_world": [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, camera_height], [0.0, 0.0, 0.0, 1.0]],
                "image_size": [image_width, image_height],
                "convention": "pinhole, camera +Z forward, source world Z-up",
            }
            image_requests = []
            for region in plan_value.regions:
                categories = ", ".join(obj.category for obj in region.objects if obj.count > 0)
                image_requests.append({
                    "id": f"composition_{region.id}",
                    "region_id": region.id,
                    "camera": camera,
                    "conditioning_image": str(condition_path),
                    "strength": 0.58,
                    "prompt": (
                        f"Terrain-faithful regional composition for {region.function}; "
                        f"preserve all terrain boundaries and camera geometry; include {categories}; "
                        f"{region.appearance}; no text, no frame"
                    ),
                })
            response_path = self.work / "region_composition_response.json"
            if response_path.is_file():
                try:
                    cached = json.loads(response_path.read_text(encoding="utf-8"))
                    if cached.get("status") == "ok":
                        return cached
                except (OSError, json.JSONDecodeError):
                    pass
            return run_worker(
                "region_composition", REGION_COMPOSITION_WORKER_ENV,
                {"image_requests": image_requests},
            )

        def export():
            response = run_worker("export", "WORLDCLAW_EXPORT_WORKER")
            blender = os.getenv(
                "BLENDER_BIN", str(Path.home() / "apps" / "blender-4.2.0-linux-x64" / "blender")
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "blender_finalize.py"
            profile = self._render_profile()
            # Detached live runs may outlive the terminal that started them.
            # Keep Blender's diagnostics in the run directory instead of
            # inheriting a closed stdout pipe, which makes Blender terminate
            # with SIGPIPE while emitting progress output.
            finalize_log = self.run_dir / "blender_finalize.log"
            with finalize_log.open("ab") as log:
                subprocess.run(
                    [blender, "--background", "--python", str(script), "--", "--run", str(self.run_dir), "--profile", profile],
                    stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=14400,
                )
            return response | {"render_profile": profile}

        def structural_view_plan():
            deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
            if deterministic.get("status") != "pass":
                raise RuntimeError("structural view planning is gated by deterministic validation")
            branch = json.loads((self.work / "structural_branch.json").read_text(encoding="utf-8"))
            structural_plan = json.loads((self.work / "structural_plan.json").read_text(encoding="utf-8"))
            plan = self._plan().model_dump(mode="json", exclude_none=True)
            views = plan_validation_views(
                plan, structural_plan, branch, deterministic, tuple(plan["world_size_m"]),
                client=planner_adapter(), model=self._runtime_records()["planner"], seed=self.seed,
                request_path=self.work / "structural_view_plan_request.json",
                response_path=self.work / "structural_view_plan_response.json",
            )
            _json(self.work / "validation_views.json", views)
            return {"provider": views["provider"], "fixed_views": len(views["fixed"]), "adaptive_views": len(views["adaptive"])}

        def structural_render():
            views = json.loads((self.work / "validation_views.json").read_text(encoding="utf-8"))
            blender = Path(os.getenv("BLENDER_BIN", str(Path.home() / "apps" / "blender-4.2.0-linux-x64" / "blender")))
            if not blender.is_file():
                raise FileNotFoundError(f"Blender executable for structural rendering is missing: {blender}")
            result = render_structural_views(self.work, views, blender=blender, run_dir=self.run_dir)
            plan = self._plan().model_dump(mode="json", exclude_none=True)
            structural_plan = json.loads((self.work / "structural_plan.json").read_text(encoding="utf-8"))
            deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
            bundle = build_validation_bundle(self.work, views, deterministic, structural_plan, plan, result)
            return result | bundle

        def structural_visual_validate():
            deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
            from .openai_api import OpenAIClient
            client = OpenAIClient()
            result = visual_validate_bundle(
                self.work / "validation_bundle", deterministic, client=client,
                model=self._runtime_records()["vlm"], seed=self.seed,
                request_path=self.work / "structural_visual_validation_request.json",
                response_path=self.work / "structural_visual_validation_response.json",
            )
            _json(self.work / "structural_visual_validation.json", result)
            return result

        def structural_adaptive_render():
            visual = json.loads((self.work / "structural_visual_validation.json").read_text(encoding="utf-8"))
            views = json.loads((self.work / "validation_views.json").read_text(encoding="utf-8"))
            if visual.get("confidence") == "low":
                blender = Path(os.getenv("BLENDER_BIN", str(Path.home() / "apps" / "blender-4.2.0-linux-x64" / "blender")))
                result = render_additional_views(self.work, views, visual.get("additional_views", []), blender=blender, run_dir=self.run_dir)
            else:
                result = {"status": "ok", "needed": False, "renders": []}
            if result.get("needed"):
                views["additional"] = result.get("requested", [])
                _json(self.work / "validation_views.json", views)
                plan = self._plan().model_dump(mode="json", exclude_none=True)
                structural_plan = json.loads((self.work / "structural_plan.json").read_text(encoding="utf-8"))
                deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
                build_validation_bundle(self.work, views, deterministic, structural_plan, plan, result, result.get("requested", []))
            _json(self.work / "structural_adaptive_render.json", result)
            return result

        def structural_final_validate():
            first = json.loads((self.work / "structural_visual_validation.json").read_text(encoding="utf-8"))
            if first.get("confidence") != "low":
                result = first | {"phase": "final", "follow_up": False}
            else:
                deterministic = json.loads((self.work / "structural_validation.json").read_text(encoding="utf-8"))
                from .openai_api import OpenAIClient
                result = visual_validate_bundle(
                    self.work / "validation_bundle", deterministic, client=OpenAIClient(),
                    model=self._runtime_records()["vlm"], seed=self.seed, include_additional=True,
                    request_path=self.work / "structural_final_validation_request.json",
                    response_path=self.work / "structural_final_validation_response.json",
                ) | {"phase": "final", "follow_up": True}
            _json(self.work / "structural_final_validation.json", result)
            return result

        def validate():
            self._write_mesh_reuse_manifest()
            report = _merge_structural_validation_metrics(
                self.run_dir, validate_run(self.run_dir, full=self._render_profile() == "full")
            )
            _json(self.run_dir / "metrics.json", report)
            if not report["valid"]:
                raise RuntimeError(str(report))
            return report

        return {
            Stage.INTENT: planning, Stage.PLAN: plan, Stage.LAYOUT: layout,
            Stage.TERRAIN_MACRO_PLAN: terrain_macro_plan, Stage.TERRAIN_MACRO_GENERATE: terrain_macro_generate,
            Stage.TERRAIN: terrain, Stage.TERRAIN_VISUAL_VALIDATE: terrain_visual_validate,
            Stage.STRUCTURAL_INPUT: structural_input,
            Stage.STRUCTURAL_REPLAN: structural_replan,
            Stage.STRUCTURAL_GENERATE: structural_generate,
            Stage.STRUCTURAL_INTEGRATE: structural_integrate,
            Stage.STRUCTURAL_VALIDATE: structural_validate,
            Stage.ENV_ASSETS: environment_assets,
            Stage.MESH_VALIDATE_ENV_ASSETS: mesh_validation(
                "mesh_validation_environment_assets", "environment_assets"
            ),
            Stage.REGION_COMPOSE: region_composition,
            Stage.SEGMENT: worker("segmentation", "WORLDCLAW_SEGMENT_WORKER"),
            Stage.RECONSTRUCT: reconstruction,
            Stage.MESH_VALIDATE_RECONSTRUCT: mesh_validation(
                "mesh_validation_reconstruct", "reconstruction"
            ),
            Stage.PLACE: worker("placement", "WORLDCLAW_PLACEMENT_WORKER"),
            Stage.REFINE: worker("refinement", "WORLDCLAW_REFINEMENT_WORKER"),
            Stage.EXPORT: export,
            Stage.STRUCTURAL_VIEW_PLAN: structural_view_plan,
            Stage.STRUCTURAL_RENDER: structural_render,
            Stage.STRUCTURAL_VISUAL_VALIDATE: structural_visual_validate,
            Stage.STRUCTURAL_ADAPTIVE_RENDER: structural_adaptive_render,
            Stage.STRUCTURAL_FINAL_VALIDATE: structural_final_validate,
            Stage.VALIDATE: validate,
        }


def new_run_id(prompt: str) -> str:
    return (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-"
        + hashlib.sha256(prompt.encode()).hexdigest()[:8] + "-" + uuid.uuid4().hex[:4]
    )
