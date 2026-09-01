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
from typing import Callable

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
    stable_seed,
    synthetic_plan,
    target_world_size_from_environment,
)
from .models import AssetTypeClassifier, CommandWorker, ModelLock, OpenAIJSONClient, Planner, VLLMClient
from .schemas import AssetInstance, RunManifest, ScenePlan, Stage
from .state import StateDB
from .structural import (
    StructuralFeatureAgent, build_structural_geometry, structural_features_from_plan, validate_structural_branch,
)
from .structural_workflow import (
    build_validation_bundle, plan_validation_views, prepare_structural_agent_input,
    render_additional_views, render_structural_views, visual_validate_bundle,
)
from .terrain import boundary_discontinuity, generate_terrain
from .validation import sha256, validate_run


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
            value = deterministic_layout(self._plan(), 128)
            np.save(self.work / "layout_labels.npy", value.labels)
            np.save(self.work / "layout_weights.npy", value.weights)
            _json(self.work / "layout.json", {"region_ids": value.region_ids, "deterministic": True})
            return {"resolution": 128}

        def terrain():
            from .layout import LayoutResult
            plan_value = self._plan()
            labels = np.load(self.work / "layout_labels.npy")
            weights = np.load(self.work / "layout_weights.npy")
            value = generate_terrain(
                plan_value,
                LayoutResult(labels, weights, tuple(r.id for r in plan_value.regions)),
                stable_seed(self.prompt, self.seed),
            )
            np.savez_compressed(
                self.work / "terrain.npz",
                height=value.height, vertices=value.vertices, triangles=value.triangles,
            )
            return {
                "vertices": len(value.vertices),
                "boundary_max_delta": boundary_discontinuity(value.height, labels),
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
            _json(self.work / "structural_branch.json", branch | {"status": "ok", "integration": "world_coordinates", "exclusion_mask": str(structural_dir / "structural_exclusion_mask.npy"), "occupied_mask": str(structural_dir / "structural_occupied_mask.npy"), "surface_type_mask": str(structural_dir / "surface_type_mask.npy"), "trail_clearance_field": str(structural_dir / "trail_clearance_field.npy"), "trail_exclusion_mask": str(structural_dir / "trail_exclusion_mask.npy"), "structural_placement_weights": str(structural_dir / "structural_placement_weights.npy"), "placement_constraints": {"avoid_exclusion": True, "modified_terrain_required": True, "base_layout_weights_immutable": True}, "terrain": str(self.work / "terrain_structural.npz")})
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
            semantic = {
                key: {index for index, region in enumerate(plan_value.regions)
                      if key in f"{region.id} {region.function}".lower()}
                for key in ("lake", "shore", "forest")
            }
            lake_mask = np.isin(labels, list(semantic["lake"])) if semantic["lake"] else np.zeros_like(labels, dtype=bool)
            lake_points = np.argwhere(lake_mask)
            if len(lake_points):
                padded = np.pad(lake_mask, 1, mode="constant", constant_values=False)
                interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
                boundary_points = np.argwhere(lake_mask & ~interior)
                if len(boundary_points):
                    lake_points = boundary_points
            distance_to_lake = np.full_like(height, np.inf, dtype=np.float32)
            if len(lake_points):
                grid = np.argwhere(np.ones_like(labels, dtype=bool))
                spacing = np.asarray([plan_value.world_size_m[1] / max(labels.shape[0] - 1, 1), plan_value.world_size_m[0] / max(labels.shape[1] - 1, 1)])
                for start in range(0, len(grid), 4096):
                    chunk = grid[start:start + 4096]
                    delta = (chunk[:, None, :] - lake_points[None, :, :]) * spacing
                    distance_to_lake[chunk[:, 0], chunk[:, 1]] = np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)
            lake_span = max(plan_value.world_size_m) if not len(lake_points) else max(float(np.ptp(lake_points[:, 0])), float(np.ptp(lake_points[:, 1]))) * min(plan_value.world_size_m) / max(labels.shape)
            rng = random.Random(stable_seed(self.prompt, self.seed))
            records, meshes = [], []
            width, depth = plan_value.world_size_m
            base_v, base_t = box_mesh()
            for region_index, region in enumerate(plan_value.regions):
                for spec in region.objects:
                    if is_structural_feature(spec.category, spec.asset_role):
                        continue
                    category = spec.category.lower()
                    gate = (layout_weights[region_index] >= 0.05) & (~exclusion)
                    if category in {"reed", "reeds"}:
                        if semantic["shore"]:
                            gate &= np.max(layout_weights[list(semantic["shore"])], axis=0) >= 0.15
                        if semantic["lake"]:
                            gate &= np.max(layout_weights[list(semantic["lake"])], axis=0) < 0.8
                        if semantic["forest"]:
                            gate &= np.max(layout_weights[list(semantic["forest"])], axis=0) < 0.8
                    if category in {"tree", "trees", "palm", "palms"}:
                        gate &= ~trail_exclusion
                    probabilities = np.clip(layout_weights[region_index] * influence, 0.0, None)
                    if category in {"reed", "reeds"}:
                        probabilities *= np.exp(-distance_to_lake / max(lake_span * 0.15, 1e-3))
                    candidates = [item for item in np.argwhere(gate).tolist() if probabilities[item[0], item[1]] > 1e-6]
                    candidates.sort(key=lambda item: -math.log(max(rng.random(), 1e-12)) / max(float(probabilities[item[0], item[1]]), 1e-6))
                    cursor = 0
                    for index in range(spec.count):
                        while cursor < len(candidates) and exclusion[candidates[cursor][0], candidates[cursor][1]]:
                            cursor += 1
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
            Stage.INTENT: intent, Stage.PLAN: plan, Stage.LAYOUT: layout, Stage.TERRAIN: terrain,
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
            value = deterministic_layout(self._plan(), 512)
            np.save(self.work / "layout_labels.npy", value.labels)
            np.save(self.work / "layout_weights.npy", value.weights)
            return {"resolution": 512}

        def terrain():
            from .layout import LayoutResult
            plan_value = self._plan()
            labels = np.load(self.work / "layout_labels.npy")
            weights = np.load(self.work / "layout_weights.npy")
            value = generate_terrain(
                plan_value, LayoutResult(labels, weights, tuple(r.id for r in plan_value.regions)),
                stable_seed(self.prompt, self.seed),
            )
            np.savez_compressed(
                self.work / "terrain.npz",
                height=value.height, vertices=value.vertices, triangles=value.triangles,
            )
            return {"vertices": len(value.vertices)}

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
            _json(self.work / "structural_branch.json", branch | {"status": "ok", "integration": "world_coordinates", "exclusion_mask": str(structural_dir / "structural_exclusion_mask.npy"), "occupied_mask": str(structural_dir / "structural_occupied_mask.npy"), "surface_type_mask": str(structural_dir / "surface_type_mask.npy"), "trail_clearance_field": str(structural_dir / "trail_clearance_field.npy"), "trail_exclusion_mask": str(structural_dir / "trail_exclusion_mask.npy"), "structural_placement_weights": str(structural_dir / "structural_placement_weights.npy"), "placement_constraints": {"avoid_exclusion": True, "modified_terrain_required": True, "base_layout_weights_immutable": True}, "terrain": str(self.work / "terrain_structural.npz")})
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
            subprocess.run(
                [blender, "--background", "--python", str(script), "--", "--run", str(self.run_dir), "--profile", profile],
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
            report = _merge_structural_validation_metrics(
                self.run_dir, validate_run(self.run_dir, full=self._render_profile() == "full")
            )
            _json(self.run_dir / "metrics.json", report)
            if not report["valid"]:
                raise RuntimeError(str(report))
            return report

        return {
            Stage.INTENT: planning, Stage.PLAN: plan, Stage.LAYOUT: layout, Stage.TERRAIN: terrain,
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
