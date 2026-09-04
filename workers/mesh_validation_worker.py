#!/usr/bin/env python3
"""Asset-level mesh quality gate: turntable renders, VLM inspection, and retries."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import subprocess
import base64
from pathlib import Path
from typing import Any

import numpy as np

from worldclaw_oss.asset_semantics import effective_asset_type, is_structural_feature
from worldclaw_oss.asset_types import AssetType
from worldclaw_oss.hunyuan_pool import HunyuanDaemonPool
from worldclaw_oss.openai_api import OpenAIClient
from worldclaw_oss.representation_fallback import (
    ValidatedAssetRegistry,
    resolve_stage3,
)
from worldclaw_oss.schemas import MeshValidationReport
from workers.common import load_stage_response, read_request, worker_args, write_response


# Bump this whenever the local geometry/VLM validation policy changes. The
# version is part of the cache key so old verdicts cannot silently survive a
# validator behavior change. It may be overridden for an explicitly versioned
# deployment or smoke test.
MESH_VALIDATOR_VERSION = os.getenv("WORLDCLAW_MESH_VALIDATOR_VERSION", "mesh-validator-v4")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_sha256(mesh: Path) -> str:
    """Hash mesh bytes, rather than its path, so aliases share validation."""
    return _sha256_file(mesh)


def _reference_sha256(reference: Any) -> str:
    """Hash reference bytes when local, otherwise hash its stable URI/value."""
    value = str(reference or "")
    candidate = Path(value)
    if value and candidate.is_file():
        return _sha256_file(candidate)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validation_cache_key(mesh_sha256: str, reference_sha256: str, validator_version: str) -> str:
    """Return the canonical cache key for the required three-part identity."""
    payload = {
        "mesh_sha256": str(mesh_sha256),
        "reference_sha256": str(reference_sha256),
        "validator_version": str(validator_version),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_validation_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_validation_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _make_fallback_asset(asset: dict[str, Any], output_root: Path) -> dict[str, Any] | None:
    """Create a small route-native replacement for failed non-solid assets.

    The replacement is deliberately marked as procedural fallback; it is never
    reported as validation pass for the original mesh.
    """
    import trimesh

    route = effective_asset_type(asset.get("category", ""), asset.get("asset_type"))
    if route not in {
        AssetType.PROCEDURAL_NATIVE,
        AssetType.LINEAR_STRUCTURE,
        AssetType.SURFACE_REGION_FEATURE,
    }:
        return None
    category = str(asset.get("category", "asset"))
    safe_id = str(asset.get("id", "asset")).replace("/", "_").replace("\\", "_")
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{safe_id}.fallback.glb"
    if route == AssetType.PROCEDURAL_NATIVE:
        trunk = trimesh.creation.cylinder(radius=0.10, height=1.0, sections=12)
        trunk.apply_translation([0.0, 0.0, 0.5])
        crown = trimesh.creation.cone(radius=0.42, height=1.15, sections=16)
        crown.apply_translation([0.0, 0.0, 1.25])
        mesh = trimesh.util.concatenate([trunk, crown])
        generator = "procedural-vegetation-prototype-v1"
    elif route == AssetType.LINEAR_STRUCTURE:
        mesh = trimesh.creation.box(extents=[4.0, 0.7, 0.12])
        generator = "procedural-linear-strip-v1"
    else:
        mesh = trimesh.creation.box(extents=[4.0, 4.0, 0.08])
        generator = "procedural-surface-feature-v1"
    mesh.export(output, file_type="glb")
    replacement = copy.deepcopy(asset)
    replacement["asset_type"] = route.value
    replacement["mesh"] = str(output)
    replacement["source_model"] = generator
    replacement["fallback"] = generator
    replacement["material"] = category
    return replacement


def _mesh_metrics(path: Path, category: str, asset_type: str | None) -> tuple[dict[str, Any], list[str]]:
    import trimesh
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    meshes = [mesh for mesh in meshes if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError(f"mesh is empty: {path}")
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    extents = np.asarray(mesh.extents, dtype=float)
    maximum = max(float(extents.max()), 1e-9)
    minimum = float(extents.min())
    ratio = minimum / maximum
    defects: list[str] = []
    route = effective_asset_type(category, asset_type)
    # Card vegetation is an intentional representation. All other routes must
    # retain meaningful volume in both horizontal axes.
    if route not in {AssetType.LINEAR_STRUCTURE, AssetType.SURFACE_REGION_FEATURE}:
        if ratio < 0.035:
            defects.append("flattened_geometry")
        if min(float(extents[0]), float(extents[1])) / maximum < 0.06:
            defects.append("radial_crossed_card_or_thin_side_profile")
    if len(mesh.vertices) < 8 or len(mesh.faces) < 4:
        defects.append("missing_volume")
    if not np.isfinite(mesh.vertices).all():
        defects.append("abnormal_stretching")
    topology_warnings: list[str] = []
    watertight = bool(getattr(mesh, "is_watertight", False))
    if not watertight:
        topology_warnings.append("non_watertight")
    try:
        incidence = np.bincount(
            np.asarray(mesh.edges_unique_inverse, dtype=np.int64),
            minlength=len(mesh.edges_unique),
        )
        non_manifold_edges = int(np.count_nonzero(incidence > 2))
    except (AttributeError, TypeError, ValueError):
        non_manifold_edges = 0
    if non_manifold_edges:
        topology_warnings.append("non_manifold_edges")
    return {
        "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
        "extents": extents.tolist(), "volume_ratio": ratio,
        "watertight": watertight,
        "non_manifold_edges": non_manifold_edges,
        "topology_warnings": topology_warnings,
    }, defects


def _render(blender: str, mesh: Path, output: Path) -> list[Path]:
    # Workers are run both from the repository (workers/../scripts) and from
    # an isolated run directory where the helper is staged as scripts/... .
    candidates = [
        Path(__file__).resolve().parents[1] / "scripts" / "blender_mesh_turntable.py",
        Path(__file__).resolve().parent / "scripts" / "blender_mesh_turntable.py",
    ]
    script = next((candidate for candidate in candidates if candidate.is_file()), None)
    if script is None:
        raise FileNotFoundError("blender_mesh_turntable.py not found beside worker")
    subprocess.run(
        [blender, "--background", "--python", str(script), "--", "--glb", str(mesh), "--output", str(output)],
        check=True, timeout=3600, capture_output=True, text=True,
    )
    views = sorted(output.glob("view_*.png"))
    if len(views) != 8 or any(not path.is_file() or path.stat().st_size == 0 for path in views):
        raise RuntimeError(f"expected eight non-empty turntable views, got {len(views)}")
    return views


def _contact_sheet(views: list[Path]) -> Path:
    """Persist one compact image containing all ordered turntable views."""
    from PIL import Image, ImageDraw

    if not views:
        raise ValueError("cannot build a contact sheet without turntable views")
    output = views[0].parent / "contact_sheet.jpg"
    tile_size = 320
    columns = 4
    rows = (len(views) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(views):
        with Image.open(path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((tile_size - 8, tile_size - 28), Image.Resampling.LANCZOS)
            x = (index % columns) * tile_size + (tile_size - tile.width) // 2
            y = (index // columns) * tile_size + 22 + (tile_size - 22 - tile.height) // 2
            sheet.paste(tile, (x, y))
        draw.text(((index % columns) * tile_size + 8, (index // columns) * tile_size + 4), f"view_{index:02d}", fill=(255, 255, 255))
    sheet.save(output, format="JPEG", quality=88, optimize=True)
    return output


def _inspect(request: dict, asset: dict, views: list[Path], metrics: dict, defects: list[str]) -> dict:
    system = """You are an asset-level 3D mesh quality inspector.

Inspect all eight turntable views and determine whether the asset is visually
and semantically usable as a 3D scene asset.

Reject assets for:
- radial crossed-card degeneration or repeated intersecting planes;
- missing 3D volume or severely flattened side profiles;
- abnormal stretching or collapsed geometry;
- large visible holes, missing major surfaces, or clearly broken geometry;
- wrong semantic structure;
- severe texture/geometry mismatch.

Important geometry-metric rules:
- Geometry metrics are auxiliary evidence only and must not override clear visual evidence.
- watertight=false alone is NOT a defect and MUST NOT cause rejection.
- A non-watertight mesh may still be completely valid for rendering.
- Open bottoms, intentionally thin surfaces, disconnected decorative parts,
  UV seams, and hidden boundary edges are acceptable if they do not produce
  visible structural defects in the turntable views.
- Only report "non_watertight_geometry" when non-watertightness corresponds
  to clearly visible major holes, missing surfaces, or broken geometry.
- Do not infer visible holes or broken geometry solely from watertight=false.
- Do not reject an otherwise coherent cabin, rock, vegetation asset, or other
  renderable mesh merely because it is not a closed manifold.

For vegetation:
- A deliberate single billboard or small controlled alpha-card representation
  may be acceptable.
- Radial crossed-card degeneration, repeated intersecting planes, or an
  unintended starburst/card structure is always a defect.
- Inspect the 90-degree and 270-degree side views carefully.

Prioritize the eight rendered views when judging reconstruction quality.
Use geometry metrics only to support or investigate a visually observed defect.

Do not infer quality from category or bounding-box extents.

Return exactly status, severity, defects, reason, retry_action as JSON."""
    contact_sheet = _contact_sheet(views)
    user = json.dumps({
        "asset_id": asset.get("id"), "category": asset.get("category"),
        "asset_type": asset.get("asset_type"), "reference": asset.get("source_image"),
        "geometry_metrics": metrics, "local_defects": defects,
        "turntable_views": {
            "count": len(views),
            "order": [path.name for path in views],
            "contact_sheet": str(contact_sheet),
        },
    }, ensure_ascii=False)
    if os.getenv("WORLDCLAW_MESH_VALIDATION_LOCAL_ONLY", "0") == "1":
        severity = min(1.0, 0.35 * len(defects))
        return {
            "status": "fail" if defects else "pass", "severity": severity,
            "defects": defects, "reason": "Local geometry-only gate; VLM disabled",
            "retry_action": "reconstruct with a new seed" if defects else "",
        }
    validation_provider = os.getenv("WORLDCLAW_VALIDATION_PROVIDER", "openai").lower()
    if validation_provider == "openai":
        client = OpenAIClient()
        value = client.vision_json(
            system=system, user=user, image_paths=[contact_sheet],
            schema=MeshValidationReport.model_json_schema(),
            model=client.config.validation_model,
        )
        return _coerce_report(value)
    url = os.getenv("WORLDCLAW_VLM_URL", os.getenv("WORLDCLAW_VLLM_URL", "")).rstrip("/")
    if not url:
        raise RuntimeError("mesh validation requires WORLDCLAW_VLM_URL or OpenAI provider")
    import requests
    content = [{"type": "text", "text": user}]
    encoded = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
    content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}})
    payload = {
        "model": request["models"]["vlm"]["model_id"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "temperature": 0.0, "seed": int(request["seed"]),
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "mesh_validation_report", "strict": True,
            "schema": MeshValidationReport.model_json_schema(),
        }},
    }
    headers = {"Content-Type": "application/json"}
    token = os.getenv("VLLM_API_KEY", "")
    if token:
        headers["Authorization"] = "Bearer " + token
    response = requests.post(url + "/v1/chat/completions", json=payload, headers=headers, timeout=900)
    response.raise_for_status()
    value = response.json()["choices"][0]["message"]["content"]
    if isinstance(value, list):
        value = "".join(str(item.get("text", "")) for item in value)
    return _coerce_report(json.loads(value))


def _coerce_report(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize compatible VLM aliases before enforcing the pipeline schema."""
    raw = dict(value)
    normalized = dict(value)
    if "status" not in normalized and "decision" in normalized:
        normalized["status"] = str(normalized.pop("decision")).lower()
    if "status" not in normalized:
        for key in ("verdict", "result", "label", "classification", "validity"):
            if key in normalized:
                normalized["status"] = str(normalized.pop(key)).lower()
                break
    if "status" in normalized:
        status = str(normalized["status"]).lower()
        normalized["status"] = {"accept": "pass", "accepted": "pass", "reject": "fail", "rejected": "fail"}.get(status, status)
    if "defects" not in normalized and "issues" in normalized:
        normalized["defects"] = normalized.pop("issues")
    if "defects" not in normalized:
        for key in ("problems", "concerns", "defect", "failure_modes"):
            if key in normalized:
                normalized["defects"] = normalized.pop(key)
                break
    if "reason" not in normalized and "summary" in normalized:
        normalized["reason"] = normalized.pop("summary")
    if "reason" not in normalized:
        for key in ("analysis", "assessment", "explanation", "rationale", "details", "comment"):
            if key in normalized:
                normalized["reason"] = normalized.pop(key)
                break
    if isinstance(normalized.get("reason"), dict):
        detail = normalized["reason"]
        if "defects" not in normalized:
            for key in ("observed_defects", "defects", "issues", "problems"):
                if key in detail:
                    normalized["defects"] = detail[key]
                    break
        reason_text = next(
            (detail[key] for key in ("summary", "explanation", "assessment", "reason") if isinstance(detail.get(key), str)),
            json.dumps(detail, ensure_ascii=False, sort_keys=True),
        )
        normalized["reason"] = reason_text
    if normalized.get("defects") is None:
        normalized["defects"] = []
    if isinstance(normalized.get("defects"), dict):
        normalized["defects"] = [str(key) for key, enabled in normalized["defects"].items() if enabled]
    elif isinstance(normalized.get("defects"), list):
        normalized["defects"] = [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True) for item in normalized["defects"]]
    if "status" not in normalized:
        # Some compatible gateways omit the verdict while returning the
        # requested defect/summary fields. Treat an empty defect list plus a
        # positive summary as pass; any reported issue remains a failure.
        issues = normalized.get("issues", normalized.get("defects", []))
        summary = str(normalized.get("reason", normalized.get("summary", ""))).lower()
        positive = any(token in summary for token in ("coherent", "acceptable", "no defect", "pass", "maintains"))
        normalized["status"] = "pass" if not issues and positive else "fail"
    normalized.setdefault("defects", [])
    if normalized.get("reason") is None:
        normalized["reason"] = ""
    if normalized.get("retry_action") is None:
        normalized["retry_action"] = ""
    normalized.setdefault("reason", "")
    normalized.setdefault("retry_action", "")
    normalized.setdefault(
        "severity", 0.0 if normalized.get("status") == "pass" else 1.0
    )
    if isinstance(normalized.get("severity"), str):
        severity_text = normalized["severity"].strip().lower()
        severity_aliases = {
            "none": 0.0, "negligible": 0.1, "low": 0.25,
            "minor": 0.35, "moderate": 0.55, "major": 0.75,
            "high": 0.85, "severe": 0.95, "critical": 1.0,
        }
        try:
            normalized["severity"] = float(severity_text)
        except ValueError:
            normalized["severity"] = severity_aliases.get(
                severity_text, 0.0 if normalized.get("status") == "pass" else 1.0
            )
    if not normalized["reason"] and normalized.get("status") == "fail":
        # Keep a useful audit trail when a gateway strips the model's prose.
        strings = [str(item) for key, item in raw.items() if key not in {"asset_id", "status", "decision"} and isinstance(item, str) and item.strip()]
        normalized["reason"] = "VLM rejected the mesh without a structured diagnosis" + (": " + " ".join(strings) if strings else "")
    allowed = {"status", "severity", "defects", "reason", "retry_action"}
    normalized = {key: item for key, item in normalized.items() if key in allowed}
    return MeshValidationReport.model_validate(normalized).model_dump(mode="json")


def _run_worker(command: str, request: dict, path: Path, response: Path) -> dict:
    path.write_text(json.dumps(request, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    if command == os.getenv("WORLDCLAW_IMAGE3D_WORKER", "") and request.get("stage") in {
        "environment_assets", "reconstruction", "refinement_reconstruction",
    }:
        pool = HunyuanDaemonPool.from_environment()
        if pool is not None:
            return pool.run(request, path, response)
    subprocess.run(
        shlex.split(command, posix=os.name != "nt") + ["--request", str(path), "--response", str(response)],
        check=True, timeout=7200,
    )
    value = json.loads(response.read_text(encoding="utf-8"))
    if value.get("status") != "ok":
        raise RuntimeError(value.get("error", "mesh retry worker failed"))
    return value


def _reference_prompt(work: Path, asset: dict[str, Any], request: dict[str, Any]) -> str:
    """Recover the original per-asset prompt without inventing a new one."""
    references = work / "environment_references_response.json"
    if references.is_file():
        value = json.loads(references.read_text(encoding="utf-8"))
        entries = value.get("images", value.get("outputs", []))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("id") == asset.get("id") or entry.get("category") == asset.get("category"):
                prompt = str(entry.get("prompt", "")).strip()
                if prompt:
                    return prompt
    for key in ("original_prompt", "source_prompt", "prompt"):
        prompt = str(asset.get(key, "")).strip()
        if prompt:
            return prompt
    return f"Isolated {asset.get('category', 'object')}, exactly one reusable instance, centered, full body"


def _rewrite_prompt(original: str, asset: dict[str, Any], report: dict[str, Any] | None) -> str:
    defects = ", ".join(str(item) for item in (report or {}).get("defects", []) if str(item).strip())
    return (
        f"{original}. Rewrite the reference to correct the failed reconstruction for "
        f"{asset.get('category', 'object')}: clear volumetric 3D silhouette, substantial "
        "side profiles, continuous major surfaces, no radial crossed cards, no repeated "
        "intersecting planes, no backdrop planes, no texture stretching or geometry mismatch."
        + (f" Specifically address: {defects}." if defects else "")
    )


def main() -> None:
    args = worker_args()
    request = read_request(args.request)
    # Every Stage 1 retry starts from this immutable request snapshot. Stage 2
    # prompt rewriting and later candidate replacements must not accumulate
    # into the next original-seed retry.
    initial_request = copy.deepcopy(request)
    work = Path(request["work_dir"])
    run_dir = Path(request["run_dir"])
    source_stage = request.get("source_stage", "environment_assets")
    upstream = load_stage_response(work, source_stage)
    assets = copy.deepcopy(upstream.get("assets", []))
    structural_assets = [item for item in assets if is_structural_feature(item.get("category", ""), item.get("asset_type"))]
    assets = [item for item in assets if not is_structural_feature(item.get("category", ""), item.get("asset_type"))]
    for asset in assets:
        if isinstance(asset, dict):
            asset["asset_type"] = effective_asset_type(
                asset.get("category", ""), asset.get("asset_type")
            ).value
    reports: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    cache_path = Path(
        os.getenv("WORLDCLAW_MESH_VALIDATION_CACHE", str(run_dir / "mesh_validation_cache.json"))
    )
    validation_cache = _load_validation_cache(cache_path)
    mesh_cache_path = Path(
        os.getenv(
            "WORLDCLAW_MESH_SHA256_CACHE",
            str(cache_path.with_name(cache_path.stem + "_by_mesh.json")),
        )
    )
    mesh_cache = _load_validation_cache(mesh_cache_path)
    registry = ValidatedAssetRegistry(
        request.get("validated_asset_registry", str(run_dir / "validated_asset_registry.json"))
    )
    cache_hits = 0
    mesh_cache_hits = 0
    unique_validation_keys: set[str] = set()
    performed_validation_calls = 0
    mesh_hashes: dict[str, str] = {}
    reference_hashes: dict[str, str] = {}
    stage3_stats = {
        "stage3_enter_count": 0, "prototype_reuse_count": 0,
        "procedural_fallback_count": 0, "asset_library_fallback_count": 0,
        "primitive_fallback_count": 0, "drop_count": 0,
        "hunyuan_calls_avoided_by_stage3": 0,
    }
    asset_library = request.get("asset_library", [])
    if isinstance(asset_library, (str, Path)):
        try:
            loaded_library = json.loads(Path(asset_library).read_text(encoding="utf-8"))
            asset_library = loaded_library.get("assets", loaded_library) if isinstance(loaded_library, dict) else loaded_library
        except (OSError, json.JSONDecodeError):
            asset_library = []
    if not isinstance(asset_library, list):
        asset_library = []

    def cached_mesh_sha256(path: Path) -> str:
        resolved = str(path.resolve())
        if resolved not in mesh_hashes:
            mesh_hashes[resolved] = _mesh_sha256(path)
        return mesh_hashes[resolved]

    def cached_reference_sha256(value: Any) -> str:
        marker = str(value or "")
        if marker not in reference_hashes:
            reference_hashes[marker] = _reference_sha256(value)
        return reference_hashes[marker]

    max_retry = max(0, min(int(request.get("mesh_retry_max", 1)), 3))
    regen_max = max(0, min(int(request.get("input_regen_max", 1)), 2))
    force_retry_ids = {str(item) for item in request.get("force_retry_asset_ids", [])}
    blender = os.getenv("BLENDER_BIN", str(Path.home() / "apps" / "blender-4.2.0-linux-x64" / "blender"))
    image_worker = os.getenv("WORLDCLAW_IMAGE3D_WORKER", "")
    # Reference regeneration must use the GPT Image 2 worker.  Regional
    # composition continues to use WORLDCLAW_FLUX_WORKER in the pipeline.
    reference_worker = os.getenv("WORLDCLAW_REFERENCE_IMAGE_WORKER", "")
    segment_worker = os.getenv("WORLDCLAW_SEGMENT_WORKER", "")
    for index, asset in enumerate(assets):
        current = asset
        initial_mesh = Path(current["mesh"])
        initial_mesh_sha256 = cached_mesh_sha256(initial_mesh)
        reference_sha256 = cached_reference_sha256(current.get("source_image"))
        cache_key = _validation_cache_key(
            initial_mesh_sha256, reference_sha256, MESH_VALIDATOR_VERSION
        )
        cached = validation_cache.get(cache_key)
        force_cached_retry = str(current.get("id")) in force_retry_ids
        cached_asset = cached.get("asset") if isinstance(cached, dict) else None
        cached_asset_available = (
            isinstance(cached_asset, dict)
            and Path(str(cached_asset.get("mesh", ""))).is_file()
        )
        if (
            isinstance(cached, dict)
            and cached.get("final_status") in {"pass", "fallback_pass", "dropped"}
            and not force_cached_retry
            and (cached.get("final_status") == "dropped" or cached_asset_available)
        ):
            # Reuse the complete prototype decision, including any successful
            # retry or route-native fallback. Only the instance id is unique.
            cache_hits += 1
            cached_report = copy.deepcopy(cached.get("report") or {})
            cached_source_asset_id = cached_report.get("asset_id")
            cached_report.update({
                "asset_id": current.get("id"),
                "cache_source_asset_id": cached_source_asset_id,
                "cache_key": cache_key,
                "mesh_sha256": initial_mesh_sha256,
                "reference_sha256": reference_sha256,
                "validator_version": MESH_VALIDATOR_VERSION,
                "cache_level": "exact",
                "cache_hit": True,
                "validation_performed": False,
            })
            reports.append(cached_report)
            final_status = cached["final_status"]
            if final_status in {"pass", "fallback_pass"} and isinstance(cached.get("asset"), dict):
                current = copy.deepcopy(cached["asset"])
                current["id"] = asset.get("id")
                assets[index] = current
                if final_status in {"pass", "fallback_pass"}:
                    registry.register(current | {"final_status": "pass"}, source_stage=source_stage)
            continue

        unique_validation_keys.add(cache_key)
        accepted = False
        last_report: dict[str, Any] | None = None
        retry_metadata: dict[str, Any] = {}
        for attempt in range(max_retry + 1):
            mesh = Path(current["mesh"])
            view_dir = run_dir / "mesh_validation_views" / source_stage / str(current.get("id", index)) / f"attempt_{attempt + 1:02d}"
            attempt_mesh_sha256 = cached_mesh_sha256(mesh)
            attempt_cache_key = _validation_cache_key(
                attempt_mesh_sha256, reference_sha256, MESH_VALIDATOR_VERSION
            )
            mesh_cached = mesh_cache.get(attempt_mesh_sha256)
            if (
                isinstance(mesh_cached, dict)
                and mesh_cached.get("validator_version") == MESH_VALIDATOR_VERSION
                and mesh_cached.get("status") in {"pass", "fail"}
                and isinstance(mesh_cached.get("report"), dict)
            ):
                # Mesh identity is the second-level cache. It intentionally
                # ignores reference/id so one validated environment prototype
                # can serve every later instance and retry stage.
                mesh_cache_hits += 1
                reused = copy.deepcopy(mesh_cached["report"])
                reused.update({
                    "asset_id": current.get("id"), "attempt": attempt + 1,
                    "mesh": str(mesh), "mesh_sha256": attempt_mesh_sha256,
                    "reference_sha256": reference_sha256,
                    "validator_version": MESH_VALIDATOR_VERSION,
                    "cache_key": attempt_cache_key,
                    "cache_level": "mesh_sha256", "cache_hit": True,
                    "validation_performed": False,
                    "reused_mesh_validation": True,
                })
                history.append(reused)
                last_report = reused
                if reused["status"] == "pass":
                    accepted = True
                # A repeated failed retry must stop this retry stage now; the
                # outer flow can advance to prompt/input regeneration.
                break
            metrics, local_defects = _mesh_metrics(mesh, current.get("category", "asset"), current.get("asset_type"))
            unique_validation_keys.add(attempt_cache_key)
            views = _render(blender, mesh, view_dir)
            performed_validation_calls += 1
            report = _inspect(request, current, views, metrics, local_defects)
            if local_defects and report["status"] == "pass":
                report["status"] = "fail"
                report["defects"] = sorted(set(report.get("defects", []) + local_defects))
                report["reason"] = "Local geometry gate detected: " + ", ".join(local_defects)
            mesh_cache[attempt_mesh_sha256] = {
                "mesh_sha256": attempt_mesh_sha256,
                "validator_version": MESH_VALIDATOR_VERSION,
                "status": report["status"],
                "report": copy.deepcopy(report),
            }
            _save_validation_cache(mesh_cache_path, mesh_cache)
            report |= {"asset_id": current.get("id"), "attempt": attempt + 1, "metrics": metrics,
                       "views": [str(path) for path in views], "mesh": str(mesh),
                       "action": report.get("retry_action") or "validate",
                       "cache_key": attempt_cache_key, "mesh_sha256": attempt_mesh_sha256,
                       "reference_sha256": reference_sha256,
                       "validator_version": MESH_VALIDATOR_VERSION,
                       "cache_hit": False, "validation_performed": True}
            if retry_metadata:
                report |= retry_metadata
            history.append(report)
            last_report = report
            force_retry = str(current.get("id")) in force_retry_ids and attempt == 0
            if report["status"] == "pass" and not force_retry:
                accepted = True
                break
            if force_retry:
                report["forced_retry"] = True
            if attempt >= max_retry or not image_worker:
                break
            retry_seed = int(request["seed"]) + (attempt + 1) * 7919
            retry_request = copy.deepcopy(initial_request)
            retry_request.update({"stage": source_stage, "seed": retry_seed})
            retry_metadata = {
                "retry_stage": "reconstruction",
                "retry_seed": retry_seed,
                "retry_input_source": "initial_request_same_reference",
            }
            # Stage 1 is reconstruction-only: preserve the immutable original
            # reference/crop and change only the sampling seed/parameters.
            retry_value = _run_worker(
                image_worker, retry_request,
                work / f"{source_stage}_mesh_retry_{attempt + 1}_request.json",
                work / f"{source_stage}_mesh_retry_{attempt + 1}_response.json",
            )
            replacement = next((item for item in retry_value.get("assets", []) if item.get("id") == current.get("id")), None)
            if replacement:
                current = replacement
        if not accepted and source_stage == "environment_assets" and regen_max and reference_worker and last_report:
            # Stage 2: rewrite the original prompt with explicit defect fixes,
            # regenerate its reference, then reconstruct once with a new seed.
            prompt = _rewrite_prompt(_reference_prompt(work, current, request), current, last_report)
            regen_request = dict(request)
            regen_request.update({"stage": "environment_references_retry", "image_requests": [{
                "id": current.get("id", f"asset_{index}"), "category": current.get("category", "asset"),
                "asset_type": current.get("asset_type"), "prompt": prompt,
            }]})
            regen = _run_worker(reference_worker, regen_request,
                                work / f"environment_references_retry_{index}_request.json",
                                work / f"environment_references_retry_{index}_response.json")
            if regen.get("images"):
                retry_request = copy.deepcopy(initial_request)
                rewrite_seed = int(request["seed"]) + 9001 + index
                retry_request.update({"stage": source_stage, "seed": rewrite_seed,
                                      "reference_override": regen["images"]})
                retry_value = _run_worker(image_worker, retry_request,
                                          work / f"{source_stage}_input_regen_{index}_request.json",
                                          work / f"{source_stage}_input_regen_{index}_response.json")
                replacement = next((item for item in retry_value.get("assets", []) if item.get("id") == current.get("id")), None)
                if replacement:
                    current = replacement
                    prompt_mesh_path = Path(current["mesh"])
                    prompt_mesh_sha256 = cached_mesh_sha256(prompt_mesh_path)
                    prompt_cache_key = _validation_cache_key(
                        prompt_mesh_sha256, reference_sha256, MESH_VALIDATOR_VERSION
                    )
                    mesh_cached = mesh_cache.get(prompt_mesh_sha256)
                    if (
                        isinstance(mesh_cached, dict)
                        and mesh_cached.get("validator_version") == MESH_VALIDATOR_VERSION
                        and mesh_cached.get("status") in {"pass", "fail"}
                        and isinstance(mesh_cached.get("report"), dict)
                    ):
                        mesh_cache_hits += 1
                        last_report = copy.deepcopy(mesh_cached["report"])
                        last_report.update({
                            "asset_id": current.get("id"), "attempt": "prompt_rewrite",
                            "mesh": str(prompt_mesh_path), "mesh_sha256": prompt_mesh_sha256,
                            "reference_sha256": reference_sha256,
                            "validator_version": MESH_VALIDATOR_VERSION,
                            "cache_key": prompt_cache_key, "cache_level": "mesh_sha256",
                            "cache_hit": True, "validation_performed": False,
                            "reused_mesh_validation": True,
                        })
                        history.append(last_report)
                        accepted = last_report["status"] == "pass"
                    else:
                        metrics, local_defects = _mesh_metrics(
                            prompt_mesh_path, current.get("category", "asset"), current.get("asset_type")
                        )
                        unique_validation_keys.add(prompt_cache_key)
                        view_dir = run_dir / "mesh_validation_views" / source_stage / str(current.get("id", index)) / "input_regen"
                        views = _render(blender, prompt_mesh_path, view_dir)
                        performed_validation_calls += 1
                        last_report = _inspect(request, current, views, metrics, local_defects)
                        if local_defects and last_report["status"] == "pass":
                            last_report["status"] = "fail"
                            last_report["defects"] = sorted(set(last_report.get("defects", []) + local_defects))
                        last_report |= {"asset_id": current.get("id"), "attempt": "prompt_rewrite", "metrics": metrics,
                                        "views": [str(path) for path in views], "mesh": str(prompt_mesh_path),
                                        "action": last_report.get("retry_action") or "prompt_rewrite",
                                        "retry_stage": "prompt_rewrite", "retry_seed": rewrite_seed,
                                        "retry_prompt": prompt, "cache_key": prompt_cache_key,
                                        "mesh_sha256": prompt_mesh_sha256,
                                        "reference_sha256": reference_sha256,
                                        "validator_version": MESH_VALIDATOR_VERSION,
                                        "cache_hit": False, "validation_performed": True}
                        mesh_cache[prompt_mesh_sha256] = {
                            "mesh_sha256": prompt_mesh_sha256,
                            "validator_version": MESH_VALIDATOR_VERSION,
                            "status": last_report["status"],
                            "report": copy.deepcopy(last_report),
                        }
                        _save_validation_cache(mesh_cache_path, mesh_cache)
                        history.append(last_report)
                        accepted = last_report["status"] == "pass"
        if accepted:
            final_status = "pass"
            final_report = (last_report or {"asset_id": current.get("id"), "status": "pass"}) | {
                "final_status": final_status,
            }
            reports.append(final_report)
            registry.register(current | {"final_status": "pass"}, source_stage=source_stage)
        else:
            # Reconstruction Stage 2 crop regeneration runs after this loop;
            # defer its Stage 3 decision until that input-changing stage has
            # definitively failed. Environment prototypes can resolve now.
            if source_stage == "reconstruction":
                final_status = "dropped"
                final_report = (last_report or {}) | {"final_status": final_status}
                reports.append(final_report)
            else:
                # Defer environment prototype resolution until every asset in
                # this batch has been validated and registered. This prevents
                # ordering from making a later same-category prototype
                # invisible to an earlier failed prototype.
                final_status = "dropped"
                final_report = (last_report or {}) | {"final_status": final_status}
                reports.append(final_report)
        # Persist the terminal prototype decision. This includes failures, so
        # duplicate instances do not independently launch the same retry loop.
        cache_record = {
            "mesh_sha256": initial_mesh_sha256,
            "reference_sha256": reference_sha256,
            "validator_version": MESH_VALIDATOR_VERSION,
            "cache_key": cache_key,
            "final_status": final_status,
            "report": copy.deepcopy(final_report),
            "asset": copy.deepcopy(current) if final_status in {"pass", "fallback_pass"} else None,
        }
        validation_cache[cache_key] = cache_record
        _save_validation_cache(cache_path, validation_cache)
        if accepted:
            assets[index] = current
    # Reconstruction inputs are crops from composition images. If retries did
    # not pass, regenerate segmentation/crops once, then reconstruct and gate
    # the replacement without falling back to environment references.
    if source_stage == "reconstruction" and regen_max and segment_worker and image_worker:
        failed_ids = {
            item.get("asset_id") for item in reports
            if item.get("status") == "fail" and item.get("final_status") == "dropped"
        }
        if failed_ids:
            segmentation_request = dict(request)
            segmentation_request.update({"stage": "segmentation", "seed": int(request["seed"]) + 12001})
            segmentation_value = _run_worker(
                segment_worker, segmentation_request,
                work / "segmentation_mesh_regen_request.json",
                work / "segmentation_mesh_regen_response.json",
            )
            # Hunyuan's reconstruction contract reads the canonical response;
            # retain the immutable retry response while switching that input.
            (work / "segmentation_response.json").write_text(
                json.dumps(segmentation_value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            retry_request = dict(request)
            retry_request.update({"stage": source_stage, "seed": int(request["seed"]) + 13001})
            regenerated = _run_worker(image_worker, retry_request,
                                       work / "reconstruction_crop_regen_request.json",
                                       work / "reconstruction_crop_regen_response.json")
            replacements = {item.get("id"): item for item in regenerated.get("assets", [])}
            regenerated_cache: dict[str, tuple[dict[str, Any], bool]] = {}
            for index, asset in enumerate(assets):
                if asset.get("id") not in failed_ids or asset.get("id") not in replacements:
                    continue
                candidate = replacements[asset["id"]]
                candidate_path = Path(candidate["mesh"])
                candidate_mesh_sha256 = cached_mesh_sha256(candidate_path)
                candidate_reference_sha256 = cached_reference_sha256(candidate.get("source_image"))
                candidate_key = _validation_cache_key(
                    candidate_mesh_sha256, candidate_reference_sha256, MESH_VALIDATOR_VERSION
                )
                cached_regen = regenerated_cache.get(candidate_key)
                if cached_regen is None:
                    mesh_cached = mesh_cache.get(candidate_mesh_sha256)
                    if (
                        isinstance(mesh_cached, dict)
                        and mesh_cached.get("validator_version") == MESH_VALIDATOR_VERSION
                        and mesh_cached.get("status") in {"pass", "fail"}
                        and isinstance(mesh_cached.get("report"), dict)
                    ):
                        cached_regen = (
                            copy.deepcopy(mesh_cached["report"]),
                            mesh_cached["status"] == "pass",
                        )
                        regenerated_cache[candidate_key] = cached_regen
                        mesh_cache_hits += 1
                if cached_regen is not None:
                    report = copy.deepcopy(cached_regen[0])
                    report.update({
                        "asset_id": asset["id"], "cache_key": candidate_key,
                        "mesh_sha256": candidate_mesh_sha256,
                        "reference_sha256": candidate_reference_sha256,
                        "cache_level": "mesh_sha256", "cache_hit": True,
                        "validation_performed": False, "reused_mesh_validation": True,
                    })
                    report["final_status"] = "pass" if cached_regen[1] else "dropped"
                    cache_hits += 1
                    if cached_regen[1]:
                        assets[index] = candidate
                else:
                    unique_validation_keys.add(candidate_key)
                    metrics, local_defects = _mesh_metrics(
                        candidate_path, candidate.get("category", "asset"), candidate.get("asset_type")
                    )
                    view_dir = run_dir / "mesh_validation_views" / source_stage / str(asset["id"]) / "crop_regen"
                    views = _render(blender, candidate_path, view_dir)
                    performed_validation_calls += 1
                    report = _inspect(request, candidate, views, metrics, local_defects)
                    report |= {"asset_id": asset["id"], "attempt": "crop_regen", "metrics": metrics,
                               "views": [str(path) for path in views], "mesh": str(candidate["mesh"]),
                               "action": report.get("retry_action") or "crop_regen",
                               "cache_key": candidate_key, "mesh_sha256": candidate_mesh_sha256,
                               "reference_sha256": candidate_reference_sha256,
                               "validator_version": MESH_VALIDATOR_VERSION,
                               "cache_hit": False, "validation_performed": True}
                    report["final_status"] = "pass" if report["status"] == "pass" and not local_defects else "dropped"
                    regenerated_cache[candidate_key] = (copy.deepcopy(report), report["final_status"] == "pass")
                    validation_cache[candidate_key] = {
                        "mesh_sha256": candidate_mesh_sha256,
                        "reference_sha256": candidate_reference_sha256,
                        "validator_version": MESH_VALIDATOR_VERSION,
                        "cache_key": candidate_key,
                        "final_status": report["final_status"],
                        "report": copy.deepcopy(report),
                        "asset": copy.deepcopy(candidate) if report["final_status"] == "pass" else None,
                    }
                    _save_validation_cache(cache_path, validation_cache)
                    history.append(report)
                reports = [report if item.get("asset_id") == asset["id"] else item for item in reports]
                if report.get("final_status") == "pass":
                    assets[index] = candidate
                    registry.register(candidate | {"final_status": "pass"}, source_stage=source_stage)
    # Reconstruction Stage 3 is intentionally after the bounded crop/
    # resegment Stage 2. No Hunyuan worker, seed, prompt, or image worker is
    # reachable from this resolver.
    if source_stage in {"environment_assets", "reconstruction"}:
        for index, asset in enumerate(assets):
            matching = [item for item in reports if item.get("asset_id") == asset.get("id")]
            if not matching or matching[-1].get("final_status") != "dropped":
                continue
            stage3_stats["stage3_enter_count"] += 1
            stage3_stats["hunyuan_calls_avoided_by_stage3"] += 1
            decision = resolve_stage3(
                asset, source_stage=source_stage, registry=registry,
                output_root=run_dir / "mesh_validation_fallbacks" / source_stage,
                asset_library=asset_library,
                failed_history=[item for item in history if item.get("asset_id") == asset.get("id")],
                original_prompt=str(asset.get("original_prompt", asset.get("prompt", ""))),
                reference=asset.get("source_image"), scene_role=str(asset.get("scene_role", "")),
                region_id=str(asset.get("region_id", "")),
            )
            report = matching[-1]
            report.update({
                "final_status": decision["final_status"], "retry_exhausted": True,
                "stage3": {key: value for key, value in decision.items() if key != "asset"},
                "fallback": decision.get("fallback_type") or "drop",
                "place_allowed": bool(decision.get("place_allowed")),
            })
            if report.get("cache_key"):
                validation_cache[report["cache_key"]] = {
                    "mesh_sha256": report.get("mesh_sha256", ""),
                    "reference_sha256": report.get("reference_sha256", ""),
                    "validator_version": MESH_VALIDATOR_VERSION,
                    "cache_key": report["cache_key"],
                    "final_status": decision["final_status"],
                    "report": copy.deepcopy(report),
                    "asset": copy.deepcopy(decision.get("asset")) if decision["final_status"] == "fallback_pass" else None,
                }
            history.append({"asset_id": asset.get("id"), "stage": "stage_3",
                            "action": decision.get("fallback_type") or "drop",
                            "status": decision["final_status"],
                            "reason": decision.get("reason", "")})
            if decision["final_status"] == "fallback_pass" and isinstance(decision.get("asset"), dict):
                assets[index] = decision["asset"]
                stage3_type = str(decision.get("fallback_type", ""))
                if stage3_type == "validated_prototype":
                    stage3_stats["prototype_reuse_count"] += 1
                elif stage3_type == "asset_library":
                    stage3_stats["asset_library_fallback_count"] += 1
                elif stage3_type == "primitive":
                    stage3_stats["primitive_fallback_count"] += 1
                else:
                    stage3_stats["procedural_fallback_count"] += 1
                registry.register(decision["asset"] | {"final_status": "pass"}, source_stage=source_stage)
            else:
                stage3_stats["drop_count"] += 1
    registry.save()
    accepted_ids = {
        item.get("asset_id") for item in reports
        if item.get("final_status") in {"pass", "fallback_pass"}
    }
    filtered = [item for item in assets if item.get("id") in accepted_ids]
    fallback_count = sum(1 for item in reports if item.get("final_status") == "fallback_pass")
    dropped_count = sum(1 for item in reports if item.get("final_status") == "dropped")
    response = {"status": "ok", "assets": filtered, "reports": reports, "retry_history": history,
                "accepted": sum(1 for item in reports if item.get("final_status") == "pass"),
                "fallback": fallback_count, "dropped": dropped_count,
                "source_stage": source_stage,
                "stage3": stage3_stats,
                "structural_features_excluded": len(structural_assets),
                "hunyuan_calls_avoided": len(structural_assets),
                "hunyuan_calls_in_stage3": 0,
                "validated_asset_registry": str(registry.path),
                "validation_cache": {
                    "path": str(cache_path),
                    "exact_path": str(cache_path),
                    "mesh_path": str(mesh_cache_path),
                    "levels": ["exact", "mesh_sha256"],
                    "exact_key_fields": ["mesh_sha256", "reference_sha256", "validator_version"],
                    "mesh_key_fields": ["mesh_sha256", "validator_version"],
                    "validator_version": MESH_VALIDATOR_VERSION,
                    "unique_validation_keys": len(unique_validation_keys),
                    "cache_hits": cache_hits,
                    "mesh_cache_hits": mesh_cache_hits,
                    "validation_calls": performed_validation_calls,
                }}
    write_response(args.response, response)


if __name__ == "__main__":
    main()
