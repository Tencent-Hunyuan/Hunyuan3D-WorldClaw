from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worldclaw_oss.schemas import DefectReport


def worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--socket", type=Path)
    return parser.parse_args()


def read_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_response(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_stage_response(work_dir: Path, name: str) -> dict[str, Any]:
    path = work_dir / f"{name}_response.json"
    if not path.is_file():
        raise FileNotFoundError(f"required stage response is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "ok":
        raise RuntimeError(f"upstream stage {name} is not successful")
    return value


def load_validated_stage_response(work_dir: Path, name: str) -> dict[str, Any]:
    """Load the asset-level quality-gated response when present."""
    validation_name = "mesh_validation_reconstruct" if name == "reconstruction" else f"mesh_validation_{name}"
    validated = work_dir / f"{validation_name}_response.json"
    if validated.is_file():
        value = json.loads(validated.read_text(encoding="utf-8"))
        if value.get("status") != "ok":
            raise RuntimeError(f"validated stage {name} is not successful")
        return value
    return load_stage_response(work_dir, name)


def artifact_key(source: Path, model_id: str, revision: str, parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(b"\0")
    digest.update(model_id.encode())
    digest.update(b"\0")
    digest.update(revision.encode())
    digest.update(b"\0")
    digest.update(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def image_entries(response: dict[str, Any]) -> list[dict[str, Any]]:
    raw = response.get("images", response.get("outputs", []))
    entries = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            entries.append({"path": item, "id": f"image_{index:03d}"})
        else:
            entries.append(dict(item))
    return entries


def resolve_model_source(record: dict[str, Any]) -> tuple[str, str | None]:
    """Use a complete pinned local snapshot when offline Hub lookup is unavailable."""
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        repo_cache = Path(hf_home) / "hub" / f"models--{record['model_id'].replace('/', '--')}"
        snapshot = repo_cache / "snapshots" / record["revision"]
        if snapshot.is_dir():
            return str(snapshot), None
    return record["model_id"], record["revision"]


def parse_defect_report(value: dict[str, Any]) -> DefectReport:
    """Normalize compatible VLM defect payloads at the worker boundary."""
    # Keep isolated image/mesh workers importable in environments that do not
    # carry the orchestrator's pydantic dependency.  Only VLM defect parsing
    # needs the schema model at runtime.
    from worldclaw_oss.schemas import DefectReport

    normalized = {key: value[key] for key in ("defects", "acceptable", "summary") if key in value}
    raw_defects = normalized.get("defects") or []
    allowed_kinds = {"floating", "penetration", "overlap", "slope", "scale", "texture", "mesh", "other"}
    severity_aliases = {
        "none": 0.0, "negligible": 0.05, "low": 0.2, "minor": 0.3,
        "moderate": 0.5, "medium": 0.5, "major": 0.75, "high": 0.8,
        "critical": 1.0, "severe": 1.0, "variable": 0.5, "unspecified": 0.5,
    }
    defects = []
    for raw in raw_defects if isinstance(raw_defects, list) else []:
        if isinstance(raw, str):
            raw = {"kind": "other", "recommendation": raw}
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "other")).strip().lower()
        if kind not in allowed_kinds:
            kind = "other"
        severity = raw.get("severity", 0.5)
        if isinstance(severity, str):
            try:
                severity = float(severity.strip())
            except ValueError:
                severity = severity_aliases.get(severity.strip().lower(), 0.5)
        try:
            severity = min(1.0, max(0.0, float(severity)))
        except (TypeError, ValueError):
            severity = 0.5
        recommendation = raw.get("recommendation", raw.get("action", raw.get("fix", "inspect reported defect")))
        defects.append({
            "asset_id": raw.get("asset_id") if isinstance(raw.get("asset_id"), str) else None,
            "kind": kind,
            "severity": severity,
            "recommendation": str(recommendation),
        })
    normalized["defects"] = defects
    normalized.setdefault("acceptable", not bool(defects))
    normalized.setdefault("summary", "Derived from the returned defect list; the VLM omitted a summary field.")
    return DefectReport.model_validate(normalized)
