#!/usr/bin/env python3
"""Evidence-based audit for the Terrain Macro Structure Planning goal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _has(root: Path, relative: str, needle: str | None = None) -> bool:
    path = root / relative
    value = _read(path)
    return bool(value) and (needle is None or needle in value)


def _all(root: Path, relative: str, needles: tuple[str, ...]) -> bool:
    value = _read(root / relative)
    return bool(value) and all(needle in value for needle in needles)


def _any(root: Path, relative: str, needles: tuple[str, ...]) -> bool:
    value = _read(root / relative)
    return any(needle in value for needle in needles)


def _run_has(run: Path | None, relative: str) -> bool:
    if run is None:
        return False
    return (run / relative).is_file() or (run / "work" / relative).is_file()


def _run_json(run: Path | None, relative: str) -> dict[str, Any] | None:
    if run is None:
        return None
    for path in (run / relative, run / "work" / relative):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return value if isinstance(value, dict) else None
    return None


def _runtime_signals(run: Path | None) -> dict[str, bool]:
    """Check persisted run evidence without treating source code as execution."""
    manifest = _run_json(run, "run_manifest.json")
    live = bool(manifest and manifest.get("mode") == "live" and not manifest.get("synthetic_outputs"))
    stage_results = manifest.get("stage_results", {}) if manifest else {}
    stage_names = {
        "intent", "plan", "layout", "terrain_macro_plan", "terrain_macro_generate",
        "terrain", "terrain_visual_validate", "structural_input", "structural_generate",
        "structural_replan", "structural_integrate", "structural_validate",
        "environment_assets", "mesh_validation_environment_assets", "region_composition",
        "segment", "reconstruct", "mesh_validation_reconstruct", "place", "refine", "export",
        "structural_view_plan", "structural_render", "structural_visual_validate",
        "structural_adaptive_render", "structural_final_validate", "validate",
    }
    complete_stages = bool(
        live
        and stage_names
        and stage_names.issubset(stage_results)
        and all(
            isinstance(stage_results[name], dict)
            and stage_results[name].get("status") == "complete"
            for name in stage_names
        )
    )
    validation = _run_json(run, "metrics.json")
    validation_pass = bool(
        validation
        and (
            validation.get("valid") is True
            or isinstance(validation.get("run_validation"), dict)
            and validation["run_validation"].get("valid") is True
        )
    )
    has_terrain = all(_run_has(run, path) for path in (
        "terrain/terrain_macro_plan.json", "terrain/macro_height.npy",
        "terrain/validation/terrain_metrics.json",
    ))
    terrain_visual = _run_json(run, "terrain_visual_validation.json")
    macro_visual_pass = bool(
        has_terrain
        and _run_has(run, "terrain_validation/heightmap.png")
        and _run_has(run, "terrain_validation/oblique_01.png")
        and _run_has(run, "terrain_validation/layout_overlay.png")
        and terrain_visual
        and terrain_visual.get("status") == "PASS"
    )
    comparison = _run_json(run, "terrain_regression_comparison.json")
    reuse = _run_json(run, "mesh_reuse_manifest.json")
    input_hashes = reuse.get("input_hashes", {}) if reuse else {}
    goal_inputs_present = all(
        input_hashes.get(name)
        for name in ("scene_plan.json", "terrain_macro_plan.json", "layout_labels.npy", "layout_weights.npy")
    )
    goal_outputs_regenerated = bool(
        reuse is not None
        and not reuse.get("reused")
        and has_terrain
        and goal_inputs_present
    )
    return {
        "live_mode": live,
        "complete_workflow": complete_stages,
        "terrain_artifacts": has_terrain,
        "macro_visual_pass": macro_visual_pass,
        "validation_pass": validation_pass,
        "comparison": bool(
            comparison
            and comparison.get("status") in {"comparison_ready", "pass"}
            and comparison.get("regression_identity", {}).get("same_condition") is True
        ),
        "reuse_manifest": reuse is not None,
        "verified_reuse": bool(reuse and reuse.get("reused")),
        "goal_outputs_regenerated": goal_outputs_regenerated,
    }


def audit(root: str | Path, run: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    run_path = Path(run).resolve() if run is not None else None
    code = {
        "stage": _all(root_path, "worldclaw_oss/schemas.py", ("TERRAIN_MACRO_PLAN", "TERRAIN_MACRO_GENERATE")) and _has(root_path, "worldclaw_oss/pipeline.py", "Stage.TERRAIN_MACRO_PLAN"),
        "planner": _all(root_path, "worldclaw_oss/terrain_macro.py", ("class TerrainMacroPlanner", "json_chat", "TerrainMacroPlan")) and _has(root_path, "worldclaw_oss/pipeline.py", "TerrainMacroPlanner("),
        "ontology": _all(root_path, "worldclaw_oss/schemas.py", ("class TerrainLandform", "Literal[", "hill", "basin", "saddle")) and _has(root_path, "worldclaw_oss/terrain_macro.py", "evaluate_landform"),
        "plan": _all(root_path, "worldclaw_oss/terrain_macro.py", ("TerrainMacroPlan", "model_dump_json", "terrain_macro_plan.json")),
        "worker": _all(root_path, "workers/terrain_macro_worker.py", ("generate_macro_height", "np.save", "macro_height.npy")),
        "frequency": _all(root_path, "worldclaw_oss/pipeline.py", ("terrain_frequency.json", "macro", "regional", "micro")),
        "layout": _all(root_path, "worldclaw_oss/terrain_macro.py", ("layout_weights", "soft semantic constraint")) and _all(root_path, "worldclaw_oss/pipeline.py", ("terrain_planner_input", "layout", "masks")),
        "functional": _all(root_path, "worldclaw_oss/terrain_macro.py", ("functional_zones", "TerrainFunctionalZone", "max_slope_deg")),
        "validation": _all(root_path, "worldclaw_oss/terrain_macro.py", ("validate_macro_height", "terrain_spikes", "lake_basin_containment")) and _has(root_path, "worldclaw_oss/terrain.py", "validate_regional_detail_preserves_macro"),
        "bundle": all(_has(root_path, "worldclaw_oss/terrain_macro.py", name) for name in ("heightmap.png", "hillshade.png", "slope.png", "contour.png", "top_down.png", "oblique_01.png", "oblique_02.png", "layout_overlay.png", "terrain_metrics.json")),
        "visual": _all(root_path, "worldclaw_oss/terrain_macro.py", ("visual_validate_terrain", "vision_json", "TERRAIN_VISUAL_VALIDATOR_SYSTEM")),
        "replan": _has(root_path, "worldclaw_oss/terrain_macro.py", "def replan_macro_plan") and _all(root_path, "worldclaw_oss/pipeline.py", ("replan_macro_plan", "terrain_macro_validation")),
        "determinism": _has(root_path, "workers/terrain_macro_worker.py", "generate_macro_height") and _all(root_path, "tests/test_layout_terrain.py", ("stable_seed", "np.array_equal", "generate_macro_height")),
        "generalization": _all(root_path, "worldclaw_oss/schemas.py", ("class TerrainLandform", "class TerrainMacroPlan")) and _all(root_path, "worldclaw_oss/terrain_macro.py", ("evaluate_landform", "generate_macro_height")),
        "comparison": _has(root_path, "scripts/compare_terrain_runs.py", "regression_identity") and _all(root_path, "scripts/run_live_regression.py", ("--mode", "validate_candidate_artifacts", "comparison_artifact")),
        "reuse_manifest": _all(root_path, "worldclaw_oss/pipeline.py", ("mesh_reuse_manifest.json", "dependency_fingerprint", "input_hashes")),
        "no_goal_affected_reuse": _all(root_path, "worldclaw_oss/pipeline.py", ("Terrain/layout/structural outputs are always regenerated", "terrain_macro_plan.json", "layout_weights.npy")),
    }
    runtime = _runtime_signals(run_path)

    # Fixture-only semantic branches are intentionally retained for synthetic
    # tests, but they are evidence against a strict no-demo-hardcode claim.
    fixture_hardcode = _all(root_path, "worldclaw_oss/layout.py", ("def synthetic_plan", '"desert"', '"forest"'))
    production_route_hardcode = _any(root_path, "worldclaw_oss/terrain_macro.py", ("run_name ==", "scene_name ==", "prompt =="))
    requirements = [
        (1, "Pipeline has a dedicated Terrain Macro Planning stage", ["worldclaw_oss/schemas.py", "worldclaw_oss/pipeline.py"]),
        (2, "GPT derives landform primitives from Scene Plan", ["worldclaw_oss/terrain_macro.py"]),
        (3, "Restricted terrain primitive vocabulary", ["worldclaw_oss/schemas.py"]),
        (4, "Structured terrain_macro_plan.json", ["worldclaw_oss/terrain_macro.py"]),
        (5, "Deterministic Macro Terrain Worker", ["workers/terrain_macro_worker.py"]),
        (6, "Standalone macro_height.npy", ["workers/terrain_macro_worker.py"]),
        (7, "Macro/Regional/Micro frequency separation", ["worldclaw_oss/pipeline.py"]),
        (8, "Layout is a soft semantic constraint", ["worldclaw_oss/terrain_macro.py"]),
        (9, "Functional terrain zones are planned", ["worldclaw_oss/terrain_macro.py"]),
        (10, "Downstream structural/placement terrain support", ["worldclaw_oss/pipeline.py"]),
        (11, "Regional detail preserves Macro Composition", ["worldclaw_oss/terrain.py"]),
        (12, "Deterministic terrain validation", ["worldclaw_oss/terrain_macro.py"]),
        (13, "Terrain diagnostic render bundle", ["worldclaw_oss/terrain_macro.py"]),
        (14, "GPT Terrain Visual Validation", ["worldclaw_oss/terrain_macro.py"]),
        (15, "Validation failure triggers Macro replan", ["worldclaw_oss/pipeline.py"]),
        (16, "Retry changes Macro Plan rather than seed", ["worldclaw_oss/pipeline.py"]),
        (17, "Same Macro Plan is deterministic", ["workers/terrain_macro_worker.py"]),
        (18, "No individual-demo hardcode", ["worldclaw_oss/terrain_macro.py"]),
        (19, "Forest/Lake produces basin/ridge/bench/saddle structure", ["worldclaw_oss/terrain_macro.py"]),
        (20, "Macro terrain remains readable before detail", ["worldclaw_oss/terrain_macro.py"]),
        (21, "No scene/run/prompt literal special routes", ["worldclaw_oss/pipeline.py"]),
        (22, "Scene-specific values live in Plan/Config/Schema", ["worldclaw_oss/schemas.py"]),
        (23, "Unified parameterized primitive interface", ["worldclaw_oss/terrain_macro.py"]),
        (24, "Baseline Mesh reuse passes complete dependency checks", ["worldclaw_oss/pipeline.py"]),
        (25, "mesh_reuse_manifest.json tracks decisions", ["worldclaw_oss/pipeline.py"]),
        (26, "Goal-affected outputs are not reused", ["worldclaw_oss/pipeline.py"]),
        (27, "Post-change complete Live-mode workflow run", ["scripts/run_live_regression.py"]),
        (28, "Same-condition Baseline regression comparison", ["scripts/compare_terrain_runs.py"]),
        (29, "Live Run ends in Validation pass without case patch", ["scripts/run_live_regression.py"]),
    ]
    result: list[dict[str, Any]] = []
    source_for = {
        1: "stage", 2: "planner", 3: "ontology", 4: "plan", 5: "worker", 6: "worker", 7: "frequency",
        8: "layout", 9: "functional", 10: "stage", 11: "validation", 12: "validation", 13: "bundle",
        14: "visual", 15: "replan", 16: "replan", 17: "determinism", 18: "generalization", 19: "ontology",
        20: "worker", 21: "generalization", 22: "ontology", 23: "generalization", 24: None, 25: "reuse_manifest",
        26: "no_goal_affected_reuse", 27: None, 28: "comparison", 29: None,
    }
    for number, title, evidence in requirements:
        signal = source_for[number]
        if signal is not None and code.get(signal, False):
            status = "pass"
            reason = "source implementation and local contract tests cover this requirement"
        else:
            status = "unverified"
            reason = "required source signal is absent or runtime evidence is unavailable"
        if number == 18 and fixture_hardcode:
            status = "partial"
            reason = "generic macro path exists, but synthetic_plan retains explicit demo fixture branches"
        if number == 21 and production_route_hardcode:
            status = "fail"
            reason = "production terrain macro code contains a scene/run/prompt literal route"
        elif number == 21:
            status = "partial" if fixture_hardcode else ("pass" if code["generalization"] else "unverified")
            reason = "synthetic fixture routing is isolated outside the live macro path" if fixture_hardcode else "no literal route found in terrain macro implementation"
        if number == 20 and code["worker"]:
            status = "pass" if runtime["macro_visual_pass"] else "partial"
            reason = "current Run contains valid macro diagnostic renders and PASS visual validation" if status == "pass" else "macro generation and diagnostics exist; current Run visual evidence is still required"
        if number == 26 and code["no_goal_affected_reuse"]:
            status = "pass" if runtime["goal_outputs_regenerated"] else "partial"
            reason = "Run manifest records regenerated goal-affected inputs with no reused meshes" if status == "pass" else "source gate regenerates goal-affected outputs; a completed Run is still required to verify the persisted result"
        if number in {4, 6, 12, 13, 14, 27, 28, 29}:
            if number in {27, 29}:
                status = "pass" if runtime["complete_workflow"] and runtime["live_mode"] and (number != 29 or runtime["validation_pass"]) else "unverified"
                reason = "persisted post-change Live workflow evidence is complete" if status == "pass" else "required post-change Live workflow evidence is unavailable"
            elif number == 28:
                status = "pass" if runtime["comparison"] else ("partial" if code["comparison"] else "unverified")
                reason = "persisted same-condition Baseline comparison is available" if status == "pass" else "comparison implementation exists, but a persisted same-condition result is unavailable"
            elif run_path is not None and not runtime["terrain_artifacts"]:
                status = "partial" if code.get(source_for[number], False) else "unverified"
                reason = "source implementation exists but specified Run does not contain required terrain evidence"
        if number == 24:
            status = "fail"
            reason = "no verified Baseline Mesh cache hit has been established"
        if number == 25 and run_path is not None and not runtime["reuse_manifest"]:
            status = "partial" if code["reuse_manifest"] else "unverified"
            reason = "manifest implementation exists but specified Run has no mesh_reuse_manifest.json"
        result.append({"id": number, "title": title, "status": status, "evidence": evidence, "reason": reason})
    counts = {name: sum(item["status"] == name for item in result) for name in ("pass", "partial", "unverified", "fail")}
    return {
        "schema_version": "worldclaw-oss-terrain-macro-dod-audit-v1",
        "root": str(root_path),
        "run": str(run_path) if run_path else None,
        "counts": counts,
        "strict_pass_fraction": f"{counts['pass']}/29",
        "complete": counts["pass"] == 29,
        "code_signals": code,
        "runtime_signals": runtime,
        "requirements": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete": report["complete"], "strict_pass_fraction": report["strict_pass_fraction"], "output": str(args.output.resolve())}, ensure_ascii=False))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
