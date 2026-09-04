"""Archive an older run that predates Pipeline's per-attempt snapshots.

This is intentionally conservative: it only copies existing files and reads
the SQLite checkpoint.  It never changes work files, stage state, or outputs.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


STAGE_PATTERNS = {
    "intent": ["intent.json"],
    "plan": ["scene_plan.json"],
    "layout": ["layout_*.npy", "layout.json"],
    "terrain": [
        "terrain.npz",
        "terrain_*.json",
        "terrain.blend",
        "terrain_preview.png",
    ],
    "environment_assets": ["environment_*", "assets.json", "asset_*"] ,
    "region_composition": ["region_composition*", "terrain_condition.png"],
    "segment": ["segmentation*"],
    "reconstruct": ["reconstruction*", "assets.json", "asset_*"],
    "place": ["placement*", "assets.json", "asset_*"],
    "refine": ["refinement*", "refine*"],
    "export": ["export*"],
    "validate": ["validation*"],
}
FINAL_OUTPUTS = [
    "scene.blend", "scene.glb", "scene.json", "preview.png", "walkthrough.mp4",
    "metrics.json",
]


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    work = run / "work"
    db_path = run / "state.sqlite3"
    if not work.is_dir() or not db_path.is_file():
        raise SystemExit(f"missing work/state files under {run}")
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT stage, status, attempts FROM stages ORDER BY rowid").fetchall()
    root = run / "stage_artifacts"
    root.mkdir(exist_ok=True)
    for stage, status, attempts in rows:
        if not attempts:
            continue
        base = root / stage / f"attempt_{attempts:02d}"
        destination = base
        if destination.exists():
            destination = root / stage / f"attempt_{attempts:02d}_posthoc"
        destination.mkdir(parents=True, exist_ok=False)
        patterns = STAGE_PATTERNS.get(stage, [f"{stage}*"])
        selected = []
        for path in sorted(work.rglob("*")):
            if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
                selected.append((path, f"work/{path.relative_to(work).as_posix()}"))
        if stage == "export":
            selected.extend((run / name, name) for name in FINAL_OUTPUTS if (run / name).is_file())
        files = []
        for source, relative in selected:
            target = destination / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append({
                "path": relative,
                "snapshot": target.relative_to(destination).as_posix(),
                "sha256": digest(source),
                "size_bytes": source.stat().st_size,
            })
        record = {
            "schema": "worldclaw-oss-stage-attempt-v1",
            "stage": stage,
            "attempt": attempts,
            "status": status,
            "posthoc": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "note": "Read-only archive of a run created before automatic per-attempt snapshots.",
        }
        (destination / "attempt.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"stage": stage, "attempt": attempts, "status": status,
                          "snapshot": destination.relative_to(run).as_posix(),
                          "files": len(files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
