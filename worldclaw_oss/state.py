from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schemas import Stage


SCHEMA = """
CREATE TABLE IF NOT EXISTS stages (
 run_id TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0, payload TEXT, error TEXT,
 updated_at TEXT NOT NULL, PRIMARY KEY(run_id, stage));
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, stage TEXT NOT NULL,
 event TEXT NOT NULL, detail TEXT, created_at TEXT NOT NULL);
"""


class StateDB:
    def __init__(self,path: Path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db: db.executescript(SCHEMA)
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30); db.row_factory=sqlite3.Row; return db
    def initialize(self,run_id: str):
        now=datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            for stage in Stage:
                db.execute("INSERT OR IGNORE INTO stages(run_id,stage,status,updated_at) VALUES(?,?,?,?)",(run_id,stage.value,"pending",now))
    def status(self,run_id: str,stage: Stage) -> str:
        with self.connect() as db:
            row=db.execute("SELECT status FROM stages WHERE run_id=? AND stage=?",(run_id,stage.value)).fetchone()
        return row[0] if row else "missing"
    def start(self,run_id: str,stage: Stage): self._set(run_id,stage,"running",increment=True)
    def complete(self,run_id: str,stage: Stage,payload: dict | None=None): self._set(run_id,stage,"complete",payload=payload)
    def fail(self,run_id: str,stage: Stage,error: str): self._set(run_id,stage,"failed",error=error)
    def _set(self,run_id: str,stage: Stage,status: str,payload=None,error=None,increment=False):
        now=datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("UPDATE stages SET status=?, attempts=attempts+?, payload=?, error=?, updated_at=? WHERE run_id=? AND stage=?",(status,int(increment),json.dumps(payload,sort_keys=True) if payload is not None else None,error,now,run_id,stage.value))
            db.execute("INSERT INTO events(run_id,stage,event,detail,created_at) VALUES(?,?,?,?,?)",(run_id,stage.value,status,error or "",now))
    def attempts(self,run_id: str,stage: Stage) -> int:
        with self.connect() as db: return int(db.execute("SELECT attempts FROM stages WHERE run_id=? AND stage=?",(run_id,stage.value)).fetchone()[0])

    def reopen(self, run_id: str, stage: Stage) -> None:
        """Reopen an exhausted failed stage for an explicit external retry.

        The failure history remains in the event log; resetting the attempt
        counter only gives the resumed run a fresh three-attempt budget.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute(
                "UPDATE stages SET status='pending', attempts=0, error=NULL, updated_at=? "
                "WHERE run_id=? AND stage=? AND status IN ('failed', 'running')",
                (now, run_id, stage.value),
            )
            db.execute(
                "INSERT INTO events(run_id,stage,event,detail,created_at) VALUES(?,?,?,?,?)",
                (run_id, stage.value, "reopened", "explicit resume", now),
            )

    def export_events(self, run_id: str, path: Path) -> None:
        """Materialize the SQLite event stream for portable run inspection."""
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,run_id,stage,event,detail,created_at FROM events "
                "WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
