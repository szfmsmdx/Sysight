"""Run ledger — SQLite-based recording of all pipeline activities.

Records: runs, findings, patches, measurements, benchmark results.
Enables: cross-session learning, SOTA tracking, candidate generation.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunRecord:
    run_id: str
    status: str = ""
    profile_hash: str = ""
    repo_root: str = ""
    repo_commit: str = ""
    prompt_version: str = ""
    memory_namespace: str = ""
    created_at: str = ""


class RunLedger:
    """SQLite-based ledger for tracking runs, findings, patches, and benchmarks."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else Path.cwd() / ".sysight" / "runs" / "runs.sqlite"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, status TEXT, profile_hash TEXT,
                    repo_root TEXT, repo_commit TEXT, prompt_version TEXT,
                    memory_namespace TEXT, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS findings (
                    run_id TEXT, finding_id TEXT, category TEXT, file_path TEXT,
                    line INTEGER, function TEXT, confidence TEXT,
                    accepted INTEGER DEFAULT 1, reject_reason TEXT DEFAULT '',
                    PRIMARY KEY (run_id, finding_id)
                );
                CREATE TABLE IF NOT EXISTS patches (
                    run_id TEXT, patch_id TEXT, finding_id TEXT, status TEXT,
                    metric_before REAL, metric_after REAL, delta_pct REAL,
                    reason TEXT DEFAULT '', PRIMARY KEY (run_id, patch_id)
                );
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    run_id TEXT, case_name TEXT, score REAL, total INTEGER,
                    matched_ids_json TEXT, PRIMARY KEY (run_id, case_name)
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY, run_id TEXT, kind TEXT,
                    scope TEXT, status TEXT DEFAULT 'new', title TEXT,
                    content_hash TEXT, created_at TEXT
                );
            """)
            conn.commit()

    def record_session(self, run: RunRecord) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, status, profile_hash, repo_root, repo_commit, "
                "prompt_version, memory_namespace, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run.run_id, run.status, run.profile_hash, run.repo_root,
                 run.repo_commit, run.prompt_version, run.memory_namespace,
                 run.created_at or _now_iso()),
            )
            conn.commit()

    def record_findings(self, run_id: str, findings: list[dict]) -> None:
        with closing(self._connect()) as conn:
            for f in findings:
                conn.execute(
                    "INSERT OR REPLACE INTO findings "
                    "(run_id, finding_id, category, file_path, line, function, "
                    "confidence, accepted, reject_reason) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, f.get("finding_id", ""), f.get("category", ""),
                     f.get("file_path"), f.get("line"), f.get("function"),
                     f.get("confidence", "unresolved"),
                     0 if f.get("status") == "rejected" else 1,
                     f.get("reject_reason", "")),
                )
            conn.commit()

    def record_patches(self, run_id: str, patches: list[dict]) -> None:
        with closing(self._connect()) as conn:
            for p in patches:
                conn.execute(
                    "INSERT OR REPLACE INTO patches "
                    "(run_id, patch_id, finding_id, status, metric_before, "
                    "metric_after, delta_pct, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, p.get("patch_id", ""), p.get("finding_id", ""),
                     p.get("status", ""), p.get("metric_before"),
                     p.get("metric_after"), p.get("delta_pct"),
                     p.get("reason", "")),
                )
            conn.commit()

    def record_benchmark(self, run_id: str, case: str, score: dict) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO benchmark_results "
                "(run_id, case_name, score, total, matched_ids_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, case, score.get("score", 0), score.get("total", 0),
                 json.dumps(score.get("matched_ids", []))),
            )
            conn.commit()

    def record_candidate(self, candidate: dict) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO candidates "
                "(candidate_id, run_id, kind, scope, status, title, content_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (candidate.get("candidate_id", ""), candidate.get("run_id", ""),
                 candidate.get("kind", ""), candidate.get("scope", "workspace"),
                 candidate.get("status", "new"), candidate.get("title", ""),
                 candidate.get("content_hash", ""), _now_iso()),
            )
            conn.commit()

    def recent_session(self, namespace: str) -> dict | None:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE memory_namespace = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (namespace,),
            ).fetchone()
            return dict(row) if row else None


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
