"""Tests for sysight.wiki.ledger — RunLedger."""

import tempfile
import unittest
from pathlib import Path

from sysight.wiki.ledger import RunLedger, RunRecord


class TestRunLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "runs.sqlite"
        self.ledger = RunLedger(self.db_path)
        self.ledger.init()

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_session(self):
        run = RunRecord(
            run_id="run-001",
            status="ok",
            profile_hash="abc123",
            repo_root="/home/user/model",
            repo_commit="def456",
            prompt_version="v1",
            memory_namespace="ns1",
        )
        self.ledger.record_session(run)
        result = self.ledger.recent_session("ns1")
        self.assertIsNotNone(result)
        self.assertEqual(result["run_id"], "run-001")
        self.assertEqual(result["status"], "ok")

    def test_recent_session_returns_latest(self):
        import time
        self.ledger.record_session(RunRecord(run_id="run-001", memory_namespace="ns1", status="ok"))
        time.sleep(0.01)
        self.ledger.record_session(RunRecord(run_id="run-002", memory_namespace="ns1", status="ok"))
        result = self.ledger.recent_session("ns1")
        self.assertEqual(result["run_id"], "run-002")

    def test_recent_session_nonexistent_namespace(self):
        self.assertIsNone(self.ledger.recent_session("nonexistent"))

    def test_record_findings(self):
        self.ledger.record_session(RunRecord(run_id="run-001"))
        self.ledger.record_findings("run-001", [
            {"finding_id": "C2:abc12345", "category": "C2", "file_path": "src/x.py",
             "line": 42, "function": "f", "confidence": "confirmed", "status": "accepted"},
            {"finding_id": "C4:def67890", "category": "C4", "file_path": "src/y.py",
             "line": 10, "function": "g", "confidence": "probable",
             "status": "rejected", "reject_reason": "no evidence"},
        ])
        # Query directly to verify
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT * FROM findings WHERE run_id = 'run-001'").fetchall()
        conn.close()
        self.assertEqual(len(rows), 2)

    def test_record_patches(self):
        self.ledger.record_session(RunRecord(run_id="run-001"))
        self.ledger.record_patches("run-001", [
            {"patch_id": "patch_001", "finding_id": "C2:abc12345",
             "status": "kept", "metric_before": 10.0, "metric_after": 15.0, "delta_pct": 50.0},
            {"patch_id": "patch_002", "finding_id": "C4:def67890",
             "status": "reverted", "reason": "tests_failed"},
        ])
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT * FROM patches WHERE run_id = 'run-001'").fetchall()
        conn.close()
        self.assertEqual(len(rows), 2)

    def test_record_benchmark(self):
        self.ledger.record_session(RunRecord(run_id="run-001"))
        self.ledger.record_benchmark("run-001", "case_1", {
            "score": 0.85, "total": 10, "matched_ids": ["C2:abc", "C4:def"],
        })
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT * FROM benchmark_results WHERE run_id = 'run-001'").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)

    def test_record_candidate(self):
        self.ledger.record_session(RunRecord(run_id="run-001"))
        self.ledger.record_candidate({
            "candidate_id": "cand-001", "run_id": "run-001",
            "kind": "experience", "scope": "global",
            "title": "D2H sync detection", "content_hash": "abc123",
        })
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT * FROM candidates WHERE candidate_id = 'cand-001'").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
