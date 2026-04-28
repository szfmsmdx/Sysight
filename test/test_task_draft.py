"""Tests for nsys analysis findings and stable_finding_id.

Covers:
  - analyze_nsys() → NsysDiag.findings (gpu_memcpy_hotspot etc.)
  - stable_finding_id determinism and format
"""

import sqlite3
import tempfile
from contextlib import closing
import textwrap
import unittest
from pathlib import Path

from sysight.analyzer.nsys import analyze_nsys
from sysight.analyzer.nsys.models import NsysAnalysisRequest, NsysFinding


# ── SQLite builder helpers ────────────────────────────────────────────────────

def _make_minimal_sqlite(path: str) -> None:
    """Build a minimal Nsight Systems SQLite with enough data for analyze_nsys.

    Creates:
      - CUPTI_ACTIVITY_KIND_KERNEL    (2 GPU kernels, 10ms each)
      - CUPTI_ACTIVITY_KIND_MEMCPY    (3 HtoD copies, 6ms total)
      - StringIds                     (kernel/memcpy names)
    """
    with closing(sqlite3.connect(path)) as conn:
        c = conn.cursor()

        # StringIds
        c.execute("""CREATE TABLE StringIds (
            id INTEGER PRIMARY KEY,
            value TEXT
        )""")
        c.executemany("INSERT INTO StringIds VALUES (?,?)", [
            (1, "void MatMul(float*, float*, float*, int)"),
            (2, "cudaMemcpy HtoD"),
        ])

        # Use shortName to match the real nsys schema expected by _extract_kernels
        c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start     INTEGER,
            end       INTEGER,
            shortName INTEGER,
            deviceId  INTEGER,
            streamId  INTEGER,
            correlationId INTEGER
        )""")
        c.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)", [
            (0,        10_000_000, 1, 0, 1, 100),
            (11_000_000, 21_000_000, 1, 0, 1, 101),
        ])

        c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start         INTEGER,
            end           INTEGER,
            copyKind      INTEGER,
            bytes         INTEGER,
            deviceId      INTEGER,
            streamId      INTEGER,
            correlationId INTEGER
        )""")
        c.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)", [
            (21_500_000, 23_500_000, 1, 64_000_000, 0, 1, 200),
            (24_000_000, 26_000_000, 1, 64_000_000, 0, 1, 201),
            (26_500_000, 28_500_000, 1, 64_000_000, 0, 1, 202),
        ])
        conn.commit()


def _write(root: Path, rel: str, src: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src).strip() + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NsysDiag.findings from analyze_nsys
# ═══════════════════════════════════════════════════════════════════════════════

class TestNsysFindings(unittest.TestCase):

    def test_memcpy_finding_produced(self):
        """analyze_nsys with memcpy data should produce findings."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _make_minimal_sqlite(sq)

            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertEqual(diag.status, "ok")
        self.assertGreater(len(diag.findings), 0)

    def test_empty_sqlite_produces_no_memcpy_finding(self):
        """If no kernel/memcpy data → no gpu_memcpy_hotspot finding."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "empty.sqlite")
            with closing(sqlite3.connect(sq)) as conn:
                conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER, deviceId INTEGER, streamId INTEGER, correlationId INTEGER)")
                conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")

            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        # No memcpy data → no gpu_memcpy_hotspot finding
        memcpy_findings = [f for f in diag.findings if "memcpy" in f.category]
        self.assertEqual(len(memcpy_findings), 0)

    def test_nsys_diag_has_no_task_drafts(self):
        """NsysDiag should not have task_drafts (removed in clean nsys refactor)."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _make_minimal_sqlite(sq)

            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertFalse(hasattr(diag, "task_drafts"),
                         "NsysDiag should not have task_drafts field")

    def test_nsys_diag_has_no_repo_context_enabled(self):
        """NsysDiag should not have repo_context_enabled (removed)."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _make_minimal_sqlite(sq)

            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertFalse(hasattr(diag, "repo_context_enabled"),
                         "NsysDiag should not have repo_context_enabled field")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. stable_finding_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestStableModels(unittest.TestCase):

    def test_stable_finding_id_is_deterministic(self):
        """stable_finding_id must return same value for same inputs."""
        from sysight.analyzer.nsys.models import stable_finding_id
        a = stable_finding_id("gpu_idle", "critical", (0, 1_000_000_000), None)
        b = stable_finding_id("gpu_idle", "critical", (0, 1_000_000_000), None)
        self.assertEqual(a, b)

    def test_stable_finding_id_differs_by_category(self):
        """Different categories must produce different IDs."""
        from sysight.analyzer.nsys.models import stable_finding_id
        a = stable_finding_id("gpu_idle", "critical", None, None)
        b = stable_finding_id("sync_wait", "critical", None, None)
        self.assertNotEqual(a, b)

    def test_stable_finding_id_prefix_is_category(self):
        """ID must start with category: prefix for readability."""
        from sysight.analyzer.nsys.models import stable_finding_id
        sid = stable_finding_id("gpu_comm_hotspot", "warning", None, None)
        self.assertTrue(sid.startswith("gpu_comm_hotspot:"))

    def test_findings_have_stable_ids_after_classify(self):
        """classify_bottlenecks must assign stable_id to all findings."""
        import tempfile
        from sysight.analyzer.nsys.extract import extract_trace, inspect_schema
        db_path = tempfile.mktemp(suffix=".sqlite")
        _make_minimal_sqlite(db_path)
        schema = inspect_schema(db_path)
        trace = extract_trace(db_path, schema)
        from sysight.analyzer.nsys.classify import classify_bottlenecks
        _, findings = classify_bottlenecks(trace)
        for f in findings:
            self.assertTrue(f.stable_id, f"Finding {f.category} has empty stable_id")
            self.assertTrue(f.stable_id.startswith(f.category + ":"))


if __name__ == "__main__":
    unittest.main()
