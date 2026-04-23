"""Focused tests for deep nsys SQL analysis semantics."""

import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path

from sysight.analyzer.nsys.classify_sql import _sql_profile_health
from sysight.analyzer.nsys.classify_sql_nvtx import (
    _sql_nvtx_hotspots,
    attribute_kernels_to_nvtx,
)
from sysight.analyzer.nsys.models import NsysFinding


class TestProfileHealthSemantics(unittest.TestCase):
    """Profile health should use union/wall semantics for headline ratios."""

    def test_overlap_and_idle_use_union_wall_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trace.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
                conn.executemany(
                    "INSERT INTO StringIds VALUES (?, ?)",
                    [(1, "compute_kernel"), (2, "ncclDevKernel_AllReduce")],
                )
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                        start INTEGER,
                        end INTEGER,
                        shortName INTEGER,
                        deviceId INTEGER,
                        streamId INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 0, 1, ?)",
                    [
                        (0, 100_000_000, 1, 10),
                        (50_000_000, 150_000_000, 2, 11),
                    ],
                )

                findings = []
                _sql_profile_health(
                    findings,
                    conn,
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    runtime_tbl=None,
                    memcpy_tbl=None,
                    nvtx_tbl=None,
                    has_strings=True,
                    total_ns=200_000_000,
                )

        health = next(f for f in findings if f.category == "sql_profile_health")
        evidence = "\n".join(health.evidence)
        self.assertIn("NCCL 重叠率(union)：50.0%", evidence)
        self.assertIn("GPU 空闲(global union)：25.0%", evidence)
        self.assertNotIn("GPU 空闲(global union)：100", evidence)


class TestSqlNvtxModuleSurface(unittest.TestCase):
    def test_nvtx_sql_api_lives_in_dedicated_module(self):
        self.assertTrue(callable(_sql_nvtx_hotspots))
        self.assertTrue(callable(attribute_kernels_to_nvtx))


class TestNvtxGilAnchor(unittest.TestCase):
    """GIL NVTX ranges should become explicit Host-side profile anchors."""

    def test_gil_nvtx_ranges_emit_host_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trace.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute(
                    """CREATE TABLE NVTX_EVENTS (
                        start INTEGER,
                        end INTEGER,
                        text TEXT
                    )"""
                )
                conn.executemany(
                    "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
                    [
                        (0, 50_000_000, "Holding GIL"),
                        (25_000_000, 75_000_000, "Waiting for GIL"),
                        (0, 100_000_000, "iter_1"),
                    ],
                )

                findings = []
                _sql_nvtx_hotspots(
                    findings,
                    conn,
                    "NVTX_EVENTS",
                    has_strings=False,
                    total_ns=100_000_000,
                )

        categories = [f.category for f in findings]
        self.assertIn("sql_nvtx_hotspots", categories)
        self.assertIn("host_gil_contention", categories)

        gil = next(f for f in findings if f.category == "host_gil_contention")
        evidence = "\n".join(gil.evidence)
        self.assertIn("GIL NVTX union 75.0ms（占 NVTX 观测窗口 75.0%）", evidence)
        self.assertIn("Waiting for GIL", gil.related_hotspots)


if __name__ == "__main__":
    unittest.main()
