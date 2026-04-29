"""Focused tests for deep nsys SQL analysis semantics."""

import importlib
import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path

from sysight.analyzer.nsys.classify_sql import (
    _sql_profile_health,
    _sql_nvtx_hotspots,
    _sql_root_cause_analysis,
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


class TestSqlSplitFacade(unittest.TestCase):
    def test_classify_sql_reexports_split_domain_modules(self):
        from sysight.analyzer.nsys import classify_sql

        sql_compute = importlib.import_module("sysight.analyzer.nsys.sql_compute")
        sql_memory = importlib.import_module("sysight.analyzer.nsys.sql_memory")
        sql_comm = importlib.import_module("sysight.analyzer.nsys.sql_comm")
        sql_sync = importlib.import_module("sysight.analyzer.nsys.sql_sync")
        sql_root_cause = importlib.import_module("sysight.analyzer.nsys.sql_root_cause")
        sql_profile = importlib.import_module("sysight.analyzer.nsys.sql_profile")
        sql_nvtx = importlib.import_module("sysight.analyzer.nsys.sql_nvtx")

        self.assertIs(classify_sql._sql_top_kernels, sql_compute._sql_top_kernels)
        self.assertIs(classify_sql._sql_gpu_idle_gaps, sql_compute._sql_gpu_idle_gaps)
        self.assertIs(classify_sql._sql_memory_bandwidth, sql_memory._sql_memory_bandwidth)
        self.assertIs(classify_sql._sql_nccl_breakdown, sql_comm._sql_nccl_breakdown)
        self.assertIs(classify_sql._sql_sync_cost, sql_sync._sql_sync_cost)
        self.assertIs(classify_sql._sql_root_cause_analysis, sql_root_cause._sql_root_cause_analysis)
        self.assertIs(classify_sql._sql_profile_health, sql_profile._sql_profile_health)
        self.assertIs(classify_sql._sql_nvtx_hotspots, sql_nvtx._sql_nvtx_hotspots)
        self.assertIs(classify_sql.attribute_kernels_to_nvtx, sql_nvtx.attribute_kernels_to_nvtx)


class TestRootCauseAnalysisSurface(unittest.TestCase):
    def test_root_cause_same_stream_nccl_still_works_after_split(self):
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
                        streamId INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
                    [
                        (0, 10_000_000, 1, 7, 10),
                        (10_000_000, 20_000_000, 2, 7, 11),
                    ],
                )

                findings = []
                _sql_root_cause_analysis(
                    findings,
                    conn,
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    runtime_tbl=None,
                    memcpy_tbl=None,
                    memset_tbl=None,
                    sync_tbl=None,
                    nvtx_tbl=None,
                    has_strings=True,
                    total_ns=20_000_000,
                )

        finding = next(f for f in findings if f.category == "sql_root_cause_analysis")
        self.assertIn("NCCL 序列化", "\n".join(finding.evidence))


class TestNvtxAttributionNameResolution(unittest.TestCase):
    def test_attribute_kernels_uses_demangled_name_without_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trace.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                        start INTEGER,
                        end INTEGER,
                        demangledName TEXT,
                        correlationId INTEGER
                    )"""
                )
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                        globalTid INTEGER,
                        start INTEGER,
                        end INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.execute(
                    """CREATE TABLE NVTX_EVENTS (
                        globalTid INTEGER,
                        start INTEGER,
                        end INTEGER,
                        eventType INTEGER,
                        text TEXT
                    )"""
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
                    (10, 30, "KernelDemangled", 7),
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                    (5, 0, 40, 7),
                )
                conn.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
                    (5, 0, 50, 59, "iter_1"),
                )

                attributions = attribute_kernels_to_nvtx(
                    conn,
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    "CUPTI_ACTIVITY_KIND_RUNTIME",
                    "NVTX_EVENTS",
                    has_strings=False,
                )

        self.assertEqual(len(attributions), 1)
        self.assertEqual(attributions[0].kernel_name, "KernelDemangled")
        self.assertEqual(attributions[0].nvtx_text, "iter_1")


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
