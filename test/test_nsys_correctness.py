"""P0 nsys correctness tests using synthetic SQLite.

Tests:
  - trace_start_ns / trace_end_ns populated correctly
  - per-device breakdown populated for multi-GPU profiles
  - single-device profile: per_device is empty (not a regression)
  - EvidenceLink extended fields (id, correlation_id, device_id, inferred_by)
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from sysight.analyzer.nsys import analyze_nsys
from sysight.analyzer.nsys.extract import extract_trace, inspect_schema
from sysight.analyzer.nsys.models import NsysAnalysisRequest


def _write_sqlite(path: str, kernels: list[tuple], memcpys: list[tuple] | None = None) -> None:
    """Write a minimal SQLite with kernel (and optional memcpy) data.

    Uses shortName (matching nsys real schema) so _extract_kernels JOIN works.
    kernels: list of (start, end, shortName_id, deviceId, streamId, correlationId)
    memcpys: list of (start, end, copyKind, bytes, deviceId, streamId, correlationId)
    """
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    c.execute("INSERT INTO StringIds VALUES (1, 'KernelA'), (2, 'KernelB')")
    # Use shortName to match the real nsys schema expected by _extract_kernels
    c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
        start INTEGER, end INTEGER, shortName INTEGER,
        deviceId INTEGER, streamId INTEGER, correlationId INTEGER
    )""")
    c.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)",
        kernels
    )
    if memcpys:
        c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, copyKind INTEGER,
            bytes INTEGER, deviceId INTEGER, streamId INTEGER, correlationId INTEGER
        )""")
        c.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)",
            memcpys
        )
    conn.commit()
    conn.close()


class TestTraceStartEnd(unittest.TestCase):

    def test_trace_start_end_ns_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            _write_sqlite(sq, [
                (1_000_000, 5_000_000, 1, 0, 1, 100),
                (6_000_000, 9_000_000, 1, 0, 1, 101),
            ])
            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)

        self.assertEqual(trace.trace_start_ns, 1_000_000)
        self.assertEqual(trace.trace_end_ns, 9_000_000)
        self.assertEqual(trace.duration_ns, 8_000_000)

    def test_trace_start_zero_for_empty_kernel_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "empty.sqlite")
            conn = sqlite3.connect(sq)
            conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, nameId INTEGER, deviceId INTEGER, streamId INTEGER, correlationId INTEGER)")
            conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            conn.commit()
            conn.close()

            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)

        self.assertEqual(trace.trace_start_ns, 0)
        self.assertEqual(trace.trace_end_ns, 0)
        self.assertEqual(trace.duration_ns, 0)


class TestPerDeviceBreakdown(unittest.TestCase):

    def test_single_device_per_device_empty(self):
        """Single device → per_device should be empty (not useful)."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            _write_sqlite(sq, [
                (0, 10_000_000, 1, 0, 1, 100),
                (11_000_000, 20_000_000, 1, 0, 1, 101),
            ])
            req = NsysAnalysisRequest(
                repo_root=tmp,
                profile_path=sq,
                sqlite_path=sq,
                include_repo_context=False,
            )
            diag = analyze_nsys(req)

        self.assertIsNotNone(diag.bottlenecks)
        self.assertEqual(diag.bottlenecks.per_device, [])

    def test_multi_device_per_device_populated(self):
        """Two devices → per_device should have two entries."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            _write_sqlite(sq, [
                (0,          10_000_000, 1, 0, 1, 100),  # device 0
                (11_000_000, 20_000_000, 1, 0, 1, 101),  # device 0
                (0,          8_000_000,  2, 1, 1, 200),  # device 1
                (9_000_000,  18_000_000, 2, 1, 1, 201),  # device 1
            ])
            req = NsysAnalysisRequest(
                repo_root=tmp,
                profile_path=sq,
                sqlite_path=sq,
                include_repo_context=False,
            )
            diag = analyze_nsys(req)

        self.assertIsNotNone(diag.bottlenecks)
        per_device = diag.bottlenecks.per_device
        self.assertEqual(len(per_device), 2)
        device_ids = {bd.device_id for bd in per_device}
        self.assertEqual(device_ids, {0, 1})

    def test_multi_device_active_ns_per_device(self):
        """Each device's active_ns should reflect only that device's kernels."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            # Device 0: 10ms, Device 1: 5ms (non-overlapping)
            _write_sqlite(sq, [
                (0,         10_000_000, 1, 0, 1, 100),  # device 0: 10ms
                (0,          5_000_000, 2, 1, 1, 200),  # device 1: 5ms
            ])
            req = NsysAnalysisRequest(
                repo_root=tmp,
                profile_path=sq,
                sqlite_path=sq,
                include_repo_context=False,
            )
            diag = analyze_nsys(req)

        per_device = {bd.device_id: bd for bd in diag.bottlenecks.per_device}
        self.assertAlmostEqual(per_device[0].active_ns, 10_000_000)
        self.assertAlmostEqual(per_device[1].active_ns, 5_000_000)


class TestEvidenceLinkExtendedFields(unittest.TestCase):

    def test_evidence_links_have_expected_fields(self):
        """EvidenceLink fields must include new audit fields without breaking."""
        from sysight.analyzer.nsys.models import EvidenceLink
        # Build an EvidenceLink using the new extended fields
        el = EvidenceLink(
            bottleneck_category="gpu_memcpy_hotspot",
            event_name="cudaMemcpy HtoD",
            event_category="gpu_memcpy",
            hotspot_function="train_step",
            hotspot_file="train.py",
            link_type="correlation_id",
            reason="API correlationId=100 → kernel",
            confidence=0.9,
            id="gpu_memcpy_hotspot:cudaMemcpy HtoD:correlation_id:0",
            correlation_id=100,
            device_id=0,
            inferred_by="deterministic",
        )
        self.assertEqual(el.correlation_id, 100)
        self.assertEqual(el.device_id, 0)
        self.assertEqual(el.inferred_by, "deterministic")
        self.assertIsNotNone(el.id)

    def test_evidence_link_default_fields(self):
        """New fields should have sensible defaults so old call sites still work."""
        from sysight.analyzer.nsys.models import EvidenceLink
        el = EvidenceLink(
            bottleneck_category="gpu_idle",
            event_name="idle_gap",
            event_category="gpu_idle",
            hotspot_function=None,
            hotspot_file=None,
            link_type="time_overlap",
            reason="large idle gap",
            confidence=0.8,
        )
        self.assertIsNone(el.id)
        self.assertIsNone(el.correlation_id)
        self.assertIsNone(el.device_id)
        self.assertEqual(el.inferred_by, "deterministic")


class TestCallstackExtraction(unittest.TestCase):
    """Tests for full callstack extraction in CPU samples."""

    def _write_sqlite_with_callstack(self, path: str) -> None:
        """Build a SQLite with CPU sampling data that has multiple stack frames."""
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        # Stack frames: depth 0 = leaf, depth 1 = caller, depth 2 = outer caller
        c.executemany("INSERT INTO StringIds VALUES (?,?)", [
            (1, "KernelA"),
            (10, "_PyEval_EvalFrameDefault"),   # leaf frame (depth 0)
            (11, "train_step"),                  # caller (depth 1)
            (12, "main_loop"),                   # outer caller (depth 2)
        ])
        c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, shortName INTEGER,
            deviceId INTEGER, streamId INTEGER, correlationId INTEGER
        )""")
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 10000000, 1, 0, 1, 100)")

        # COMPOSITE_EVENTS: one CPU sample
        c.execute("""CREATE TABLE COMPOSITE_EVENTS (
            id INTEGER PRIMARY KEY,
            timestamp INTEGER,
            globalTid INTEGER
        )""")
        c.execute("INSERT INTO COMPOSITE_EVENTS VALUES (1, 5000000, 42)")

        # SAMPLING_CALLCHAINS: three frames for sample id=1
        c.execute("""CREATE TABLE SAMPLING_CALLCHAINS (
            id INTEGER,
            stackDepth INTEGER,
            symbol INTEGER
        )""")
        c.executemany("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,?,?)", [
            (1, 0, 10),   # leaf: _PyEval_EvalFrameDefault
            (1, 1, 11),   # caller: train_step
            (1, 2, 12),   # outer: main_loop
        ])
        conn.commit()
        conn.close()

    def test_cpu_sample_has_full_callstack_in_extra(self):
        """CPU sample events must have extra['callstack'] with all frames."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            self._write_sqlite_with_callstack(sq)
            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)

        cpu_samples = [ev for ev in trace.events if ev.is_sample]
        self.assertGreater(len(cpu_samples), 0, "Expected at least one CPU sample")

        sample = cpu_samples[0]
        # Name should be the leaf frame
        self.assertEqual(sample.name, "_PyEval_EvalFrameDefault")
        # Callstack must have all 3 frames
        callstack = sample.extra.get("callstack", [])
        self.assertEqual(len(callstack), 3,
                         f"Expected 3 stack frames, got: {callstack}")
        self.assertIn("_PyEval_EvalFrameDefault", callstack)
        self.assertIn("train_step", callstack)
        self.assertIn("main_loop", callstack)

    def test_leaf_frame_is_name(self):
        """Leaf frame (stackDepth=0) must be used as event.name."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            self._write_sqlite_with_callstack(sq)
            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)

        cpu_samples = [ev for ev in trace.events if ev.is_sample]
        self.assertGreater(len(cpu_samples), 0)
        # Leaf frame is depth 0: _PyEval_EvalFrameDefault
        self.assertEqual(cpu_samples[0].name, "_PyEval_EvalFrameDefault")


if __name__ == "__main__":
    unittest.main()
