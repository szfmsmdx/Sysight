"""Tests for nsys-sql CLI subcommands.

These commands provide structured JSON output for agent consumption.
Each command is narrow, read-only, and outputs bounded JSON.
"""

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from sysight.analyzer.cli import main


def _write_minimal_nsys_sqlite(path: str) -> None:
    """Create a minimal valid nsys SQLite for testing."""
    with closing(sqlite3.connect(path)) as conn:
        cur = conn.cursor()
        cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        cur.execute("INSERT INTO StringIds VALUES (1, 'KernelA')")
        cur.execute("INSERT INTO StringIds VALUES (2, 'cudaLaunchKernel')")
        cur.execute(
            """CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER,
                end INTEGER,
                shortName INTEGER,
                deviceId INTEGER,
                streamId INTEGER,
                correlationId INTEGER
            )"""
        )
        cur.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 10000000, 1, 0, 1, 10)"
        )
        cur.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (15000000, 25000000, 1, 0, 1, 11)"
        )
        cur.execute(
            """CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER,
                end INTEGER,
                nameId INTEGER,
                correlationId INTEGER
            )"""
        )
        cur.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 1000, 2, 10)"
        )
        conn.commit()


class TestNsysSqlSchemaCommand(unittest.TestCase):
    """Test `sysight nsys-sql schema` command."""

    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        return json.loads(buf.getvalue())

    def test_schema_returns_capabilities_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            data = self._run_json(["nsys-sql", "schema", db, "--json"])

        self.assertIn("capabilities", data)
        self.assertIn("tables", data)
        self.assertIn("cuda_kernel", data["capabilities"])
        self.assertIn("CUPTI_ACTIVITY_KIND_KERNEL", data["tables"])

    def test_schema_includes_gpu_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            data = self._run_json(["nsys-sql", "schema", db, "--json"])

        # gpu_devices may be empty for minimal DB
        self.assertIn("gpu_devices", data)


class TestNsysSqlKernelsCommand(unittest.TestCase):
    """Test `sysight nsys-sql kernels` command."""

    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        return json.loads(buf.getvalue())

    def test_kernels_returns_top_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            data = self._run_json(["nsys-sql", "kernels", db, "--json"])

        self.assertIn("kernels", data)
        self.assertGreater(len(data["kernels"]), 0)
        kernel = data["kernels"][0]
        self.assertIn("name", kernel)
        self.assertIn("count", kernel)
        self.assertIn("total_ns", kernel)


class TestNsysSqlGapsCommand(unittest.TestCase):
    """Test `sysight nsys-sql gaps` command."""

    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        return json.loads(buf.getvalue())

    def test_gaps_returns_idle_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            data = self._run_json(["nsys-sql", "gaps", db, "--json"])

        self.assertIn("gaps", data)
        # Gap between the two kernels: 10000000 to 15000000
        self.assertGreater(len(data["gaps"]), 0)


class TestNsysSqlSyncCommand(unittest.TestCase):
    """Test `sysight nsys-sql sync` command."""

    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        return json.loads(buf.getvalue())

    def test_sync_returns_sync_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            data = self._run_json(["nsys-sql", "sync", db, "--json"])

        # May be empty for minimal DB without sync events
        self.assertIn("sync_events", data)


class TestNsysSqlNvtxCommand(unittest.TestCase):
    """Test `sysight nsys-sql nvtx` command."""

    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        return json.loads(buf.getvalue())

    def test_nvtx_returns_nvtx_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            data = self._run_json(["nsys-sql", "nvtx", db, "--json"])

        # May be empty for minimal DB without NVTX
        self.assertIn("nvtx_ranges", data)


if __name__ == "__main__":
    unittest.main()
