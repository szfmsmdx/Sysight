"""Tests for nsys-sql CLI subcommands.

These commands provide structured JSON output for agent consumption.
Each command is narrow, read-only, and outputs bounded JSON.
"""

import argparse
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

    def test_kernels_uses_demangled_name_without_stringids(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trace.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                        start INTEGER,
                        end INTEGER,
                        shortName INTEGER,
                        demangledName TEXT,
                        deviceId INTEGER,
                        streamId INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (0, 10_000_000, 1, "KernelDemangled", 0, 1, 10),
                )
                conn.commit()

            data = self._run_json(["nsys-sql", "kernels", str(db), "--json"])

        self.assertEqual(data["kernels"][0]["name"], "KernelDemangled")


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

    def test_gaps_uses_demangled_name_without_stringids(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trace.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                        start INTEGER,
                        end INTEGER,
                        shortName INTEGER,
                        demangledName TEXT,
                        deviceId INTEGER,
                        streamId INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (0, 10_000_000, 1, "BeforeKernel", 0, 3, 10),
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (15_000_000, 25_000_000, 2, "AfterKernel", 0, 3, 11),
                )
                conn.commit()

            data = self._run_json(["nsys-sql", "gaps", str(db), "--json"])

        self.assertEqual(data["gaps"][0]["before_kernel"], "BeforeKernel")
        self.assertEqual(data["gaps"][0]["after_kernel"], "AfterKernel")


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


class TestNsysSqlAdvancedCommands(unittest.TestCase):
    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        return json.loads(buf.getvalue())

    def test_kernel_launch_uses_demangled_name_without_stringids(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trace.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                        start INTEGER,
                        end INTEGER,
                        shortName INTEGER,
                        demangledName TEXT,
                        deviceId INTEGER,
                        streamId INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                        start INTEGER,
                        end INTEGER,
                        nameId INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (2_000, 10_002_000, 1, "KernelDemangled", 0, 1, 10),
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                    (1_000, 1_500, 2, 10),
                )
                conn.commit()

            data = self._run_json(["nsys-sql", "kernel-launch", str(db), "--json"])

        self.assertEqual(data["entries"][0]["kernel_name"], "KernelDemangled")

    def test_nccl_uses_demangled_name_without_stringids(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trace.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                        start INTEGER,
                        end INTEGER,
                        shortName INTEGER,
                        demangledName TEXT,
                        deviceId INTEGER,
                        streamId INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (0, 10_000_000, 1, "ncclAllReduceKernel", 0, 7, 10),
                )
                conn.commit()

            data = self._run_json(["nsys-sql", "nccl", str(db), "--json"])

        self.assertEqual(data["total_ops"], 1)
        self.assertEqual(data["streams"][0]["stream_id"], 7)

    def test_overlap_uses_demangled_name_without_stringids(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "trace.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.execute(
                    """CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                        start INTEGER,
                        end INTEGER,
                        shortName INTEGER,
                        demangledName TEXT,
                        deviceId INTEGER,
                        streamId INTEGER,
                        correlationId INTEGER
                    )"""
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (0, 10_000_000, 1, "ComputeKernel", 0, 1, 10),
                )
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (5_000_000, 15_000_000, 2, "ncclAllReduceKernel", 0, 7, 11),
                )
                conn.commit()

            data = self._run_json(["nsys-sql", "overlap", str(db), "--json"])

        self.assertEqual(data["compute_kernels"], 1)
        self.assertEqual(data["nccl_kernels"], 1)
        self.assertEqual(data["overlap_ns"], 5_000_000)


class TestNsysSqlCliModule(unittest.TestCase):
    def test_module_can_register_and_dispatch_subparser(self):
        from sysight.analyzer.nsys import sql_cli

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            parser = argparse.ArgumentParser()
            sub = parser.add_subparsers(dest="subcmd")
            sql_cli.add_nsys_sql_subparser(sub)
            args = parser.parse_args(["nsys-sql", "kernels", db, "--json"])

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                handled = sql_cli.dispatch_nsys_sql(args)

        self.assertTrue(handled)
        data = json.loads(buf.getvalue())
        self.assertIn("kernels", data)


if __name__ == "__main__":
    unittest.main()
