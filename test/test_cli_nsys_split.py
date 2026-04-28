"""CLI tests for keeping nsys analysis independent from repo analysis."""

import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
import unittest
from pathlib import Path
from unittest import mock

from sysight.analyzer.cli import main


def _write_minimal_nsys_sqlite(path: str) -> None:
    with closing(sqlite3.connect(path)) as conn:
        cur = conn.cursor()
        cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        cur.execute("INSERT INTO StringIds VALUES (1, 'KernelA')")
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
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 1000000, 1, 0, 1, 10)"
        )
        conn.commit()


class TestNsysCliSplit(unittest.TestCase):
    def test_package_import_does_not_preload_cli_module(self):
        import importlib
        import sys

        sys.modules.pop("sysight.analyzer", None)
        sys.modules.pop("sysight.analyzer.cli", None)

        importlib.import_module("sysight.analyzer")

        self.assertNotIn("sysight.analyzer.cli", sys.modules)

    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        return json.loads(buf.getvalue())

    def test_python_m_cli_module_emits_stdout_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sysight.analyzer.cli",
                    "nsys",
                    db,
                    "--json",
                    "--no-codex",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**dict(), **__import__("os").environ, "PYTHONPATH": "src"},
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.strip())
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "ok")

    def test_no_codex_prints_terminal_without_json_artifact(self):
        """--no-codex: terminal output, no JSON artifact written."""
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sysight.analyzer.cli",
                    "nsys",
                    db,
                    "--no-codex",
                ],
                cwd=tmp,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )
            json_path = Path(tmp) / ".sysight" / "nsys" / "trace.json"

            self.assertEqual(result.returncode, 0)
            # Terminal mode: output is human-readable text, not JSON
            self.assertNotEqual(result.stdout.lstrip()[:1], "{")
            # No JSON artifact written in --no-codex mode
            self.assertFalse(json_path.exists())

    def test_standalone_nsys_does_not_have_repo_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            data = self._run_json(["nsys", db, "--json", "--no-codex"])

        self.assertEqual(data["status"], "ok")
        # No repo_context_enabled field (removed)
        self.assertNotIn("repo_context_enabled", data)
        # No repo-related fields
        self.assertNotIn("evidence_links", data)
        self.assertNotIn("task_drafts", data)
        # hotspots field exists but is now SampleHotspot (CPU samples),
        # not the old MappedHotspot format
        self.assertIn("hotspots", data)
        self.assertIn("windows", data)
        self.assertNotIn("tasks", data)
        self.assertNotIn("optimizer_handoff", data)
        if data["windows"]:
            self.assertIn("callstack_summaries", data["windows"][0])
            self.assertIn("window_rank_in_iter", data["windows"][0])
            self.assertIn("actionable_chain", data["windows"][0])

    def test_no_codex_skips_windows(self):
        """--no-codex: fast mode, no windows, SQL findings still present."""
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            data = self._run_json(["nsys", db, "--json", "--no-codex"])

        self.assertEqual(data["windows"], [])
        self.assertTrue(any(f["category"].startswith("sql_") for f in data.get("findings", [])))

    def test_default_with_repo_root_enables_windows(self):
        """Default mode (Codex on): windows should be populated."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            class _FakePopen:
                def __init__(self, command, stdin=None, stdout=None, stderr=None, text=None, env=None, start_new_session=None):
                    out_idx = command.index("--output-last-message") + 1
                    Path(command[out_idx]).write_text("{}", encoding="utf-8")
                    self.pid = 99
                    self.returncode = 0
                def communicate(self, prompt_text=None):
                    return ("", "")

            with mock.patch("sysight.analyzer.nsys.investigation.subprocess.Popen", side_effect=_FakePopen):
                data = self._run_json(["nsys", db, "--json", "--repo-root", str(root)])

        self.assertGreater(len(data["windows"]), 0)
        self.assertTrue(any(f["category"].startswith("sql_") for f in data.get("findings", [])))

    def test_no_codex_flag_skips_codex(self):
        """--no-codex: even with repo-root, Codex must not be invoked."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "train.py").write_text("def train():\n    return 1\n", encoding="utf-8")
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            out = io.StringIO()
            err = io.StringIO()
            with mock.patch("sysight.analyzer.nsys.investigation.subprocess.Popen") as popen:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    main(["nsys", db, "--json", "--repo-root", str(root), "--no-codex"])

        data = json.loads(out.getvalue())
        self.assertEqual(data["status"], "ok")
        self.assertIsNone(data["investigation"])
        popen.assert_not_called()
        self.assertNotIn("codex", err.getvalue().lower())

    def test_full_report_waits_for_codex_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "train.py").write_text("def train():\n    return 1\n", encoding="utf-8")
            db = str(Path(tmp) / "trace.sqlite")
            _write_minimal_nsys_sqlite(db)

            class _FakeSyncPopen:
                def __init__(self, command, stdin=None, stdout=None, stderr=None, text=None, env=None, start_new_session=None):
                    out_idx = command.index("--output-last-message") + 1
                    self.output_path = command[out_idx]
                    self.pid = 34567
                    self.returncode = 0

                def communicate(self, prompt_text=None):
                    Path(self.output_path).write_text(
                        '{"summary":"同步细定位完成","questions":[{"question_id":"Q1","problem_id":"gpu_idle","category":"gpu_idle","title":"GPU 空闲占比高","status":"mapped","file_path":"train.py","line":1,"function":"train","rationale":"训练入口附近触发该问题","suggestion":"先检查 train 里的主循环","window_ids":["W1"]}],"anchors":[{"window_id":"W1","problem_id":"gpu_idle","category":"gpu_idle","event_name":"KernelA","status":"mapped","file_path":"train.py","line":1,"function":"train","rationale":"训练入口附近触发该窗口","suggestion":"先检查 train 里的主循环"}]}',
                        encoding="utf-8",
                    )
                    return ("", "")

            out = io.StringIO()
            err = io.StringIO()
            with mock.patch("sysight.analyzer.nsys.investigation.subprocess.Popen", side_effect=_FakeSyncPopen):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    main(["nsys", db, "--json", "--repo-root", str(root)])

        data = json.loads(out.getvalue())
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["investigation"]["status"], "ok")
        self.assertEqual(data["investigation"]["summary"], "同步细定位完成")
        self.assertNotIn("optimizer_handoff", data)
        self.assertEqual(len(data["investigation"]["questions"]), 1)
        self.assertEqual(data["investigation"]["questions"][0]["question_id"], "Q1")
        self.assertEqual(data["investigation"]["questions"][0]["file_path"], "train.py")
        self.assertEqual(len(data["investigation"]["anchors"]), 1)
        self.assertEqual(data["investigation"]["anchors"][0]["file_path"], "train.py")
        self.assertTrue(data["investigation"]["artifact_dir"])
        self.assertTrue(data["investigation"]["prompt_path"])
        self.assertTrue(data["investigation"]["stdout_path"])
        self.assertTrue(data["investigation"]["stderr_path"])
        # New prompt format: TASK.txt template
        self.assertIn("python3 -m sysight.analyzer.cli nsys-sql", data["investigation"]["prompt"])
        self.assertIn("输出格式：", data["investigation"]["prompt"])
        self.assertIn("workspace-write", data["investigation"]["command"])
        # Old prompt artifacts should NOT be present
        self.assertNotIn("workspace_structure.md", data["investigation"]["prompt"])
        self.assertNotIn("=== 预计算统计", data["investigation"]["prompt"])
        self.assertNotIn("待回答问题：", data["investigation"]["prompt"])
        self.assertIn("codex 调查进行中", err.getvalue())
        self.assertIn("Codex 调查结果已回填", err.getvalue())


if __name__ == "__main__":
    unittest.main()
