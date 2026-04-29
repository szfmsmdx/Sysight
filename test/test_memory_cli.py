from __future__ import annotations

import contextlib
import io
import json
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from benchmark import extract_findings, run_sysight_streaming
from sysight.analyzer.cli import main
from sysight.shared.memory.store import (
    apply_memory_updates,
    build_memory_brief,
    read_memory_file,
    search_memory,
    workspace_namespace,
    write_memory_file,
)


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


class TestMemoryStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(self.root, "workspace.md", "## 文件链路\ntrain.py -> trainer.py\n")
        _write(self.root, "experience.md", "## 用 D2H 次数反推同步\n- 场景：...\n- 规则：检查 item()\n---\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_search_memory_honors_scope(self):
        workspace_hits = search_memory(str(self.root), "train.py", scope="workspace")
        experience_hits = search_memory(str(self.root), "item()", scope="experience")

        self.assertEqual(workspace_hits.total_matches, 1)
        self.assertEqual(workspace_hits.matches[0].path, "workspace.md")
        self.assertEqual(experience_hits.total_matches, 1)
        self.assertEqual(experience_hits.matches[0].path, "experience.md")

    def test_read_memory_rejects_path_outside_root(self):
        result = read_memory_file(str(self.root), "/etc/passwd")
        self.assertIsNotNone(result.error)
        self.assertIn("outside", result.error)

    def test_write_memory_supports_append_and_replace(self):
        appended = write_memory_file(str(self.root), "workspace", "\n追加记录", append=True)
        replaced = write_memory_file(str(self.root), "notes/session.md", "fresh", append=False)

        self.assertEqual(appended.path, "workspace.md")
        self.assertIn("追加记录", (self.root / "workspace.md").read_text(encoding="utf-8"))
        self.assertEqual(replaced.path, "notes/session.md")
        self.assertEqual((self.root / "notes" / "session.md").read_text(encoding="utf-8"), "fresh")

    def test_apply_memory_updates_creates_layered_pages_and_signal_graph(self):
        apply_memory_updates(
            str(self.root),
            [
                {"path": "workspace", "content": "## 文件链路\nentry=train.py", "append": True, "category": "C7"},
                {
                    "path": "experience",
                    "title": "Find sync via D2H count",
                    "content": "## Find sync via D2H count\n- 场景：同步\n- 规则：检查 item()\n---",
                    "append": True,
                    "category": "C3",
                    "tags": ["sync", "d2h"],
                },
            ],
            repo_root="/repo/demo",
            namespace="bench/case_1",
            raw_run={"run_id": "run-123", "manifest_path": "raw/runs/run-123/manifest.json"},
        )

        ns = workspace_namespace(repo_root="/repo/demo", namespace="bench/case_1")
        workspace_page = self.root / "wiki" / "workspaces" / ns / "overview.md"
        experience_page = self.root / "wiki" / "experiences" / "find-sync-via-d2h-count.md"
        signal_page = self.root / "wiki" / "signals" / "C3-synchronization.md"
        index_page = self.root / "wiki" / "INDEX.md"

        self.assertTrue(workspace_page.exists())
        self.assertTrue(experience_page.exists())
        self.assertTrue(signal_page.exists())
        self.assertTrue(index_page.exists())
        self.assertIn("raw/runs/run-123/manifest.json", experience_page.read_text(encoding="utf-8"))
        self.assertIn("wiki/experiences/find-sync-via-d2h-count.md", signal_page.read_text(encoding="utf-8"))
        self.assertIn("Stale Pages", index_page.read_text(encoding="utf-8"))

    def test_search_memory_scope_honors_workspace_namespace(self):
        apply_memory_updates(
            str(self.root),
            [{"path": "workspace", "content": "entry=case1.py", "append": True}],
            repo_root="/repo/demo",
            namespace="bench/case_1",
        )
        apply_memory_updates(
            str(self.root),
            [{"path": "workspace", "content": "entry=case2.py", "append": True}],
            repo_root="/repo/demo",
            namespace="bench/case_2",
        )

        hits = search_memory(str(self.root), "entry=case1.py", scope="workspace", namespace="bench/case_1")
        self.assertEqual(hits.total_matches, 1)
        self.assertIn("bench/case_1", hits.matches[0].path)

    def test_build_memory_brief_mentions_namespace_signal_and_raw_run(self):
        apply_memory_updates(
            str(self.root),
            [{
                "path": "experience",
                "title": "Kernel Launch Loop",
                "content": "## Kernel Launch Loop\n- 场景：小 kernel\n- 规则：检查 Python 循环\n---",
                "append": True,
                "category": "C2",
            }],
            repo_root="/repo/demo",
            namespace="bench/case_9",
            raw_run={"run_id": "run-brief", "manifest_path": "raw/runs/run-brief/manifest.json"},
        )

        brief = build_memory_brief(str(self.root), repo_root="/repo/demo", namespace="bench/case_9")
        self.assertIn("bench/case_9", brief)
        self.assertIn("wiki/signals/C2-kernel-launch-overhead.md", brief)
        self.assertIn("raw/runs/run-brief/manifest.json", brief)


class TestMemoryCli(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(self.root, "workspace.md", "## 文件链路\nentry=train.py\n")
        _write(self.root, "experience.md", "## Kernel Launch 过碎\n- 规则：检查 Python 循环\n---\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_json(self, argv: list[str], stdin_text: str | None = None) -> dict:
        out = io.StringIO()
        stdin = io.StringIO(stdin_text or "")
        with mock.patch("sys.stdin", stdin):
            with contextlib.redirect_stdout(out):
                main(argv)
        return json.loads(out.getvalue())

    def test_memory_search_cli_json(self):
        data = self._run_json([
            "memory", "search", "train.py", "--root", str(self.root), "--scope", "workspace", "--json"
        ])
        self.assertEqual(data["total_matches"], 1)
        self.assertEqual(data["matches"][0]["path"], "workspace.md")

    def test_memory_read_cli_json(self):
        data = self._run_json([
            "memory", "read", "workspace", "--root", str(self.root), "--json"
        ])
        self.assertEqual(data["path"], "workspace.md")
        self.assertTrue(any("entry=train.py" in line["text"] for line in data["lines"]))

    def test_memory_write_cli_append_from_stdin(self):
        data = self._run_json([
            "memory", "write", "experience", "--root", str(self.root), "--append", "--stdin", "--json"
        ], stdin_text="\n## 新经验\n- 规则：减少同步\n---\n")
        self.assertEqual(data["path"], "experience.md")
        text = (self.root / "experience.md").read_text(encoding="utf-8")
        self.assertIn("## 新经验", text)

    def test_memory_search_cli_supports_namespace(self):
        apply_memory_updates(
            str(self.root),
            [{"path": "workspace", "content": "entry=cli.py", "append": True}],
            repo_root="/repo/demo",
            namespace="bench/case_cli",
        )

        data = self._run_json([
            "memory", "search", "entry=cli.py", "--root", str(self.root), "--scope", "workspace", "--namespace", "bench/case_cli", "--json"
        ])
        self.assertEqual(data["total_matches"], 1)
        self.assertIn("bench/case_cli", data["matches"][0]["path"])

    def test_memory_brief_cli_json(self):
        apply_memory_updates(
            str(self.root),
            [{"path": "experience", "title": "Loop", "content": "## Loop\n- 规则：批处理\n---", "append": True, "category": "C2"}],
            repo_root="/repo/demo",
            namespace="bench/case_cli",
        )

        data = self._run_json([
            "memory", "brief", "--root", str(self.root), "--repo-root", "/repo/demo", "--namespace", "bench/case_cli", "--json"
        ])
        self.assertIn("bench/case_cli", data["brief"])


class TestBenchmarkMemoryNamespace(unittest.TestCase):
    def test_extract_findings_prefers_localization_output(self):
        diag = {
            "localization": {
                "status": "ok",
                "output": json.dumps({
                    "findings": [
                        {
                            "category": "C1",
                            "file": "src/runtime/device.py",
                            "function": "choose_device",
                            "line": 10,
                        }
                    ]
                })
            },
            "findings": [],
        }

        findings = extract_findings(diag)

        self.assertEqual(findings, [{
            "category": "C1",
            "file": "src/runtime/device.py",
            "function": "choose_device",
            "line": 10,
        }])

    def test_run_sysight_streaming_passes_memory_namespace(self):
        import tempfile

        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, cmd, stdout=None, stderr=None, text=None, env=None, cwd=None):
                captured["cmd"] = cmd
                self.stdout = io.StringIO("{}")
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "case.log"
            case_dir = Path(tmp) / "case_1"
            case_dir.mkdir()
            with mock.patch("benchmark.subprocess.Popen", side_effect=_FakePopen):
                rc, stdout_text = run_sysight_streaming("/tmp/test.sqlite", case_dir, log_path, "case_1", False)

        self.assertEqual(rc, 0)
        self.assertEqual(stdout_text, "{}")
        self.assertIn("--memory-namespace", captured["cmd"])
        self.assertIn("bench/case_1", captured["cmd"])


if __name__ == "__main__":
    unittest.main()
