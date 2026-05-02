"""Real integration tests — AgentLoop + ToolRegistry + live HTTP.

These tests exercise the actual code paths that matter:
  - Tool calling works across both OpenAI and Anthropic protocols
  - Tools actually execute and return data
  - LLM can call multiple tools in sequence
  - Final structured output is produced

Requires: .sysight/config.yaml with valid API keys.
"""

import json
import tempfile
import unittest
from pathlib import Path

from sysight.tools.registry import ToolRegistry, ToolPolicy
from sysight.tools import register_all_tools
from sysight.agent.config_loader import load_config
from sysight.agent.provider import create_provider
from sysight.agent.loop import AgentLoop, AgentTask
from sysight.agent.prompts.loader import PromptLoader


class TestAgentLoopIntegration(unittest.TestCase):
    """Full AgentLoop integration with real HTTP calls and tool execution."""

    @classmethod
    def setUpClass(cls):
        # Skip if no config
        try:
            cls._configs = load_config()
        except FileNotFoundError:
            raise unittest.SkipTest(".sysight/config.yaml not found")

        cls._registry = ToolRegistry()
        register_all_tools(cls._registry)

        cls._tmp = tempfile.TemporaryDirectory()
        cls._repo = Path(cls._tmp.name)
        (cls._repo / "src").mkdir(parents=True)
        (cls._repo / "src" / "train.py").write_text("""\
import torch
from torch.utils.data import DataLoader

def create_dataloader(batch_size=32, num_workers=0):
    dataset = FakeDataset()
    return DataLoader(dataset, batch_size=batch_size,
                      num_workers=num_workers)

class FakeDataset:
    def __len__(self): return 1000
    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.tensor(0)

def train_one_epoch(model, loader):
    for batch in loader:
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        out = model(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
""")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _make_task(self, task_type="analyze", max_turns=8):
        loader = PromptLoader()
        system = loader.build_system_prompt(task_type)
        user = (
            f"Repo: {self._repo}\n"
            "1. scanner_files to list files\n"
            "2. scanner_read src/train.py\n"
            "3. Identify GPU issues (num_workers=0, cuda.synchronize)\n"
            "4. Return findings as JSON"
        )
        return AgentTask(
            run_id="test", task_id="t1", task_type=task_type,
            system_prompt=system, user_prompt=user,
            max_turns=max_turns, max_wall_seconds=120,
        )

    def test_openai_agent_loop_tool_calling(self):
        """OpenAI protocol: LLM calls tools, returns structured output."""
        cfg = self._configs.get("analyze")
        if not cfg or not cfg.api_key:
            raise unittest.SkipTest("analyze config not set")

        provider = create_provider(cfg)
        policy = ToolPolicy(allowed_tools={"scanner_*"}, read_only=True)
        loop = AgentLoop(provider, self._registry, policy)
        task = self._make_task()

        result = loop.run(task)

        self.assertEqual(result.status, "ok", f"Errors: {result.errors}")
        self.assertGreater(len(result.tool_calls), 0,
                           "Should have made at least 1 tool call")
        self.assertGreater(result.turns, 1,
                           "Should take multiple turns (tool calls + final answer)")

        # Check that scanner_files was actually called and returned data
        tool_names = [tc["name"] for tc in result.tool_calls]
        self.assertIn("scanner_files", tool_names)
        self.assertIn("scanner_read", tool_names)

        # All tool calls should be successful
        for tc in result.tool_calls:
            self.assertEqual(tc["status"], "ok",
                             f"Tool {tc['name']} failed: {tc.get('error')}")

        # Should have output or raw content
        has_output = bool(result.output) or bool(result.raw_content)
        self.assertTrue(has_output, "Should produce output")

    def test_anthropic_agent_loop_tool_calling(self):
        """Anthropic protocol: LLM calls tools, returns structured output."""
        cfg = self._configs.get("optimize")
        if not cfg or not cfg.api_key:
            raise unittest.SkipTest("optimize config not set")

        provider = create_provider(cfg)
        policy = ToolPolicy(allowed_tools={"scanner_*"}, read_only=True)
        loop = AgentLoop(provider, self._registry, policy)
        task = self._make_task(task_type="optimize")

        result = loop.run(task)

        self.assertEqual(result.status, "ok", f"Errors: {result.errors}")
        self.assertGreater(len(result.tool_calls), 0)

        tool_names = [tc["name"] for tc in result.tool_calls]
        self.assertIn("scanner_files", tool_names)
        self.assertIn("scanner_read", tool_names)

        for tc in result.tool_calls:
            self.assertEqual(tc["status"], "ok",
                             f"Tool {tc['name']} failed: {tc.get('error')}")

        has_output = bool(result.output) or bool(result.raw_content)
        self.assertTrue(has_output, "Should produce output")

    def test_output_contains_gpu_findings(self):
        """LLM should identify num_workers=0 and cuda.synchronize."""
        cfg = self._configs.get("analyze")
        if not cfg or not cfg.api_key:
            raise unittest.SkipTest("analyze config not set")

        provider = create_provider(cfg)
        policy = ToolPolicy(allowed_tools={"scanner_*"}, read_only=True)
        loop = AgentLoop(provider, self._registry, policy)
        task = self._make_task()

        result = loop.run(task)
        content = (
            json.dumps(result.output, ensure_ascii=False).lower()
            if result.output else result.raw_content.lower()
        )

        # The code has obvious issues — LLM should catch them
        found_sync = "synchronize" in content or "sync" in content
        found_workers = "num_workers" in content or "worker" in content or "dataloader" in content

        self.assertTrue(
            found_sync or found_workers,
            f"LLM should mention at least one obvious GPU issue. "
            f"Found sync={found_sync}, workers={found_workers}"
        )


class TestScannerToolsEndToEnd(unittest.TestCase):
    """Tests scanner tools do real file I/O, not mocks."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._repo = Path(self._tmp.name)
        (self._repo / "src").mkdir(parents=True)
        (self._repo / "src" / "main.py").write_text("def foo():\n    bar()\n")
        (self._repo / "src" / "utils.py").write_text("def bar():\n    pass\n")
        (self._repo / "README.md").write_text("# Test Repo\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_files_lists_all(self):
        from sysight.tools.scanner.files import list_files
        result = list_files(str(self._repo))
        self.assertGreaterEqual(result.total, 3)
        paths = {f.path for f in result.files}
        self.assertIn("src/main.py", paths)
        self.assertIn("README.md", paths)

    def test_files_filter_by_ext(self):
        from sysight.tools.scanner.files import list_files
        result = list_files(str(self._repo), ext="py")
        self.assertEqual(result.total, 2)

    def test_search_finds_symbol(self):
        from sysight.tools.scanner.search import search
        result = search(str(self._repo), "bar", fixed=True)
        self.assertGreaterEqual(result.total_matches, 1)
        self.assertEqual(result.matches[0].path, "src/main.py")

    def test_read_returns_correct_lines(self):
        from sysight.tools.scanner.read import read_file
        result = read_file(str(self._repo), "src/main.py")
        self.assertEqual(result.total_lines, 2)
        self.assertEqual(result.lines[0].text, "def foo():")

    def test_symbols_parses_python(self):
        from sysight.tools.scanner.symbols import list_symbols
        result = list_symbols(str(self._repo), "src/main.py")
        names = {s.name for s in result.symbols}
        self.assertIn("foo", names)

    def test_callers_finds_calls(self):
        from sysight.tools.scanner.callers import find_callsites
        result = find_callsites(str(self._repo), symbol="bar")
        self.assertGreaterEqual(result.total, 1)


class TestLedgerEndToEnd(unittest.TestCase):
    """Tests RunLedger actually persists and retrieves data."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        from sysight.wiki.ledger import RunLedger, RunRecord
        self._db = Path(self._tmp.name) / "runs.sqlite"
        self._ledger = RunLedger(self._db)
        self._ledger.init()

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_and_retrieve_session(self):
        from sysight.wiki.ledger import RunRecord
        self._ledger.record_session(RunRecord(
            run_id="run-001", status="ok", memory_namespace="ns1",
            profile_hash="abc", repo_root="/repo",
        ))
        result = self._ledger.recent_session("ns1")
        self.assertIsNotNone(result)
        self.assertEqual(result["run_id"], "run-001")

    def test_record_findings_persists(self):
        from sysight.wiki.ledger import RunRecord
        import sqlite3
        self._ledger.record_session(RunRecord(run_id="run-002"))
        self._ledger.record_findings("run-002", [
            {"finding_id": "C2:abc12345", "category": "C2",
             "file_path": "src/x.py", "line": 42, "function": "f",
             "confidence": "confirmed"},
        ])
        conn = sqlite3.connect(str(self._db))
        count = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)


class TestWikiRepositoryEndToEnd(unittest.TestCase):
    """Tests WikiRepository actually reads/writes files."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        from sysight.wiki.store import WikiRepository
        self._repo = WikiRepository(root=Path(self._tmp.name) / ".sysight" / "memory")

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_and_read_page(self):
        self._repo.write_page("workspaces/ns1/overview.md",
                              "# Overview\nEntry: train.py",
                              title="Test Overview")
        content = self._repo.read_page("workspaces/ns1/overview.md")
        self.assertIn("Entry: train.py", content)

    def test_append_worklog(self):
        self._repo.append_worklog("ns1", "Fixed loader")
        wl = self._repo.read_page("workspaces/ns1/worklog.md")
        self.assertIn("Fixed loader", wl)

    def test_list_experiences(self):
        self._repo.write_page("wiki/experiences/test.md", "content",
                              title="Test", scope="global", category="C3")
        exps = self._repo.list_experiences()
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0]["category"], "C3")

    def test_path_traversal_blocked(self):
        with self.assertRaises(ValueError):
            self._repo._resolve_path("../../../etc/passwd")


if __name__ == "__main__":
    unittest.main()
