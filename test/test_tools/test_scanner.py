"""Tests for sysight.tools.scanner — ToolDef wrappers + function behavior."""

import tempfile
import unittest
from pathlib import Path

from sysight.tools.registry import ToolRegistry, ToolPolicy
from sysight.tools.scanner import register_scanner_tools


class TestScannerToolsRegistration(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()
        register_scanner_tools(self.reg)

    def test_all_tools_registered(self):
        expected = {
            "scanner_files", "scanner_search", "scanner_read",
            "scanner_callers", "scanner_symbols", "scanner_symbol_callers",
            "scanner_callees", "scanner_trace", "scanner_variants",
        }
        registered = {t.name for t in self.reg.list_read_only()}
        for name in expected:
            self.assertIn(name, registered, f"Missing tool: {name}")

    def test_all_scanner_tools_are_read_only(self):
        for t in self.reg.list_read_only():
            self.assertTrue(t.read_only, f"{t.name} should be read_only")

    def test_policy_allows_scanner_wildcard(self):
        policy = ToolPolicy(allowed_tools={"scanner_*"}, read_only=True)
        tools = self.reg.list_for_policy(policy)
        self.assertEqual(len(tools), 9)


class TestScannerFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "main.py").write_text("print('hello')")
        (self.repo / "README.md").write_text("# Repo")

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_all_files(self):
        from sysight.tools.scanner.files import list_files
        result = list_files(str(self.repo))
        self.assertGreaterEqual(result.total, 2)
        paths = {f.path for f in result.files}
        self.assertIn("src/main.py", paths)

    def test_filter_by_ext(self):
        from sysight.tools.scanner.files import list_files
        result = list_files(str(self.repo), ext="py")
        self.assertEqual(result.total, 1)
        self.assertEqual(result.files[0].ext, ".py")

    def test_filter_by_pattern(self):
        from sysight.tools.scanner.files import list_files
        result = list_files(str(self.repo), pattern="src/*")
        self.assertEqual(result.total, 1)


class TestScannerSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "train.py").write_text("model = Model()\nmodel.train()\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_search_literal(self):
        from sysight.tools.scanner.search import search
        result = search(str(self.repo), "model", fixed=True)
        self.assertGreaterEqual(result.total_matches, 2)

    def test_search_regex(self):
        from sysight.tools.scanner.search import search
        result = search(str(self.repo), r"model\.train", fixed=False)
        self.assertEqual(result.total_matches, 1)

    def test_search_no_match(self):
        from sysight.tools.scanner.search import search
        result = search(str(self.repo), "nonexistent_xyz", fixed=True)
        self.assertEqual(result.total_matches, 0)


class TestScannerRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
        (self.repo / "test.py").write_text(content)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_whole_file(self):
        from sysight.tools.scanner.read import read_file
        result = read_file(str(self.repo), "test.py")
        self.assertEqual(result.total_lines, 5)
        self.assertEqual(len(result.lines), 5)

    def test_read_with_range(self):
        from sysight.tools.scanner.read import read_file
        result = read_file(str(self.repo), "test.py", start=2, end=4)
        self.assertEqual(result.shown_start, 2)
        self.assertEqual(result.shown_end, 4)
        self.assertEqual(len(result.lines), 3)

    def test_read_around(self):
        from sysight.tools.scanner.read import read_file
        result = read_file(str(self.repo), "test.py", around=3, context=1)
        self.assertEqual(result.shown_start, 2)
        self.assertEqual(result.shown_end, 4)


if __name__ == "__main__":
    unittest.main()
