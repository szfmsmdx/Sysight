"""Unit tests for sysight.analyzer.scanner modules.

Each test creates a minimal synthetic repo in a temp directory.
No external repos required; tests are deterministic and fast.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sysight.analyzer.scanner.fs import list_files
from sysight.analyzer.scanner.search import search
from sysight.analyzer.scanner.reader import read_file
from sysight.analyzer.scanner.callsites import find_callsites
from sysight.analyzer.scanner.symbols import (
    list_symbols, find_callers, find_callees, trace_symbol,
)
from sysight.analyzer.scanner.variants import find_variants


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")


# ── fs tests ──────────────────────────────────────────────────────────────────

class TestFs(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(self.root, "src/a.py", "x = 1\n")
        _write(self.root, "src/b.py", "y = 2\n")
        _write(self.root, "src/data/c.csv", "col1,col2\n")
        _write(self.root, "__pycache__/x.pyc", "binary\n")  # should be skipped

    def tearDown(self):
        self._tmp.cleanup()

    def test_list_all(self):
        r = list_files(str(self.root))
        paths = {f.path for f in r.files}
        self.assertIn("src/a.py", paths)
        self.assertIn("src/b.py", paths)
        self.assertIn("src/data/c.csv", paths)
        # __pycache__ is skipped
        self.assertFalse(any("pycache" in p for p in paths))

    def test_ext_filter(self):
        r = list_files(str(self.root), ext="py")
        exts = {f.ext for f in r.files}
        self.assertEqual(exts, {".py"})
        self.assertEqual(r.total, 2)

    def test_pattern_filter(self):
        r = list_files(str(self.root), pattern="*/data/*")
        self.assertEqual(r.total, 1)
        self.assertEqual(r.files[0].path, "src/data/c.csv")

    def test_max_results(self):
        r = list_files(str(self.root), max_results=1)
        self.assertEqual(r.total, 1)


# ── search tests ──────────────────────────────────────────────────────────────

class TestSearch(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(self.root, "src/a.py", """\
            def foo():
                bar()
                baz()
        """)
        _write(self.root, "src/b.py", """\
            def bar():
                pass
        """)

    def tearDown(self):
        self._tmp.cleanup()

    def test_basic_search(self):
        r = search(str(self.root), "bar")
        self.assertGreaterEqual(r.total_matches, 2)  # called in a.py, defined in b.py

    def test_ext_filter(self):
        r = search(str(self.root), "bar", ext="py")
        self.assertGreaterEqual(r.total_matches, 1)

    def test_fixed_string(self):
        r = search(str(self.root), "baz()", fixed=True)
        self.assertEqual(r.total_matches, 1)  # only call in a.py

    def test_no_match(self):
        r = search(str(self.root), "no_such_symbol_xyz")
        self.assertEqual(r.total_matches, 0)

    def test_case_insensitive(self):
        r = search(str(self.root), "BAR", case_sensitive=False)
        self.assertGreaterEqual(r.total_matches, 1)

    def test_result_fields(self):
        r = search(str(self.root), "def foo")
        self.assertEqual(r.total_matches, 1)
        m = r.matches[0]
        self.assertEqual(m.line, 1)
        self.assertIn("foo", m.text)
        self.assertGreaterEqual(m.column, 1)

    def test_max_results(self):
        r = search(str(self.root), ".", max_results=2)
        self.assertLessEqual(r.total_matches, 2)


# ── reader tests ──────────────────────────────────────────────────────────────

class TestReader(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        content = "\n".join(f"line{i}" for i in range(1, 21))
        _write(self.root, "src/target.py", content)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_whole(self):
        r = read_file(str(self.root), "src/target.py")
        self.assertIsNone(r.error)
        self.assertEqual(r.total_lines, 20)
        self.assertEqual(len(r.lines), 20)
        self.assertEqual(r.lines[0].line, 1)
        self.assertEqual(r.lines[0].text, "line1")

    def test_read_slice(self):
        r = read_file(str(self.root), "src/target.py", start=5, end=8)
        self.assertEqual(r.shown_start, 5)
        self.assertEqual(r.shown_end, 8)
        self.assertEqual(len(r.lines), 4)
        self.assertEqual(r.lines[0].text, "line5")

    def test_read_around(self):
        r = read_file(str(self.root), "src/target.py", around=10, context=2)
        self.assertEqual(r.shown_start, 8)
        self.assertEqual(r.shown_end, 12)

    def test_file_not_found(self):
        r = read_file(str(self.root), "nonexistent.py")
        self.assertIsNotNone(r.error)

    def test_path_outside_root(self):
        r = read_file(str(self.root), "/etc/passwd")
        self.assertIsNotNone(r.error)
        self.assertIn("outside", r.error)


# ── callsites tests ───────────────────────────────────────────────────────────

class TestCallsites(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(self.root, "src/a.py", """\
            def train():
                optimizer.step()
                compute_loss()
                compute_loss()

            class Trainer:
                def run(self):
                    compute_loss()
        """)
        _write(self.root, "src/b.py", """\
            def compute_loss():
                pass
        """)

    def tearDown(self):
        self._tmp.cleanup()

    def test_find_callsites(self):
        r = find_callsites(str(self.root), "compute_loss")
        # 3 calls in a.py (two in train, one in run) + 1 def in b.py
        # But def lines match our pattern too; let's just check >= 3
        self.assertGreaterEqual(r.total, 3)

    def test_enclosing_scope(self):
        r = find_callsites(str(self.root), "compute_loss")
        # Sites from a.py should have enclosing scopes
        a_sites = [s for s in r.sites if "a.py" in s.path]
        scopes = {s.enclosing for s in a_sites}
        self.assertIn("train", scopes)
        self.assertIn("run", scopes)

    def test_file_filter(self):
        r = find_callsites(str(self.root), "compute_loss", file_filter="src/a.py")
        for s in r.sites:
            self.assertIn("a.py", s.path)

    def test_ext_filter(self):
        r = find_callsites(str(self.root), "compute_loss", ext="py")
        self.assertGreater(r.total, 0)

    def test_max_results(self):
        r = find_callsites(str(self.root), "compute_loss", max_results=1)
        self.assertEqual(len(r.sites), 1)


# ── symbols tests ─────────────────────────────────────────────────────────────

class TestSymbols(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(self.root, "src/ops.py", '''\
            """ops module"""

            def compute_loss(pred, target):
                """Compute cross-entropy loss."""
                return cross_entropy(pred, target)

            async def async_step():
                pass

            class Optimizer:
                def step(self):
                    self.update()
                    self.zero_grad()
        ''')

    def tearDown(self):
        self._tmp.cleanup()

    def test_list_symbols(self):
        r = list_symbols(str(self.root), "src/ops.py")
        self.assertIsNone(r.error)
        names = [s.name for s in r.symbols]
        self.assertIn("compute_loss", names)
        self.assertIn("async_step", names)
        self.assertIn("Optimizer", names)
        self.assertIn("step", names)

    def test_symbol_kinds(self):
        r = list_symbols(str(self.root), "src/ops.py")
        by_name = {s.name: s for s in r.symbols}
        self.assertEqual(by_name["compute_loss"].kind, "function")
        self.assertEqual(by_name["async_step"].kind, "async_function")
        self.assertEqual(by_name["Optimizer"].kind, "class")
        self.assertEqual(by_name["step"].kind, "method")

    def test_docstring(self):
        r = list_symbols(str(self.root), "src/ops.py")
        by_name = {s.name: s for s in r.symbols}
        self.assertEqual(by_name["compute_loss"].docstring, "Compute cross-entropy loss.")

    def test_find_callees(self):
        r = find_callees(str(self.root), "src/ops.py", "compute_loss")
        self.assertIn("cross_entropy", r.callees)

    def test_find_callees_method(self):
        r = find_callees(str(self.root), "src/ops.py", "step")
        self.assertIn("update", r.callees)
        self.assertIn("zero_grad", r.callees)

    def test_find_callers(self):
        # Put a caller in a separate file
        _write(self.root, "src/trainer.py", """\
            from ops import compute_loss
            def train():
                loss = compute_loss(pred, target)
        """)
        r = find_callers(str(self.root), "compute_loss")
        self.assertGreater(r.total, 0)
        paths = {s["path"] for s in r.sites}
        self.assertTrue(any("trainer.py" in p for p in paths))

    def test_trace_symbol(self):
        r = trace_symbol(str(self.root), "compute_loss", max_depth=1)
        self.assertEqual(r.root_symbol, "compute_loss")
        self.assertGreater(len(r.chain), 0)
        root_entry = r.chain[0]
        self.assertEqual(root_entry["symbol"], "compute_loss")
        self.assertIn("cross_entropy", root_entry["callees"])

    def test_trace_symbol_not_found(self):
        r = trace_symbol(str(self.root), "no_such_func_xyz", max_depth=1)
        self.assertEqual(len(r.chain), 1)
        self.assertTrue(r.chain[0]["external"])

    def test_syntax_error_file(self):
        _write(self.root, "src/broken.py", "def (")
        r = list_symbols(str(self.root), "src/broken.py")
        self.assertIsNotNone(r.error)


# ── variants tests ────────────────────────────────────────────────────────────

class TestScannerCliModule(unittest.TestCase):
    def test_module_can_register_and_dispatch_files_command(self):
        from sysight.analyzer.scanner import scanner_cli

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "src/a.py", "x = 1\n")

            parser = argparse.ArgumentParser()
            sub = parser.add_subparsers(dest="subcmd")
            scanner_cli.add_scanner_subparser(sub)
            args = parser.parse_args(["scanner", "files", str(root), "--json"])

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                scanner_cli.dispatch_scanner(args)

        data = json.loads(buf.getvalue())
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["files"][0]["path"], "src/a.py")


class TestVariants(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_dict_registry(self):
        _write(self.root, "src/registry.py", """\
            from models.v1 import ModelV1
            from models.v2 import ModelV2

            MODEL_REGISTRY = {
                "v1": ModelV1,
                "v2": ModelV2,
            }

            def build(name):
                return MODEL_REGISTRY[name]()
        """)

    def _make_if_registry(self):
        _write(self.root, "src/factory.py", """\
            def create_optimizer(name, params):
                if name == "adam":
                    return AdamOptimizer(params)
                elif name == "sgd":
                    return SGDOptimizer(params)
        """)

    def _make_decorator_registry(self):
        _write(self.root, "src/plugins.py", """\
            @register("plugin_a")
            class PluginA:
                pass

            @register("plugin_b")
            class PluginB:
                pass
        """)

    def test_dict_literal(self):
        self._make_dict_registry()
        r = find_variants(str(self.root))
        keys = {e.key for e in r.entries}
        self.assertIn("v1", keys)
        self.assertIn("v2", keys)
        targets = {e.target for e in r.entries if e.key == "v1"}
        self.assertIn("ModelV1", targets)

    def test_if_elif_chain(self):
        self._make_if_registry()
        r = find_variants(str(self.root))
        keys = {e.key for e in r.entries}
        self.assertIn("adam", keys)
        self.assertIn("sgd", keys)
        targets = {e.target for e in r.entries if e.key == "adam"}
        self.assertIn("AdamOptimizer", targets)
        kinds = {e.kind for e in r.entries if e.key in {"adam", "sgd"}}
        self.assertIn("if_elif", kinds)

    def test_decorator(self):
        self._make_decorator_registry()
        r = find_variants(str(self.root))
        keys = {e.key for e in r.entries}
        self.assertIn("plugin_a", keys)
        self.assertIn("plugin_b", keys)
        kinds = {e.kind for e in r.entries}
        self.assertIn("decorator", kinds)

    def test_key_filter(self):
        self._make_dict_registry()
        r = find_variants(str(self.root), key="v1")
        self.assertEqual(r.total, 1)
        self.assertEqual(r.entries[0].key, "v1")
        self.assertEqual(r.entries[0].target, "ModelV1")

    def test_file_filter(self):
        self._make_dict_registry()
        self._make_if_registry()
        r = find_variants(str(self.root), file_filter="src/registry.py")
        for e in r.entries:
            self.assertEqual(e.file, "src/registry.py")

    def test_no_variants(self):
        _write(self.root, "src/plain.py", "x = 1\ny = 2\n")
        r = find_variants(str(self.root))
        # No dict key→Class mappings
        class_targets = [e for e in r.entries if e.target[0].isupper()]
        self.assertEqual(len(class_targets), 0)

    def test_mixed_all_three(self):
        self._make_dict_registry()
        self._make_if_registry()
        self._make_decorator_registry()
        r = find_variants(str(self.root))
        kinds = {e.kind for e in r.entries}
        self.assertTrue(kinds >= {"dict_literal", "if_elif", "decorator"})


if __name__ == "__main__":
    unittest.main()
