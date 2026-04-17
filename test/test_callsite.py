"""Tests for callsite index, search_calls, derive_analysis_scope, and get_callsite_context.

All tests use synthetic in-memory Python source — no real repos needed.
"""

import tempfile
import textwrap
import unittest
from pathlib import Path

from sysight.analyzer.scanners import PythonScanner
from sysight.analyzer.analyzer import (
    build_callsite_index,
    search_calls,
    derive_analysis_scope,
    get_callsite_context,
    RepoScope,
    scan_repo,
)


def _write(root: Path, rel: str, src: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src).strip() + "\n", encoding="utf-8")


def _scan_py(root: Path, rel: str) -> object:
    """Parse a single file and return its FileFacts."""
    scanner = PythonScanner(root)
    return scanner._parse(root / rel)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CallSiteFacts collection
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallSiteFacts(unittest.TestCase):

    def test_loop_depth_inside_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", """
                def train_step(loader, device):
                    for batch in loader:
                        x = batch.to(device)
            """)
            facts = _scan_py(root, "train.py")
            to_sites = [cs for cs in facts.callsites if cs.call_name == "to"]
            self.assertEqual(len(to_sites), 1)
            self.assertEqual(to_sites[0].loop_depth, 1)
            self.assertEqual(to_sites[0].receiver, "batch")

    def test_runfiles_are_opt_in_for_scanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "trainer.py", """
                def main():
                    pass
            """)
            _write(root, "trainer.runfiles/pkg/train_loop.py", """
                def train_one_iter(loss):
                    loss.backward()
            """)

            default_files, _ = scan_repo(root)
            self.assertIn("trainer.py", default_files)
            self.assertNotIn("trainer.runfiles/pkg/train_loop.py", default_files)

            scope = RepoScope(mode="full", include_runfiles=True)
            runfiles_files, _ = scan_repo(root, scope=scope)
            self.assertIn("trainer.runfiles/pkg/train_loop.py", runfiles_files)

    def test_loop_depth_nested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", """
                def train_step(loader, device):
                    for batch in loader:
                        for item in batch:
                            x = item.cuda()
            """)
            facts = _scan_py(root, "train.py")
            cuda_sites = [cs for cs in facts.callsites if cs.call_name == "cuda"]
            self.assertEqual(len(cuda_sites), 1)
            self.assertEqual(cuda_sites[0].loop_depth, 2)

    def test_init_not_in_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "model.py", """
                def setup(model, device):
                    model.to(device)
            """)
            facts = _scan_py(root, "model.py")
            to_sites = [cs for cs in facts.callsites if cs.call_name == "to"]
            self.assertEqual(len(to_sites), 1)
            self.assertEqual(to_sites[0].loop_depth, 0)
            self.assertEqual(to_sites[0].receiver, "model")

    def test_keywords_captured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", """
                def train_step(batch, device):
                    for b in batch:
                        x = b.to(device, non_blocking=True)
            """)
            facts = _scan_py(root, "train.py")
            to_sites = [cs for cs in facts.callsites if cs.call_name == "to"]
            self.assertEqual(len(to_sites), 1)
            self.assertIn("non_blocking", to_sites[0].keywords)
            self.assertEqual(to_sites[0].keywords["non_blocking"], "True")

    def test_full_call_name_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "t.py", """
                def f(batch, device):
                    for b in batch:
                        b.to(device)
            """)
            facts = _scan_py(root, "t.py")
            to_sites = [cs for cs in facts.callsites if cs.call_name == "to"]
            self.assertTrue(to_sites[0].full_call_name.endswith(".to"))

    def test_stable_id_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "t.py", """
                def f(x, d):
                    x.cuda()
            """)
            facts = _scan_py(root, "t.py")
            cuda_sites = [cs for cs in facts.callsites if cs.call_name == "cuda"]
            self.assertEqual(len(cuda_sites), 1)
            # id must be "<path>:<line>:<col>:<call_name>"
            parts = cuda_sites[0].id.split(":")
            self.assertEqual(parts[0], "t.py")
            self.assertEqual(parts[-1], "cuda")

    def test_enclosing_function_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "t.py", """
                class Trainer:
                    def step(self, batch, device):
                        for b in batch:
                            b.to(device)
            """)
            facts = _scan_py(root, "t.py")
            to_sites = [cs for cs in facts.callsites if cs.call_name == "to"]
            self.assertEqual(len(to_sites), 1)
            self.assertIn("Trainer.step", to_sites[0].enclosing_function)

    def test_callsites_default_empty_for_no_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "empty.py", """
                def nothing():
                    x = 1 + 2
            """)
            facts = _scan_py(root, "empty.py")
            self.assertEqual(facts.callsites, [])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. build_callsite_index + search_calls
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchCalls(unittest.TestCase):

    def _build(self, root: Path):
        files, _ = scan_repo(root)
        return files, build_callsite_index(files)

    def test_loop_call_ranked_higher_than_init(self):
        """Loop-inner .to() must score higher than init-phase model.to()."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", """
                def setup(model, device):
                    model.to(device)

                def train_step(loader, device):
                    for batch in loader:
                        x = batch.to(device)
            """)
            files, index = self._build(root)
            scope = derive_analysis_scope("gpu_memcpy_h2d", files)
            results = search_calls(index, scope, names=["to"], keywords=["device"])
            self.assertGreater(len(results), 0)
            # first result must be the loop-inner one
            self.assertGreater(results[0].loop_depth, 0)

    def test_search_finds_cuda_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", """
                def train_step(loader):
                    for batch in loader:
                        x = batch[0].cuda()
            """)
            files, index = self._build(root)
            scope = derive_analysis_scope("gpu_memcpy_h2d", files)
            results = search_calls(index, scope, names=["cuda"])
            self.assertTrue(any(r.call_name == "cuda" for r in results))

    def test_search_returns_empty_for_unknown_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "t.py", "def f(): pass")
            files, index = self._build(root)
            scope = derive_analysis_scope("gpu_memcpy_h2d", files)
            results = search_calls(index, scope, names=["nonexistent_call_xyz"])
            self.assertEqual(results, [])

    def test_limit_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "t.py", """
                def f(loader, device):
                    for batch in loader:
                        a = batch.to(device)
                        b = batch.to(device)
                        c = batch.to(device)
            """)
            files, index = self._build(root)
            scope = derive_analysis_scope("gpu_memcpy_h2d", files)
            results = search_calls(index, scope, names=["to"], limit=2)
            self.assertLessEqual(len(results), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. derive_analysis_scope
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeriveAnalysisScope(unittest.TestCase):

    def _files(self, root: Path):
        files, _ = scan_repo(root)
        return files

    def test_memcpy_h2d_scope_contains_train_and_data_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", "def train(): pass")
            _write(root, "data/loader.py", "def load(): pass")
            _write(root, "config.py", "def cfg(): pass")
            files = self._files(root)
            scope = derive_analysis_scope("gpu_memcpy_h2d", files)
            # train.py and data/loader.py should be selected; config.py may or may not be
            self.assertIn("train.py", scope.selected_files)
            self.assertIn("data/loader.py", scope.selected_files)

    def test_memcpy_h2d_scope_call_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "t.py", "def f(): pass")
            files = self._files(root)
            scope = derive_analysis_scope("gpu_memcpy_h2d", files)
            for name in ("to", "cuda", "copy_"):
                self.assertIn(name, scope.call_names)

    def test_scope_has_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", "def f(): pass")
            files = self._files(root)
            scope = derive_analysis_scope("gpu_memcpy_h2d", files)
            self.assertTrue(len(scope.reason) > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. get_callsite_context
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetCallsiteContext(unittest.TestCase):

    def test_returns_function_snippet_not_full_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", """
                def unrelated():
                    pass

                def train_step(loader, device):
                    for batch in loader:
                        x = batch.to(device)
                    return x
            """)
            files, _ = scan_repo(root)
            index = build_callsite_index(files)
            scope = derive_analysis_scope("gpu_memcpy_h2d", files)
            results = search_calls(index, scope, names=["to"])
            self.assertGreater(len(results), 0)
            ctx = get_callsite_context(results[0].id, files, repo_root=root)
            self.assertIsNotNone(ctx)
            # Must contain the callsite line
            self.assertIn("to", ctx.source_snippet)
            # Must NOT be the whole file (unrelated fn should not appear)
            self.assertNotIn("def unrelated", ctx.source_snippet)

    def test_returns_none_for_unknown_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "t.py", "def f(): pass")
            files, _ = scan_repo(root)
            ctx = get_callsite_context("nonexistent:999:0:to", files, repo_root=root)
            self.assertIsNone(ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NVTX region → repo callsite mapping
# ═══════════════════════════════════════════════════════════════════════════════

class TestNvtxRegionMapping(unittest.TestCase):
    """_map_nvtx_regions_to_repo finds user-defined nvtx.range_push callsites."""

    def _make_trace_with_nvtx(self, region_names: list[str]):
        """Build a minimal NsysTrace with NVTX events."""
        from sysight.analyzer.nsys.models import NsysTrace, TimelineEvent
        events = [
            TimelineEvent(
                category="nvtx", name=name,
                start_ns=i * 1_000_000, dur_ns=500_000,
                is_sample=False,
            )
            for i, name in enumerate(region_names)
        ]
        return NsysTrace(
            tool="nsys",
            profile_path="test.sqlite",
            sqlite_path="test.sqlite",
            events=events,
            duration_ns=max(len(region_names) * 1_000_000, 1),
        )

    def test_finds_range_push_callsite_by_string_arg(self):
        """range_push('forward') callsite is found when 'forward' NVTX region exists."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "trainer.py", """
                import torch
                def train_step(model, batch):
                    torch.cuda.nvtx.range_push('forward')
                    out = model(batch)
                    torch.cuda.nvtx.range_pop()
                    return out
            """)
            files, _ = scan_repo(root, scope=RepoScope(
                mode="full", max_files=100, max_file_bytes=1_000_000,
                include_runfiles=False,
            ))
            trace = self._make_trace_with_nvtx(["forward", "forward", "backward"])

            from sysight.analyzer.nsys import _map_nvtx_regions_to_repo
            hotspots = _map_nvtx_regions_to_repo(trace, files)

            self.assertTrue(len(hotspots) > 0, "Should find at least one NVTX hotspot")
            found = [h for h in hotspots if h.sample.frame.symbol == "forward"]
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].match_reason, "nvtx_region")
            self.assertIsNotNone(found[0].repo_file)
            self.assertGreater(found[0].match_confidence, 0)

    def test_noise_regions_filtered_out(self):
        """Holding GIL, iter_NNN regions are not emitted as hotspots."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "trainer.py", """
                def f(): pass
            """)
            files, _ = scan_repo(root, scope=RepoScope(
                mode="full", max_files=100, max_file_bytes=1_000_000,
                include_runfiles=False,
            ))
            noisy = ["Holding GIL", "Waiting for GIL", "iter_100", "iter_101",
                     "0xffffffffb03f9078"]
            trace = self._make_trace_with_nvtx(noisy)

            from sysight.analyzer.nsys import _map_nvtx_regions_to_repo
            hotspots = _map_nvtx_regions_to_repo(trace, files)

            self.assertEqual(hotspots, [], "All noisy regions should be filtered")

    def test_empty_files_returns_empty(self):
        """Empty file dict yields no hotspots."""
        trace = self._make_trace_with_nvtx(["forward"])
        from sysight.analyzer.nsys import _map_nvtx_regions_to_repo
        hotspots = _map_nvtx_regions_to_repo(trace, {})
        self.assertEqual(hotspots, [])

    def test_no_nvtx_events_returns_empty(self):
        """Trace with no NVTX events yields no hotspots."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "trainer.py", "def f(): pass")
            files, _ = scan_repo(root, scope=RepoScope(
                mode="full", max_files=100, max_file_bytes=1_000_000,
                include_runfiles=False,
            ))
            trace = self._make_trace_with_nvtx([])  # no NVTX events
            from sysight.analyzer.nsys import _map_nvtx_regions_to_repo
            hotspots = _map_nvtx_regions_to_repo(trace, files)
            self.assertEqual(hotspots, [])


if __name__ == "__main__":
    unittest.main()
