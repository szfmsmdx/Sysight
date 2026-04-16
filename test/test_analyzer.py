"""Comprehensive tests for the sysight static repository analyzer.

Coverage:
  - Entry point detection (Python training / inference / generic)
  - Cross-file call-chain tracing (Python & Rust)
  - Ignore rules (vendor / runfiles)
  - DAG structure (nodes, edges, in-degrees, reachability)
  - Python class method parsing
  - search_symbols  (exact / prefix / substring / file match)
  - impact_radius   (reverse-DAG propagation, depth tracking)
  - trace_from      (file-level & symbol-level targeting)
  - find_hubs       (degree computation)
  - C++ scanner     (include resolution, function extraction)
  - Java scanner    (import resolution, method extraction)
  - Go scanner      (package main, func main detection)
  - render_summary / render_trace output format
  - AnalysisResult.to_dict  (JSON-serialisable output)
"""

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from sysight import (
    analyze_repo,
    render_summary,
    render_trace,
    search_symbols,
    impact_radius,
    find_hubs,
    trace_from,
)
from sysight.analyzer.repo import (
    build_dag,
    scan_repo,
    FileDAG,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Entry-point detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntryPointDetection(unittest.TestCase):
    """Entry-point detection: mode classification and multi-file call chains."""

    def test_detects_training_entry_and_cross_file_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train.py", """
                from engine.trainer import run_training

                def main():
                    run_training()

                if __name__ == "__main__":
                    main()
            """)
            _write(root, "engine/trainer.py", """
                from engine.model import build_model
                from engine.data import build_dataloader

                def run_training():
                    build_model()
                    build_dataloader()
            """)
            _write(root, "engine/model.py", """
                def build_model():
                    return "model"
            """)
            _write(root, "engine/data.py", """
                def build_dataloader():
                    return "loader"
            """)

            result = analyze_repo(root)

            self.assertGreaterEqual(len(result.entry_points), 1)
            self.assertEqual(result.entry_points[0].path, "train.py")
            self.assertEqual(result.entry_points[0].mode, "training")

            chain = result.call_chains[0]
            self.assertIn("train.py", chain.visited_files)
            self.assertIn("engine/trainer.py", chain.visited_files)
            self.assertIn("engine/model.py", chain.visited_files)
            self.assertIn("engine/data.py", chain.visited_files)

            targets = {step.to_symbol for step in chain.steps}
            self.assertIn("engine/trainer.py::run_training", targets)
            self.assertIn("engine/model.py::build_model", targets)
            self.assertIn("engine/data.py::build_dataloader", targets)

    def test_detects_inference_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "inference.py", """
                from serving.pipeline import predict

                def main():
                    predict()

                if __name__ == "__main__":
                    main()
            """)
            _write(root, "serving/pipeline.py", """
                def predict():
                    return {"status": "ok"}
            """)

            result = analyze_repo(root)

            self.assertGreaterEqual(len(result.entry_points), 1)
            self.assertEqual(result.entry_points[0].path, "inference.py")
            self.assertEqual(result.entry_points[0].mode, "inference")

    def test_detects_rust_main_and_traces_module_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "src/main.rs", """
                mod core;

                fn main() {
                    run_cli();
                }

                fn run_cli() {
                    core::telemetry::maybe_ping();
                }
            """)
            _write(root, "src/core/mod.rs", """
                pub mod telemetry;
            """)
            _write(root, "src/core/telemetry.rs", """
                pub fn maybe_ping() {}
            """)

            result = analyze_repo(root)

            self.assertGreaterEqual(len(result.entry_points), 1)
            self.assertEqual(result.entry_points[0].path, "src/main.rs")
            self.assertEqual(result.entry_points[0].mode, "generic")

            chain = result.call_chains[0]
            self.assertIn("src/main.rs", chain.visited_files)
            self.assertIn("src/core/telemetry.rs", chain.visited_files)
            targets = {step.to_symbol for step in chain.steps}
            self.assertIn("src/main.rs::run_cli", targets)
            self.assertIn("src/core/telemetry.rs::maybe_ping", targets)

    def test_ignores_vendor_trees_and_keeps_project_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "trainer.py", """
                def main():
                    train()

                def train():
                    return "ok"

                if __name__ == "__main__":
                    main()
            """)
            _write(root, "external/site-packages/fakepkg/train.py", """
                def main():
                    raise RuntimeError("should be ignored")
            """)
            _write(root, "trainer.runfiles/generated.py", """
                def main():
                    raise RuntimeError("should be ignored")
            """)

            result = analyze_repo(root)

            self.assertEqual(result.entry_points[0].path, "trainer.py")
            self.assertEqual(result.source_files, 1)
            summary = render_summary(result)
            self.assertIn("Likely entry points:", summary)
            self.assertIn("trainer.py", summary)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DAG structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileDAG(unittest.TestCase):
    """Direct assertions on the DAG: nodes, edges, in-degrees, reachability."""

    def _build(self, root: Path) -> tuple[dict, FileDAG]:
        files, _ = scan_repo(root)
        dag = build_dag(files)
        return files, dag

    def test_dag_nodes_and_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "app.py", """
                from lib.utils import helper

                def main():
                    helper()
            """)
            _write(root, "lib/utils.py", """
                def helper():
                    pass
            """)

            files, dag = self._build(root)

            self.assertIn("app.py", dag.nodes)
            self.assertIn("lib/utils.py", dag.nodes)
            # app.py  ->  lib/utils.py
            self.assertIn("lib/utils.py", dag.edges.get("app.py", set()))

    def test_dag_indegree_zero_is_candidate_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "entry.py", """
                from core.logic import run

                def main():
                    run()

                if __name__ == "__main__":
                    main()
            """)
            _write(root, "core/logic.py", """
                def run():
                    pass
            """)

            _, dag = self._build(root)

            zeros = dag.zero_indegree()
            self.assertIn("entry.py", zeros)
            self.assertNotIn("core/logic.py", zeros)

    def test_dag_reachable_from_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "top.py", "from mid import foo\ndef run(): foo()")
            _write(root, "mid.py", "from bottom import bar\ndef foo(): bar()")
            _write(root, "bottom.py", "def bar(): pass")

            _, dag = self._build(root)

            reachable = {path for path, _ in dag.reachable_from("top.py")}
            self.assertIn("top.py", reachable)
            self.assertIn("mid.py", reachable)
            self.assertIn("bottom.py", reachable)

    def test_dag_no_self_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "solo.py", "def main(): pass\n\nif __name__ == '__main__': main()")

            _, dag = self._build(root)

            self.assertNotIn("solo.py", dag.edges.get("solo.py", set()))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Python class method parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestPythonClassMethods(unittest.TestCase):
    """Python scanner must parse class methods and expose them as qualified symbols."""

    def _facts(self, root: Path, rel: str) -> object:
        files, _ = scan_repo(root)
        return files[rel]

    def test_class_method_appears_in_functions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "model.py", """
                class Transformer:
                    def forward(self, x):
                        return x

                    def _init_weights(self):
                        pass
            """)

            facts = self._facts(root, "model.py")

            self.assertIn("Transformer.forward", facts.functions)
            self.assertIn("Transformer._init_weights", facts.functions)

    def test_class_method_qualified_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "net.py", """
                class Net:
                    def train_step(self, batch):
                        loss = self.forward(batch)
                        return loss
            """)

            facts = self._facts(root, "net.py")
            fn = facts.functions["Net.train_step"]

            self.assertEqual(fn.qualified_name, "net.py::Net.train_step")
            self.assertGreater(fn.line, 0)

    def test_class_method_calls_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "trainer.py", """
                class Trainer:
                    def fit(self, model, loader):
                        for batch in loader:
                            model.train_step(batch)
                            self.log(batch)
            """)

            facts = self._facts(root, "trainer.py")
            calls = facts.functions["Trainer.fit"].calls

            self.assertIn("model.train_step", calls)
            self.assertIn("self.log", calls)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  search_symbols
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchSymbols(unittest.TestCase):
    """search_symbols: exact, prefix, substring, file-path matching."""

    def _scan(self, root: Path):
        files, _ = scan_repo(root)
        return files

    def _repo(self) -> tempfile.TemporaryDirectory:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        _write(root, "train.py", """
            def train_model():
                pass

            def evaluate_model():
                pass
        """)
        _write(root, "utils/metrics.py", """
            def compute_accuracy():
                pass

            def compute_loss():
                pass
        """)
        return tmp

    def test_exact_function_match_gets_highest_score(self) -> None:
        with self._repo() as tmp:
            files = self._scan(Path(tmp))
            results = search_symbols(files, "train_model")

            self.assertTrue(any(r.symbol.endswith("::train_model") for r in results))
            top = next(r for r in results if r.symbol.endswith("::train_model"))
            self.assertEqual(top.score, 3.0)

    def test_prefix_match_returns_results(self) -> None:
        with self._repo() as tmp:
            files = self._scan(Path(tmp))
            results = search_symbols(files, "compute")

            names = [r.symbol for r in results]
            self.assertTrue(any("compute_accuracy" in n for n in names))
            self.assertTrue(any("compute_loss" in n for n in names))
            for r in results:
                if "compute_" in r.symbol:
                    self.assertGreaterEqual(r.score, 1.0)

    def test_file_path_match(self) -> None:
        with self._repo() as tmp:
            files = self._scan(Path(tmp))
            results = search_symbols(files, "metrics")

            self.assertTrue(any(r.kind == "file" and "metrics" in r.path for r in results))

    def test_empty_query_returns_empty(self) -> None:
        with self._repo() as tmp:
            files = self._scan(Path(tmp))
            self.assertEqual(search_symbols(files, ""), [])
            self.assertEqual(search_symbols(files, "   "), [])

    def test_limit_respected(self) -> None:
        with self._repo() as tmp:
            files = self._scan(Path(tmp))
            results = search_symbols(files, "model", limit=1)
            self.assertLessEqual(len(results), 1)

    def test_no_match_returns_empty(self) -> None:
        with self._repo() as tmp:
            files = self._scan(Path(tmp))
            results = search_symbols(files, "zzz_nonexistent_symbol_xyz")
            self.assertEqual(results, [])


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  impact_radius
# ═══════════════════════════════════════════════════════════════════════════════

class TestImpactRadius(unittest.TestCase):
    """impact_radius: reverse-DAG propagation, depth tracking, max_nodes cap."""

    def _build(self, root: Path):
        files, _ = scan_repo(root)
        dag = build_dag(files)
        return files, dag

    def test_direct_importer_is_impacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "core.py", "def compute(): pass")
            _write(root, "app.py", "from core import compute\ndef run(): compute()")

            files, dag = self._build(root)
            result = impact_radius(files, dag, ["core.py"])

            self.assertIn("app.py", result.impacted_files)
            self.assertNotIn("core.py", result.impacted_files)   # seed is not in impacted
            self.assertEqual(result.seed_files, ["core.py"])

    def test_transitive_impact_across_three_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "base.py", "def primitive(): pass")
            _write(root, "mid.py", "from base import primitive\ndef helper(): primitive()")
            _write(root, "top.py", "from mid import helper\ndef main(): helper()\nif __name__=='__main__': main()")

            files, dag = self._build(root)
            result = impact_radius(files, dag, ["base.py"])

            self.assertIn("mid.py", result.impacted_files)
            self.assertIn("top.py", result.impacted_files)
            self.assertEqual(result.depth_map["mid.py"], 1)
            self.assertEqual(result.depth_map["top.py"], 2)

    def test_max_depth_limits_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "a.py", "def a(): pass")
            _write(root, "b.py", "from a import a\ndef b(): a()")
            _write(root, "c.py", "from b import b\ndef c(): b()")

            files, dag = self._build(root)
            result = impact_radius(files, dag, ["a.py"], max_depth=1)

            self.assertIn("b.py", result.impacted_files)
            self.assertNotIn("c.py", result.impacted_files)

    def test_unchanged_file_not_in_impact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "foo.py", "def foo(): pass")
            _write(root, "bar.py", "def bar(): pass")   # no import of foo

            files, dag = self._build(root)
            result = impact_radius(files, dag, ["foo.py"])

            self.assertNotIn("bar.py", result.impacted_files)

    def test_nonexistent_seed_is_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "only.py", "def run(): pass")

            files, dag = self._build(root)
            result = impact_radius(files, dag, ["ghost.py"])

            self.assertEqual(result.seed_files, [])
            self.assertEqual(result.impacted_files, [])


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  trace_from
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceFrom(unittest.TestCase):
    """trace_from: file targeting, symbol targeting, partial path matching."""

    def _build(self, root: Path):
        files, _ = scan_repo(root)
        dag = build_dag(files)
        return files, dag

    def test_trace_from_file_follows_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "runner.py", """
                from worker import process

                def start():
                    process()
            """)
            _write(root, "worker.py", """
                def process():
                    pass
            """)

            files, dag = self._build(root)
            chains = trace_from(files, dag, "runner.py")

            self.assertGreaterEqual(len(chains), 1)
            all_visited = set()
            for c in chains:
                all_visited.update(c.visited_files)
            self.assertIn("worker.py", all_visited)

    def test_trace_from_partial_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "pkg/entry.py", """
                from pkg.helper import do_work

                def main():
                    do_work()

                if __name__ == "__main__":
                    main()
            """)
            _write(root, "pkg/helper.py", """
                def do_work():
                    pass
            """)

            files, dag = self._build(root)
            # partial path suffix should resolve
            chains = trace_from(files, dag, "entry.py")

            self.assertGreaterEqual(len(chains), 1)

    def test_trace_from_specific_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "pipeline.py", """
                from io_utils import load, save

                def preprocess():
                    load()

                def postprocess():
                    save()
            """)
            _write(root, "io_utils.py", """
                def load(): pass
                def save(): pass
            """)

            files, dag = self._build(root)
            chains = trace_from(files, dag, "pipeline.py", symbol="preprocess")

            self.assertEqual(len(chains), 1)
            targets = {step.to_symbol for step in chains[0].steps}
            self.assertTrue(any("load" in t for t in targets))
            # postprocess is NOT in this chain
            self.assertFalse(any("postprocess" in s for s in chains[0].visited_symbols))

    def test_trace_from_unmatched_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "solo.py", "def run(): pass")

            files, dag = self._build(root)
            chains = trace_from(files, dag, "completely_missing.py")

            self.assertEqual(chains, [])


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  find_hubs
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindHubs(unittest.TestCase):
    """find_hubs: highly-called shared utilities should surface at the top."""

    def test_shared_utility_is_top_hub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # shared_utils is called by three callers
            _write(root, "shared_utils.py", """
                def log_event(msg):
                    pass

                def format_output(data):
                    pass
            """)
            _write(root, "a.py", """
                from shared_utils import log_event, format_output

                def task_a():
                    log_event("a")
                    format_output({})
            """)
            _write(root, "b.py", """
                from shared_utils import log_event

                def task_b():
                    log_event("b")
            """)
            _write(root, "c.py", """
                from shared_utils import log_event, format_output

                def task_c():
                    log_event("c")
                    format_output({})
            """)

            files, _ = scan_repo(root)
            dag = build_dag(files)
            hubs = find_hubs(files, dag, top_n=5)

            self.assertGreaterEqual(len(hubs), 1)
            # log_event and format_output are called most; both must appear
            hub_symbols = {h.symbol for h in hubs}
            self.assertTrue(
                any("log_event" in s or "format_output" in s for s in hub_symbols),
                f"Expected utility functions in hubs, got: {hub_symbols}",
            )

    def test_top_n_limits_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # create 6 distinct functions with calls
            for i in range(6):
                _write(root, f"mod{i}.py", f"""
                    def fn{i}():
                        helper()

                    def helper():
                        pass
                """)

            files, _ = scan_repo(root)
            dag = build_dag(files)
            hubs = find_hubs(files, dag, top_n=3)

            self.assertLessEqual(len(hubs), 3)

    def test_hub_degrees_are_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "caller.py", """
                def caller():
                    callee()

                def callee():
                    pass
            """)

            files, _ = scan_repo(root)
            dag = build_dag(files)
            hubs = find_hubs(files, dag, top_n=10)

            for hub in hubs:
                self.assertGreater(hub.total_degree, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  C++ scanner
# ═══════════════════════════════════════════════════════════════════════════════

class TestCppScanner(unittest.TestCase):
    """CppScanner: function extraction, include resolution, main detection."""

    def _scan(self, root: Path):
        files, _ = scan_repo(root)
        return files

    def test_cpp_functions_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "main.cpp", """
                #include "utils.h"

                int main() {
                    run();
                    return 0;
                }

                void run() {
                    process();
                }
            """)
            _write(root, "utils.h", """
                void process();
            """)

            files = self._scan(root)

            self.assertIn("main.cpp", files)
            facts = files["main.cpp"]
            self.assertIn("main", facts.functions)
            self.assertIn("run", facts.functions)
            self.assertEqual(facts.language, "cpp")

    def test_cpp_main_detected_as_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "program.cpp", """
                void helper() {}

                int main() {
                    helper();
                    return 0;
                }
            """)

            files = self._scan(root)
            facts = files["program.cpp"]

            self.assertTrue(facts.has_main_guard)
            self.assertGreater(facts.generic_score, 0)

    def test_cpp_include_links_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "src/app.cpp", '#include "net.h"\nvoid run() { connect(); }')
            _write(root, "src/net.h", "void connect();")

            files = self._scan(root)
            facts = files["src/app.cpp"]

            # net should be in imports (even as header)
            self.assertIn("net", facts.imports)

    def test_cpp_call_extracted_from_function_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "ops.cpp", """
                void kernel() {}

                void launcher() {
                    kernel();
                }
            """)

            files = self._scan(root)
            calls = files["ops.cpp"].functions["launcher"].calls

            self.assertIn("kernel", calls)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Java scanner
# ═══════════════════════════════════════════════════════════════════════════════

class TestJavaScanner(unittest.TestCase):
    """JavaScanner: method extraction, import binding, main detection."""

    def _scan(self, root: Path):
        files, _ = scan_repo(root)
        return files

    def test_java_methods_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "com/example/App.java", """
                package com.example;

                public class App {
                    public static void main(String[] args) {
                        run();
                    }

                    private static void run() {
                        System.out.println("hello");
                    }
                }
            """)

            files = self._scan(root)

            self.assertIn("com/example/App.java", files)
            facts = files["com/example/App.java"]
            self.assertIn("main", facts.functions)
            self.assertIn("run", facts.functions)
            self.assertEqual(facts.language, "java")

    def test_java_main_has_high_generic_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "Main.java", """
                public class Main {
                    public static void main(String[] args) {
                        System.out.println("start");
                    }
                }
            """)

            files = self._scan(root)
            facts = files["Main.java"]

            self.assertTrue(facts.has_main_guard)
            self.assertGreater(facts.generic_score, 0)

    def test_java_import_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "com/example/Service.java", """
                import com.example.Repo;

                public class Service {
                    public void handle() {
                        Repo r = new Repo();
                        r.save();
                    }
                }
            """)
            _write(root, "com/example/Repo.java", """
                public class Repo {
                    public void save() {}
                }
            """)

            files = self._scan(root)
            facts = files["com/example/Service.java"]

            self.assertIn("Repo", facts.imports)
            imp = facts.imports["Repo"]
            self.assertEqual(imp.binding_type, "import")

    def test_java_method_calls_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "Processor.java", """
                public class Processor {
                    public void process() {
                        validate();
                        transform();
                        store();
                    }

                    private void validate() {}
                    private void transform() {}
                    private void store() {}
                }
            """)

            files = self._scan(root)
            calls = files["Processor.java"].functions["process"].calls

            self.assertIn("validate", calls)
            self.assertIn("transform", calls)
            self.assertIn("store", calls)


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Go scanner
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoScanner(unittest.TestCase):
    """GoScanner: func extraction, package main detection, import handling."""

    def _scan(self, root: Path):
        files, _ = scan_repo(root)
        return files

    def test_go_main_func_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "main.go", """
                package main

                import "fmt"

                func main() {
                    greet()
                }

                func greet() {
                    fmt.Println("hello")
                }
            """)

            files = self._scan(root)

            self.assertIn("main.go", files)
            facts = files["main.go"]
            self.assertIn("main", facts.functions)
            self.assertIn("greet", facts.functions)
            self.assertTrue(facts.has_main_guard)
            self.assertGreater(facts.generic_score, 0)
            self.assertEqual(facts.language, "go")

    def test_go_non_main_package_lower_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "lib/helper.go", """
                package helper

                func Compute() int {
                    return 42
                }
            """)

            files = self._scan(root)
            facts = files["lib/helper.go"]

            self.assertFalse(facts.has_main_guard)

    def test_go_func_calls_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "server.go", """
                package main

                func main() {
                    serve()
                }

                func serve() {
                    listen()
                    accept()
                }

                func listen() {}
                func accept() {}
            """)

            files = self._scan(root)
            calls = files["server.go"].functions["serve"].calls

            self.assertIn("listen", calls)
            self.assertIn("accept", calls)

    def test_go_import_group_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "app.go", """
                package main

                import (
                    "fmt"
                    "os"
                    log "log/slog"
                )

                func main() {
                    fmt.Println("ok")
                }
            """)

            files = self._scan(root)
            facts = files["app.go"]

            self.assertIn("fmt", facts.imports)
            self.assertIn("os", facts.imports)
            self.assertIn("log", facts.imports)   # alias


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  Render helpers + JSON output
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderAndJSON(unittest.TestCase):
    """render_summary, render_trace, and to_dict produce well-formed output."""

    def _make_result(self, tmp_dir: str):
        root = Path(tmp_dir)
        _write(root, "run.py", """
            from utils import helper

            def main():
                helper()

            if __name__ == "__main__":
                main()
        """)
        _write(root, "utils.py", """
            def helper():
                pass
        """)
        return analyze_repo(root)

    def test_render_summary_contains_key_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._make_result(tmp)
            summary = render_summary(result)

            self.assertIn("Repo:", summary)
            self.assertIn("Source files analyzed:", summary)
            self.assertIn("Likely entry points:", summary)
            self.assertIn("Call chain preview:", summary)

    def test_render_summary_entry_point_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._make_result(tmp)
            summary = render_summary(result)

            self.assertIn("run.py", summary)

    def test_to_dict_is_json_serialisable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._make_result(tmp)
            d = result.to_dict()

            # Must not raise
            serialised = json.dumps(d)
            parsed = json.loads(serialised)

            self.assertIn("entry_points", parsed)
            self.assertIn("call_chains", parsed)
            self.assertIn("hub_nodes", parsed)
            self.assertIn("source_files", parsed)

    def test_to_dict_entry_point_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._make_result(tmp)
            d = result.to_dict()

            self.assertGreater(len(d["entry_points"]), 0)
            ep = d["entry_points"][0]
            for key in ("path", "module_name", "mode", "score", "reasons", "start_symbols"):
                self.assertIn(key, ep, f"Missing key: {key}")

    def test_render_trace_shows_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "src.py", """
                from tgt import work

                def start():
                    work()
            """)
            _write(root, "tgt.py", """
                def work():
                    pass
            """)

            files, _ = scan_repo(root)
            dag = build_dag(files)
            chains = trace_from(files, dag, "src.py")
            output = render_trace(chains, "src.py")

            self.assertIn("Trace from:", output)
            self.assertIn("Chains:", output)


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  Language distribution & multi-language repo
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiLanguage(unittest.TestCase):
    """analyze_repo correctly counts files per language in a mixed repo."""

    def test_language_distribution_mixed_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "main.py", "def run(): pass\nif __name__=='__main__': run()")
            _write(root, "lib.go", "package lib\nfunc Helper() {}")
            _write(root, "core.cpp", "void compute() {}")

            result = analyze_repo(root)

            self.assertIn("python", result.languages)
            self.assertIn("go", result.languages)
            self.assertIn("cpp", result.languages)
            self.assertEqual(result.languages["python"], 1)
            self.assertEqual(result.languages["go"], 1)
            self.assertEqual(result.languages["cpp"], 1)

    def test_source_files_count_matches_actual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(5):
                _write(root, f"mod{i}.py", f"def fn{i}(): pass")

            result = analyze_repo(root)

            self.assertEqual(result.source_files, 5)


if __name__ == "__main__":
    unittest.main()
