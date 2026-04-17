"""End-to-end test: synthetic SQLite (with memcpy data) + synthetic repo
(with .to() in training loop) → NsysDiag.task_drafts is populated.

Tests the full pipeline:
  analyze_nsys() → NsysDiag.findings (gpu_memcpy_hotspot)
  → _generate_task_drafts() → TaskDraft (inferred_by=deterministic)

Separately tests repo callsite layer:
  derive_analysis_scope("gpu_memcpy_hotspot") + search_calls → finds .to() in loop
"""

import sqlite3
import tempfile
import textwrap
import unittest
from pathlib import Path

from sysight.analyzer.nsys import analyze_nsys
from sysight.analyzer.nsys.models import NsysAnalysisRequest
from sysight.analyzer.analyzer import (
    build_callsite_index,
    derive_analysis_scope,
    scan_repo,
    search_calls,
)


# ── SQLite builder helpers ────────────────────────────────────────────────────

def _make_minimal_sqlite(path: str) -> None:
    """Build a minimal Nsight Systems SQLite with enough data for analyze_nsys.

    Creates:
      - CUPTI_ACTIVITY_KIND_KERNEL    (2 GPU kernels, 10ms each)
      - CUPTI_ACTIVITY_KIND_MEMCPY    (3 HtoD copies, 6ms total)
      - StringIds                     (kernel/memcpy names)
    """
    conn = sqlite3.connect(path)
    c = conn.cursor()

    # StringIds
    c.execute("""CREATE TABLE StringIds (
        id INTEGER PRIMARY KEY,
        value TEXT
    )""")
    c.executemany("INSERT INTO StringIds VALUES (?,?)", [
        (1, "void MatMul(float*, float*, float*, int)"),
        (2, "cudaMemcpy HtoD"),
    ])

    # Use shortName to match the real nsys schema expected by _extract_kernels
    c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
        start     INTEGER,
        end       INTEGER,
        shortName INTEGER,
        deviceId  INTEGER,
        streamId  INTEGER,
        correlationId INTEGER
    )""")
    c.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)", [
        (0,        10_000_000, 1, 0, 1, 100),  # 10ms
        (11_000_000, 21_000_000, 1, 0, 1, 101),  # 10ms
    ])

    # CUPTI_ACTIVITY_KIND_MEMCPY  (start_ns, end_ns, copyKind, bytes, deviceId, streamId, correlationId)
    c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
        start         INTEGER,
        end           INTEGER,
        copyKind      INTEGER,
        bytes         INTEGER,
        deviceId      INTEGER,
        streamId      INTEGER,
        correlationId INTEGER
    )""")
    # 3 HtoD copies (copyKind=1), 2ms each → 6ms total out of 30ms trace → ~20%
    c.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)", [
        (21_500_000, 23_500_000, 1, 64_000_000, 0, 1, 200),
        (24_000_000, 26_000_000, 1, 64_000_000, 0, 1, 201),
        (26_500_000, 28_500_000, 1, 64_000_000, 0, 1, 202),
    ])

    conn.commit()
    conn.close()


def _write(root: Path, rel: str, src: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src).strip() + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NsysDiag.task_drafts is generated from findings
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskDraftGeneration(unittest.TestCase):

    def test_memcpy_finding_produces_task_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _make_minimal_sqlite(sq)

            req = NsysAnalysisRequest(
                repo_root=tmp,
                profile_path=sq,
                sqlite_path=sq,
                include_repo_context=False,
            )
            diag = analyze_nsys(req)

        self.assertEqual(diag.status, "ok")
        # There must be at least one finding
        self.assertGreater(len(diag.findings), 0)
        # There must be at least one task draft
        self.assertGreater(len(diag.task_drafts), 0)
        # All task drafts must be deterministic
        for td in diag.task_drafts:
            self.assertEqual(td.inferred_by, "deterministic")

    def test_task_draft_for_memcpy_has_correct_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _make_minimal_sqlite(sq)
            req = NsysAnalysisRequest(
                repo_root=tmp,
                profile_path=sq,
                sqlite_path=sq,
                include_repo_context=False,
            )
            diag = analyze_nsys(req)

        memcpy_drafts = [td for td in diag.task_drafts
                         if "memcpy" in td.id or "memcpy" in td.finding_id]
        self.assertGreater(len(memcpy_drafts), 0, "Expected a memcpy TaskDraft")
        draft = memcpy_drafts[0]
        self.assertIn("memcpy", draft.hypothesis.lower())
        self.assertIsNotNone(draft.verification_metric)
        self.assertNotEqual(draft.verification_metric, "")
        self.assertEqual(draft.target_locations, [])  # empty until LLM fills it
        self.assertEqual(draft.candidate_callsites, [])

    def test_no_duplicate_category_drafts(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _make_minimal_sqlite(sq)
            req = NsysAnalysisRequest(
                repo_root=tmp,
                profile_path=sq,
                sqlite_path=sq,
                include_repo_context=False,
            )
            diag = analyze_nsys(req)

        # Task draft ids must be unique
        ids = [td.id for td in diag.task_drafts]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate task draft ids found")

    def test_empty_sqlite_produces_no_task_drafts(self):
        """If no kernel/memcpy data → no findings → no task drafts."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "empty.sqlite")
            conn = sqlite3.connect(sq)
            conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER, deviceId INTEGER, streamId INTEGER, correlationId INTEGER)")
            conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            conn.commit()
            conn.close()

            req = NsysAnalysisRequest(
                repo_root=tmp,
                profile_path=sq,
                sqlite_path=sq,
                include_repo_context=False,
            )
            diag = analyze_nsys(req)

        # No memcpy data → no gpu_memcpy_hotspot finding → no task draft for it
        memcpy_drafts = [td for td in diag.task_drafts if "memcpy" in td.id]
        self.assertEqual(len(memcpy_drafts), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. End-to-end: profile finding → scope → search_calls finds .to() in loop
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2EMemcpyToRepo(unittest.TestCase):

    def test_memcpy_finding_scope_finds_loop_inner_to(self):
        """Full path: gpu_memcpy_hotspot scope selects train.py and finds .to() in loop."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # A repo with two files: only train.py is in scope for memcpy
            _write(root, "train.py", """
                def setup(model, device):
                    model.to(device)  # init phase, loop_depth=0

                def train_step(loader, device):
                    for batch in loader:
                        x = batch.to(device)  # loop, loop_depth=1
            """)
            _write(root, "config.py", """
                def get_config():
                    return {"lr": 0.001}
            """)

            files, _ = scan_repo(root)
            scope = derive_analysis_scope("gpu_memcpy_hotspot", files)
            index = build_callsite_index(files)
            results = search_calls(index, scope, names=["to"])

            # train.py should be in scope, config.py should not
            self.assertIn("train.py", scope.selected_files)
            self.assertNotIn("config.py", scope.selected_files)

            # The top-ranked result must be the loop-inner one (loop_depth=1)
            self.assertGreater(len(results), 0)
            self.assertGreater(results[0].loop_depth, 0,
                               f"Expected loop-inner .to() first, got {results[0]}")

    def test_scope_call_names_for_memcpy_hotspot(self):
        """gpu_memcpy_hotspot scope must include .to, .cuda, .copy_, pin_memory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "t.py", "def f(): pass")
            files, _ = scan_repo(root)
            scope = derive_analysis_scope("gpu_memcpy_hotspot", files)
            for name in ("to", "cuda", "copy_", "pin_memory"):
                self.assertIn(name, scope.call_names)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TaskDraft new fields: evidence_windows and search_specs
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskDraftNewFields(unittest.TestCase):

    def _run_analysis(self) -> object:
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _make_minimal_sqlite(sq)
            req = NsysAnalysisRequest(
                repo_root=tmp,
                profile_path=sq,
                sqlite_path=sq,
                include_repo_context=False,
            )
            return analyze_nsys(req)

    def test_memcpy_task_draft_has_search_specs(self):
        """gpu_memcpy_hotspot TaskDraft must have at least one search_spec."""
        diag = self._run_analysis()
        memcpy_drafts = [td for td in diag.task_drafts
                         if "memcpy" in td.id or "memcpy" in td.finding_id]
        self.assertGreater(len(memcpy_drafts), 0)
        draft = memcpy_drafts[0]
        self.assertIsInstance(draft.search_specs, list)
        self.assertGreater(len(draft.search_specs), 0,
                           "Expected at least one search_spec for memcpy finding")
        spec = draft.search_specs[0]
        self.assertIn("pattern", spec)
        self.assertIn("kind", spec)
        self.assertIn("rationale", spec)
        self.assertEqual(spec["kind"], "rg")
        # Pattern must mention memcpy-related keywords
        self.assertTrue(
            any(kw in spec["pattern"] for kw in (".to", ".cuda", "pin_memory")),
            f"Expected memcpy-related pattern, got: {spec['pattern']}"
        )

    def test_memcpy_task_draft_has_evidence_windows(self):
        """gpu_memcpy_hotspot TaskDraft must have evidence_windows from profile."""
        diag = self._run_analysis()
        memcpy_drafts = [td for td in diag.task_drafts
                         if "memcpy" in td.id or "memcpy" in td.finding_id]
        self.assertGreater(len(memcpy_drafts), 0)
        draft = memcpy_drafts[0]
        self.assertIsInstance(draft.evidence_windows, list)
        self.assertGreater(len(draft.evidence_windows), 0,
                           "Expected at least one evidence_window for memcpy finding")
        w = draft.evidence_windows[0]
        # Required keys
        for key in ("start_ms", "end_ms", "duration_ms", "event"):
            self.assertIn(key, w, f"Missing key '{key}' in evidence_window")
        # start < end
        self.assertLess(w["start_ms"], w["end_ms"])
        self.assertGreater(w["duration_ms"], 0)

    def test_evidence_window_overlap_nvtx_is_list(self):
        """overlap_nvtx in evidence_window must be a list (even if empty)."""
        diag = self._run_analysis()
        for td in diag.task_drafts:
            for w in td.evidence_windows:
                self.assertIsInstance(w.get("overlap_nvtx", []), list)

    def test_search_specs_are_entry_points_not_conclusions(self):
        """search_specs must have kind='rg' and non-empty pattern and rationale."""
        diag = self._run_analysis()
        for td in diag.task_drafts:
            for spec in td.search_specs:
                self.assertIn(spec.get("kind"), ("rg",),
                              f"Unexpected search spec kind: {spec.get('kind')}")
                self.assertTrue(spec.get("pattern", "").strip(),
                                "search_spec pattern must not be empty")
                self.assertTrue(spec.get("rationale", "").strip(),
                                "search_spec rationale must not be empty")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OptimizeTask new fields: rejected_candidates and confidence_breakdown
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptimizeTaskNewFields(unittest.TestCase):

    def test_optimize_task_has_rejected_candidates_and_breakdown(self):
        """OptimizeTask must accept rejected_candidates and confidence_breakdown (strong types)."""
        from sysight.analyzer.nsys.models import (
            OptimizeTask, TargetLocation, RejectedCandidate, ConfidenceBreakdown,
        )
        breakdown = ConfidenceBreakdown(deterministic_finding=0.85, callsite_score=0.91)
        task = OptimizeTask(
            id="memcpy_001",
            finding_id="gpu_memcpy_hotspot:a1b2c3d4",
            hypothesis="H2D memcpy at train.py:84",
            evidence_links=["ev_link_0"],
            target_files=["train.py"],
            target_locations=[
                TargetLocation(
                    file="train.py", line=84, call="batch.to(device)",
                    callsite_id="train.py:84:4:to",
                    anchor_type="callsite",
                )
            ],
            proposed_change_kind="add non_blocking=True",
            verification_metric="H2D memcpy total_ms 下降",
            confidence=breakdown.composite(),
            risk="low",
            rejected_candidates=[
                RejectedCandidate(
                    callsite_id="model.py:12:8:to",
                    file="model.py", line=12, call="model.to(device)",
                    reason="init-time only, loop_depth=0",
                )
            ],
            confidence_breakdown=breakdown,
        )
        self.assertEqual(len(task.rejected_candidates), 1)
        self.assertEqual(task.rejected_candidates[0].file, "model.py")
        self.assertEqual(task.confidence_breakdown.deterministic_finding, 0.85)  # type: ignore[union-attr]

    def test_optimize_task_defaults_empty(self):
        """rejected_candidates defaults to empty list; confidence_breakdown defaults to None."""
        from sysight.analyzer.nsys.models import OptimizeTask
        task = OptimizeTask(
            id="x",
            finding_id="y",
            hypothesis="h",
            evidence_links=[],
            target_files=[],
            target_locations=[],
            proposed_change_kind="k",
            verification_metric="m",
            confidence=0.5,
            risk="low",
        )
        self.assertEqual(task.rejected_candidates, [])
        self.assertIsNone(task.confidence_breakdown)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. stable_finding_id, ConfidenceBreakdown.composite, EvidenceLink id
# ═══════════════════════════════════════════════════════════════════════════════

class TestStableModels(unittest.TestCase):

    def test_stable_finding_id_is_deterministic(self):
        """stable_finding_id must return same value for same inputs."""
        from sysight.analyzer.nsys.models import stable_finding_id
        a = stable_finding_id("gpu_idle", "critical", (0, 1_000_000_000), None)
        b = stable_finding_id("gpu_idle", "critical", (0, 1_000_000_000), None)
        self.assertEqual(a, b)

    def test_stable_finding_id_differs_by_category(self):
        """Different categories must produce different IDs."""
        from sysight.analyzer.nsys.models import stable_finding_id
        a = stable_finding_id("gpu_idle", "critical", None, None)
        b = stable_finding_id("sync_wait", "critical", None, None)
        self.assertNotEqual(a, b)

    def test_stable_finding_id_prefix_is_category(self):
        """ID must start with category: prefix for readability."""
        from sysight.analyzer.nsys.models import stable_finding_id
        sid = stable_finding_id("gpu_comm_hotspot", "warning", None, None)
        self.assertTrue(sid.startswith("gpu_comm_hotspot:"))

    def test_findings_have_stable_ids_after_classify(self):
        """classify_bottlenecks must assign stable_id to all findings."""
        import tempfile
        from sysight.analyzer.nsys.extract import extract_trace, inspect_schema
        db_path = tempfile.mktemp(suffix=".sqlite")
        _make_minimal_sqlite(db_path)
        schema = inspect_schema(db_path)
        trace = extract_trace(db_path, schema)
        from sysight.analyzer.nsys.classify import classify_bottlenecks
        _, findings = classify_bottlenecks(trace)
        for f in findings:
            self.assertTrue(f.stable_id, f"Finding {f.category} has empty stable_id")
            self.assertTrue(f.stable_id.startswith(f.category + ":"))

    def test_confidence_breakdown_composite_bounded(self):
        """composite() must not raise deterministic_finding by more than +0.15."""
        from sysight.analyzer.nsys.models import ConfidenceBreakdown
        bd = ConfidenceBreakdown(
            deterministic_finding=0.85,
            callsite_score=1.0,
            llm_verify=1.0,
        )
        composite = bd.composite()
        self.assertLessEqual(composite, 0.85 + 0.15 + 1e-9)
        self.assertGreaterEqual(composite, 0.85)

    def test_confidence_breakdown_no_code_evidence_returns_deterministic(self):
        """If no code evidence, composite equals deterministic_finding."""
        from sysight.analyzer.nsys.models import ConfidenceBreakdown
        bd = ConfidenceBreakdown(deterministic_finding=0.75)
        self.assertAlmostEqual(bd.composite(), 0.75)

    def test_target_location_requires_anchor(self):
        """TargetLocation can be created with callsite_id anchor (no runtime enforcement yet,
        but fields must exist and be settable)."""
        from sysight.analyzer.nsys.models import TargetLocation
        loc = TargetLocation(
            file="train.py", line=84, call="batch.to(device)",
            callsite_id="train.py:84:4:to",
            anchor_type="callsite",
        )
        self.assertEqual(loc.anchor_type, "callsite")
        self.assertIsNotNone(loc.callsite_id)

    def test_evidence_window_has_event_category(self):
        """Evidence windows from _generate_task_drafts must include event_category field."""
        import sqlite3, tempfile
        db_path = tempfile.mktemp(suffix=".sqlite")
        _make_minimal_sqlite(db_path)
        req = NsysAnalysisRequest(
            repo_root=tempfile.mkdtemp(),
            profile_path=db_path,
            sqlite_path=db_path,
            include_repo_context=False,
        )
        from sysight.analyzer.nsys import analyze_nsys
        diag = analyze_nsys(req)
        for draft in diag.task_drafts:
            for w in draft.evidence_windows:
                self.assertIn("event_category", w, f"Missing event_category in window: {w}")


if __name__ == "__main__":
    unittest.main()
