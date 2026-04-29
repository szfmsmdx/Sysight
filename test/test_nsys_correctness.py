"""P0 nsys correctness tests using synthetic SQLite.

Tests:
  - trace_start_ns / trace_end_ns populated correctly
  - per-device breakdown populated for multi-GPU profiles
  - single-device profile: per_device is empty (not a regression)
  - NsysFinding has no confidence field (aligned with nsys-ai)
"""

import contextlib
import io
import json
import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path
from unittest import mock

from sysight.analyzer.nsys import analyze_nsys
from sysight.analyzer.nsys.extract import extract_trace, inspect_schema
from sysight.analyzer.nsys.localization import (
    _build_localization_prompt,
    _flush_memory,
    _memory_dir,
    _parse_localization_output,
)
from sysight.shared.memory.store import apply_memory_updates
from sysight.analyzer.nsys.models import NsysAnalysisRequest, NsysFinding


def _write_sqlite(
    path: str,
    kernels: list[tuple],
    memcpys: list[tuple] | None = None,
    gpu_infos: list[tuple] | None = None,
) -> None:
    """Write a minimal SQLite with kernel, optional memcpy, and GPU inventory data.

    Uses shortName (matching nsys real schema) so _extract_kernels JOIN works.
    kernels: list of (start, end, shortName_id, deviceId, streamId, correlationId)
    memcpys: list of (start, end, copyKind, bytes, deviceId, streamId, correlationId)
    gpu_infos: list of (id, name, totalMemory, memoryBandwidth, smCount, computeMajor, computeMinor)
    """
    with closing(sqlite3.connect(path)) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        c.execute("INSERT INTO StringIds VALUES (1, 'KernelA'), (2, 'KernelB')")
        c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, shortName INTEGER,
            deviceId INTEGER, streamId INTEGER, correlationId INTEGER
        )""")
        c.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)",
            kernels
        )
        if memcpys:
            c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER, end INTEGER, copyKind INTEGER,
                bytes INTEGER, deviceId INTEGER, streamId INTEGER, correlationId INTEGER
            )""")
            c.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)",
                memcpys
            )
        if gpu_infos:
            c.execute("""CREATE TABLE TARGET_INFO_GPU (
                id INTEGER, name TEXT, totalMemory INTEGER, memoryBandwidth INTEGER,
                smCount INTEGER, computeMajor INTEGER, computeMinor INTEGER
            )""")
            c.executemany(
                "INSERT INTO TARGET_INFO_GPU VALUES (?,?,?,?,?,?,?)",
                gpu_infos,
            )
        conn.commit()


class TestTraceStartEnd(unittest.TestCase):

    def test_trace_start_end_ns_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            _write_sqlite(sq, [
                (1_000_000, 5_000_000, 1, 0, 1, 100),
                (6_000_000, 9_000_000, 1, 0, 1, 101),
            ])
            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)

        self.assertEqual(trace.trace_start_ns, 1_000_000)
        self.assertEqual(trace.trace_end_ns, 9_000_000)
        self.assertEqual(trace.duration_ns, 8_000_000)

    def test_trace_start_zero_for_empty_kernel_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "empty.sqlite")
            with closing(sqlite3.connect(sq)) as conn:
                conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, nameId INTEGER, deviceId INTEGER, streamId INTEGER, correlationId INTEGER)")
                conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")

            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)

        self.assertEqual(trace.trace_start_ns, 0)
        self.assertEqual(trace.trace_end_ns, 0)
        self.assertEqual(trace.duration_ns, 0)


class TestPerDeviceBreakdown(unittest.TestCase):

    def test_single_device_per_device_empty(self):
        """Single device → per_device should be empty (not useful)."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            _write_sqlite(sq, [
                (0, 10_000_000, 1, 0, 1, 100),
                (11_000_000, 20_000_000, 1, 0, 1, 101),
            ])
            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertIsNotNone(diag.bottlenecks)
        self.assertEqual(diag.bottlenecks.per_device, [])

    def test_multi_device_per_device_populated(self):
        """Two devices → per_device should have two entries."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            _write_sqlite(sq, [
                (0,          10_000_000, 1, 0, 1, 100),  # device 0
                (11_000_000, 20_000_000, 1, 0, 1, 101),  # device 0
                (0,          8_000_000,  2, 1, 1, 200),  # device 1
                (9_000_000,  18_000_000, 2, 1, 1, 201),  # device 1
            ])
            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertIsNotNone(diag.bottlenecks)
        per_device = diag.bottlenecks.per_device
        self.assertEqual(len(per_device), 2)
        device_ids = {bd.device_id for bd in per_device}
        self.assertEqual(device_ids, {0, 1})

    def test_multi_device_active_ns_per_device(self):
        """Each device's active_ns should reflect only that device's kernels."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            # Device 0: 10ms, Device 1: 5ms (non-overlapping)
            _write_sqlite(sq, [
                (0,         10_000_000, 1, 0, 1, 100),  # device 0: 10ms
                (0,          5_000_000, 2, 1, 1, 200),  # device 1: 5ms
            ])
            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        per_device = {bd.device_id: bd for bd in diag.bottlenecks.per_device}
        self.assertAlmostEqual(per_device[0].active_ns, 10_000_000)
        self.assertAlmostEqual(per_device[1].active_ns, 5_000_000)


class TestNsysFindingNoConfidence(unittest.TestCase):
    """NsysFinding must NOT have a confidence field (aligned with nsys-ai)."""

    def test_finding_has_no_confidence(self):
        f = NsysFinding(
            category="gpu_idle",
            severity="critical",
            title="GPU idle",
            description="GPU was idle",
        )
        self.assertFalse(hasattr(f, "confidence"),
                         "NsysFinding should not have confidence field")

    def test_diag_findings_have_no_confidence(self):
        """All findings from a real analysis should have no confidence field."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            _write_sqlite(sq, [
                (0, 10_000_000, 1, 0, 1, 100),
            ])
            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        for f in diag.findings:
            self.assertFalse(hasattr(f, "confidence"),
                             f"Finding {f.category} should not have confidence field")


class TestGpuInventoryExtraction(unittest.TestCase):

    def test_schema_extracts_gpu_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "gpu.sqlite")
            _write_sqlite(
                sq,
                [(0, 10_000_000, 1, 0, 1, 100)],
                gpu_infos=[
                    (0, "NVIDIA A100-SXM4-80GB", 84_987_740_160, 2_039_040_000_000, 108, 8, 0),
                    (1, "NVIDIA A100-SXM4-80GB", 84_987_740_160, 2_039_040_000_000, 108, 8, 0),
                ],
            )
            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertEqual(diag.status, "ok")
        self.assertEqual(len(diag.gpu_devices), 2)
        self.assertEqual(diag.gpu_devices[0].name, "NVIDIA A100-SXM4-80GB")
        self.assertEqual(diag.gpu_devices[0].sm_count, 108)
        self.assertEqual(diag.gpu_devices[0].compute_capability, "8.0")


class TestAnalyzePipelineInput(unittest.TestCase):

    def test_sqlite_only_request_uses_sqlite_as_profile_path(self):
        """sqlite-only requests should not report '.' as profile_path."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _write_sqlite(sq, [
                (0, 10_000_000, 1, 0, 1, 100),
            ])
            req = NsysAnalysisRequest(
                profile_path="",
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertEqual(diag.status, "ok")
        self.assertEqual(diag.profile_path, sq)
        self.assertEqual(diag.sqlite_path, sq)

    def test_schema_warnings_preserved_on_success(self):
        """Successful analysis should keep schema inspection warnings."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "warn.sqlite")
            _write_sqlite(sq, [
                (0, 10_000_000, 1, 0, 1, 100),
            ])
            with closing(sqlite3.connect(sq)) as conn:
                conn.execute("CREATE TABLE MYSTERY_TABLE (id INTEGER)")

            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertEqual(diag.status, "ok")
        self.assertTrue(
            any("发现未知表" in warning for warning in diag.warnings),
            f"Expected schema warning in diag.warnings, got: {diag.warnings}",
        )


class TestMemoryWriteback(unittest.TestCase):

    def test_parse_localization_output_supports_memory_updates(self):
        text = json.dumps(
            {
                "summary": "done",
                "memory_updates": [
                    {"path": "workspace", "content": "## 基本配置\nactive_config=a", "append": True},
                    {"path": "experience", "content": "## 新经验\n- 场景：x\n- 规则：y\n---", "append": True},
                ],
            },
            ensure_ascii=False,
        )
        summary, questions, anchors, workspace_mem, experience_mem, memory_updates = _parse_localization_output(text)

        self.assertEqual(summary, "done")
        self.assertEqual(questions, [])
        self.assertEqual(anchors, [])
        self.assertIsNone(workspace_mem)
        self.assertIsNone(experience_mem)
        self.assertEqual(len(memory_updates), 2)
        self.assertEqual(memory_updates[0]["path"], "workspace")
        self.assertEqual(memory_updates[1]["path"], "experience")

    def test_flush_memory_migrates_legacy_files_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_dir = root / "sysight" / "memory"
            runtime_dir = root / ".sysight" / "memory"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "workspace.md").write_text("legacy workspace", encoding="utf-8")
            (legacy_dir / "experience.md").write_text("legacy experience", encoding="utf-8")
            with mock.patch("sysight.analyzer.nsys.localization._memory_dir", return_value=runtime_dir):
                _flush_memory(None, None, [])

            self.assertEqual((runtime_dir / "workspace.md").read_text(encoding="utf-8"), "legacy workspace")
            self.assertEqual((runtime_dir / "experience.md").read_text(encoding="utf-8"), "legacy experience")

    def test_flush_memory_dedups_same_experience_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_dir = Path(tmp) / ".sysight" / "memory"
            runtime_dir.mkdir(parents=True)
            update = [{"path": "experience", "content": "## 新经验\n- 场景：x\n- 规则：y\n---", "append": True}]
            with mock.patch("sysight.analyzer.nsys.localization._memory_dir", return_value=runtime_dir):
                _flush_memory(None, None, update)
                _flush_memory(None, None, update)

            text = (runtime_dir / "experience.md").read_text(encoding="utf-8")
            self.assertEqual(text.count("## 新经验"), 1)


class TestLocalizationPrompt(unittest.TestCase):

    def test_memory_root_uses_runtime_sysight_dir(self):
        self.assertEqual(_memory_dir().name, "memory")
        self.assertEqual(_memory_dir().parent.name, ".sysight")
        self.assertNotEqual(_memory_dir().parent.name, "sysight")

    def test_prompt_does_not_inline_memory_files_or_paths(self):
        req = NsysAnalysisRequest(
            profile_path="trace.sqlite",
            sqlite_path="trace.sqlite",
            repo_root="/tmp/repo",
        )
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            workspace_path = memory_dir / "workspace.md"
            experience_path = memory_dir / "experience.md"
            workspace_path.write_text("WORKSPACE_SENTINEL", encoding="utf-8")
            experience_path.write_text("EXPERIENCE_SENTINEL", encoding="utf-8")
            with mock.patch("sysight.analyzer.nsys.localization._memory_dir", return_value=memory_dir):
                prompt = _build_localization_prompt(
                    req,
                    summary="summary",
                    findings=[],
                    windows=[],
                    sqlite_path="trace.sqlite",
                )

        self.assertNotIn("WORKSPACE_SENTINEL", prompt)
        self.assertNotIn("EXPERIENCE_SENTINEL", prompt)
        self.assertNotIn(str(workspace_path), prompt)
        self.assertNotIn(str(experience_path), prompt)
        self.assertNotIn("workspace.md（当前 workspace 专属", prompt)
        self.assertNotIn("experience.md（通用经验", prompt)

    def test_prompt_limits_questions_and_context_volume(self):
        from sysight.analyzer.nsys.models import EvidenceWindow

        findings = [
            NsysFinding(
                category="sync_wait",
                severity="warning",
                title=f"sync {idx}",
                description="sync",
                stable_id=f"sync_wait:{idx}",
            )
            for idx in range(10)
        ]
        windows = [
            EvidenceWindow(
                problem_id=f"sync_wait:{idx}",
                category="sync_wait",
                start_ns=idx,
                end_ns=idx + 1,
                duration_ns=1,
                device_id=None,
                stream_id=None,
                correlation_id=None,
                event_name="cuda_sync",
                event_category="sync_wait",
                nvtx_labels=[f"iter_{idx}"],
            )
            for idx in range(10)
        ]
        req = NsysAnalysisRequest(
            profile_path="trace.sqlite",
            sqlite_path="trace.sqlite",
            repo_root="/tmp/repo",
        )

        prompt = _build_localization_prompt(
            req,
            summary="summary",
            findings=findings,
            windows=windows,
            sqlite_path="trace.sqlite",
        )

        # New prompt format: TASK.txt template, no pre-generated Q-items
        self.assertIn("python3 -m sysight.analyzer.cli nsys-sql", prompt)
        self.assertIn("输出格式：", prompt)
        self.assertIn("/tmp/repo", prompt)
        self.assertIn("trace.sqlite", prompt)
        self.assertIn("python3 -m sysight.analyzer.cli memory search", prompt)
        self.assertIn("python3 -m sysight.analyzer.cli memory read", prompt)
        self.assertIn("python3 -m sysight.analyzer.cli memory write", prompt)
        # Old prompt artifacts should NOT be present
        self.assertNotIn("Q1. problem_id=", prompt)
        self.assertNotIn("待回答问题：", prompt)
        self.assertNotIn("Analyzer harness（必须先阅读并遵守）：", prompt)

    def test_prompt_includes_memory_brief_and_namespace(self):
        req = NsysAnalysisRequest(
            profile_path="trace.sqlite",
            sqlite_path="trace.sqlite",
            repo_root="/tmp/repo",
            memory_namespace="bench/case_prompt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / ".sysight" / "memory"
            apply_memory_updates(
                str(memory_dir),
                [{"path": "experience", "title": "Loop", "content": "## Loop\n- 规则：批处理\n---", "append": True, "category": "C2"}],
                repo_root="/tmp/repo",
                namespace="bench/case_prompt",
                raw_run={"run_id": "run-prompt", "manifest_path": "raw/runs/run-prompt/manifest.json"},
            )
            with mock.patch("sysight.analyzer.nsys.localization._memory_dir", return_value=memory_dir):
                prompt = _build_localization_prompt(
                    req,
                    summary="summary",
                    findings=[],
                    windows=[],
                    sqlite_path="trace.sqlite",
                )

        self.assertIn("bench/case_prompt", prompt)
        self.assertIn("raw/runs/run-prompt/manifest.json", prompt)
        # MEMORY_BRIEF is replaced with actual brief content (namespace, paths, etc.)
        # so we check for characteristic brief content, not the literal placeholder
        self.assertIn("namespace: bench/case_prompt", prompt)
        self.assertIn("查询工具", prompt)

    def test_prompt_includes_precomputed_context(self):
        from sysight.analyzer.nsys.models import (
            BottleneckSummary,
            BottleneckLabel,
            EventStat,
            SampleHotspot,
            SourceFrame,
        )

        req = NsysAnalysisRequest(
            profile_path="trace.sqlite",
            sqlite_path="trace.sqlite",
            repo_root="/tmp/repo",
        )
        findings = [
            NsysFinding(
                category="sql_sync_cost",
                severity="warning",
                title="sync",
                description="sync",
                evidence=["STREAM_WAIT_EVENT: 10 count 1.0ms"],
                stable_id="sql_sync_cost:1",
            )
        ]
        windows = []
        bottlenecks = BottleneckSummary(
            total_ns=100,
            gpu_active_ns=90,
            gpu_idle_ns=10,
            labels=[
                BottleneckLabel(
                    category="gpu_compute",
                    active_ns=90,
                    inclusive_ns=120,
                    pct_of_trace=0.9,
                    pct_of_gpu_active=1.0,
                    evidence=[],
                )
            ],
            top_events=[
                EventStat(
                    name="KernelA",
                    category="gpu_compute",
                    count=3,
                    total_ns=30,
                    max_ns=15,
                    avg_ns=10.0,
                    inclusive_pct=0.3,
                )
            ],
        )
        hotspots = [
            SampleHotspot(
                frame=SourceFrame(symbol="cudaStreamSynchronize", source_file=None, source_line=None),
                count=12,
                pct=0.12,
                coarse_location="Torch 张量拷贝/同步路径",
            )
        ]

        prompt = _build_localization_prompt(
            req,
            summary="summary",
            findings=findings,
            windows=windows,
            sqlite_path="trace.sqlite",
            bottleneck_summary=bottlenecks,
            hotspots=hotspots,
        )

        # New prompt format: TASK.txt template, raw stats come from profile report
        self.assertIn("python3 -m sysight.analyzer.cli nsys-sql", prompt)
        self.assertIn("输出格式：", prompt)
        # Old pre-computed stats block should NOT be present
        self.assertNotIn("=== 预计算统计", prompt)


class TestStage6Localization(unittest.TestCase):

    def test_stage6_requires_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _write_sqlite(sq, [
                (0, 10_000_000, 1, 0, 1, 100),
            ])
            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
                run_localization=True,
                localization_backend="codex",
            )
            diag = analyze_nsys(req)

        self.assertEqual(diag.status, "action_required")
        self.assertIn("repo_root", diag.required_action or "")

    def test_stage6_codex_sync_returns_ok_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "train.py").write_text("def train():\n    return 1\n", encoding="utf-8")
            sq = str(Path(tmp) / "trace.sqlite")
            _write_sqlite(sq, [
                (0, 10_000_000, 1, 0, 1, 100),
            ])

            class _FakePopen:
                def __init__(self, command, stdin=None, stdout=None, stderr=None, text=None, env=None, start_new_session=None):
                    out_idx = command.index("--output-last-message") + 1
                    self.output_path = command[out_idx]
                    self.command = command
                    self.pid = 54321
                    self.returncode = 0

                def communicate(self, prompt_text=None):
                    Path(self.output_path).write_text(
                        '{"summary":"定位到训练主循环","questions":[{"question_id":"Q1","problem_id":"gpu_idle","category":"gpu_idle","title":"GPU 空闲占比高","status":"mapped","file_path":"train.py","line":1,"function":"train","rationale":"入口函数 train 触发该问题对应的主要调用","suggestion":"先检查 train 函数中的训练主循环","window_ids":["W1"]}],"anchors":[{"window_id":"W1","problem_id":"gpu_idle","category":"gpu_idle","event_name":"KernelA","status":"mapped","file_path":"train.py","line":1,"function":"train","rationale":"入口函数 train 触发该窗口附近的主要调用","suggestion":"先检查 train 函数中的训练主循环"}]}',
                        encoding="utf-8",
                    )
                    return ("", "")

            err = io.StringIO()
            with mock.patch("sysight.analyzer.nsys.localization.subprocess.Popen", side_effect=_FakePopen):
                with contextlib.redirect_stderr(err):
                    req = NsysAnalysisRequest(
                        profile_path=sq,
                        sqlite_path=sq,
                        repo_root=str(root),
                        run_localization=True,
                        localization_backend="codex",
                        emit_progress_info=True,
                    )
                    diag = analyze_nsys(req)

        self.assertEqual(diag.status, "ok")
        self.assertIsNotNone(diag.localization)
        self.assertEqual(diag.localization.backend, "codex")
        self.assertEqual(diag.localization.status, "ok")
        self.assertTrue(diag.localization.output_path)
        self.assertIn("定位到训练主循环", diag.localization.output)
        self.assertEqual(diag.localization.summary, "定位到训练主循环")
        self.assertEqual(len(diag.localization.questions), 1)
        self.assertEqual(diag.localization.questions[0].question_id, "Q1")
        self.assertEqual(diag.localization.questions[0].file_path, "train.py")
        self.assertEqual(len(diag.localization.anchors), 1)
        self.assertEqual(diag.localization.anchors[0].file_path, "train.py")
        self.assertTrue(diag.localization.artifact_dir)
        self.assertTrue(diag.localization.prompt_path)
        self.assertTrue(diag.localization.stdout_path)
        self.assertTrue(diag.localization.stderr_path)
        self.assertIn("codex 子进程启动", err.getvalue())
        self.assertIn("codex 调查进行中", err.getvalue())
        self.assertIn("Codex 调查结果已回填", err.getvalue())
        # New prompt format: TASK.txt template
        self.assertIn("python3 -m sysight.analyzer.cli nsys-sql", diag.localization.prompt)
        self.assertIn("输出格式：", diag.localization.prompt)
        # Old prompt artifacts should NOT be present
        self.assertNotIn("待回答问题：", diag.localization.prompt)
        self.assertNotIn("Q1. problem_id=", diag.localization.prompt)
        self.assertNotIn("Analyzer harness（必须先阅读并遵守）：", diag.localization.prompt)
        self.assertIn("--cd", diag.localization.command)
        self.assertIn("workspace-write", diag.localization.command)


class TestStage4AndStage7(unittest.TestCase):

    def test_windows_rank_sync_events_within_iter(self):
        from sysight.analyzer.nsys.models import NsysFinding, NsysTrace, TimelineEvent
        from sysight.analyzer.nsys.windows import extract_evidence_windows

        trace = NsysTrace(
            tool="nsys",
            profile_path="trace.sqlite",
            sqlite_path="trace.sqlite",
            duration_ns=100,
            trace_start_ns=0,
            trace_end_ns=100,
            events=[
                TimelineEvent(category="nvtx", name="iter_7", start_ns=0, dur_ns=100),
                TimelineEvent(category="sync_wait", name="cudaStreamSynchronize", start_ns=10, dur_ns=5),
                TimelineEvent(category="sync_wait", name="cudaStreamSynchronize", start_ns=20, dur_ns=5),
            ],
        )
        finding = NsysFinding(
            category="sync_wait",
            severity="warning",
            title="sync wait",
            description="sync dominates",
            stable_id="sync_wait:test",
        )

        windows = extract_evidence_windows(trace, [finding], top_n=2)

        by_start = {window.start_ns: window for window in windows}
        self.assertEqual(by_start[10].window_rank_in_iter, 1)
        self.assertEqual(by_start[20].window_rank_in_iter, 2)

    def test_diag_includes_windows_and_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "trace.sqlite")
            _write_sqlite(sq, [
                (0, 10_000_000, 1, 0, 1, 100),
                (11_000_000, 20_000_000, 1, 0, 1, 101),
            ])
            req = NsysAnalysisRequest(
                profile_path=sq,
                sqlite_path=sq,
            )
            diag = analyze_nsys(req)

        self.assertEqual(diag.status, "ok")
        self.assertGreater(len(diag.windows), 0)
        self.assertFalse(hasattr(diag, "tasks"))


class TestCallstackExtraction(unittest.TestCase):
    """Tests for full callstack extraction in CPU samples."""

    def _write_sqlite_with_callstack(self, path: str) -> None:
        """Build a SQLite with CPU sampling data that has multiple stack frames."""
        with closing(sqlite3.connect(path)) as conn:
            c = conn.cursor()
            c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            c.executemany("INSERT INTO StringIds VALUES (?,?)", [
                (1, "KernelA"),
                (10, "_PyEval_EvalFrameDefault"),
                (11, "train_step"),
                (12, "main_loop"),
            ])
            c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, shortName INTEGER,
                deviceId INTEGER, streamId INTEGER, correlationId INTEGER
            )""")
            c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 10000000, 1, 0, 1, 100)")

            c.execute("""CREATE TABLE COMPOSITE_EVENTS (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                globalTid INTEGER
            )""")
            c.execute("INSERT INTO COMPOSITE_EVENTS VALUES (1, 5000000, 42)")

            c.execute("""CREATE TABLE SAMPLING_CALLCHAINS (
                id INTEGER,
                stackDepth INTEGER,
                symbol INTEGER
            )""")
            c.executemany("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,?,?)", [
                (1, 0, 10),
                (1, 1, 11),
                (1, 2, 12),
            ])
            conn.commit()

    def test_cpu_sample_has_full_callstack_in_extra(self):
        """CPU sample events must have extra['callstack'] with all frames."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            self._write_sqlite_with_callstack(sq)
            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)

        cpu_samples = [ev for ev in trace.events if ev.is_sample]
        self.assertGreater(len(cpu_samples), 0, "Expected at least one CPU sample")

        sample = cpu_samples[0]
        # Name should be the leaf frame
        self.assertEqual(sample.name, "_PyEval_EvalFrameDefault")
        # Callstack must have all 3 frames
        callstack = sample.extra.get("callstack", [])
        self.assertEqual(len(callstack), 3,
                         f"Expected 3 stack frames, got: {callstack}")
        self.assertIn("_PyEval_EvalFrameDefault", callstack)
        self.assertIn("train_step", callstack)
        self.assertIn("main_loop", callstack)

    def test_leaf_frame_is_name(self):
        """Leaf frame (stackDepth=0) must be used as event.name."""
        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            self._write_sqlite_with_callstack(sq)
            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)

        cpu_samples = [ev for ev in trace.events if ev.is_sample]
        self.assertGreater(len(cpu_samples), 0)
        # Leaf frame is depth 0: _PyEval_EvalFrameDefault
        self.assertEqual(cpu_samples[0].name, "_PyEval_EvalFrameDefault")

    def test_cpu_hotspots_prefer_complete_user_callstacks(self):
        """One-frame unresolved samples should not dominate CPU hotspot output."""
        from sysight.analyzer.nsys import _build_cpu_hotspots

        with tempfile.TemporaryDirectory() as tmp:
            sq = str(Path(tmp) / "t.sqlite")
            with closing(sqlite3.connect(sq)) as conn:
                c = conn.cursor()
                c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
                c.executemany("INSERT INTO StringIds VALUES (?,?)", [
                    (1, "KernelA"),
                    (10, "0xffffffffb03f9078"),
                    (11, "cudaStreamSynchronize"),
                    (12, "c10::cuda::memcpy_and_sync(void*, void const*, long, cudaMemcpyKind, CUstream_st*)"),
                    (13, "train_step"),
                    (14, "main_loop"),
                ])
                c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER, end INTEGER, shortName INTEGER,
                    deviceId INTEGER, streamId INTEGER, correlationId INTEGER
                )""")
                c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 10000000, 1, 0, 1, 100)")
                c.execute("""CREATE TABLE COMPOSITE_EVENTS (
                    id INTEGER PRIMARY KEY,
                    start INTEGER,
                    globalTid INTEGER
                )""")
                c.execute("""CREATE TABLE SAMPLING_CALLCHAINS (
                    id INTEGER,
                    stackDepth INTEGER,
                    symbol INTEGER
                )""")
                for sample_id in range(1, 6):
                    c.execute("INSERT INTO COMPOSITE_EVENTS VALUES (?,?,?)", (sample_id, sample_id * 1000, 42))
                    c.execute("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,?,?)", (sample_id, 0, 10))
                for sample_id in range(6, 9):
                    c.execute("INSERT INTO COMPOSITE_EVENTS VALUES (?,?,?)", (sample_id, sample_id * 1000, 42))
                    c.executemany("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,?,?)", [
                        (sample_id, 0, 11),
                        (sample_id, 1, 12),
                        (sample_id, 2, 13),
                        (sample_id, 3, 14),
                    ])
                conn.commit()

            schema = inspect_schema(sq)
            trace = extract_trace(sq, schema)
            hotspots = _build_cpu_hotspots(trace, top_n=3)

        self.assertGreater(len(hotspots), 0)
        self.assertNotIn("0xffffffff", hotspots[0].frame.raw or "")
        self.assertIn("cudaStreamSynchronize", hotspots[0].frame.raw or "")
        self.assertIn("train_step", hotspots[0].frame.raw or "")
        self.assertGreaterEqual(len(hotspots[0].callstack), 2)
        self.assertNotIn("__clock_gettime", hotspots[0].frame.raw or "")


if __name__ == "__main__":
    unittest.main()
