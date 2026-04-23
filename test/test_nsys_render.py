"""Terminal rendering tests for nsys reports."""

import unittest

from sysight.analyzer.nsys.render import _group_findings, _kv_block, _render_cpu_hotspots, render_nsys_terminal
from sysight.analyzer.nsys.text import display_width, format_table


def _marker_offset(line: str, marker: str) -> int:
    return display_width(line[:line.index(marker)])


class TestTerminalTables(unittest.TestCase):

    def test_table_columns_align_with_cjk_headers_and_cells(self) -> None:
        lines = format_table(
            ["内核名称", "次数", "总(ms)"],
            [
                ["nccl", "440", "4808.52"],
                ["注意力后向", "528", "2536.64"],
            ],
            col_widths=[12, 6, 8],
        )

        second_col_offsets = [
            _marker_offset(lines[0], "次数"),
            _marker_offset(lines[2], "440"),
            _marker_offset(lines[3], "528"),
        ]
        self.assertEqual(second_col_offsets, [14, 14, 14])

    def test_table_clips_by_display_width(self) -> None:
        lines = format_table(
            ["名称", "次数"],
            [["超长中文内核名称", "7"]],
            col_widths=[8, 4],
        )

        self.assertEqual(_marker_offset(lines[2], "7"), 10)
        self.assertEqual(display_width(lines[2]), 14)

    def test_key_value_block_aligns_cjk_and_ascii_keys(self) -> None:
        lines = _kv_block(
            [
                ("GPU 内核", "A"),
                ("GPU memcpy", "B"),
                ("代码仓库映射", "C"),
            ],
            key_width=16,
        )

        offsets = [
            _marker_offset(lines[0], "A"),
            _marker_offset(lines[1], "B"),
            _marker_offset(lines[2], "C"),
        ]
        self.assertEqual(offsets, [20, 20, 20])


class TestFindingDomainGrouping(unittest.TestCase):
    """SQL findings should be grouped into useful report domains."""

    def test_global_sql_findings_are_triage_not_other(self):
        from sysight.analyzer.nsys.models import NsysFinding

        groups = _group_findings([
            NsysFinding(
                category="sql_profile_health",
                severity="warning",
                title="health",
                description="",
            ),
            NsysFinding(
                category="sql_root_cause_analysis",
                severity="critical",
                title="root cause",
                description="",
            ),
        ])

        triage_primary, triage_aux = groups["triage"]
        other_primary, other_sql = groups["other"]
        self.assertEqual([f.category for f in triage_primary], ["sql_root_cause_analysis"])
        self.assertEqual([f.category for f in triage_aux], ["sql_profile_health"])
        self.assertEqual(other_primary, [])
        self.assertEqual(other_sql, [])

    def test_host_sql_hotspot_becomes_aux_when_host_primary_exists(self):
        from sysight.analyzer.nsys.models import NsysFinding

        groups = _group_findings([
            NsysFinding(
                category="host_gil_contention",
                severity="warning",
                title="gil",
                description="",
            ),
            NsysFinding(
                category="sql_nvtx_hotspots",
                severity="info",
                title="nvtx hot",
                description="",
            ),
        ])

        host_primary, host_aux = groups["host"]
        self.assertEqual([f.category for f in host_primary], ["host_gil_contention"])
        self.assertEqual([f.category for f in host_aux], ["sql_nvtx_hotspots"])

    def test_nvtx_layer_and_gil_findings_have_specific_domains(self):
        from sysight.analyzer.nsys.models import NsysFinding

        groups = _group_findings([
            NsysFinding(
                category="sql_nvtx_layer_breakdown",
                severity="warning",
                title="nvtx layer",
                description="",
            ),
            NsysFinding(
                category="host_gil_contention",
                severity="warning",
                title="gil",
                description="",
            ),
        ])

        compute_primary, _ = groups["compute"]
        host_primary, _ = groups["host"]
        other_primary, _ = groups["other"]
        self.assertEqual([f.category for f in compute_primary], ["sql_nvtx_layer_breakdown"])
        self.assertEqual([f.category for f in host_primary], ["host_gil_contention"])
        self.assertEqual(other_primary, [])


class TestStage4AndStage7Rendering(unittest.TestCase):

    def test_render_includes_window_and_task_sections(self):
        from sysight.analyzer.nsys.models import (
            EvidenceWindow,
            InvestigationAnchor,
            InvestigationQuestion,
            InvestigationResult,
            NsysDiag,
            NsysFinding,
        )

        window = EvidenceWindow(
            problem_id="gpu_compute:1",
            category="gpu_compute_hotspot",
            start_ns=0,
            end_ns=10_000,
            duration_ns=10_000,
            device_id=0,
            stream_id=1,
            correlation_id=10,
            event_name="KernelA",
            event_category="gpu_compute",
            runtime_api="cudaLaunchKernel_v7000",
            callstack_summaries=["train_step <- forward <- loss"],
            coarse_location="Python 函数 `train_step` 触发 CUDA kernel launch",
        )
        diag = NsysDiag(
            status="ok",
            profile_path="test.sqlite",
            sqlite_path="test.sqlite",
            required_action=None,
            bottlenecks=None,
            findings=[
                NsysFinding(
                    category="gpu_compute_hotspot",
                    severity="critical",
                    title="GPU compute hotspot",
                    description="compute dominates",
                    stable_id="gpu_compute:1",
                )
            ],
            hotspots=[],
            warnings=[],
            summary="summary",
            windows=[window],
            investigation=InvestigationResult(
                backend="codex",
                status="ok",
                prompt="prompt",
                output='{"summary":"定位到训练主循环","questions":[{"question_id":"Q1","problem_id":"gpu_compute:1","category":"gpu_compute_hotspot","title":"GPU compute hotspot","status":"mapped","file_path":"trainer.py","line":42,"function":"train_step","rationale":"CUDA kernel launch 来自 train_step","suggestion":"优先检查该函数内的 kernel 发射路径","window_ids":["W1"]}],"anchors":[{"window_id":"W1","problem_id":"gpu_compute:1","category":"gpu_compute_hotspot","event_name":"KernelA","status":"mapped","file_path":"trainer.py","line":42,"function":"train_step","rationale":"CUDA kernel launch 来自 train_step","suggestion":"优先检查该函数内的 kernel 发射路径"}]}',
                command=["codex", "exec"],
                summary="定位到训练主循环",
                questions=[
                    InvestigationQuestion(
                        question_id="Q1",
                        problem_id="gpu_compute:1",
                        category="gpu_compute_hotspot",
                        title="GPU compute hotspot",
                        file_path="trainer.py",
                        line=42,
                        function="train_step",
                        rationale="CUDA kernel launch 来自 train_step",
                        suggestion="优先检查该函数内的 kernel 发射路径",
                        status="mapped",
                        window_ids=["W1"],
                    )
                ],
                anchors=[
                    InvestigationAnchor(
                        window_id="W1",
                        problem_id="gpu_compute:1",
                        category="gpu_compute_hotspot",
                        event_name="KernelA",
                        file_path="trainer.py",
                        line=42,
                        function="train_step",
                        rationale="CUDA kernel launch 来自 train_step",
                        suggestion="优先检查该函数内的 kernel 发射路径",
                        status="mapped",
                    )
                ],
                artifact_dir="/tmp/codex-run",
                prompt_path="/tmp/codex-run/prompt.txt",
                stdout_path="/tmp/codex-run/stdout.txt",
                stderr_path="/tmp/codex-run/stderr.txt",
            ),
        )

        compact = render_nsys_terminal(diag)
        self.assertNotIn("粗定位窗口", compact)
        self.assertNotIn("Stage 4", compact)
        self.assertIn("Codex 调查结果", compact)
        self.assertNotIn("结构化结果（按问题回填）", compact)
        self.assertIn("Codex 已回填到下方优化建议", compact)
        self.assertIn("优化建议", compact)
        self.assertIn("Q1/P0", compact)
        self.assertIn("细定位", compact)
        self.assertIn("trainer.py:42:train_step", compact)
        self.assertIn("原因", compact)
        self.assertIn("动作", compact)
        self.assertNotIn("原始输出", compact)
        self.assertNotIn("采样栈1", compact)
        self.assertNotIn("调用栈1", compact)

        verbose = render_nsys_terminal(diag, verbose=True)
        self.assertNotIn("采样栈1", verbose)
        stage7 = verbose.split("优化建议", 1)[1]
        self.assertIn("KernelA@0.000ms", stage7)
        self.assertIn("trainer.py:42:train_step", stage7)
        self.assertEqual(stage7.count("trainer.py:42:train_step"), 1)

    def test_render_shows_window_identity_rank(self):
        from sysight.analyzer.nsys.models import EvidenceWindow, InvestigationQuestion, InvestigationResult, NsysDiag, NsysFinding

        window = EvidenceWindow(
            problem_id="sync_wait:test-rank",
            category="sync_wait",
            start_ns=10_000,
            end_ns=15_000,
            duration_ns=5_000,
            device_id=0,
            stream_id=1,
            correlation_id=10,
            event_name="cudaStreamSynchronize",
            event_category="sync_wait",
            runtime_api="cudaStreamSynchronize",
            nvtx_labels=["iter_7"],
            window_rank_in_iter=2,
        )
        diag = NsysDiag(
            status="ok",
            profile_path="test.sqlite",
            sqlite_path="test.sqlite",
            required_action=None,
            bottlenecks=None,
            findings=[
                NsysFinding(
                    category="sync_wait",
                    severity="warning",
                    title="sync wait",
                    description="sync dominates",
                    stable_id="sync_wait:test-rank",
                )
            ],
            hotspots=[],
            warnings=[],
            summary="summary",
            windows=[window],
            investigation=InvestigationResult(
                backend="codex",
                status="ok",
                prompt="prompt",
                questions=[
                    InvestigationQuestion(
                        question_id="Q1",
                        problem_id="sync_wait:test-rank",
                        category="sync_wait",
                        title="sync wait",
                        file_path="trainer.py",
                        line=24,
                        function="train_step",
                        rationale="同步点位于训练主循环内",
                        suggestion="优先检查该同步前是否存在可提前发射的工作",
                        status="mapped",
                        window_ids=["W1"],
                    )
                ],
            ),
        )

        output = render_nsys_terminal(diag)
        self.assertIn("iter_7 中第 2 个同步", output)

class TestCallstackReadability(unittest.TestCase):
    """Tests for callstack readability and coarse location extraction.

    These tests ensure that raw syscall/runtime wrappers are not shipped as
    the final coarse location. See AGENTS.md for bad case examples.
    """

    def test_focus_prefers_watchdog_and_gil_context(self):
        """NCCL watchdog with GIL context must produce readable location."""
        from sysight.analyzer.nsys.models import SourceFrame
        from sysight.analyzer.nsys.stacks import summarize_callstack_focus

        focus = summarize_callstack_focus(
            [
                SourceFrame(symbol="pthread_cond_timedwait", source_file=None, source_line=None),
                SourceFrame(symbol="PyEval_RestoreThread", source_file=None, source_line=None),
                SourceFrame(symbol="PyGILState_Ensure", source_file=None, source_line=None),
                SourceFrame(symbol="c10d::ProcessGroupNCCL::watchdogHandler()", source_file=None, source_line=None),
                SourceFrame(symbol="c10d::ProcessGroupNCCL::ncclCommWatchdog()", source_file=None, source_line=None),
            ],
            nvtx_labels=["iter_110", "Holding GIL", "Waiting for GIL"],
        )

        self.assertIsNotNone(focus)
        self.assertIn("NCCL watchdog", focus)
        self.assertIn("GIL", focus)

    def test_gil_wait_without_user_python(self):
        """GIL wait must not show raw PyEval_RestoreThread as primary location."""
        from sysight.analyzer.nsys.models import SourceFrame
        from sysight.analyzer.nsys.stacks import summarize_callstack_focus

        focus = summarize_callstack_focus(
            [
                SourceFrame(symbol="pthread_cond_timedwait", source_file=None, source_line=None),
                SourceFrame(symbol="PyEval_RestoreThread", source_file=None, source_line=None),
                SourceFrame(symbol="PyGILState_Ensure", source_file=None, source_line=None),
            ],
        )
        self.assertIsNotNone(focus)
        # Must NOT be raw syscall name
        self.assertNotIn("pthread_cond_timedwait", focus)
        self.assertIn("GIL", focus)

    def test_cuda_kernel_launch_with_python_entry(self):
        """CUDA kernel launch must point to Python caller."""
        from sysight.analyzer.nsys.models import SourceFrame
        from sysight.analyzer.nsys.stacks import summarize_callstack_focus

        focus = summarize_callstack_focus(
            [
                SourceFrame(symbol="cudaLaunchKernel", source_file=None, source_line=None),
                SourceFrame(symbol="_PyEval_EvalFrameDefault", source_file=None, source_line=None),
                SourceFrame(symbol="train_step", source_file=None, source_line=None, module="libpython3.11.so.1.0"),
            ],
        )
        self.assertIsNotNone(focus)
        self.assertIn("train_step", focus)
        self.assertIn("CUDA kernel launch", focus)

    def test_tensor_copy_with_python_entry(self):
        """Tensor copy must point to Python caller."""
        from sysight.analyzer.nsys.models import SourceFrame
        from sysight.analyzer.nsys.stacks import summarize_callstack_focus

        focus = summarize_callstack_focus(
            [
                SourceFrame(symbol="cudaMemcpyAsync", source_file=None, source_line=None),
                SourceFrame(symbol="copy_impl", source_file=None, source_line=None),
                SourceFrame(symbol="forward", source_file=None, source_line=None, module="libpython3.11.so.1.0"),
            ],
        )
        self.assertIsNotNone(focus)
        self.assertIn("forward", focus)
        self.assertIn("拷贝", focus)

    def test_nccl_watchdog_without_gil(self):
        """NCCL watchdog without GIL must show communication wait hint."""
        from sysight.analyzer.nsys.models import SourceFrame
        from sysight.analyzer.nsys.stacks import summarize_callstack_focus

        focus = summarize_callstack_focus(
            [
                SourceFrame(symbol="pthread_cond_timedwait", source_file=None, source_line=None),
                SourceFrame(symbol="c10d::ProcessGroupNCCL::watchdogHandler()", source_file=None, source_line=None),
                SourceFrame(symbol="c10d::ProcessGroupNCCL::ncclCommWatchdog()", source_file=None, source_line=None),
            ],
        )
        self.assertIsNotNone(focus)
        self.assertIn("NCCL watchdog", focus)
        self.assertIn("通信", focus)

    def test_nvtx_label_as_fallback(self):
        """When no actionable frame exists, NVTX label provides context."""
        from sysight.analyzer.nsys.models import SourceFrame
        from sysight.analyzer.nsys.stacks import summarize_callstack_focus

        focus = summarize_callstack_focus(
            [],
            nvtx_labels=["iter_110", "forward", "attention"],
        )
        self.assertIsNotNone(focus)
        self.assertIn("NVTX", focus)
        self.assertIn("forward", focus)


class TestGpuInventoryRendering(unittest.TestCase):

    def test_summary_renders_grouped_gpu_inventory(self):
        from sysight.analyzer.nsys.models import BottleneckSummary, GpuDeviceInfo, NsysDiag

        diag = NsysDiag(
            status="ok",
            profile_path="test.sqlite",
            sqlite_path="test.sqlite",
            required_action=None,
            bottlenecks=BottleneckSummary(
                total_ns=10_000_000,
                gpu_active_ns=8_000_000,
                gpu_idle_ns=2_000_000,
                labels=[],
                top_events=[],
                per_device=[],
            ),
            findings=[],
            hotspots=[],
            warnings=[],
            summary="",
            gpu_devices=[
                GpuDeviceInfo(
                    device_id=i,
                    name="NVIDIA A100-SXM4-80GB",
                    total_memory_bytes=84_987_740_160,
                    memory_bandwidth_bytes_per_s=2_039_040_000_000,
                    sm_count=108,
                    compute_capability="8.0",
                )
                for i in range(8)
            ],
        )

        output = render_nsys_terminal(diag)
        self.assertIn("GPU 信息", output)
        self.assertIn("A100-SXM4-80GB*8", output)
        self.assertIn("HBM 80GB", output)
        self.assertIn("SM 108", output)


class TestCpuHotspotRendering(unittest.TestCase):
    """Tests for CPU hotspot section in the terminal report."""

    def _make_diag(self, hotspots):
        """Build a minimal NsysDiag with given hotspots."""
        from sysight.analyzer.nsys.models import NsysDiag
        return NsysDiag(
            status="ok",
            profile_path="test.sqlite",
            sqlite_path="test.sqlite",
            required_action=None,
            bottlenecks=None,
            findings=[],
            hotspots=hotspots,
            warnings=[],
            summary="",
        )

    def test_no_crash_on_empty_hotspots(self):
        """_render_cpu_hotspots must return empty list for no hotspots."""
        diag = self._make_diag([])
        lines = _render_cpu_hotspots(diag, width=100)
        self.assertEqual(lines, [])

    def test_cpu_hotspot_renders_symbol_and_count(self):
        from sysight.analyzer.nsys.models import SampleHotspot, SourceFrame
        hs = SampleHotspot(
            frame=SourceFrame(symbol="_PyEval_EvalFrameDefault", source_file=None, source_line=None),
            count=500,
            pct=0.42,
        )
        diag = self._make_diag([hs])
        lines = _render_cpu_hotspots(diag, width=100)
        output = "\n".join(lines)
        self.assertIn("_PyEval_EvalFrameDefault", output)
        self.assertIn("500", output)
        self.assertIn("42.0%", output)

    def test_cpu_hotspot_hint_prefers_wait_reason_over_syscall(self):
        """Coarse location must be readable, not raw syscall names."""
        from sysight.analyzer.nsys.models import SampleHotspot, SourceFrame

        hs = SampleHotspot(
            frame=SourceFrame(symbol="pthread_cond_timedwait", source_file=None, source_line=None, raw="watchdogHandler <- ncclCommWatchdog"),
            count=12,
            pct=0.3,
            callstack=[
                "c10d::ProcessGroupNCCL::watchdogHandler()",
                "c10d::ProcessGroupNCCL::ncclCommWatchdog()",
            ],
            coarse_location="NCCL watchdog 线程，疑似通信/监控等待",
        )
        output = "\n".join(_render_cpu_hotspots(self._make_diag([hs]), width=100))
        # Must show readable coarse_location instead of raw syscall
        self.assertIn("NCCL watchdog", output)
        self.assertIn("监控等待", output)
        # Must NOT show raw syscall leaf as primary hint
        self.assertNotIn("pthread_cond_timedwait", output)


if __name__ == "__main__":
    unittest.main()
