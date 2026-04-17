"""Terminal rendering tests for nsys reports."""

import unittest

from sysight.analyzer.nsys.render import _kv_block, _render_task_drafts
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


class TestTaskDraftRendering(unittest.TestCase):
    """Tests for the TaskDraft section in the terminal report."""

    def _make_diag(self, task_drafts):
        """Build a minimal NsysDiag with given task_drafts."""
        from sysight.analyzer.nsys.models import NsysDiag
        return NsysDiag(
            status="ok",
            profile_path="test.sqlite",
            sqlite_path="test.sqlite",
            required_action=None,
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            task_drafts=task_drafts,
            repo_warnings=[],
            summary="",
        )

    def test_rg_command_rendered_for_search_specs(self):
        """If a TaskDraft has search_specs, rg command must appear in output."""
        from sysight.analyzer.nsys.models import TaskDraft
        td = TaskDraft(
            id="sync_wait_001_draft",
            finding_id="sync_wait_001",
            hypothesis="同步等待时间过长",
            verification_metric="同步等待时间下降",
            candidate_callsites=[],
            target_locations=[],
            search_specs=[{
                "pattern": r"synchronize\(|\.item\(",
                "kind": "rg",
                "rationale": "sync_wait seed",
            }],
        )
        diag = self._make_diag([td])
        lines = _render_task_drafts(diag, width=100)
        output = "\n".join(lines)

        self.assertIn("rg -n", output, "Expected rg command in task draft output")
        self.assertIn("synchronize", output, "Expected pattern in rg command")
        self.assertIn("搜索入口", output, "Expected '搜索入口' label to distinguish from conclusions")
        self.assertIn("非定论", output, "Expected '非定论' disclaimer")

    def test_evidence_windows_rendered(self):
        """If a TaskDraft has evidence_windows, timestamps and event must appear."""
        from sysight.analyzer.nsys.models import TaskDraft
        td = TaskDraft(
            id="memcpy_001_draft",
            finding_id="gpu_memcpy_hotspot_001",
            hypothesis="memcpy 量异常",
            verification_metric="memcpy 下降",
            candidate_callsites=[],
            target_locations=[],
            evidence_windows=[{
                "start_ms": 1234.5,
                "end_ms": 1250.2,
                "duration_ms": 15.7,
                "device": 0,
                "stream": 7,
                "event": "memcpy_HtoD",
                "before_kernel": "volta_sgemm_128x64_nt",
                "after_kernel": "nccl_AllReduce",
                "overlap_nvtx": ["iter_106", "Holding GIL"],
            }],
        )
        diag = self._make_diag([td])
        lines = _render_task_drafts(diag, width=100)
        output = "\n".join(lines)

        self.assertIn("1234.5", output, "Expected start_ms in evidence window output")
        self.assertIn("1250.2", output, "Expected end_ms in evidence window output")
        self.assertIn("memcpy_HtoD", output, "Expected event name in evidence window output")
        self.assertIn("volta_sgemm", output, "Expected before_kernel in output")
        self.assertIn("iter_106", output, "Expected NVTX label in output")
        self.assertIn("Holding GIL", output, "Expected Holding GIL NVTX label in output")
        self.assertIn("Top 证据窗口", output, "Expected section header '证据窗口'")

    def test_no_crash_on_empty_task_drafts(self):
        """_render_task_drafts must return empty list for no task_drafts."""
        diag = self._make_diag([])
        lines = _render_task_drafts(diag, width=100)
        self.assertEqual(lines, [])

    def test_nvtx_holding_gil_shown_not_filtered(self):
        """'Holding GIL' NVTX label must NOT be filtered; it appears as-is."""
        from sysight.analyzer.nsys.models import TaskDraft
        td = TaskDraft(
            id="sync_001_draft",
            finding_id="sync_wait_001",
            hypothesis="同步等待",
            verification_metric="",
            candidate_callsites=[],
            target_locations=[],
            evidence_windows=[{
                "start_ms": 100.0,
                "end_ms": 110.0,
                "duration_ms": 10.0,
                "device": None,
                "stream": None,
                "event": "cuda_sync",
                "before_kernel": None,
                "after_kernel": None,
                "overlap_nvtx": ["Holding GIL", "iter_42"],
            }],
        )
        diag = self._make_diag([td])
        lines = _render_task_drafts(diag, width=100)
        output = "\n".join(lines)
        self.assertIn("Holding GIL", output,
                      "Holding GIL must be shown, not filtered")


if __name__ == "__main__":
    unittest.main()
