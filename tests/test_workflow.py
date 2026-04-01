from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sysight.workflow import load_program_contract, resolve_workflow_route


class WorkflowContractTest(unittest.TestCase):
    def test_load_program_contract_extracts_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            program = root / "program.md"
            program.write_text(
                """
# Task
- 分析训练主路径上的 GPU idle 问题

# Project
- 一个 PyTorch 推理项目

# Framework
- PyTorch 2.6 + CUDA + Triton

# Entry
- python main.py --config configs/infer.yaml

# Performance Goal
- 优先降低 P99 latency

# Important Paths
- `src/model/decoder.py`
- `src/runtime/launcher.py`

# Constraints
- 当前阶段不要自动修改代码

# Success Criteria
- 至少给出 3 个有证据支撑的问题与下一步建议

# Output Contract
- 输出终端摘要和 report.md
""".strip()
                + "\n",
                encoding="utf-8",
            )

            contract = load_program_contract(root)
            self.assertTrue(contract.is_complete)
            self.assertEqual(contract.task, "分析训练主路径上的 GPU idle 问题")
            self.assertEqual(contract.framework, "PyTorch 2.6 + CUDA + Triton")
            self.assertIn("python main.py", contract.entry)
            self.assertEqual(contract.success_criteria, "至少给出 3 个有证据支撑的问题与下一步建议")
            self.assertIn("src/model/decoder.py", contract.important_paths)

    def test_resolve_workflow_route_prefers_workspace_mode_when_program_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile = root / "trace.sqlite"
            profile.write_text("placeholder", encoding="utf-8")
            (root / "program.md").write_text(
                """
# Task
- 分析 bottleneck

# Project
- Demo project

# Framework
- PyTorch training stack

# Entry
- python run.py

# Performance Goal
- 提升 throughput

# Important Paths
- `src/train.py`

# Constraints
- 只做分析

# Success Criteria
- 给出优先级最高的问题

# Output Contract
- 输出 report
""".strip()
                + "\n",
                encoding="utf-8",
            )

            route = resolve_workflow_route(profile_path=str(profile), workspace_root=str(root))
            self.assertEqual(route.mode, "workspace-aware")
            self.assertTrue(route.ready)
            self.assertEqual(route.program_contract["framework"], "PyTorch training stack")
            self.assertEqual(route.program_contract["entry"], "python run.py")

    def test_resolve_workflow_route_without_program_stays_profile_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile = root / "trace.sqlite"
            profile.write_text("placeholder", encoding="utf-8")

            route = resolve_workflow_route(profile_path=str(profile))
            self.assertEqual(route.mode, "profile-only")
            self.assertTrue(route.ready)
            self.assertFalse(route.program_contract)


if __name__ == "__main__":
    unittest.main()
