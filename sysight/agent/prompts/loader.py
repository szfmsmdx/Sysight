"""PromptLoader — assemble prompt fragments by task type.

Builder pattern — fragments are assembled based on task_type,
with benchmark hints excluded from production prompts by default.
"""

from __future__ import annotations

from pathlib import Path


class PromptLoader:
    """Loads and assembles prompt fragments for a given task."""

    def __init__(self, fragments_dir: str | Path | None = None):
        if fragments_dir:
            self._fragments_dir = Path(fragments_dir)
        else:
            self._fragments_dir = Path(__file__).resolve().parent / "fragments"

    def build_system_prompt(
        self,
        task_type: str,
        include_benchmark_hints: bool = False,
    ) -> str:
        """Assemble system prompt from fragments.

        task_type: "analyze"  → analyze_system.md (role + SOP + schema)
                   "optimize" → optimize_system.md
                   "learn"    → learn_system.md (wiki knowledge extraction)
        """
        if task_type == "analyze":
            content = self._load("analyze_system")
        elif task_type == "learn":
            content = self._load("learn_system")
        elif task_type == "instrument":
            content = self._load("instrument_system")
        elif task_type == "optimize":
            content = self._load("optimize_system")
        else:
            content = ""

        if include_benchmark_hints:
            hints = self._load("benchmark_hints")
            if hints:
                content = content + "\n\n" + hints if content else hints

        return content.strip()

    def build_user_prompt(
        self,
        task_type: str,
        profile_summary: str = "",
        pre_injected_sql: str = "",
        memory_brief: str = "",
        findings_json: str = "",
        learn_stage: str = "",
    ) -> str:
        """Assemble the user message with context."""

        parts: list[str] = []

        if task_type == "analyze":
            parts.append("以下是 profile 统计报告（已由 Sysight analyzer 生成）：")
            if profile_summary:
                parts.append(profile_summary)
            if pre_injected_sql:
                parts.append(f"\n────────────────────────────────────────────────────────────────\n  预注入 Profile 数据（已覆盖主要维度，无新疑问时无需重调 CLI）\n────────────────────────────────────────────────────────────────\n\n{pre_injected_sql}")
            if memory_brief:
                parts.append(f"\n## Memory Context\n\n{memory_brief}")
            parts.append("请输出 LocalizedFindingSet JSON。")

        elif task_type == "instrument":
            parts.append("以下是 Analyzer 产出的 findings，请根据这些 findings 进行针对性打标。")
            if findings_json:
                parts.append(f"## Analyzer Findings\n\n{findings_json}")
            if memory_brief:
                parts.append(f"## Memory Context\n\n{memory_brief}")
            parts.append("请按 SOP 逐步调查，最后输出 Instrumentation JSON。")

        elif task_type == "learn":
            if learn_stage == "post_analyze":
                parts.append(
                    "## 当前阶段：LEARN(1) — analyze 后知识沉淀\n\n"
                    "本次 LEARN 在 ANALYZE 之后运行，**没有 patch 试验结果**。\n\n"
                    "**写入范围（本阶段严格限定）**：\n"
                    "- `workspaces/<namespace>/overview.md`：只写入从 findings 中归纳出的"
                    "长期稳定事实（repo 结构、入口链路、核心模块），**禁止写入性能数字或行号**。\n"
                    "- `experiences/<slug>.md`：只写跨 workspace 可复用的通用经验。\n"
                    "- **禁止**写入 worklog、具体优化结论，或预先描述尚未发生的优化。\n"
                )
            elif learn_stage == "post_optimize":
                parts.append(
                    "## 当前阶段：LEARN(2) — optimize 后知识沉淀\n\n"
                    "本次 LEARN 在 OPTIMIZE 之后运行，输入包含 findings **和** patch 试验结果。\n\n"
                    "**写入范围（本阶段严格限定）**：\n"
                    "- `workspaces/<namespace>/overview.md`：**仅修正**因 patch 而失效的已有描述"
                    "（用 `memory_replace`），不新增大段内容。\n"
                    "- `workspaces/<namespace>/worklog.md`：补充 accepted/rejected 试验摘要，"
                    "包括效果和拒绝原因，格式为追加行。\n"
                    "- `experiences/<slug>.md`：把本次试验中发现的通用优化规律写成经验条目，"
                    "禁止写入具体行号或数字。\n"
                    "- **禁止**重复写入 LEARN(1) 已沉淀的内容，避免与上一阶段产生冲突或覆盖。\n"
                )
            else:
                parts.append("请根据以下分析结果更新 wiki 知识。")

            if findings_json:
                parts.append(f"## 分析结果\n\n{findings_json}")
            if memory_brief:
                parts.append(f"## 当前 Memory 线索\n\n{memory_brief}")

        return "\n\n".join(parts)

    def _load(self, name: str) -> str:
        """Load a single fragment .md file."""
        path = self._fragments_dir / f"{name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""
