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
