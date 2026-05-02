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

        task_type: "analyze" | "optimize" | "learn"

        Production prompt includes:
          - common_role (always)
          - evidence_sop (analyze) or optimizer_sop (optimize)
          - output_schema (localized or patch)
          - safety_read_only (analyze only)

        Production prompt excludes:
          - benchmark_hints (unless include_benchmark_hints=True)
        """
        fragments: list[str] = []

        # Common role — always included
        fragments.append(self._load("common_role"))

        # Task-specific SOP
        if task_type == "analyze":
            fragments.append(self._load("evidence_sop"))
            fragments.append(self._load("output_schema_localized"))
            fragments.append(self._load("safety_read_only"))
        elif task_type == "optimize":
            fragments.append(self._load("optimizer_sop"))
            fragments.append(self._load("output_schema_patch"))
        elif task_type == "learn":
            fragments.append(self._load("evidence_sop"))  # for reflection on findings

        # Benchmark hints — only when explicitly enabled
        if include_benchmark_hints:
            hints = self._load("benchmark_hints")
            if hints:
                fragments.append(hints)

        return "\n\n".join(f.strip() for f in fragments if f.strip())

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
            parts.append("请分析以下 Nsight Systems profile 的性能瓶颈。")
            if profile_summary:
                parts.append(f"## Profile Summary\n\n{profile_summary}")
            if pre_injected_sql:
                parts.append(f"## Pre-Injected SQL Results\n\n{pre_injected_sql}")
            if memory_brief:
                parts.append(f"## Memory Context\n\n{memory_brief}")
            parts.append("请输出 LocalizedFindingSet JSON。")

        elif task_type == "optimize":
            parts.append("请根据以下 findings 生成优化 patch。")
            if findings_json:
                parts.append(f"## Findings\n\n{findings_json}")
            if memory_brief:
                parts.append(f"## Memory Context\n\n{memory_brief}")
            parts.append("请输出 PatchCandidate JSON 数组。")

        elif task_type == "learn":
            parts.append("请根据以下 session 结果生成经验总结。")
            if findings_json:
                parts.append(f"## Session Results\n\n{findings_json}")

        return "\n\n".join(parts)

    def _load(self, name: str) -> str:
        """Load a single fragment .md file."""
        path = self._fragments_dir / f"{name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""
