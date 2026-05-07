"""LEARN stage — agentic wiki learning from analysis results.

Single LLM call with memory read/write tools. The LLM agent:
1. Reads current wiki state (workspace overview + global experience)
2. Compares against findings/patches from the previous stage
3. Writes/updates wiki with new stable knowledge

Called after both analyze and optimize stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class LearnResult:
    run_id: str = ""
    summary: str = ""
    errors: list[str] = field(default_factory=list)


def run_learn(
    run_id: str,
    knowledge,
    provider=None,
    findings_json: str = "",
    patches_json: str = "",
) -> LearnResult:
    """Agentic wiki learning from analysis results.

    Args:
        run_id: The run to process
        knowledge: WikiRepository
        provider: LLMProvider (skips LLM call if None)
        findings_json: JSON string of findings from analyze stage
        patches_json: JSON string of patches from optimize stage (may be empty)
    """
    if not provider:
        return LearnResult(run_id=run_id)

    errors: list[str] = []

    try:
        from sysight.agent.loop import AgentLoop, AgentTask
        from sysight.agent.prompts.loader import PromptLoader
        from sysight.tools.registry import LEARN_POLICY

        loader = PromptLoader()
        system = loader.build_system_prompt("learn")

        # Build input: findings + patches (if any)
        input_parts: list[str] = []
        if findings_json:
            input_parts.append(f"## Findings\n\n{findings_json}")
        if patches_json:
            input_parts.append(f"## Patches\n\n{patches_json}")

        memory_brief = _build_memory_brief(run_id, knowledge)

        user = loader.build_user_prompt(
            "learn",
            findings_json="\n\n".join(input_parts) if input_parts else "",
            memory_brief=memory_brief,
        )

        loop = AgentLoop(provider, None, LEARN_POLICY)
        task = AgentTask(
            run_id=run_id,
            task_id=f"{run_id}-learn",
            task_type="learn",
            system_prompt=system,
            user_prompt=user,
            max_turns=10,
            max_wall_seconds=120,
        )
        result = loop.run(task)

        # Parse memory_updates from LLM output and apply them
        summary = ""
        if result.output:
            data = result.output
            summary = data.get("summary", "")
            for update in data.get("memory_updates", []):
                _apply_memory_update(update, knowledge)
        else:
            # Fallback: try raw content
            raw = result.raw_content or ""
            summary = raw[:200] if raw else ""

        return LearnResult(run_id=run_id, summary=summary, errors=errors)

    except Exception as e:
        errors.append(str(e))
        return LearnResult(run_id=run_id, errors=errors)


def _build_memory_brief(run_id: str, knowledge) -> str:
    """Build a brief memory context for the LLM."""
    from sysight.wiki.ledger import RunLedger

    ledger = RunLedger()
    ledger.init()
    ns_record = ledger.recent_session(run_id)
    namespace = ns_record.get("memory_namespace", "default") if ns_record else "default"

    parts = [f"当前 workspace namespace: {namespace}"]

    # Read current workspace overview if exists
    try:
        overview = knowledge.read_page(f"workspaces/{namespace}/overview.md")
        if overview:
            parts.append(f"\n当前 workspace overview:\n```\n{overview[:2000]}\n```")
    except Exception:
        pass

    # Read global experience if exists
    try:
        exp = knowledge.read_page("experiences/experience.md")
        if exp:
            parts.append(f"\n当前全局 experience（摘要）:\n```\n{exp[:1500]}\n```")
    except Exception:
        pass

    return "\n".join(parts)


def _apply_memory_update(update: dict, knowledge) -> None:
    """Apply a single memory update from the LLM's output."""
    path = update.get("path", "")
    action = update.get("action", "")

    if not path or not action:
        return

    # Validate path is within allowed boundaries
    if not (path.startswith("workspaces/") or path.startswith("experiences/")):
        return

    try:
        if action == "write":
            content = update.get("content", "")
            knowledge.write_page(path, content)
        elif action == "append":
            content = update.get("content", "")
            knowledge.append_page(path, content)
        elif action == "replace":
            old = update.get("old", "")
            new = update.get("new", "")
            if old and new:
                knowledge.replace_in_page(path, old, new)
    except Exception as e:
        import sys
        print(f"  ⚠ learn memory update failed [{action} {path}]: {e}", file=sys.stderr)
