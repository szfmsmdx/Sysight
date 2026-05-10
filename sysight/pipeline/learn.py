"""LEARN stage: turn pipeline results into durable wiki knowledge."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LearnResult:
    run_id: str = ""
    summary: str = ""
    worklog: str = ""
    worklog_path: str = ""
    memory_updates: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_learn(
    run_id: str,
    knowledge,
    provider=None,
    findings_json: str = "",
    patches_json: str = "",
    repo: str = "",
    max_wall_seconds: int = 0,
) -> LearnResult:
    """Learn from one analyze/optimize step and update wiki/worklog.

    A deterministic workspace worklog entry is always attempted. When a provider
    is available, an LLM can additionally update workspace or experience wiki
    pages using memory tools.
    """
    errors: list[str] = []
    namespace = _namespace_for_run(run_id, knowledge, repo)
    worklog = _build_deterministic_worklog(
        run_id,
        findings_json=findings_json,
        patches_json=patches_json,
    )
    worklog_path = ""
    try:
        if knowledge and worklog:
            worklog_path = str(knowledge.append_worklog(namespace, worklog))
    except Exception as e:
        errors.append(f"worklog append failed: {e}")

    if not provider:
        return LearnResult(
            run_id=run_id,
            summary="learn provider not configured; deterministic worklog appended",
            worklog=worklog,
            worklog_path=worklog_path,
            errors=errors,
        )

    try:
        from sysight.agent.loop import AgentLoop, AgentTask
        from sysight.agent.prompts.loader import PromptLoader
        from sysight.tools.memory import register_memory_tools
        from sysight.tools.registry import LEARN_POLICY, ToolRegistry

        loader = PromptLoader()
        system = loader.build_system_prompt("learn")

        input_parts: list[str] = []
        if findings_json:
            input_parts.append(f"## Findings\n\n{findings_json}")
        if patches_json:
            input_parts.append(f"## Optimize Trials\n\n{patches_json}")

        memory_brief = _build_memory_brief(
            run_id,
            knowledge,
            repo=repo,
            namespace=namespace,
        )
        user = loader.build_user_prompt(
            "learn",
            findings_json="\n\n".join(input_parts) if input_parts else "",
            memory_brief=memory_brief,
        )

        registry = ToolRegistry()
        register_memory_tools(registry)
        loop = AgentLoop(provider, registry, LEARN_POLICY)
        task = AgentTask(
            run_id=run_id,
            task_id=f"{run_id}-learn",
            task_type="learn",
            system_prompt=system,
            user_prompt=user,
            max_turns=20,
            max_wall_seconds=max_wall_seconds,
            max_tokens=8192,
        )
        result = loop.run(task)

        summary = ""
        updates: list[dict] = []
        if result.output:
            data = result.output
            summary = str(data.get("summary", ""))
            updates = [u for u in data.get("memory_updates", []) if isinstance(u, dict)]
            for update in updates:
                _apply_memory_update(update, knowledge)
        else:
            raw = result.raw_content or ""
            summary = raw[:200] if raw else ""

        errors.extend(result.errors)
        return LearnResult(
            run_id=run_id,
            summary=summary,
            worklog=worklog,
            worklog_path=worklog_path,
            memory_updates=updates,
            errors=errors,
        )
    except Exception as e:
        errors.append(str(e))
        return LearnResult(
            run_id=run_id,
            worklog=worklog,
            worklog_path=worklog_path,
            errors=errors,
        )


def _namespace_for_run(run_id: str, knowledge, repo: str = "") -> str:
    if knowledge and repo:
        try:
            return knowledge.workspace_namespace(repo_root=repo)
        except Exception:
            pass
    try:
        from sysight.wiki.ledger import RunLedger
        ledger = RunLedger()
        ledger.init()
        with ledger._connect() as conn:
            row = conn.execute(
                "SELECT memory_namespace FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row and row[0]:
                return str(row[0])
    except Exception:
        pass
    return Path(repo).resolve().name if repo else "default"


def _build_memory_brief(
    run_id: str,
    knowledge,
    *,
    repo: str = "",
    namespace: str = "",
) -> str:
    namespace = namespace or _namespace_for_run(run_id, knowledge, repo)
    parts = [f"Current workspace namespace: {namespace}"]
    if not knowledge:
        return "\n".join(parts)

    for label, path, limit in (
        ("Current workspace overview", f"workspaces/{namespace}/overview.md", 2500),
        ("Recent workspace worklog", f"workspaces/{namespace}/worklog.md", 2500),
        ("Global experience excerpt", "experiences/experience.md", 1500),
    ):
        try:
            text = knowledge.read_page(path) or ""
        except Exception:
            text = ""
        if text:
            excerpt = text[-limit:] if label.startswith("Recent") else text[:limit]
            parts.append(f"\n{label}:\n```\n{excerpt}\n```")
    return "\n".join(parts)


def _build_deterministic_worklog(
    run_id: str,
    *,
    findings_json: str = "",
    patches_json: str = "",
) -> str:
    parts = [f"- run_id: `{run_id}`"]
    findings = _loads(findings_json)
    patches = _loads(patches_json)
    finding_count = len(findings.get("findings", [])) if isinstance(findings, dict) else 0
    parts.append(f"- findings: {finding_count}")

    if isinstance(patches, dict) and patches:
        baseline = patches.get("baseline", {}) or {}
        best = patches.get("best_measurement", {}) or {}
        parts.extend([
            f"- accepted_trials: {int(patches.get('accepted_count', 0) or 0)}",
            f"- rejected_trials: {int(patches.get('rejected_count', 0) or 0)}",
            f"- best_commit: `{patches.get('best_commit', '')}`",
            (
                f"- metric: {best.get('primary_metric', '')} "
                f"{baseline.get('primary_value', '')} -> {best.get('primary_value', '')}"
            ),
        ])
        summaries = [
            str(t.get("summary", ""))
            for t in patches.get("trials", [])
            if isinstance(t, dict) and t.get("summary")
        ]
        if summaries:
            parts.append(f"- coding_summary: {summaries[-1][:500]}")
    return "\n".join(parts)


def _loads(text: str) -> dict:
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _apply_memory_update(update: dict, knowledge) -> None:
    path = str(update.get("path", ""))
    action = str(update.get("action", ""))
    if not knowledge or not path or not action:
        return
    if not (path.startswith("workspaces/") or path.startswith("experiences/")):
        return

    try:
        if action == "write":
            knowledge.write_page(path, str(update.get("content", "")))
        elif action == "append":
            knowledge.append_page(path, str(update.get("content", "")))
        elif action == "replace":
            old = str(update.get("old", ""))
            new = str(update.get("new", ""))
            if old and new:
                knowledge.replace_in_page(path, old, new)
    except Exception as e:
        import sys
        print(f"  learn memory update failed [{action} {path}]: {e}", file=sys.stderr)
