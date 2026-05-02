"""ANALYZE stage — per-profile, 1 AgentLoop → LocalizedFindingSet.

1. Extract + Classify (deterministic): SQLite → classify() → C1-C7 bottleneck report
2. Build prompt: system + profile_summary + memory_brief
3. Build prompt: system + profile_summary + memory_brief
4. LLM Investigate (1 AgentLoop, N turns)
5. Validate (deterministic): schema + path + dedup
6. Apply Memory (deterministic): parent writes memory_updates
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sysight.types.findings import LocalizedFinding, LocalizedFindingSet, MemoryUpdate


@dataclass
class AnalyzeResult:
    run_id: str = ""
    finding_set: LocalizedFindingSet = field(default_factory=lambda: LocalizedFindingSet(run_id=""))
    errors: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0


def run_analyze(
    profile: str,
    repo: str,
    registry,
    provider,
    knowledge=None,
    run_id: str = "",
) -> AnalyzeResult:
    """Analyze a profile and produce a LocalizedFindingSet."""
    errors: list[str] = []
    root = Path(repo).resolve()
    sqlite_path = Path(profile).resolve()
    ns = knowledge.workspace_namespace(repo_root=str(root)) if knowledge else "default"

    if not run_id:
        from datetime import datetime, timezone
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # 1. Build profile summary (deterministic: classify + nsys_sql)
    profile_summary = _build_profile_summary(sqlite_path)

    # 2. Build memory brief
    memory_brief = ""
    if knowledge:
        from sysight.wiki.brief import build_memory_brief
        memory_brief = build_memory_brief(knowledge, namespace=ns)

    # 3. Build prompt
    from sysight.agent.prompts.loader import PromptLoader
    loader = PromptLoader()
    system_prompt = loader.build_system_prompt("analyze")
    user_prompt = loader.build_user_prompt(
        "analyze",
        profile_summary=profile_summary,
        memory_brief=memory_brief,
    )
    user_prompt += f"\n\nSQLite path: {sqlite_path}\nRepo root: {root}"

    # 4. Run AgentLoop
    from sysight.agent.loop import AgentLoop, AgentTask
    from sysight.tools.registry import ToolPolicy

    policy = ToolPolicy(
        allowed_tools={"scanner_*", "nsys_sql_*", "memory_*"},
        read_only=True,
    )
    loop = AgentLoop(provider, registry, policy)
    task = AgentTask(
        run_id=run_id, task_id=f"{run_id}-analyze",
        task_type="analyze",
        system_prompt=system_prompt, user_prompt=user_prompt,
        max_turns=30, max_wall_seconds=600,
    )

    result = loop.run(task)

    # 5. Parse output
    finding_set = LocalizedFindingSet(run_id=run_id)
    if result.output:
        finding_set = _parse_finding_set(result.output, run_id)

    # 6. Validate findings
    accepted, rejected = _validate_findings(finding_set.findings, root)
    finding_set.findings = accepted
    finding_set.rejected = rejected

    # 7. Apply memory updates (parent-only write)
    if knowledge and finding_set.memory_updates:
        _apply_memory_updates(knowledge, finding_set.memory_updates, ns)

    # 8. Record to ledger
    if knowledge:
        try:
            from sysight.wiki.ledger import RunLedger, RunRecord
            ledger = RunLedger()
            ledger.init()
            ledger.record_session(RunRecord(
                run_id=run_id, status="ok", repo_root=str(root),
                profile_hash=str(sqlite_path), memory_namespace=ns,
            ))
            ledger.record_findings(run_id, [
                {"finding_id": f.finding_id, "category": f.category,
                 "file_path": f.file_path, "line": f.line, "function": f.function,
                 "confidence": f.confidence, "status": f.status,
                 "reject_reason": f.reject_reason}
                for f in (accepted + rejected)
            ])
        except Exception as e:
            errors.append(f"ledger write failed: {e}")

    errors.extend(result.errors)
    return AnalyzeResult(
        run_id=run_id, finding_set=finding_set,
        errors=errors, tool_calls=result.tool_calls,
        elapsed_ms=result.elapsed_ms,
    )


def _build_profile_summary(sqlite_path: Path) -> str:
    """Build profile summary using the deterministic classify tool."""
    try:
        from sysight.tools.classify import classify
        result = classify(str(sqlite_path))
        return result.summary
    except Exception:
        import sqlite3
        lines = [f"Profile: {sqlite_path}"]
        try:
            conn = sqlite3.connect(str(sqlite_path))
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            lines.append(f"Tables: {', '.join(t[0] for t in tables)}")
            conn.close()
        except Exception:
            lines.append("(unable to read profile)")
        return "\n".join(lines)


def _parse_finding_set(data: dict, run_id: str) -> LocalizedFindingSet:
    """Parse LLM output into LocalizedFindingSet."""
    findings = []
    memory_updates = []

    for f in data.get("findings", []):
        findings.append(LocalizedFinding(
            finding_id=f.get("finding_id", ""),
            category=f.get("category", ""),
            title=f.get("title", ""),
            priority=f.get("priority", "medium"),
            confidence=f.get("confidence", "unresolved"),
            evidence_refs=f.get("evidence_refs", []),
            file_path=f.get("file_path"),
            function=f.get("function"),
            line=f.get("line"),
            description=f.get("description", ""),
            suggestion=f.get("suggestion", ""),
            status=f.get("status", "accepted"),
            reject_reason=f.get("reject_reason", ""),
        ))

    for mu in data.get("memory_updates", []):
        memory_updates.append(MemoryUpdate(
            path=mu.get("path", ""),
            content=mu.get("content", ""),
            action=mu.get("action", "append"),
            category=mu.get("category"),
            scope=mu.get("scope", "workspace"),
            reason=mu.get("reason", ""),
        ))

    return LocalizedFindingSet(
        run_id=run_id,
        summary=data.get("summary", ""),
        findings=findings,
        memory_updates=memory_updates,
    )


def _validate_findings(
    findings: list[LocalizedFinding],
    repo_root: Path,
) -> tuple[list[LocalizedFinding], list[LocalizedFinding]]:
    """Validate findings: path containment, file existence, dedup."""
    accepted: list[LocalizedFinding] = []
    rejected: list[LocalizedFinding] = []
    seen: set[str] = set()

    for f in findings:
        # Dedup
        key = f"{f.category}:{f.file_path}:{f.line}:{f.function}"
        if key in seen:
            continue
        seen.add(key)

        # Path containment
        if f.file_path:
            try:
                full = (repo_root / f.file_path).resolve()
                if not str(full).startswith(str(repo_root.resolve())):
                    f.status = "rejected"
                    f.reject_reason = "path outside repo"
                    rejected.append(f)
                    continue
                if not full.exists():
                    f.status = "rejected"
                    f.reject_reason = "file not found"
                    rejected.append(f)
                    continue
            except (OSError, ValueError):
                f.status = "rejected"
                f.reject_reason = "invalid path"
                rejected.append(f)
                continue

        accepted.append(f)

    return accepted, rejected


def _apply_memory_updates(knowledge, updates: list[MemoryUpdate], namespace: str) -> None:
    """Apply MemoryUpdate suggestions to wiki (parent-side write)."""
    for mu in updates:
        path = mu.path
        if not path.startswith("workspaces/") and not path.startswith("experiences/"):
            path = f"workspaces/{namespace}/{path}"
        try:
            knowledge.write_page(
                path, mu.content, category=mu.category, scope=mu.scope,
                source_run=getattr(mu, 'source_run', ''),
            )
        except Exception:
            pass
