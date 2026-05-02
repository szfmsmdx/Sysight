"""OPTIMIZE stage — per-session, N AgentLoops → PatchResult[].

Per-finding: LLM → PatchCandidate → sandbox.apply → validate → measure → keep/revert.
Context between subloops: previous_patches summary (not chat history).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from sysight.types.findings import LocalizedFindingSet
from sysight.types.optimization import PatchResult


@dataclass
class OptimizeResult:
    run_id: str = ""
    patches: list[PatchResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0


def run_optimize(
    findings: LocalizedFindingSet,
    repo: str,
    registry,
    provider,
    knowledge=None,
) -> OptimizeResult:
    """Optimize findings and produce PatchResults.

    Subloop 0: Instrumentation (optional, 1 LLM — TODO)
    Subloop 1..N: Per-Finding
      - LLM → PatchCandidate
      - sandbox_apply → sandbox_validate → sandbox_measure
      - keep: sandbox_commit + log / revert: sandbox_revert + log
    """
    errors: list[str] = []
    patch_results: list[PatchResult] = []

    if not findings.findings:
        return OptimizeResult(run_id=findings.run_id, patches=[], errors=["no findings to optimize"])

    # Sort findings by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_findings = sorted(
        [f for f in findings.findings if f.status == "accepted"],
        key=lambda f: priority_order.get(f.priority, 2),
    )

    from sysight.agent.prompts.loader import PromptLoader
    from sysight.agent.loop import AgentLoop, AgentTask
    from sysight.tools.registry import ToolPolicy

    loader = PromptLoader()
    policy = ToolPolicy(
        allowed_tools={"scanner_*", "sandbox_*"},
        read_only=False,
    )

    previous_summaries: list[str] = []

    for finding in sorted_findings:
        # Build context: previous patches
        prev_context = ""
        if previous_summaries:
            prev_context = "## Previous Patches\n" + "\n".join(
                f"- {s}" for s in previous_summaries[-5:]
            )

        # Build finding info
        finding_json = json.dumps({
            "finding_id": finding.finding_id,
            "category": finding.category,
            "title": finding.title,
            "file_path": finding.file_path,
            "function": finding.function,
            "line": finding.line,
            "description": finding.description,
            "suggestion": finding.suggestion,
        }, indent=2, ensure_ascii=False)

        # Memory brief
        memory_brief = ""
        if knowledge:
            from sysight.wiki.brief import build_memory_brief
            ns = knowledge.workspace_namespace(repo_root=repo)
            memory_brief = build_memory_brief(knowledge, namespace=ns)

        system_prompt = loader.build_system_prompt("optimize")
        user_prompt = loader.build_user_prompt(
            "optimize",
            findings_json=finding_json,
            memory_brief=memory_brief,
        )
        user_prompt += f"\n\n{prev_context}\nRepo root: {repo}"

        loop = AgentLoop(provider, registry, policy)
        task = AgentTask(
            run_id=findings.run_id, task_id=f"{findings.run_id}-opt-{finding.finding_id}",
            task_type="optimize",
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_turns=10, max_wall_seconds=300,
        )

        result = loop.run(task)

        if result.status != "ok":
            patch_result = PatchResult(
                patch_id=f"patch-{finding.finding_id}",
                finding_id=finding.finding_id,
                status="reverted",
                reason="llm_error",
            )
            patch_results.append(patch_result)
            previous_summaries.append(f"FAILED: {finding.title} — LLM error")
            errors.extend(result.errors)
            continue

        # Parse patches from output
        patches_data = result.output.get("patches", []) if result.output else []
        if not patches_data and result.raw_content:
            try:
                parsed = AgentLoop._extract_json(result.raw_content)
                patches_data = parsed.get("patches", [])
            except Exception:
                pass

        if not patches_data:
            patch_result = PatchResult(
                patch_id=f"patch-{finding.finding_id}",
                finding_id=finding.finding_id,
                status="reverted",
                reason="no_patch_generated",
            )
            patch_results.append(patch_result)
            previous_summaries.append(f"SKIPPED: {finding.title} — no patch generated")
            continue

        for pd in patches_data:
            patch_id = pd.get("patch_id", f"patch-{finding.finding_id}")
            summary = pd.get("rationale", "")[:100]

            # TODO: when sandbox is implemented:
            # 1. sandbox_create → sandbox_id
            # 2. sandbox_apply(sandbox_id, file_path, old_span_start, old_span_end, old_span_hash, replacement)
            # 3. sandbox_validate(sandbox_id, validation_commands)
            # 4. sandbox_measure(sandbox_id, run_cmd, grep, runs=5)
            # 5. compare → sandbox_commit or sandbox_revert
            # For now, record as "pending" since sandbox tools are not implemented

            patch_result = PatchResult(
                patch_id=patch_id,
                finding_id=finding.finding_id,
                status="kept",
                reason="patch_generated_sandbox_pending",
            )
            patch_results.append(patch_result)
            previous_summaries.append(f"OK: {finding.title} — {summary}")

    # Record to ledger
    if knowledge:
        try:
            from sysight.wiki.ledger import RunLedger
            ledger = RunLedger()
            ledger.init()
            ledger.record_patches(findings.run_id, [
                {"patch_id": p.patch_id, "finding_id": p.finding_id,
                 "status": p.status, "reason": p.reason,
                 "metric_before": p.metric_before, "metric_after": p.metric_after,
                 "delta_pct": p.delta_pct}
                for p in patch_results
            ])
        except Exception as e:
            errors.append(f"ledger write failed: {e}")

    return OptimizeResult(
        run_id=findings.run_id,
        patches=patch_results,
        errors=errors,
    )
