"""OPTIMIZE stage — single AgentLoop with tool access → PatchCandidate[].

Architecture:
  Phase 1: Plan — single AgentLoop (≤20 turns, tools enabled)
           LLM reads findings, evaluates them, reads source files,
           decides which to fix, generates PatchCandidate[]
  Phase 2: Fill hashes — code-side compute old_span_hash for each patch

Does NOT modify source files.  That is the EXECUTE stage's job.
"""

from __future__ import annotations

import json
import shlex
import sys
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from sysight.types.findings import LocalizedFindingSet
from sysight.types.optimization import (
    MeasurementPlan,
    MeasurementResult,
    PatchCandidate,
    TrialResult,
    compute_span_hash,
)
from sysight.types.repo_setup import RepoSetup


@dataclass
class OptimizeLoopResult:
    run_id: str = ""
    run_dir: Path | None = None
    worktree_path: str = ""
    baseline: MeasurementResult = field(default_factory=MeasurementResult)
    best_measurement: MeasurementResult = field(default_factory=MeasurementResult)
    best_commit: str = ""
    trials: list[TrialResult] = field(default_factory=list)
    accepted_count: int = 0
    rejected_count: int = 0
    errors: list[str] = field(default_factory=list)


# ── Main entry point ──

def run_optimize_trials(
    findings: LocalizedFindingSet,
    measurement_plan: MeasurementPlan,
    repo: str,
    registry,
    provider,
    *,
    repo_setup: RepoSetup | None = None,
    knowledge=None,
    verbose: bool = False,
    run_dir: Path | None = None,
    max_trials: int = 5,
    max_agent_wall_seconds: int = 0,
    memory_repo: str = "",
) -> OptimizeLoopResult:
    """Run iterative patch trials against an end-to-end measurement plan."""
    import hashlib
    from sysight.benchmark.debug import DebugProvider
    from sysight.pipeline.measure import compare_measurements, run_measurement
    from sysight.tools.patcher import PatchApplier

    root = Path(repo).resolve()
    errors: list[str] = []
    repo_setup = repo_setup or RepoSetup()
    max_trials = max(1, int(max_trials or 1))

    if run_dir is None:
        digest = hashlib.sha1(f"{root}|{findings.run_id}|trials".encode()).hexdigest()[:8]
        from sysight.utils.cache import cache_dir
        run_dir = cache_dir("optimizer-runs", f"run-{digest}-{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if not measurement_plan.run_command:
        measurement_plan.run_command = _repo_setup_run_command(repo_setup)
    if not measurement_plan.is_valid():
        errors.append("invalid measurement_plan: run_command and primary metric regex are required")
        result = OptimizeLoopResult(run_id=findings.run_id, run_dir=run_dir, errors=errors)
        _write_loop_result_json(result, run_dir)
        return result

    debug_log: list[dict] = []
    wrapped_provider = DebugProvider(
        provider, debug_log, verbose=verbose, log_file=str(run_dir / "optimize_debug.log")
    )

    try:
        worktree_root = _trial_worktree_root(run_dir)
        worktree = _create_trial_worktree(root, findings.run_id, worktree_root)
        best_commit = _git_rev_parse(worktree)
    except RuntimeError as e:
        errors.append(str(e))
        result = OptimizeLoopResult(run_id=findings.run_id, run_dir=run_dir, errors=errors)
        _write_loop_result_json(result, run_dir)
        return result

    bootstrap_error = _run_repo_setup_bootstrap(worktree, repo_setup)
    if bootstrap_error:
        errors.append(bootstrap_error)
        result = OptimizeLoopResult(
            run_id=findings.run_id,
            run_dir=run_dir,
            worktree_path=str(worktree),
            best_commit=best_commit,
            errors=errors,
        )
        _write_loop_result_json(result, run_dir)
        return result

    baseline = run_measurement(worktree, measurement_plan, run_dir / "measurements", "baseline")
    if baseline.status != "ok":
        fallback_plan = _fallback_measurement_plan(measurement_plan, repo_setup)
        if fallback_plan:
            fallback = run_measurement(
                worktree,
                fallback_plan,
                run_dir / "measurements",
                "baseline-fallback",
            )
            if fallback.status == "ok":
                errors.append(
                    "measurement_plan run_command failed; using warmup minimal_run fallback"
                )
                measurement_plan = fallback_plan
                baseline = fallback
    best_measurement = baseline
    if baseline.status != "ok":
        errors.extend(baseline.errors)
        result = OptimizeLoopResult(
            run_id=findings.run_id,
            run_dir=run_dir,
            worktree_path=str(worktree),
            baseline=baseline,
            best_measurement=best_measurement,
            best_commit=best_commit,
            errors=errors,
        )
        _write_loop_result_json(result, run_dir)
        return result

    result = OptimizeLoopResult(
        run_id=findings.run_id,
        run_dir=run_dir,
        worktree_path=str(worktree),
        baseline=baseline,
        best_measurement=best_measurement,
        best_commit=best_commit,
        errors=errors,
    )

    rejected_streak = 0
    for idx in range(1, max_trials + 1):
        trial_id = f"trial-{idx:03d}"
        action, summary, patches = _run_trial_optimize_loop(
            findings, measurement_plan, best_measurement, result.trials,
            worktree, wrapped_provider, registry, errors, trial_id,
            knowledge=knowledge,
            max_wall_seconds=max_agent_wall_seconds,
            memory_repo=memory_repo,
        )
        if action == "stop" or not patches:
            trial = TrialResult(
                trial_id=trial_id,
                status="skipped",
                commit_before=best_commit,
                primary_metric=best_measurement.primary_metric,
                metric_before=best_measurement.primary_value,
                reason="agent_stop" if action == "stop" else "no_patches",
                summary=summary,
            )
            result.trials.append(trial)
            _append_trial_logs(run_dir, trial)
            break

        _fill_span_hashes(patches, worktree, errors)
        trial = TrialResult(
            trial_id=trial_id,
            patch_ids=[p.patch_id for p in patches],
            finding_ids=sorted({fid for p in patches for fid in p.finding_ids}),
            commit_before=best_commit,
            primary_metric=best_measurement.primary_metric,
            metric_before=best_measurement.primary_value,
            summary=summary,
        )

        applier = PatchApplier(worktree)
        apply_error = _apply_trial_patches(applier, patches, worktree)
        if apply_error:
            _git_reset_to(worktree, best_commit)
            trial.status = "rejected"
            trial.reason = apply_error
            result.rejected_count += 1
            rejected_streak += 1
            result.trials.append(trial)
            _append_trial_logs(run_dir, trial)
            if rejected_streak >= 2:
                break
            continue

        validation_error = _run_trial_validation(worktree, patches, repo_setup)
        if validation_error:
            _git_reset_to(worktree, best_commit)
            trial.status = "rejected"
            trial.reason = validation_error
            result.rejected_count += 1
            rejected_streak += 1
            result.trials.append(trial)
            _append_trial_logs(run_dir, trial)
            if rejected_streak >= 2:
                break
            continue

        after = run_measurement(worktree, measurement_plan, run_dir / "measurements", trial_id)
        accepted, delta_pct, reason = compare_measurements(best_measurement, after, measurement_plan)
        trial.metric_after = after.primary_value
        trial.delta_pct = delta_pct
        trial.measurement_errors = after.errors[:]
        if after.status != "ok" and after.errors:
            trial.reason = f"{reason}: {after.errors[0]}"
        else:
            trial.reason = reason

        if accepted:
            try:
                commit = _commit_trial(worktree, trial_id, patches, delta_pct)
                trial.status = "accepted"
                trial.commit_after = commit
                best_commit = commit
                best_measurement = after
                result.best_commit = best_commit
                result.best_measurement = best_measurement
                result.accepted_count += 1
                rejected_streak = 0
            except RuntimeError as e:
                _git_reset_to(worktree, best_commit)
                trial.status = "rejected"
                trial.reason = str(e)
                result.rejected_count += 1
                rejected_streak += 1
        else:
            _git_reset_to(worktree, best_commit)
            trial.status = "rejected"
            result.rejected_count += 1
            rejected_streak += 1

        result.trials.append(trial)
        _append_trial_logs(run_dir, trial)
        _write_loop_result_json(result, run_dir)
        if rejected_streak >= 2:
            break

    _write_loop_result_json(result, run_dir)
    print(f"\n  Optimizer trial output -> {run_dir}", file=sys.stderr)
    print(f"  Worktree             -> {worktree}", file=sys.stderr)
    return result


def run_optimize(
    findings: LocalizedFindingSet,
    repo: str,
    registry,
    provider,
    *,
    verbose: bool = False,
    run_dir: Path | None = None,
) -> list[PatchCandidate]:
    """Run the optimizer: AgentLoop → PatchCandidate[] (plan only, no file changes).

    Writes patches to optimize_result.json in run_dir.

    Args:
        findings: The FindingSet from the analyze stage.
        repo: Path to repo root.
        registry: ToolRegistry instance.
        provider: LLM provider.
        verbose: Print LLM I/O to terminal.
        run_dir: Override output directory (used by benchmark runner).
    """
    import hashlib
    from sysight.benchmark.debug import DebugProvider

    t0 = time.monotonic()
    root = Path(repo).resolve()
    errors: list[str] = []

    if not root.is_dir():
        print(f"Error: repo 不是目录: {root}", file=sys.stderr)
        return []

    if not findings.findings:
        print("Warning: no findings to optimize", file=sys.stderr)
        return []

    # Create output directory
    if run_dir is None:
        digest = hashlib.sha1(
            f"{root}|{findings.run_id}".encode()
        ).hexdigest()[:8]
        opt_run_id = f"run-{digest}"
        from sysight.utils.cache import cache_dir
        run_dir = cache_dir("optimizer-runs", opt_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup debug logging
    debug_log: list[dict] = []
    log_file = str(run_dir / "optimize_debug.log")

    wrapped_provider = DebugProvider(provider, debug_log, verbose=verbose, log_file=log_file)

    # ── Phase 1: Plan — single AgentLoop with tools ──
    patches = _run_optimize_loop(
        findings, root, wrapped_provider, registry, errors,
    )

    if not patches:
        # Write empty result
        _write_patches_json([], findings.run_id, run_dir, errors)
        print(f"\n  Optimizer output → {run_dir}", file=sys.stderr)
        return []

    # ── Phase 2: Fill hashes — code-side, not LLM ──
    _fill_span_hashes(patches, root, errors)

    elapsed_ms = (time.monotonic() - t0) * 1000

    # Write patches JSON
    _write_patches_json(patches, findings.run_id, run_dir, errors, elapsed_ms)

    # Print output location
    print(f"\n  Optimizer output → {run_dir}", file=sys.stderr)

    # Print summary
    print(f"\n  OPTIMIZE COMPLETE  run_id={findings.run_id}", file=sys.stderr)
    print(f"  Patches:      {len(patches)}", file=sys.stderr)
    print(f"  Elapsed:      {elapsed_ms:.0f} ms", file=sys.stderr)
    if errors:
        print(f"  Errors:       {len(errors)}", file=sys.stderr)

    return patches


# ── Phase 1: Plan ──

def _run_optimize_loop(
    findings: LocalizedFindingSet,
    root: Path,
    provider,
    registry,
    errors: list[str],
) -> list[PatchCandidate]:
    """Single AgentLoop — LLM evaluates findings, reads files, generates patches.

    The LLM has access to scanner_read and scanner_search so it can:
    - Read source files referenced by findings
    - Search for cross-file dependencies
    - Evaluate whether each finding is worth fixing
    - Generate minimal patches only for findings it deems actionable
    """
    from sysight.agent.loop import AgentLoop, AgentTask
    from sysight.agent.prompts.loader import PromptLoader
    from sysight.tools.registry import ToolPolicy

    loader = PromptLoader()
    system_prompt = loader.build_system_prompt("optimize")

    # Build findings JSON — only fields optimizer actually needs
    findings_data = []
    for f in findings.findings:
        if f.status != "accepted":
            continue
        findings_data.append({
            "finding_id": f.finding_id,
            "title": f.title,
            "file_path": f.file_path,
            "function": f.function,
            "line": f.line,
            "description": f.description,
            "suggestion": f.suggestion,
        })

    if not findings_data:
        return []

    findings_json = json.dumps(findings_data, indent=2, ensure_ascii=False)

    user_parts = [
        "## Findings 列表\n",
        findings_json,
        f"\n\nRepo root: {root}",
        "\n\n## 指引",
        "\n- 用 `scanner_read` 的 `start`/`end` 参数只看 finding 指向的行及周围上下文",
        "\n- 先评判每个 finding 是否值得修，再对确认的 finding 生成 patch",
        "\n- 不确定的 finding 直接跳过，不需要解释",
        "\n- 只输出最终 JSON，不要输出其他内容",
    ]
    user_prompt = "\n".join(user_parts)

    # Allow read tools — LLM decides what to read
    policy = ToolPolicy(
        allowed_tools={"scanner_read", "scanner_search", "scanner_files"},
        read_only=True,
        max_calls_per_task=30,
    )

    loop = AgentLoop(provider, registry, policy)
    task = AgentTask(
        run_id=f"optimize-{root.name}",
        task_id=f"optimize-{findings.run_id}",
        task_type="optimize",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_turns=20,
        max_wall_seconds=600,
        max_tokens=16384,
    )

    result = loop.run(task)

    if result.status != "ok":
        errors.append(f"optimize LLM failed: {result.status}")
        errors.extend(result.errors)
        return []

    output = result.output
    if not output:
        errors.append("optimize LLM returned empty output")
        return []

    return _parse_patch_candidates(output)


def _run_trial_optimize_loop(
    findings: LocalizedFindingSet,
    measurement_plan: MeasurementPlan,
    best_measurement: MeasurementResult,
    previous_trials: list[TrialResult],
    root: Path,
    provider,
    registry,
    errors: list[str],
    trial_id: str,
    *,
    knowledge=None,
    max_wall_seconds: int = 0,
    memory_repo: str = "",
) -> tuple[str, str, list[PatchCandidate]]:
    """Ask the LLM for one optimization experiment."""
    from sysight.agent.loop import AgentLoop, AgentTask
    from sysight.agent.prompts.loader import PromptLoader
    from sysight.tools.registry import ToolPolicy

    loader = PromptLoader()
    system_prompt = loader.build_system_prompt("optimize")
    constraints = _derive_environment_constraints(previous_trials)
    memory_root = Path(memory_repo).resolve() if memory_repo else root
    memory_brief = _build_optimizer_memory_brief(memory_root, knowledge)
    user_prompt = _build_trial_user_prompt(
        root,
        measurement_plan,
        best_measurement,
        findings,
        previous_trials,
        constraints=constraints,
        memory_brief=memory_brief,
        compact=False,
    )

    policy = ToolPolicy(
        allowed_tools={"scanner_read", "scanner_search", "scanner_files", "memory_search", "memory_read"},
        read_only=True,
        max_calls_per_task=60,
    )
    loop = AgentLoop(provider, registry, policy)
    task = AgentTask(
        run_id=f"optimize-{root.name}",
        task_id=f"optimize-{findings.run_id}-{trial_id}",
        task_type="optimize",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_turns=30,
        max_wall_seconds=max_wall_seconds,
        max_tokens=16384,
    )
    result = loop.run(task)
    if result.status == "timeout":
        retry_prompt = _build_trial_user_prompt(
            root,
            measurement_plan,
            best_measurement,
            findings,
            previous_trials,
            constraints=constraints,
            memory_brief=memory_brief,
            compact=True,
        )
        retry_task = AgentTask(
            run_id=f"optimize-{root.name}",
            task_id=f"optimize-{findings.run_id}-{trial_id}-retry",
            task_type="optimize",
            system_prompt=system_prompt,
            user_prompt=retry_prompt,
            max_turns=12,
            max_wall_seconds=max_wall_seconds,
            max_tokens=8192,
        )
        result = loop.run(retry_task)

    if result.status != "ok":
        if result.status == "tool_error" and any("max_turns exceeded" in str(e) for e in result.errors):
            forced_prompt = "\n".join([
                user_prompt,
                "",
                "## Finalization",
                "Tool-call budget is exhausted. Do not call any tools.",
                "Return one final JSON object now, using only context already gathered.",
            ])
            no_tool_policy = ToolPolicy(
                allowed_tools=set(),
                read_only=True,
                max_calls_per_task=0,
            )
            no_tool_loop = AgentLoop(provider, registry, no_tool_policy)
            final_task = AgentTask(
                run_id=f"optimize-{root.name}",
                task_id=f"optimize-{findings.run_id}-{trial_id}-final",
                task_type="optimize",
                system_prompt=system_prompt,
                user_prompt=forced_prompt,
                max_turns=2,
                max_wall_seconds=max_wall_seconds,
                max_tokens=4096,
            )
            final_result = no_tool_loop.run(final_task)
            if final_result.status == "ok":
                output = final_result.output or {}
                action = str(output.get("action") or ("patch" if output.get("patches") else "stop"))
                summary = str(output.get("summary") or output.get("rationale") or "")
                return action, summary, _parse_patch_candidates(output)

        errors.append(f"{trial_id}: optimizer LLM failed: {result.status}")
        errors.extend(result.errors)
        return "stop", "optimizer LLM failed", []
    output = result.output or {}
    action = str(output.get("action") or ("patch" if output.get("patches") else "stop"))
    summary = str(output.get("summary") or output.get("rationale") or "")
    return action, summary, _parse_patch_candidates(output)


def _findings_for_optimizer(findings: LocalizedFindingSet) -> list[dict]:
    return [
        {
            "finding_id": f.finding_id,
            "title": f.title,
            "category": f.category,
            "file_path": f.file_path,
            "function": f.function,
            "line": f.line,
            "metric": f.metric,
            "description": f.description,
            "suggestion": f.suggestion,
        }
        for f in findings.findings
        if f.status == "accepted"
    ]


def _trials_for_prompt(previous_trials: list[TrialResult], *, compact: bool) -> list[dict]:
    kept = previous_trials[-6:]
    items: list[dict] = []
    for trial in kept:
        item = {
            "trial_id": trial.trial_id,
            "status": trial.status,
            "patch_ids": trial.patch_ids,
            "finding_ids": trial.finding_ids,
            "metric_before": trial.metric_before,
            "metric_after": trial.metric_after,
            "delta_pct": trial.delta_pct,
            "reason": trial.reason,
            "summary": trial.summary,
        }
        if trial.measurement_errors:
            item["measurement_errors"] = trial.measurement_errors[:3]
        if not compact:
            item["commit_before"] = trial.commit_before
            item["commit_after"] = trial.commit_after
        items.append(item)
    return items


def _derive_environment_constraints(previous_trials: list[TrialResult]) -> list[str]:
    constraints: list[str] = []
    for trial in previous_trials:
        if trial.status != "rejected":
            continue
        joined = " ".join([trial.reason, *trial.measurement_errors]).lower()
        if "triton_key" in joined:
            msg = "torch.compile is currently broken in this environment (triton_key ImportError); avoid compile-based experiments."
            if msg not in constraints:
                constraints.append(msg)
        if "timed out" in joined:
            msg = "Prefer low-risk experiments that keep the measurement command within timeout."
            if msg not in constraints:
                constraints.append(msg)
    return constraints


def _build_optimizer_memory_brief(root: Path, knowledge) -> dict:
    if not knowledge:
        return {}
    try:
        namespace = knowledge.workspace_namespace(repo_root=str(root))
    except Exception:
        namespace = root.name

    brief: dict = {
        "namespace": namespace,
        "read_first": [
            f"workspaces/{namespace}/overview.md",
            f"workspaces/{namespace}/worklog.md",
            "INDEX.md",
        ],
        "search_hints": [
            "Use memory_search for prior accepted/rejected optimization attempts.",
            "Use workspace worklog to avoid repeating failed experiments.",
            "Use experience pages for reusable performance rules.",
        ],
    }
    for key, path in (
        ("overview_excerpt", f"workspaces/{namespace}/overview.md"),
        ("worklog_excerpt", f"workspaces/{namespace}/worklog.md"),
        ("index_excerpt", "INDEX.md"),
    ):
        try:
            text = knowledge.read_page(path) or ""
        except Exception:
            text = ""
        if text:
            brief[key] = text[:2000]
    return brief


def _build_trial_user_prompt(
    root: Path,
    measurement_plan: MeasurementPlan,
    best_measurement: MeasurementResult,
    findings: LocalizedFindingSet,
    previous_trials: list[TrialResult],
    *,
    constraints: list[str],
    memory_brief: dict,
    compact: bool,
) -> str:
    parts = [
        "## Objective",
        "Generate exactly one small optimization experiment for this trial, or stop.",
        "Do not optimize for a single finding. Use findings as clues and optimize the end-to-end metric.",
        "You may freely inspect the repo and wiki. Prefer a focused experiment, but do not stop early if more context is needed.",
        "",
        f"Repo root: {root}",
        "",
        "## Measurement Plan",
        json.dumps(measurement_plan.to_dict(), indent=2, ensure_ascii=False),
        "",
        "## Current Best Measurement",
        json.dumps(best_measurement.to_dict(), indent=2, ensure_ascii=False),
        "",
        "## Findings",
        json.dumps(_findings_for_optimizer(findings), indent=2, ensure_ascii=False),
        "",
        "## Previous Trials",
        json.dumps(_trials_for_prompt(previous_trials, compact=compact), indent=2, ensure_ascii=False),
    ]
    if memory_brief:
        parts.extend([
            "",
            "## Wiki / Memory Context",
            json.dumps(memory_brief, indent=2, ensure_ascii=False),
        ])
    if constraints:
        parts.extend([
            "",
            "## Environment Lessons From Rejected Trials",
            json.dumps(constraints, indent=2, ensure_ascii=False),
        ])
    parts.extend([
        "",
        "## Output JSON",
        "Return one object only:",
        '{"action":"patch","summary":"why this experiment should improve the primary metric","patches":[...]}',
        'or {"action":"stop","summary":"why no useful next experiment remains","patches":[]}.',
        "Patch objects use the existing PatchCandidate schema. Do not compute old_span_hash.",
    ])
    return "\n".join(parts)


def _parse_patch_candidates(output: dict) -> list[PatchCandidate]:
    """Parse LLM output into PatchCandidate list.

    The LLM outputs finding_ids (list), old_span_start/end, replacement.
    old_span_hash is NOT expected from the LLM — it will be filled later.
    """
    patches = []
    for item in output.get("patches", []):
        finding_ids = item.get("finding_ids", [])
        if isinstance(finding_ids, str):
            finding_ids = [finding_ids]

        patches.append(PatchCandidate(
            patch_id=item.get("patch_id", f"patch-{len(patches)+1}"),
            finding_ids=finding_ids,
            file_path=item.get("file_path", ""),
            old_span_start=int(item.get("old_span_start", 0)),
            old_span_end=int(item.get("old_span_end", 0)),
            old_span_hash="",  # filled by _fill_span_hashes
            replacement=item.get("replacement", ""),
            rationale=item.get("rationale", ""),
            validation_commands=item.get("validation_commands", []),
        ))
    return patches


# ── Phase 2: Fill hashes ──

def _fill_span_hashes(
    patches: list[PatchCandidate],
    root: Path,
    errors: list[str],
) -> None:
    """Compute old_span_hash for each patch from actual file content.

    This replaces the LLM-provided hash (which is unreliable) with a
    deterministic code-side computation.
    """
    for patch in patches:
        file_path = root / patch.file_path
        if not file_path.exists():
            errors.append(
                f"hash fill: file not found {patch.file_path} for {patch.patch_id}"
            )
            continue

        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            errors.append(f"hash fill: cannot read {patch.file_path}: {e}")
            continue

        total = len(lines)
        if patch.old_span_start < 1 or patch.old_span_end > total:
            errors.append(
                f"hash fill: invalid span {patch.old_span_start}-{patch.old_span_end} "
                f"in {patch.file_path} (file has {total} lines)"
            )
            continue

        span_lines = lines[patch.old_span_start - 1: patch.old_span_end]
        span_text = "\n".join(span_lines)
        patch.old_span_hash = compute_span_hash(span_text)


# ── Output ──

def _repo_setup_run_command(repo_setup: RepoSetup) -> list[str]:
    if repo_setup.minimal_run:
        return repo_setup.minimal_run[:]
    if repo_setup.entry_point:
        try:
            return shlex.split(repo_setup.entry_point)
        except ValueError:
            return repo_setup.entry_point.split()
    return []


def _fallback_measurement_plan(
    plan: MeasurementPlan,
    repo_setup: RepoSetup,
) -> MeasurementPlan | None:
    fallback_cmd = _repo_setup_run_command(repo_setup)
    if not fallback_cmd or fallback_cmd == plan.run_command:
        return None
    data = plan.to_dict()
    data["run_command"] = fallback_cmd
    if repo_setup.env_vars:
        env = dict(data.get("env_vars", {}) or {})
        env.update(repo_setup.env_vars)
        data["env_vars"] = env
    fallback = MeasurementPlan.from_dict(data)
    return fallback if fallback.is_valid() else None


def _trial_worktree_root(run_dir: Path) -> Path:
    path = run_dir / "worktrees"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_trial_worktree(root: Path, run_id: str, worktree_root: Path | None = None) -> Path:
    base = _run_git(root, ["rev-parse", "HEAD"]).strip()
    if not base:
        raise RuntimeError(f"not a git repository or no HEAD commit: {root}")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in (run_id or "run"))
    suffix = str(int(time.time() * 1000))
    branch = f"sysight/opt-{safe}-{suffix}"
    worktree_base = worktree_root or root.parent
    worktree_base.mkdir(parents=True, exist_ok=True)
    worktree = worktree_base / f".sysight-opt-{safe}-{suffix}"
    _run_git(root, ["worktree", "add", "-B", branch, str(worktree), base])
    return worktree.resolve()


def _run_git(repo: Path, args: list[str]) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return proc.stdout


def _git_rev_parse(repo: Path) -> str:
    return _run_git(repo, ["rev-parse", "HEAD"]).strip()


def _git_reset_to(repo: Path, commit: str) -> None:
    _run_git(repo, ["reset", "--hard", commit])
    _run_git(repo, ["clean", "-fd"])


def _apply_trial_patches(applier, patches: list[PatchCandidate], root: Path) -> str:
    for patch in sorted(patches, key=lambda p: (p.file_path, -p.old_span_start)):
        _normalize_patch_indentation(root, patch)
        if not patch.old_span_hash:
            return f"missing old_span_hash for {patch.patch_id}"
        ok, msg = applier.apply(
            file_path=patch.file_path,
            old_span_start=patch.old_span_start,
            old_span_end=patch.old_span_end,
            old_span_hash=patch.old_span_hash,
            replacement=patch.replacement,
        )
        if not ok:
            return f"apply_failed:{patch.patch_id}: {msg}"
    return ""


def _normalize_patch_indentation(root: Path, patch: PatchCandidate) -> None:
    """Best-effort indentation repair for Python replacements in indented scopes."""
    if not patch.file_path.endswith(".py"):
        return
    if not patch.replacement.strip():
        return

    file_path = root / patch.file_path
    try:
        src_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    if patch.old_span_start < 1 or patch.old_span_start > len(src_lines):
        return

    anchor = src_lines[patch.old_span_start - 1]
    indent = anchor[: len(anchor) - len(anchor.lstrip(" \t"))]
    import textwrap

    rebased = textwrap.dedent(patch.replacement).splitlines()
    patch.replacement = "\n".join((indent + line) if line.strip() else line for line in rebased)


def _run_trial_validation(root: Path, patches: list[PatchCandidate], repo_setup: RepoSetup) -> str:
    commands: list[list[str]] = []
    for patch in patches:
        commands.extend(_sanitize_validation_commands(patch.validation_commands))
    if not commands:
        commands.extend(_sanitize_validation_commands(repo_setup.test_commands))
    if not commands:
        commands.extend(_compile_commands_for_patches(patches))

    for cmd in commands:
        if not cmd:
            continue
        normalized = _normalize_python_command(cmd)
        try:
            proc = subprocess.run(
                normalized,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return f"validation_failed:{' '.join(normalized)}:{e}"
        if proc.returncode != 0:
            return f"validation_failed:{' '.join(normalized)}:{proc.stderr[:300]}"
    return ""


def _sanitize_validation_commands(commands: list[list[str]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for cmd in commands:
        if not cmd:
            continue
        head = str(cmd[0]).strip()
        if not head or head.startswith("-"):
            continue
        cleaned.append([str(x) for x in cmd])
    return cleaned


def _run_repo_setup_bootstrap(root: Path, repo_setup: RepoSetup) -> str:
    commands = [cmd for cmd in repo_setup.build_commands if cmd]
    for cmd in commands:
        normalized = _normalize_python_command(cmd)
        try:
            proc = subprocess.run(
                normalized,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return f"bootstrap_failed:{' '.join(normalized)}:{e}"
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            return f"bootstrap_failed:{' '.join(normalized)}:{stderr[:300]}"
    return ""


def _compile_commands_for_patches(patches: list[PatchCandidate]) -> list[list[str]]:
    files = sorted({p.file_path for p in patches if p.file_path.endswith(".py")})
    return [["python", "-m", "py_compile", fp] for fp in files]


def _normalize_python_command(cmd: list[str]) -> list[str]:
    if cmd and cmd[0] == "python":
        import shutil
        if shutil.which("python") is None and shutil.which("python3") is not None:
            return ["python3"] + cmd[1:]
    return cmd


def _commit_trial(
    root: Path,
    trial_id: str,
    patches: list[PatchCandidate],
    delta_pct: float | None,
) -> str:
    files = sorted({p.file_path for p in patches})
    _run_git(root, ["add", "--", *files])
    delta = 0.0 if delta_pct is None else delta_pct
    msg = f"sysight: accept {trial_id} {delta:+.3f}%"
    proc = subprocess.run(
        [
            "git", "-C", str(root),
            "-c", "user.name=Sysight",
            "-c", "user.email=sysight@example.invalid",
            "commit", "-m", msg,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git commit failed: {(proc.stderr or proc.stdout).strip()}")
    return _git_rev_parse(root)


def _append_trial_logs(run_dir: Path, trial: TrialResult) -> None:
    item = trial.to_dict()
    with (run_dir / "trials.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    lines = [
        f"## {trial.trial_id} - {trial.status}",
        "",
        f"- summary: {trial.summary}",
        f"- reason: {trial.reason}",
        f"- measurement_errors: {trial.measurement_errors}",
        f"- metric: {trial.primary_metric}",
        f"- before: {trial.metric_before}",
        f"- after: {trial.metric_after}",
        f"- delta_pct: {trial.delta_pct}",
        f"- commit_before: {trial.commit_before}",
        f"- commit_after: {trial.commit_after}",
        "",
    ]
    with (run_dir / "worklog.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        from sysight.pipeline.worklog import WorklogRow, append_many, default_worklog_path
        row = WorklogRow(
            loop_id=run_dir.parent.name if run_dir.parent.name.startswith("loop-") else "",
            iteration=0,
            stage="optimize_trial",
            status=trial.status,
            run_id=run_dir.name,
            artifact_dir=str(run_dir),
            commit_before=trial.commit_before,
            commit_after=trial.commit_after,
            metric_name=trial.primary_metric,
            metric_before=trial.metric_before,
            metric_after=trial.metric_after,
            metric_delta_pct=trial.delta_pct,
            accepted_count=1 if trial.status == "accepted" else 0,
            rejected_count=1 if trial.status == "rejected" else 0,
            coding_summary=trial.summary,
            reason=trial.reason,
            errors=trial.measurement_errors,
        )
        targets = [run_dir / "worklog.csv"]
        try:
            sysight_root = (Path.cwd() / ".sysight").resolve()
            if str(run_dir.resolve()).startswith(str(sysight_root)):
                targets.append(default_worklog_path())
        except OSError:
            pass
        append_many(targets, row)
    except Exception:
        pass


def _write_loop_result_json(result: OptimizeLoopResult, run_dir: Path) -> None:
    data = {
        "run_id": result.run_id,
        "worktree_path": result.worktree_path,
        "best_commit": result.best_commit,
        "accepted_count": result.accepted_count,
        "rejected_count": result.rejected_count,
        "baseline": result.baseline.to_dict(),
        "best_measurement": result.best_measurement.to_dict(),
        "trials": [t.to_dict() for t in result.trials],
        "errors": result.errors,
    }
    (run_dir / "optimize_loop_result.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _write_patches_json(
    patches: list[PatchCandidate],
    run_id: str,
    run_dir: Path,
    errors: list[str],
    elapsed_ms: float = 0,
) -> None:
    """Write optimize_result.json to the run directory."""
    data = {
        "run_id": run_id,
        "elapsed_ms": elapsed_ms,
        "patches": [
            {
                "patch_id": p.patch_id,
                "finding_ids": p.finding_ids,
                "file_path": p.file_path,
                "old_span_start": p.old_span_start,
                "old_span_end": p.old_span_end,
                "old_span_hash": p.old_span_hash,
                "replacement": p.replacement,
                "rationale": p.rationale,
                "validation_commands": p.validation_commands,
            }
            for p in patches
        ],
        "errors": errors,
    }
    (run_dir / "optimize_result.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
