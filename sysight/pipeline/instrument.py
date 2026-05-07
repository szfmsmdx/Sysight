"""INSTRUMENT stage — targeted NVTX tagging driven by analyzer findings.

Runs AFTER analyze. Takes the FindingSet and determines where to add
NVTX markers to precisely measure the hotspots identified by the analyzer.
Also validates the smoke test, discovers metrics, and checks profile files.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sysight.pipeline.warmup import _warmup_cache_path
from sysight.types.repo_setup import RepoSetup
from sysight.types.findings import LocalizedFindingSet


@dataclass
class InstrumentResult:
    repo_setup: RepoSetup = field(default_factory=RepoSetup)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run_instrument(
    findings: LocalizedFindingSet,
    repo: str,
    registry,
    provider,
    knowledge=None,
    repo_setup: RepoSetup | None = None,
    *,
    verbose: bool = False,
    run_dir: Path | None = None,
) -> InstrumentResult:
    """Run targeted instrumentation based on analyzer findings.

    Args:
        findings: The FindingSet from the analyze stage.
        repo: Path to repo root.
        registry: ToolRegistry instance.
        provider: LLM provider.
        knowledge: WikiRepository instance (optional).
        repo_setup: Existing RepoSetup from warmup Phase 1 (optional).
        verbose: Print LLM I/O to terminal.
        run_dir: Output directory (analysis-runs/<run_id>/) to co-locate logs.
    """
    root = Path(repo).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if repo_setup is None:
        repo_setup = RepoSetup(source="warmup_partial")

    if not root.is_dir():
        errors.append(f"repo 不是目录: {root}")
        return InstrumentResult(repo_setup=repo_setup, errors=errors)

    if not findings.findings:
        warnings.append("no findings to instrument — skipping")
        return InstrumentResult(
            repo_setup=repo_setup,
            warnings=warnings,
            summary={"status": "skipped", "reason": "no findings"},
        )

    # Run LLM-guided instrumentation
    try:
        _run_llm_instrument(
            root, registry, provider, knowledge,
            repo_setup, findings, warnings,
            verbose=verbose,
            run_dir=run_dir,
        )
    except Exception as e:
        errors.append(f"instrument LLM failed: {e}")
        repo_setup.warmup_llm_errors.append(str(e))

    # Persist updated cache
    cache_path = _warmup_cache_path(root)
    try:
        repo_setup.save_cache(cache_path)
        print(f"  [instrument] cache saved: {cache_path}")
    except Exception as e:
        warnings.append(f"instrument cache save failed: {e}")

    summary = {
        "repo": str(root),
        "source": repo_setup.source,
        "entry_point": repo_setup.entry_point,
        "smoke_test_passed": repo_setup.smoke_test_passed,
        "has_nvtx": repo_setup.has_nvtx,
        "suggested_tags": len(repo_setup.suggested_instrumentation),
        "human_action_items": repo_setup.human_action_items,
        "warnings": warnings,
        "errors": errors,
    }

    return InstrumentResult(
        repo_setup=repo_setup,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


# ── LLM Instrumentation ──

def _run_llm_instrument(
    root: Path,
    registry,
    provider,
    knowledge,
    repo_setup: RepoSetup,
    findings: LocalizedFindingSet,
    warnings: list[str],
    *,
    verbose: bool = False,
    run_dir: Path | None = None,
) -> None:
    """Run LLM to determine targeted instrumentation based on findings."""
    from sysight.agent.loop import AgentLoop, AgentTask
    from sysight.agent.prompts.loader import PromptLoader
    from sysight.benchmark.debug import DebugProvider
    from sysight.tools.registry import ToolPolicy

    debug_log: list[dict] = []
    if run_dir is not None:
        log_file = str(run_dir / "instrument_debug.log")
    else:
        log_file = str(Path.cwd() / ".sysight" / "instrument_debug.log")
    wrapped_provider = DebugProvider(provider, debug_log, verbose=verbose, log_file=log_file)

    # Build findings JSON
    findings_json = _build_findings_json(findings)

    # Build memory brief
    memory_brief = ""
    if knowledge:
        try:
            ns = knowledge.workspace_namespace(repo_root=str(root))
            memory_brief = knowledge.build_brief(
                f"workspaces/{ns}/overview.md",
                max_tokens=800,
            )
        except Exception:
            pass

    loader = PromptLoader()
    system_prompt = loader.build_system_prompt("instrument")
    user_prompt = loader.build_user_prompt(
        "instrument",
        findings_json=findings_json,
        memory_brief=memory_brief,
    )
    user_prompt += f"\n\nRepo root: {root}"

    policy = ToolPolicy(
        allowed_tools={"scanner_*", "shell_exec", "memory_search", "memory_read"},
        read_only=False,
        max_calls_per_task=30,
        max_wall_seconds=300,
    )

    loop = AgentLoop(wrapped_provider, registry, policy)
    task = AgentTask(
        run_id=f"instrument-{root.name}",
        task_id=f"instrument-{root.name}",
        task_type="instrument",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_wall_seconds=300,
    )

    print(f"  [instrument] LLM analyzing {len(findings.findings)} findings for targeted tagging...", file=sys.stderr)
    if verbose:
        print(f"  [instrument] debug log: {log_file}", file=sys.stderr)
    result = loop.run(task)

    if result.status != "ok":
        warnings.append(f"instrument LLM failed: {result.status}")
        repo_setup.warmup_llm_errors.extend(result.errors)
        return

    output = result.output
    if not output:
        warnings.append("instrument LLM returned empty output")
        return

    repo_setup.warmup_raw_output = output
    _apply_instrument_output(repo_setup, output, warnings)


def _build_findings_json(findings: LocalizedFindingSet) -> str:
    """Serialize findings as JSON for the LLM."""
    data = {
        "summary": findings.summary,
        "total_findings": len(findings.findings),
        "findings": [
            {
                "finding_id": f.finding_id,
                "category": f.category,
                "title": f.title,
                "priority": f.priority,
                "file_path": f.file_path,
                "function": f.function,
                "line": f.line,
                "description": f.description,
                "suggestion": f.suggestion,
            }
            for f in findings.findings
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _apply_instrument_output(repo_setup: RepoSetup, output: dict, warnings: list[str]) -> None:
    """Apply instrument LLM output to RepoSetup."""

    def _apply(section: str, *fields: tuple):
        data = output.get(section, {})
        if not data:
            return
        for attr, key, cast, default in fields:
            setattr(repo_setup, attr, cast(data.get(key, default)))

    _apply("environment",
        ("python_version", "python_version", str, ""),
        ("python_bin", "python_bin", str, "python"),
        ("gpu_available", "gpu_available", bool, False),
        ("nsys_available", "nsys_available", bool, False),
    )
    if output.get("environment", {}).get("key_packages"):
        repo_setup.key_packages = {str(k): str(v) for k, v in output["environment"]["key_packages"].items()}

    _apply("smoke_test",
        ("smoke_test_passed", "passed", bool, False),
        ("smoke_test_command", "command", str, ""),
        ("smoke_test_exit_code", "exit_code", int, -1),
        ("smoke_test_stdout_tail", "stdout_tail", str, ""),
        ("smoke_test_stderr_tail", "stderr_tail", str, ""),
        ("smoke_test_elapsed_ms", "elapsed_ms", float, 0),
        ("smoke_test_notes", "notes", str, ""),
    )

    instr = output.get("instrumentation", {})
    if instr:
        repo_setup.has_nvtx = bool(instr.get("has_nvtx", False))
        repo_setup.has_custom_timer = bool(instr.get("has_custom_timer", False))
        repo_setup.existing_nvtx_tags = instr.get("existing_tags", [])
        repo_setup.suggested_instrumentation = instr.get("suggested_insertions", [])
        repo_setup.needs_instrumentation = (
            not repo_setup.has_nvtx and len(repo_setup.suggested_instrumentation) > 0
        )

    _apply("profile",
        ("profile_sqlite", "sqlite_path", str, repo_setup.profile_sqlite),
        ("profile_sqlite_valid", "sqlite_valid", bool, False),
        ("profile_nsys_rep", "nsys_rep_path", str, ""),
        ("profile_nsys_rep_valid", "nsys_rep_valid", bool, False),
        ("profile_notes", "notes", str, ""),
    )

    repo_setup.warmup_summary = str(output.get("summary", ""))
    hai = output.get("human_action_items", [])
    if isinstance(hai, list):
        repo_setup.human_action_items = [str(item) for item in hai]

    if repo_setup.smoke_test_passed:
        repo_setup.source = "warmup_verified"
        from datetime import datetime, timezone
        repo_setup.verified_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
