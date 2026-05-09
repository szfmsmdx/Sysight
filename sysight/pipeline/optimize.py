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
import sys
import time
from pathlib import Path

from sysight.types.findings import LocalizedFindingSet
from sysight.types.optimization import PatchCandidate, compute_span_hash


# ── Main entry point ──

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
        run_dir = Path.cwd() / ".sysight" / "optimizer-runs" / opt_run_id
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
