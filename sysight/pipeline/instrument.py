"""INSTRUMENT stage — LLM-driven cuda_timer placement based on analyzer findings.

Runs AFTER analyze. Takes the FindingSet and:
1. Sends findings + source files to LLM for timer placement decisions.
2. LLM reads source files, determines precise wrap_start/wrap_end for each finding.
3. Programmatically inserts cuda_timer calls into source files.
4. Writes instrument_result.json for the optimizer verify step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from sysight.utils.cuda_timer import (
    CUDA_TIMER_IMPORT_LINE,
    CUDA_TIMER_MODULE_CONTENT,
    CUDA_TIMER_MODULE_NAME,
)
from sysight.types.findings import LocalizedFindingSet


@dataclass
class TimerSpec:
    """A single timer placement specification."""
    finding_id: str
    timer_label: str
    file: str
    wrap_start: int      # 1-based
    wrap_end: int        # 1-based, inclusive
    reason: str = ""


@dataclass
class InstrumentResult:
    run_id: str = ""
    timers: list[TimerSpec] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


_MARKER_BEGIN = "# ── SYSIGHT_TIMER_BEGIN:{label} ──"
_MARKER_END   = "# ── SYSIGHT_TIMER_END:{label} ──"


def run_instrument(
    findings: LocalizedFindingSet,
    repo: str,
    *,
    provider=None,
    registry=None,
    verbose: bool = False,
    run_dir: Path | None = None,
) -> InstrumentResult:
    """Run LLM-driven instrumentation based on analyzer findings.

    Args:
        findings: The FindingSet from the analyze stage.
        repo: Path to repo root.
        provider: LLM provider (required for LLM-driven mode).
        registry: ToolRegistry instance (required for LLM-driven mode).
        verbose: Print LLM I/O to terminal.
        run_dir: Output directory for instrument_result.json.
    """
    root = Path(repo).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        errors.append(f"repo 不是目录: {root}")
        return InstrumentResult(errors=errors)

    if not findings.findings:
        return InstrumentResult(
            run_id=findings.run_id,
            warnings=["no findings to instrument — skipping"],
            summary={"status": "skipped", "reason": "no findings"},
        )

    # ── LLM-driven timer inference ──
    timer_specs = _llm_infer_timer_specs(
        findings, root, provider, registry, warnings, verbose,
    )

    if not timer_specs:
        errors.append("LLM inferred no timer specs")
        return InstrumentResult(
            run_id=findings.run_id, errors=errors,
            summary={"status": "failed", "reason": "no timers generated"},
        )

    # ── Insert timers into source files ──
    modified_files = _insert_timers(root, timer_specs, warnings)

    result = InstrumentResult(
        run_id=findings.run_id,
        timers=timer_specs,
        modified_files=modified_files,
        errors=errors,
        warnings=warnings,
        summary={
            "status": "ok",
            "method": "llm",
            "timer_count": len(timer_specs),
            "modified_files": modified_files,
        },
    )

    if run_dir is not None:
        _write_result_json(result, run_dir)

    return result


# ── LLM-driven timer inference ───────────────────────────────────────────────


def _llm_infer_timer_specs(
    findings: LocalizedFindingSet,
    root: Path,
    provider,
    registry,
    warnings: list[str],
    verbose: bool = False,
) -> list[TimerSpec]:
    """Use LLM to read source files and determine timer placement.

    The LLM has access to scanner_read so it can:
    - Read source files referenced by findings
    - Determine precise wrap_start/wrap_end for each finding
    - Handle overlapping ranges by merging them
    """
    if provider is None or registry is None:
        warnings.append("no LLM provider/registry — cannot infer timer specs")
        return []

    from sysight.agent.loop import AgentLoop, AgentTask
    from sysight.agent.prompts.loader import PromptLoader
    from sysight.tools.registry import ToolPolicy
    from sysight.benchmark.debug import DebugProvider

    loader = PromptLoader()
    system_prompt = loader.build_system_prompt("instrument")

    # Build findings JSON — only fields instrument LLM needs
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
        })

    if not findings_data:
        return []

    findings_json = json.dumps(findings_data, indent=2, ensure_ascii=False)

    user_parts = [
        "## Analyzer Findings\n",
        findings_json,
        f"\n\nRepo root: {root}",
        "\n\n## 指引",
        "\n- 用 `scanner_read` 的 `start`/`end` 参数只看 finding 指向的行及周围上下文",
        "\n- 为每个 finding 确定精确的 wrap_start/wrap_end",
        "\n- 重叠的 timer 范围必须合并（用 `+` 连接 timer_label）",
        "\n- 只输出最终 JSON，不要输出其他内容",
    ]
    user_prompt = "\n".join(user_parts)

    # Allow read tools — LLM decides what to read
    policy = ToolPolicy(
        allowed_tools={"scanner_read", "scanner_search", "scanner_files"},
        read_only=True,
        max_calls_per_task=30,
    )

    wrapped_provider = DebugProvider(provider, [], verbose=verbose)

    loop = AgentLoop(wrapped_provider, registry, policy)
    task = AgentTask(
        run_id=f"instrument-{root.name}",
        task_id=f"instrument-{findings.run_id}",
        task_type="instrument",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_turns=10,
        max_wall_seconds=300,
        max_tokens=8192,
    )

    result = loop.run(task)

    if result.status != "ok":
        warnings.append(f"instrument LLM failed: {result.status}")
        return []

    output = result.output
    if not output:
        warnings.append("instrument LLM returned empty output")
        return []

    return _parse_timer_specs(output)


def _parse_timer_specs(output: dict) -> list[TimerSpec]:
    """Parse LLM output into TimerSpec list."""
    specs = []
    for item in output.get("timers", []):
        specs.append(TimerSpec(
            finding_id=item.get("finding_id", ""),
            timer_label=item.get("timer_label", ""),
            file=item.get("file", ""),
            wrap_start=int(item.get("wrap_start", 0)),
            wrap_end=int(item.get("wrap_end", 0)),
            reason=item.get("reason", ""),
        ))
    return specs


# ── Timer insertion ──────────────────────────────────────────────────────────


def _insert_timers(root: Path, specs: list[TimerSpec], warnings: list[str]) -> list[str]:
    """Write _sysight_timer.py once, then inject import + with-blocks per file."""
    _ensure_timer_module(root, warnings)

    file_specs: dict[str, list[TimerSpec]] = {}
    for spec in specs:
        file_specs.setdefault(spec.file, []).append(spec)

    modified: list[str] = []

    for file_rel, file_timer_specs in file_specs.items():
        file_path = root / file_rel
        try:
            original = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            warnings.append(f"cannot read {file_rel}: {e}")
            continue

        pending: list[TimerSpec] = []
        for spec in file_timer_specs:
            if _MARKER_BEGIN.format(label=spec.timer_label) in original:
                warnings.append(f"__skip__:{spec.timer_label} ({file_rel})")
            else:
                pending.append(spec)

        if not pending:
            continue

        new_content = _apply_timer_insertions(original, pending, warnings)
        if new_content is None:
            continue

        if f"from {CUDA_TIMER_MODULE_NAME} import cuda_timer" not in new_content:
            new_content = _insert_import(new_content, CUDA_TIMER_IMPORT_LINE)

        try:
            file_path.write_text(new_content, encoding="utf-8")
            modified.append(file_rel)
        except OSError as e:
            warnings.append(f"cannot write {file_rel}: {e}")

    return modified


def _ensure_timer_module(root: Path, warnings: list[str]) -> None:
    module_path = root / f"{CUDA_TIMER_MODULE_NAME}.py"
    if module_path.exists():
        if module_path.read_text(encoding="utf-8", errors="replace").strip() == CUDA_TIMER_MODULE_CONTENT.strip():
            return
        warnings.append(f"{CUDA_TIMER_MODULE_NAME}.py exists but differs — overwriting")
    try:
        module_path.write_text(CUDA_TIMER_MODULE_CONTENT, encoding="utf-8")
    except OSError as e:
        warnings.append(f"cannot write {CUDA_TIMER_MODULE_NAME}.py: {e}")


def _insert_import(source: str, import_line: str) -> str:
    """Insert import_line after any module docstring and __future__ imports."""
    lines = source.splitlines(keepends=True)
    i = 0

    # Skip blank lines / comments
    while i < len(lines) and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
        i += 1

    # Skip module docstring
    if i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith(('"""', "'''", '"', "'")):
            quote = stripped[:3] if stripped[:3] in ('"""', "'''") else stripped[0]
            rest = stripped[len(quote):]
            i += 1
            if quote not in rest:  # multi-line docstring
                while i < len(lines) and quote not in lines[i]:
                    i += 1
                i += 1

    # Skip __future__ imports and blank lines
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("from __future__") or s == "":
            i += 1
        else:
            break

    lines.insert(i, import_line + "\n")
    return "".join(lines)


def _apply_timer_insertions(
    original: str,
    specs: list[TimerSpec],
    warnings: list[str],
) -> str | None:
    """Wrap each spec's line range with ``with cuda_timer(label)():``.

    Processes specs bottom-up so earlier insertions don't shift later indices.
    Preserves relative indentation within the wrapped block.
    """
    lines = original.splitlines()
    total = len(lines)

    for spec in sorted(specs, key=lambda s: s.wrap_start, reverse=True):
        start, end = spec.wrap_start, spec.wrap_end
        if start < 1 or end > total or start > end:
            warnings.append(f"skipping {spec.timer_label}: invalid range {start}-{end}")
            continue

        # Detect base indentation from the first wrapped line
        raw_first = lines[start - 1]
        indent = raw_first[:len(raw_first) - len(raw_first.lstrip())] if raw_first.strip() else "    "
        inner = indent + "    "

        new_lines = [
            f"{indent}{_MARKER_BEGIN.format(label=spec.timer_label)}",
            f'{indent}with cuda_timer("{spec.timer_label}")():',
        ]
        for old in lines[start - 1:end]:
            if not old.strip():
                new_lines.append("")
            elif old.startswith(indent):
                new_lines.append(inner + old[len(indent):])
            else:
                new_lines.append(inner + old.lstrip())
        new_lines.append(f"{indent}{_MARKER_END.format(label=spec.timer_label)}")

        lines[start - 1:end] = new_lines

    return "\n".join(lines)


# ── Result JSON ──────────────────────────────────────────────────────────────


def _write_result_json(result: InstrumentResult, run_dir: Path) -> None:
    data = {
        "run_id": result.run_id,
        "status": result.summary.get("status", "unknown"),
        "timers": [
            {
                "finding_id": t.finding_id,
                "timer_label": t.timer_label,
                "file": t.file,
                "wrap_start": t.wrap_start,
                "wrap_end": t.wrap_end,
                "reason": t.reason,
            }
            for t in result.timers
        ],
        "modified_files": result.modified_files,
        "warnings": result.warnings,
        "errors": result.errors,
        "verify_hint": (
            "Run the program and grep stdout for [SYSIGHT_TIMER] lines. "
            "Each line reports: <label>: <elapsed_ms> ms."
        ),
    }
    (run_dir / "instrument_result.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )