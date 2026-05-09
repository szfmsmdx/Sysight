"""INSTRUMENT stage — targeted cuda_timer insertion driven by analyzer findings.

Runs AFTER analyze. Takes the FindingSet and:
1. Uses AST analysis to determine timer placement for each finding.
2. Programmatically inserts cuda_timer calls into source files.
3. Writes instrument_result.json for the optimizer verify step.
"""

from __future__ import annotations

import ast
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
    verbose: bool = False,
    run_dir: Path | None = None,
) -> InstrumentResult:
    """Run targeted instrumentation based on analyzer findings (AST path)."""
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

    timer_specs = infer_timer_specs(findings, root, warnings)

    if not timer_specs:
        errors.append("AST inferred no timer specs")
        return InstrumentResult(
            run_id=findings.run_id, errors=errors,
            summary={"status": "failed", "reason": "no timers generated"},
        )

    modified_files = _insert_timers(root, timer_specs, warnings)

    result = InstrumentResult(
        run_id=findings.run_id,
        timers=timer_specs,
        modified_files=modified_files,
        errors=errors,
        warnings=warnings,
        summary={
            "status": "ok",
            "method": "ast",
            "timer_count": len(timer_specs),
            "modified_files": modified_files,
        },
    )

    if run_dir is not None:
        _write_result_json(result, run_dir)

    return result


# ── AST-driven timer inference ──────────────────────────────────────────────


def infer_timer_specs(
    findings: LocalizedFindingSet,
    root: Path,
    warnings: list[str],
) -> list[TimerSpec]:
    """Derive TimerSpec list from findings using AST analysis.

    Strategy per category:
      - C1 (DataLoader config): find the collate_fn body, then largest For loop,
        then enclosing function body.
      - C2 / C7 (loops): wrap the innermost enclosing For/AsyncFor statement.
      - Others: wrap the smallest statement containing finding.line.
    """
    by_file: dict[str, list] = {}
    for f in findings.findings:
        if f.file_path and f.line:
            by_file.setdefault(f.file_path, []).append(f)

    specs: list[TimerSpec] = []
    seq = 1

    for file_rel, file_findings in by_file.items():
        file_path = root / file_rel
        if not file_path.exists():
            warnings.append(f"AST: file not found: {file_rel}")
            continue
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            warnings.append(f"AST: cannot parse {file_rel}: {exc}")
            continue

        for finding in file_findings:
            spec = _infer_one(tree, finding, file_rel, seq, warnings)
            if spec is not None:
                specs.append(spec)
                seq += 1

    return specs


def _infer_one(
    tree: ast.AST,
    finding,
    file_rel: str,
    seq: int,
    warnings: list[str],
) -> TimerSpec | None:
    line = finding.line
    category = finding.category
    label = f"F{seq:02d}_{_slug(finding.title or finding.finding_id)}"

    if category == "C1":
        span = _c1_span(tree, line, warnings, file_rel)
        if span is None:
            warnings.append(
                f"AST: C1 finding {finding.finding_id} — no For loop found "
                f"near line {line} in {file_rel}, skipping"
            )
            return None
        start, end = span
        return TimerSpec(
            finding_id=finding.finding_id, timer_label=label, file=file_rel,
            wrap_start=start, wrap_end=end,
            reason=f"C1: DataLoader config at line {line}; timing enclosing For loop",
        )

    if category in ("C2", "C7"):
        node = _find_enclosing_for(tree, line)
        if node is not None:
            end = node.end_lineno  # type: ignore[attr-defined]
            return TimerSpec(
                finding_id=finding.finding_id, timer_label=label, file=file_rel,
                wrap_start=node.lineno, wrap_end=end,
                reason=f"{category}: loop at lines {node.lineno}-{end}",
            )

    stmt = _find_stmt_at_line(tree, line)
    if stmt is None:
        warnings.append(
            f"AST: no statement found at line {line} in {file_rel} "
            f"(finding {finding.finding_id}), skipping"
        )
        return None

    start = stmt.lineno
    end = getattr(stmt, "end_lineno", stmt.lineno)
    return TimerSpec(
        finding_id=finding.finding_id, timer_label=label, file=file_rel,
        wrap_start=start, wrap_end=end,
        reason=f"{category}: statement at lines {start}-{end}",
    )


def _slug(text: str, max_len: int = 30) -> str:
    return re.sub(r"[^\w]+", "_", text.lower()).strip("_")[:max_len]


def _find_enclosing_for(tree: ast.AST, line: int) -> ast.For | ast.AsyncFor | None:
    """Return the innermost For/AsyncFor whose range contains *line*."""
    best: ast.For | ast.AsyncFor | None = None
    best_size = float("inf")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if node.lineno <= line <= end:
            size = end - node.lineno
            if size < best_size:
                best, best_size = node, size  # type: ignore[assignment]
    return best


def _find_stmt_at_line(tree: ast.AST, line: int) -> ast.stmt | None:
    """Return the smallest non-compound statement whose range contains *line*.

    Compound nodes (For/If/With/…) are used only as fallback when no
    non-compound statement matches.
    """
    _COMPOUND = (
        ast.For, ast.AsyncFor, ast.If, ast.While, ast.With,
        ast.AsyncWith, ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef, ast.Try,
    )
    best_non_compound: ast.stmt | None = None
    best_non_compound_size = float("inf")
    best_compound: ast.stmt | None = None
    best_compound_size = float("inf")
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if not (node.lineno <= line <= end):
            continue
        size = end - node.lineno
        if isinstance(node, _COMPOUND):
            if size < best_compound_size:
                best_compound, best_compound_size = node, size
        else:
            if size < best_non_compound_size:
                best_non_compound, best_non_compound_size = node, size
    return best_non_compound if best_non_compound is not None else best_compound


# ── C1 (DataLoader) span helpers ────────────────────────────────────────────


def _c1_span(
    tree: ast.AST, config_line: int, warnings: list[str], file_rel: str
) -> tuple[int, int] | None:
    """Three-level fallback for C1 DataLoader findings.

    1. collate_fn function body.
    2. Largest For loop in the file.
    3. Body of the enclosing function.
    """
    span = _find_collate_fn_span(tree)
    if span:
        return span

    best: ast.For | ast.AsyncFor | None = None
    best_size = -1
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        size = end - node.lineno
        if size > best_size:
            best, best_size = node, size  # type: ignore[assignment]
    if best is not None:
        return (best.lineno, getattr(best, "end_lineno", best.lineno))

    enclosing: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    enc_size = float("inf")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        size = end - node.lineno
        if node.lineno <= config_line <= end and size < enc_size:
            enclosing, enc_size = node, size  # type: ignore[assignment]

    if enclosing and enclosing.body:
        body = enclosing.body
        return (body[0].lineno, getattr(body[-1], "end_lineno", body[-1].lineno))
    return None


def _find_collate_fn_span(tree: ast.AST) -> tuple[int, int] | None:
    """Find DataLoader(..., collate_fn=<name>) and return the referenced function's body span."""
    local_funcs: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body:
                local_funcs[node.name] = (
                    body[0].lineno,
                    getattr(body[-1], "end_lineno", body[-1].lineno),
                )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            (isinstance(func, ast.Name) and func.id == "DataLoader")
            or (isinstance(func, ast.Attribute) and func.attr == "DataLoader")
        ):
            continue
        for kw in node.keywords:
            if kw.arg != "collate_fn":
                continue
            val = kw.value
            if isinstance(val, ast.Name) and val.id in local_funcs:
                return local_funcs[val.id]
            if isinstance(val, ast.Lambda):
                lbody = val.body
                called: str | None = None
                if isinstance(lbody, ast.Call):
                    cf = lbody.func
                    called = cf.id if isinstance(cf, ast.Name) else (
                        cf.attr if isinstance(cf, ast.Attribute) else None
                    )
                if called and called in local_funcs:
                    return local_funcs[called]
                start = lbody.lineno
                return (start, getattr(lbody, "end_lineno", start))
    return None


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

        pending = _merge_overlapping_specs(pending, warnings)
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


def _merge_overlapping_specs(specs: list[TimerSpec], warnings: list[str]) -> list[TimerSpec]:
    """Merge overlapping [wrap_start, wrap_end] ranges into a single spec."""
    if not specs:
        return specs

    sorted_specs = sorted(specs, key=lambda s: (s.wrap_start, -s.wrap_end))
    merged: list[TimerSpec] = []
    cur = sorted_specs[0]

    for nxt in sorted_specs[1:]:
        if nxt.wrap_start <= cur.wrap_end:
            combined = cur.timer_label + "+" + nxt.timer_label
            warnings.append(f"__merge__:{cur.timer_label}|{nxt.timer_label}|{combined}")
            cur = TimerSpec(
                finding_id=cur.finding_id,
                timer_label=combined,
                file=cur.file,
                wrap_start=min(cur.wrap_start, nxt.wrap_start),
                wrap_end=max(cur.wrap_end, nxt.wrap_end),
                reason=cur.reason + " | " + nxt.reason,
            )
        else:
            merged.append(cur)
            cur = nxt

    merged.append(cur)
    return merged


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
