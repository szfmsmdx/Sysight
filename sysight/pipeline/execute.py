"""EXECUTE stage — apply patches, smoke test, timer comparison.

Architecture:
  Phase 0: Baseline  — run instrumented program, capture timer_before
  Phase 1: Apply     — apply PatchCandidate[] to source files
  Phase 2: Verify    — smoke test, timer comparison (revert on failure)

Verify Level 1 (always): smoke test (import check + test_commands)
Verify Level 2 (when timer data available): compare before/after [SYSIGHT_TIMER]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from sysight.types.optimization import PatchCandidate, PatchResult, compute_span_hash
from sysight.tools.patcher import PatchApplier


# ── Data structures ──

@dataclass
class VerifyResult:
    """Result of verifying all patches together."""
    smoke_passed: bool = False
    smoke_stdout: str = ""
    smoke_stderr: str = ""
    timer_before: dict[str, float] = field(default_factory=dict)  # label → ms
    timer_after: dict[str, float] = field(default_factory=dict)   # label → ms
    delta_pct: dict[str, float] = field(default_factory=dict)     # label → %
    status: str = "pending"  # "committed" | "reverted"
    revert_reason: str = ""


@dataclass
class ExecuteResult:
    """Result of the EXECUTE stage."""
    run_id: str = ""
    patches: list[PatchResult] = field(default_factory=list)
    verify: VerifyResult = field(default_factory=VerifyResult)
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0


# ── Timer log parsing ──

_TIMER_RE = re.compile(
    r"\[SYSIGHT_TIMER\]\s+(.+?):\s+([\d.]+)\s+ms"
)


def parse_timer_log(log_text: str) -> dict[str, float]:
    """Parse [SYSIGHT_TIMER] lines from log output.

    Returns dict mapping timer_label → average ms (across all occurrences).
    """
    samples: dict[str, list[float]] = {}
    for m in _TIMER_RE.finditer(log_text):
        label = m.group(1).strip()
        ms = float(m.group(2))
        samples.setdefault(label, []).append(ms)

    result = {}
    for label, vals in samples.items():
        result[label] = sum(vals) / len(vals)
    return result


# ── Main entry point ──

def run_execute(
    patches: list[PatchCandidate],
    repo: str,
    *,
    run_id: str = "",
    analyze_run_dir: Path | None = None,
    run_dir: Path | None = None,
    exec_config: dict | None = None,
    inject_timers: bool = False,
) -> ExecuteResult:
    """Apply patches, run smoke test, compare timers.

    If any patch fails to apply or smoke test fails, all patches are
    reverted and source files are restored to their original state.

    Args:
        patches: PatchCandidate[] from the OPTIMIZE stage (hashes filled).
        repo: Path to repo root.
        run_id: Run identifier for output naming.
        analyze_run_dir: Analyzer's run_dir where instrument_result.json lives.
        run_dir: Output directory for execute_result.json.
        exec_config: Execution config dict. If None, loaded from warmup cache.
    """
    import hashlib

    t0 = time.monotonic()
    root = Path(repo).resolve()
    errors: list[str] = []

    if not root.is_dir():
        errors.append(f"repo 不是目录: {root}")
        return ExecuteResult(run_id=run_id, errors=errors)

    if not patches:
        return ExecuteResult(
            run_id=run_id,
            errors=["no patches to execute"],
        )

    # Create output directory
    if run_dir is None:
        digest = hashlib.sha1(
            f"{root}|{run_id}|execute".encode()
        ).hexdigest()[:8]
        exec_run_id = f"run-{digest}"
        run_dir = Path.cwd() / ".sysight" / "execute-runs" / exec_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load execution config: explicit arg > warmup cache > empty
    if exec_config is None:
        repo_setup = _load_repo_setup(root)
        exec_config = repo_setup.to_execution_config() if repo_setup else {}

    # Load instrument result for timer labels
    timer_labels = _load_timer_labels(analyze_run_dir)

    # ── Phase 0: Baseline measurement ──
    # Prefer pre-built timer_before.json from artifacts (e.g. optimizer-bench).
    # Fall back to live measurement only when no pre-built data exists.
    timer_before: dict[str, float] = {}
    if analyze_run_dir:
        timer_before_path = analyze_run_dir / "timer_before.json"
        if timer_before_path.exists():
            try:
                timer_before = json.loads(timer_before_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    if not timer_before and (exec_config.get("entry_point") or exec_config.get("minimal_run")):
        timer_before = _run_and_measure(root, exec_config, errors)

    # ── Phase 1+2: Apply + Verify ──
    applier = PatchApplier(root)
    patch_results, verify = _apply_and_verify(
        patches, root, exec_config, timer_before, timer_labels, applier, errors,
        inject_timers=inject_timers,
        analyze_run_dir=analyze_run_dir,
    )

    elapsed_ms = (time.monotonic() - t0) * 1000

    result = ExecuteResult(
        run_id=run_id,
        patches=patch_results,
        verify=verify,
        errors=errors,
        elapsed_ms=elapsed_ms,
    )

    # Write execute_result.json
    _write_result_json(result, run_dir)

    # Print output location
    print(f"\n  Execute output → {run_dir}", file=sys.stderr)

    # Print summary
    _print_summary(result, timer_before)

    return result


# ── Apply + Verify ──

def _apply_and_verify(
    patches: list[PatchCandidate],
    root: Path,
    exec_config: dict,
    timer_before: dict[str, float],
    timer_labels: dict[str, str],
    applier: PatchApplier,
    errors: list[str],
    *,
    inject_timers: bool = False,
    analyze_run_dir: Path | None = None,
) -> tuple[list[PatchResult], VerifyResult]:
    """Apply all patches, run smoke test, compare timers.

    If any patch fails to apply or smoke test fails, revert all.

    Patches targeting the same file are applied in descending line order
    (bottom-up) so that earlier patches don't shift line numbers for later
    ones.  This is critical when the LLM generates multiple non-contiguous
    patches for the same file (e.g. remove barrier at line 47 + add no_sync
    at line 59).
    """
    verify = VerifyResult(timer_before=timer_before)

    # Sort patches: within each file, apply bottom-up (descending old_span_start)
    # to prevent earlier patches from shifting line numbers of later ones.
    def _patch_sort_key(p: PatchCandidate) -> tuple[str, int]:
        return (p.file_path, -p.old_span_start)
    sorted_patches = sorted(patches, key=_patch_sort_key)

    # Apply all patches (bottom-up within each file)
    for patch in sorted_patches:
        ok, msg = applier.apply(
            file_path=patch.file_path,
            old_span_start=patch.old_span_start,
            old_span_end=patch.old_span_end,
            old_span_hash=patch.old_span_hash,
            replacement=patch.replacement,
        )
        if not ok:
            errors.append(f"apply failed for {patch.patch_id}: {msg}")
            # Revert all
            applier.revert_all()
            verify.status = "reverted"
            verify.revert_reason = f"apply_failed: {patch.patch_id}"
            return [
                PatchResult(
                    patch_id=p.patch_id,
                    finding_ids=p.finding_ids,
                    status="reverted",
                    reason="apply_failed",
                )
                for p in patches
            ], verify

    # Smoke test
    smoke_ok, smoke_stdout, smoke_stderr = _run_smoke_test(
        root, patches, exec_config,
    )
    verify.smoke_passed = smoke_ok
    verify.smoke_stdout = smoke_stdout
    verify.smoke_stderr = smoke_stderr

    if not smoke_ok:
        applier.revert_all()
        verify.status = "reverted"
        verify.revert_reason = "smoke_test_failed"
        errors.append(f"smoke test failed: {smoke_stderr[:200]}")
        return [
            PatchResult(
                patch_id=p.patch_id,
                finding_ids=p.finding_ids,
                status="reverted",
                reason="smoke_test_failed",
            )
            for p in patches
        ], verify

    # Timer comparison
    if timer_before and (exec_config.get("entry_point") or exec_config.get("minimal_run")):
        # Inject cuda_timer instrumentation AFTER patches are applied so
        # _run_and_measure can capture [SYSIGHT_TIMER] output.  This must
        # happen here (not before patch apply) because _fill_span_hashes
        # computed hashes against the original source files.
        if inject_timers:
            _inject_timers_from_artifacts(root, analyze_run_dir, patches=patches)
        timer_after = _run_and_measure(root, exec_config, errors)
        verify.timer_after = timer_after

        for label in timer_before:
            if label in timer_after:
                before_ms = timer_before[label]
                after_ms = timer_after[label]
                if before_ms > 0:
                    verify.delta_pct[label] = (
                        (after_ms - before_ms) / before_ms * 100
                    )

    verify.status = "committed"

    # Build per-patch results with timer data
    patch_results = []
    for patch in patches:
        # Collect delta_pct for all finding_ids this patch touches
        combined_delta = None
        combined_before = None
        combined_after = None
        for fid in patch.finding_ids:
            label = _finding_to_timer_label(fid, timer_labels)
            if label in verify.delta_pct:
                combined_delta = verify.delta_pct[label]
                combined_before = verify.timer_before.get(label)
                combined_after = verify.timer_after.get(label)
                break  # Use first match

        # Compute lines_changed for Minimality scoring
        replacement_lines = len(patch.replacement.splitlines()) if patch.replacement else 0
        span_lines = patch.old_span_end - patch.old_span_start + 1
        lines_changed = max(replacement_lines, span_lines)

        patch_results.append(PatchResult(
            patch_id=patch.patch_id,
            finding_ids=patch.finding_ids,
            status="kept",
            reason="smoke_passed",
            metric_before=combined_before,
            metric_after=combined_after,
            delta_pct=combined_delta,
            lines_changed=lines_changed,
        ))

    return patch_results, verify


# ── Smoke test ──

def _normalize_python(python_bin: str) -> str:
    """Normalize 'python' → 'python3' on systems where 'python' doesn't exist (macOS)."""
    if python_bin == "python":
        import shutil
        if shutil.which("python") is None and shutil.which("python3") is not None:
            return "python3"
    return python_bin


def _run_smoke_test(
    root: Path,
    patches: list[PatchCandidate],
    exec_config: dict,
) -> tuple[bool, str, str]:
    """Run smoke test for all patches.

    Tries validation_commands from patches first, then falls back to
    import checks and test_commands from warmup.
    """
    # Try patch-provided validation commands first
    for patch in patches:
        for cmd in patch.validation_commands:
            try:
                # Normalize 'python' → 'python3' for macOS compatibility
                normalized_cmd = [_normalize_python(cmd[0])] + cmd[1:] if cmd else cmd
                r = subprocess.run(
                    normalized_cmd,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if r.returncode != 0:
                    return False, r.stdout, r.stderr
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return False, "", str(e)

    # Fallback: import check for each modified file
    seen_modules: set[str] = set()
    for patch in patches:
        fp = patch.file_path
        if not fp.endswith(".py"):
            continue
        module = fp.replace("/", ".").replace("\\", ".").removesuffix(".py")
        if module in seen_modules:
            continue
        seen_modules.add(module)
        try:
            r = subprocess.run(
                [_normalize_python("python"), "-c", f"import {module}"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode != 0:
                return False, r.stdout, r.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Also run test_commands from warmup if available
    test_commands = exec_config.get("test_commands", [])
    for cmd in test_commands:
        try:
            r = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode != 0:
                return False, r.stdout, r.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, "", str(e)

    return True, "", ""


# ── Measurement ──

def _run_and_measure(
    root: Path,
    exec_config: dict,
    errors: list[str],
) -> dict[str, float]:
    """Run the program and capture [SYSIGHT_TIMER] output.

    Returns parsed timer data: {label: avg_ms}.
    """
    run_cmd = exec_config.get("minimal_run") or []
    if not run_cmd:
        entry = exec_config.get("entry_point", "")
        if entry:
            run_cmd = [_normalize_python("python"), entry]

    if not run_cmd:
        return {}

    # Normalize 'python' → 'python3' in the command
    run_cmd = [_normalize_python(run_cmd[0])] + run_cmd[1:]

    env_vars = exec_config.get("env_vars", {})
    timeout = 300  # generous timeout for CPU-only runs (e.g. macOS CI)

    import os
    env = os.environ.copy()
    env.update(env_vars)

    try:
        r = subprocess.run(
            run_cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        # Parse timer output even if returncode != 0 (program may have
        # printed [SYSIGHT_TIMER] lines before hitting a non-fatal error).
        combined_output = r.stdout + "\n" + r.stderr
        return parse_timer_log(combined_output)
    except subprocess.TimeoutExpired:
        errors.append(f"measurement run timed out after {timeout}s")
        return {}
    except FileNotFoundError as e:
        errors.append(f"measurement run command not found: {e}")
        return {}


# ── Helpers ──

def _load_repo_setup(root: Path):
    """Load RepoSetup from warmup cache."""
    try:
        from sysight.pipeline.warmup import _warmup_cache_path
        from sysight.types.repo_setup import RepoSetup
        cache_path = _warmup_cache_path(root)
        return RepoSetup.load_cache(cache_path)
    except Exception:
        return None


def _load_timer_labels(run_dir: Path | None) -> dict[str, str]:
    """Load timer label → finding_id mapping from instrument_result.json.

    Returns {finding_id: timer_label}.
    When timers are merged (e.g. "F01+F02"), each individual finding_id
    maps to the combined label so scoring can match any of them.
    """
    if run_dir is None:
        return {}

    instrument_json = run_dir / "instrument_result.json"
    if not instrument_json.exists():
        return {}

    try:
        data = json.loads(instrument_json.read_text(encoding="utf-8"))
        result = {}
        for t in data.get("timers", []):
            finding_ids = t.get("finding_id", "")
            timer_label = t.get("timer_label", "")
            if finding_ids and timer_label:
                # Split on "+" to handle merged timers — each individual
                # finding_id maps to the same combined timer_label.
                for fid in finding_ids.split("+"):
                    fid = fid.strip()
                    if fid:
                        result[fid] = timer_label
        return result
    except (json.JSONDecodeError, OSError):
        return {}


def _finding_to_timer_label(finding_id: str, timer_labels: dict[str, str]) -> str:
    """Map a finding_id to its timer label, or return finding_id as fallback."""
    return timer_labels.get(finding_id, finding_id)


def _write_result_json(result: ExecuteResult, run_dir: Path) -> None:
    """Write execute_result.json to the run directory."""
    data = {
        "run_id": result.run_id,
        "elapsed_ms": result.elapsed_ms,
        "patches": [
            {
                "patch_id": p.patch_id,
                "finding_ids": p.finding_ids,
                "status": p.status,
                "reason": p.reason,
                "metric_before": p.metric_before,
                "metric_after": p.metric_after,
                "delta_pct": p.delta_pct,
            }
            for p in result.patches
        ],
        "verify": {
            "smoke_passed": result.verify.smoke_passed,
            "timer_before": result.verify.timer_before,
            "timer_after": result.verify.timer_after,
            "delta_pct": result.verify.delta_pct,
            "status": result.verify.status,
            "revert_reason": result.verify.revert_reason,
        },
        "errors": result.errors,
    }
    (run_dir / "execute_result.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _inject_timers_from_artifacts(
    root: Path,
    analyze_run_dir: Path | None = None,
    patches: list | None = None,
) -> None:
    """Inject cuda_timer instrumentation AFTER patches are applied.

    Reads instrument_result.json (generated by LLM-driven instrument stage)
    to get precise timer specs.  Injects timer wrappers into patched source
    files, adjusting line numbers for patch-induced shifts.

    This must happen AFTER patch application because _fill_span_hashes
    computed hashes against the original (pre-injection) source files.

    The injection is NOT tracked by PatchApplier — the caller is responsible
    for restoring source files (e.g. via _snapshot_sources / _restore_sources).
    """
    # Find instrument_result.json — look in common locations
    instrument_json = None
    candidates = []
    if analyze_run_dir:
        candidates.append(analyze_run_dir / "instrument_result.json")
    candidates.append(root / "artifacts" / "instrument_result.json")
    for candidate in candidates:
        if candidate.exists():
            instrument_json = candidate
            break

    if instrument_json is None:
        return

    try:
        data = json.loads(instrument_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    timers = data.get("timers", [])
    if not timers:
        return

    # Compute per-file line offset caused by patches.
    # For each file, sum (replacement_lines - span_lines) for patches
    # whose old_span_end is BEFORE the timer's wrap_start.
    # Patches after the timer region don't affect its line numbers.
    file_offsets: dict[str, list[tuple[int, int]]] = {}
    # file_path → [(old_span_end, line_delta), ...] sorted by old_span_end
    if patches:
        for p in patches:
            fp = p.file_path
            replacement_lines = len(p.replacement.splitlines()) if p.replacement else 0
            span_lines = p.old_span_end - p.old_span_start + 1
            delta = replacement_lines - span_lines
            if delta != 0:
                file_offsets.setdefault(fp, []).append((p.old_span_end, delta))
        for offsets in file_offsets.values():
            offsets.sort(key=lambda x: x[0])  # sort by old_span_end ascending

    def _adjust_line(original_line: int, offsets: list[tuple[int, int]]) -> int:
        """Adjust an original line number by accumulated offsets from patches
        whose old_span_end is BEFORE original_line."""
        adjusted = original_line
        for old_end, delta in offsets:
            if old_end < original_line:
                adjusted += delta
        return adjusted

    from sysight.utils.cuda_timer import (
        CUDA_TIMER_MODULE_CONTENT,
        CUDA_TIMER_MODULE_NAME,
        CUDA_TIMER_IMPORT_LINE,
    )

    # 1. Write _sysight_timer.py to repo root
    timer_module_path = root / f"{CUDA_TIMER_MODULE_NAME}.py"
    timer_module_path.write_text(CUDA_TIMER_MODULE_CONTENT, encoding="utf-8")

    # 2. Group timers by file, sort bottom-up within each file
    by_file: dict[str, list[dict]] = {}
    for t in timers:
        fp = t.get("file", "")
        if fp:
            by_file.setdefault(fp, []).append(t)

    for file_path, specs in by_file.items():
        target = root / file_path
        if not target.exists():
            continue

        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        offsets = file_offsets.get(file_path, [])

        # Add import line at top (after any existing __future__ / docstring)
        insert_pos = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if stripped.startswith("from __future__"):
                insert_pos = i + 1
                continue
            if stripped == "":
                insert_pos = i + 1
                continue
            insert_pos = i
            break
        lines.insert(insert_pos, CUDA_TIMER_IMPORT_LINE + "\n")

        # Apply wraps bottom-up (descending wrap_start) to preserve line numbers
        specs_sorted = sorted(specs, key=lambda s: -s.get("wrap_start", 0))
        for spec in specs_sorted:
            ws = spec.get("wrap_start", 0)
            we = spec.get("wrap_end", 0)
            label = spec.get("timer_label", "region")

            # Adjust line numbers for patch-induced shifts
            if offsets:
                ws = _adjust_line(ws, offsets)
                we = _adjust_line(we, offsets)

            if ws <= 0 or we <= 0 or ws > len(lines) or we > len(lines):
                continue

            # Indentation from the first line of the wrapped region
            first_line = lines[ws - 1]
            indent = first_line[: len(first_line) - len(first_line.lstrip())]

            # Insert `with cuda_timer("label")():` before wrap_start,
            # and indent the wrapped body by one level.
            lines.insert(ws - 1, f'{indent}with cuda_timer("{label}")():\n')
            for j in range(ws, we + 1):
                if lines[j].strip():
                    lines[j] = "    " + lines[j]

        target.write_text("".join(lines), encoding="utf-8")


def _print_summary(result: ExecuteResult, timer_before: dict[str, float]) -> None:
    """Print execution summary to stderr."""
    sep = "=" * 60
    kept = sum(1 for p in result.patches if p.status == "kept")
    reverted = sum(1 for p in result.patches if p.status == "reverted")

    print(f"\n{sep}", file=sys.stderr)
    print(f"  EXECUTE COMPLETE  run_id={result.run_id}", file=sys.stderr)
    print(f"  Patches:      {len(result.patches)}", file=sys.stderr)
    print(f"  Kept:         {kept}", file=sys.stderr)
    print(f"  Reverted:     {reverted}", file=sys.stderr)
    print(f"  Elapsed:      {result.elapsed_ms:.0f} ms", file=sys.stderr)

    if timer_before and result.verify.delta_pct:
        print(f"\n  Timer comparison:", file=sys.stderr)
        for label, pct in result.verify.delta_pct.items():
            arrow = "↓" if pct < 0 else "↑"
            print(
                f"    {label}: {pct:+.1f}% {arrow} "
                f"({result.verify.timer_before.get(label, 0):.1f}ms → "
                f"{result.verify.timer_after.get(label, 0):.1f}ms)",
                file=sys.stderr,
            )

    if result.errors:
        print(f"\n  Errors:", file=sys.stderr)
        for e in result.errors[:5]:
            print(f"    ✗ {e}", file=sys.stderr)

    print(f"{sep}\n", file=sys.stderr)
