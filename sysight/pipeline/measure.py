"""Deterministic end-to-end measurement helpers."""

from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path

from sysight.types.optimization import MeasurementPlan, MeasurementResult, MetricSpec


def run_measurement(
    repo: str | Path,
    plan: MeasurementPlan,
    run_dir: str | Path,
    label: str,
) -> MeasurementResult:
    """Run a MeasurementPlan and aggregate metrics from stdout/stderr."""
    root = Path(repo).resolve()
    out_dir = Path(run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    if not root.is_dir():
        return MeasurementResult(errors=[f"repo is not a directory: {root}"])
    if not plan.is_valid():
        return MeasurementResult(errors=["invalid measurement plan"])

    workdir = _resolve_workdir(root, plan.cwd, errors)
    if workdir is None:
        return MeasurementResult(errors=errors)

    stdout_path = out_dir / f"{label}.stdout.log"
    stderr_path = out_dir / f"{label}.stderr.log"
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    env = os.environ.copy()
    env.update(plan.env_vars)
    cmd = _normalize_command(plan.run_command)

    total_runs = plan.warmup_runs + plan.repeats
    for idx in range(total_runs):
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=plan.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            errors.append(f"run {idx + 1} timed out after {plan.timeout_s}s")
            continue
        except FileNotFoundError as e:
            errors.append(f"command not found: {e}")
            break

        tag = "warmup" if idx < plan.warmup_runs else "measure"
        stdout_chunks.append(f"\n===== {tag} run {idx + 1} exit={proc.returncode} =====\n{proc.stdout}")
        stderr_chunks.append(f"\n===== {tag} run {idx + 1} exit={proc.returncode} =====\n{proc.stderr}")
        if proc.returncode != 0:
            errors.append(f"run {idx + 1} exited with {proc.returncode}")
            stderr_tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            if stderr_tail:
                snippet = " | ".join(stderr_tail[-3:])[:300]
                errors.append(f"run {idx + 1} error snippet: {snippet}")

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")

    text = stdout_text + "\n" + stderr_text
    samples = {metric.name: _extract_samples(text, metric, errors) for metric in plan.metrics}
    values: dict[str, float] = {}
    for metric in plan.metrics:
        vals = samples.get(metric.name, [])
        if vals:
            values[metric.name] = _aggregate(vals, metric.aggregation)
        else:
            errors.append(f"metric {metric.name!r} produced no samples")

    primary = plan.primary_metric
    primary_name = primary.name if primary else ""
    primary_value = values.get(primary_name)
    status = "ok" if primary_value is not None and not errors else "error"

    return MeasurementResult(
        status=status,
        metric_values=values,
        samples=samples,
        primary_metric=primary_name,
        primary_value=primary_value,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        errors=errors,
    )


def compare_measurements(
    before: MeasurementResult,
    after: MeasurementResult,
    plan: MeasurementPlan,
) -> tuple[bool, float | None, str]:
    """Compare measurements. Positive delta_pct means improvement."""
    primary = plan.primary_metric
    if not primary:
        return False, None, "no primary metric"
    if before.status != "ok":
        return False, None, "before measurement failed"
    if after.status != "ok":
        return False, None, "after measurement failed"
    if before.primary_value is None or after.primary_value is None:
        return False, None, "primary metric missing"
    if before.primary_value == 0:
        return False, None, "before primary metric is zero"

    if primary.lower_is_better:
        delta_pct = (before.primary_value - after.primary_value) / before.primary_value * 100.0
    else:
        delta_pct = (after.primary_value - before.primary_value) / before.primary_value * 100.0

    threshold = plan.success_threshold_pct
    if delta_pct >= threshold:
        return True, delta_pct, f"primary metric improved by {delta_pct:.3f}%"
    return False, delta_pct, f"primary metric improved by {delta_pct:.3f}% < threshold {threshold:.3f}%"


def _resolve_workdir(root: Path, cwd: str, errors: list[str]) -> Path | None:
    workdir = (root / cwd).resolve() if cwd else root
    if not str(workdir).startswith(str(root)):
        errors.append(f"measurement cwd escapes repo: {cwd}")
        return None
    if not workdir.is_dir():
        errors.append(f"measurement cwd is not a directory: {cwd}")
        return None
    return workdir


def _normalize_command(cmd: list[str]) -> list[str]:
    """Normalize the first token of cmd to a usable python interpreter.

    Priority:
    1. If cmd[0] is an absolute path, use as-is.
    2. 'python' / 'python3' → prefer the active venv's python if $VIRTUAL_ENV is set.
    3. 'python' → 'python3' on systems where bare 'python' doesn't exist (macOS).
    """
    if not cmd:
        return []
    first = cmd[0]
    if first in ("python", "python3"):
        # Prefer the active virtual environment interpreter
        import os
        venv = os.environ.get("VIRTUAL_ENV", "")
        if venv:
            venv_python = Path(venv) / "bin" / "python3"
            if venv_python.exists():
                return [str(venv_python)] + cmd[1:]
        # Fallback: python → python3 when bare python doesn't exist
        if first == "python" and shutil.which("python") is None and shutil.which("python3"):
            return ["python3"] + cmd[1:]
    return cmd


def _extract_samples(text: str, metric: MetricSpec, errors: list[str]) -> list[float]:
    try:
        pattern = re.compile(metric.regex)
    except re.error as e:
        errors.append(f"metric {metric.name!r} regex error: {e}")
        return []

    values: list[float] = []
    for match in pattern.finditer(text):
        try:
            raw = match.group(metric.group)
        except IndexError:
            errors.append(f"metric {metric.name!r} group {metric.group} not found")
            return []
        value = _parse_float(raw)
        if value is not None:
            values.append(value)

    if metric.drop_first_n:
        values = values[metric.drop_first_n:]
    return values


def _parse_float(text: str) -> float | None:
    cleaned = str(text).replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _aggregate(values: list[float], mode: str) -> float:
    if mode == "median":
        return float(statistics.median(values))
    if mode == "min":
        return float(min(values))
    if mode == "max":
        return float(max(values))
    if mode == "last":
        return float(values[-1])
    return float(statistics.mean(values))
