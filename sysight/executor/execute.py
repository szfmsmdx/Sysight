"""Orchestration logic for sysight.executor.

Reads patch_plan.json, sequentially git applies each patch,
runs the metric_probe to verify the score, and commits or reverts
based on whether the score improved.

After each patch, one row is appended to patch_log.csv in the artifact dir.
CSV columns: patch_id, finding_id, summary, base_hash, after_hash,
             status, score_before, score_after, delta, adopted
"""

from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sysight.optimizer.models import PatchPlan, Patch, MetricProbe
from .models import ExecutionReport, ExecutedPatch

logger = logging.getLogger(__name__)

# CSV columns written to patch_log.csv
_CSV_FIELDS = [
    "patch_id",
    "finding_id",
    "summary",
    "base_hash",
    "after_hash",
    "status",
    "score_before",
    "score_after",
    "delta",
    "adopted",
]


def is_better_score(score_after: float, baseline: float, is_lower_better: bool) -> bool:
    """Compare the new score with the baseline."""
    if is_lower_better:
        return score_after < baseline
    return score_after > baseline


def determine_lower_is_better(grep_pattern: str) -> bool:
    """Heuristic to guess if lower is better based on metric name."""
    pattern_lower = grep_pattern.lower()
    if "time" in pattern_lower or "latency" in pattern_lower:
        return True
    # Assume things like MFU, score, throughput, iter/s are higher-is-better
    return False


def run_probe(repo_root: str, probe: MetricProbe) -> float | None:
    """Run the user-defined metric command and extract the float value via grep.

    Returns the float value if successful, else None.
    """
    logger.info("Running metric probe: %s", probe.run_cmd)

    try:
        res = subprocess.run(
            probe.run_cmd,
            shell=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as e:
        logger.error("Metric probe execution failed: %s", e)
        return None

    if res.returncode != 0:
        logger.warning("Metric probe returned non-zero exit code: %d", res.returncode)
        # Even if non-zero, sometimes the output might still contain the metric. We can still try to parse.

    # Combine stdout and stderr for grepping
    output = res.stdout + "\n" + res.stderr

    try:
        pattern = re.compile(probe.grep_pattern, re.IGNORECASE)
    except re.error as e:
        logger.error("Invalid grep pattern %r: %s", probe.grep_pattern, e)
        return None

    float_re = re.compile(r"[-+]?[0-9]*\.?[0-9]+")

    for line in output.splitlines():
        if pattern.search(line):
            matches = float_re.findall(line)
            if matches:
                try:
                    val = float(matches[0])
                    logger.info("Extracted metric value %f from line: %s", val, line.strip())
                    return val
                except ValueError:
                    pass

    logger.warning("Could not extract a float value using grep pattern %r", probe.grep_pattern)
    return None


def _get_git_head(repo_root: str) -> str:
    """Return the current HEAD commit hash, or empty string on failure."""
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return res.stdout.strip()
    return ""


def _write_patch_log_row(csv_path: Path, row: dict) -> None:
    """Append one row to patch_log.csv, creating with header if needed."""
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def execute_patch_plan(
    repo_root: str,
    plan: PatchPlan,
    csv_path: Path | None = None,
) -> ExecutionReport:
    """Iterate over patches, apply, verify, and conditionally commit.

    If csv_path is provided, one row per patch is appended to that CSV file.
    """
    probe = plan.metric_probe
    is_lower_better = determine_lower_is_better(probe.grep_pattern)

    logger.info("Recording baseline score...")
    baseline_score = run_probe(repo_root, probe)
    if baseline_score is None:
        logger.error("Failed to acquire baseline score. Aborting execution.")
        return ExecutionReport(
            baseline_score=None,
            patches=[],
            final_score=None,
        )
    logger.info("Baseline score: %s", baseline_score)

    executed_patches = []
    current_score = baseline_score

    # Warn if working tree is dirty before we start
    status_res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if status_res.stdout.strip():
        logger.warning("Working tree is not clean. Executor prefers a clean repo state.")

    for patch in plan.patches:
        logger.info("Applying patch %s for finding %s...", patch.id, patch.finding_id)

        base_hash = _get_git_head(repo_root)

        with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
            f.write(patch.diff)
            if not patch.diff.endswith("\n"):
                f.write("\n")
            diff_path = f.name

        after_hash = ""
        try:
            # 1. Apply patch
            apply_res = subprocess.run(
                ["git", "apply", diff_path],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            if apply_res.returncode != 0:
                logger.error("Failed to git apply patch %s: %s", patch.id, apply_res.stderr)
                ep = ExecutedPatch(
                    id=patch.id,
                    status="skipped",
                    score_after=None,
                    delta="Apply failed",
                    adopted=False,
                    base_hash=base_hash,
                    after_hash="",
                    summary=patch.summary,
                )
                executed_patches.append(ep)
                if csv_path:
                    _write_patch_log_row(csv_path, _ep_to_csv_row(ep, patch, current_score))
                continue

            # 2. Run probe
            score_after = run_probe(repo_root, probe)
            if score_after is None:
                logger.warning("Failed to extract score after patch %s. Reverting.", patch.id)
                _revert_changes(repo_root)
                ep = ExecutedPatch(
                    id=patch.id,
                    status="skipped",
                    score_after=None,
                    delta="Metric extraction failed",
                    adopted=False,
                    base_hash=base_hash,
                    after_hash="",
                    summary=patch.summary,
                )
                executed_patches.append(ep)
                if csv_path:
                    _write_patch_log_row(csv_path, _ep_to_csv_row(ep, patch, current_score))
                continue

            delta_val = score_after - current_score
            delta_str = f"{delta_val:+.4f}"

            # 3. Compare and Commit / Revert
            if is_better_score(score_after, current_score, is_lower_better):
                logger.info("Score improved! (%s). Committing.", delta_str)
                commit_msg = (
                    f"sysight: {patch.id} — {patch.summary}\n\n"
                    f"finding: {patch.finding_id}\n"
                    f"score: {current_score:.4f} → {score_after:.4f} ({delta_str})\n"
                    f"rationale: {patch.rationale}"
                )
                _commit_changes(repo_root, commit_msg)
                after_hash = _get_git_head(repo_root)
                current_score = score_after
                ep = ExecutedPatch(
                    id=patch.id,
                    status="committed",
                    score_after=score_after,
                    delta=delta_str,
                    adopted=True,
                    base_hash=base_hash,
                    after_hash=after_hash,
                    summary=patch.summary,
                )
            else:
                logger.info("Score did not improve (%s). Reverting.", delta_str)
                _revert_changes(repo_root)
                ep = ExecutedPatch(
                    id=patch.id,
                    status="reverted",
                    score_after=score_after,
                    delta=delta_str,
                    adopted=False,
                    base_hash=base_hash,
                    after_hash="",
                    summary=patch.summary,
                )

            executed_patches.append(ep)
            if csv_path:
                _write_patch_log_row(csv_path, _ep_to_csv_row(ep, patch, current_score if ep.adopted else score_after))

        finally:
            Path(diff_path).unlink(missing_ok=True)

    return ExecutionReport(
        baseline_score=baseline_score,
        patches=executed_patches,
        final_score=current_score,
    )


def _ep_to_csv_row(ep: ExecutedPatch, patch: Patch, score_before: float | None) -> dict:
    """Convert an ExecutedPatch to a CSV row dict."""
    score_after_val = ep.score_after
    if score_after_val is not None and score_before is not None:
        delta = f"{score_after_val - score_before:+.4f}"
    else:
        delta = ep.delta
    return {
        "patch_id": ep.id,
        "finding_id": patch.finding_id,
        "summary": ep.summary,
        "base_hash": ep.base_hash,
        "after_hash": ep.after_hash,
        "status": ep.status,
        "score_before": score_before if score_before is not None else "",
        "score_after": score_after_val if score_after_val is not None else "",
        "delta": delta,
        "adopted": ep.adopted,
    }


def _revert_changes(repo_root: str) -> None:
    """Discard all uncommitted changes in the repo."""
    subprocess.run(["git", "restore", "."], cwd=repo_root, capture_output=True)
    subprocess.run(["git", "clean", "-f"], cwd=repo_root, capture_output=True)


def _commit_changes(repo_root: str, message: str) -> None:
    """Commit the current changes in the repo."""
    subprocess.run(["git", "add", "."], cwd=repo_root, capture_output=True)
    with tempfile.NamedTemporaryFile("w", suffix=".msg", delete=False) as f:
        f.write(message)
        msg_path = f.name
    try:
        subprocess.run(["git", "commit", "-F", msg_path], cwd=repo_root, capture_output=True)
    finally:
        Path(msg_path).unlink(missing_ok=True)
