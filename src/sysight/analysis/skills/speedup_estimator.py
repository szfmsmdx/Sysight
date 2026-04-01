"""Speedup estimator skill adapted from nsys-ai ideas."""

from __future__ import annotations

from sysight.analysis.iterations import detect_iterations, iteration_summary
from sysight.analysis.queries import overlap_analysis
from sysight.analysis.summary import gpu_summary

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    summary = gpu_summary(prof, target, trim)
    overlap = overlap_analysis(prof, target, trim)
    iterations = detect_iterations(prof, target, trim)
    iteration_stats = iteration_summary(iterations)

    if "error" in summary:
        return [{"error": summary["error"]}]

    iter_count = max(iteration_stats["count"], 1) if iteration_stats else 1
    iteration_ms = iteration_stats["avg_ms"] if iteration_stats else summary["timing"]["span_ms"]
    idle_per_iter_ms = (overlap.get("idle_ms", 0.0) / iter_count) if "error" not in overlap else 0.0
    exposed_nccl_per_iter_ms = (
        overlap.get("nccl_only_ms", 0.0) / iter_count if "error" not in overlap else 0.0
    )

    rows = []
    if idle_per_iter_ms > 0:
        new_iter = max(iteration_ms - idle_per_iter_ms, 1e-6)
        rows.append(
            {
                "optimization": "Eliminate GPU Idle Gaps",
                "method": "Remove unnecessary syncs, move input staging earlier, or improve dispatch pipelining",
                "saved_ms": round(idle_per_iter_ms, 2),
                "new_iteration_ms": round(new_iter, 2),
                "speedup": round(iteration_ms / new_iter, 3),
                "confidence": "medium",
            }
        )

    if exposed_nccl_per_iter_ms > 0:
        new_iter = max(iteration_ms - exposed_nccl_per_iter_ms, 1e-6)
        rows.append(
            {
                "optimization": "Hide Exposed NCCL Time",
                "method": "Overlap allreduce with compute, rebalance buckets, or reduce communication frequency",
                "saved_ms": round(exposed_nccl_per_iter_ms, 2),
                "new_iteration_ms": round(new_iter, 2),
                "speedup": round(iteration_ms / new_iter, 3),
                "confidence": "medium",
            }
        )

    sync_hint_ms = min(idle_per_iter_ms, iteration_ms * 0.25)
    if summary["timing"]["utilization_pct"] < 95 and sync_hint_ms > 0:
        new_iter = max(iteration_ms - sync_hint_ms, 1e-6)
        rows.append(
            {
                "optimization": "Reduce Synchronization in Hot Path",
                "method": "Remove .item() / .cpu() / stream-wide waits and prefer event-based dependencies",
                "saved_ms": round(sync_hint_ms, 2),
                "new_iteration_ms": round(new_iter, 2),
                "speedup": round(iteration_ms / new_iter, 3),
                "confidence": "low",
            }
        )

    if not rows:
        rows.append(
            {
                "optimization": "No clear estimate",
                "method": "Profile looks balanced enough that deeper kernel or MFU analysis is needed",
                "saved_ms": 0.0,
                "new_iteration_ms": round(iteration_ms, 2),
                "speedup": 1.0,
                "confidence": "n/a",
            }
        )
    return rows


def _format(rows) -> str:
    if not rows:
        return "(No estimates)"
    if "error" in rows[0]:
        return f"(Error: {rows[0]['error']})"
    lines = ["── Speedup Estimates ──"]
    for row in rows:
        lines.append(f"- {row['optimization']}")
        lines.append(f"  Method: {row['method']}")
        lines.append(
            f"  Potential savings: {row['saved_ms']:.2f}ms | New iter: {row['new_iteration_ms']:.2f}ms | "
            f"Speedup: {row['speedup']:.3f}x | Confidence: {row['confidence']}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="speedup_estimator",
    title="Speedup Estimator",
    description="Estimates iteration-time savings from removing idle gaps, sync stalls, and exposed NCCL time.",
    runner=_run,
    formatter=_format,
)
