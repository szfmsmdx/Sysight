"""Deterministic bottleneck classification — C1-C7 categories from profile data.

Uses nsys_sql tools to extract evidence, then classifies bottlenecks
without LLM involvement. The output feeds into the ANALYZE stage's prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql.kernels import kernels
from sysight.tools.nsys_sql.sync import sync
from sysight.tools.nsys_sql.memcpy import memcpy
from sysight.tools.nsys_sql.nccl import nccl
from sysight.tools.nsys_sql.overlap import overlap
from sysight.tools.nsys_sql.gaps import gaps
from sysight.tools.nsys_sql.launch import launch


@dataclass
class ClassifyResult:
    sqlite: str = ""
    categories: list[dict] = field(default_factory=list)
    summary: str = ""


def classify(sqlite: str) -> ClassifyResult:
    """Classify a profile into C1-C7 bottleneck categories.

    Returns a structured report suitable for injection into the ANALYZE prompt.
    """
    result = ClassifyResult(sqlite=sqlite)
    categories: list[dict] = []

    # Gather evidence from all nsys_sql tools
    k = kernels(sqlite, limit=10)
    s = sync(sqlite)
    m = memcpy(sqlite)
    n = nccl(sqlite)
    o = overlap(sqlite)
    g = gaps(sqlite)
    l = launch(sqlite)

    trace_ns = k.trace_duration_ns or 1

    # C1: GPU Idle / Host Scheduling
    if g.total_gap_ns > 0:
        gap_pct = g.total_gap_ns / trace_ns * 100
        categories.append({
            "category": "C1", "title": "GPU Idle Gaps",
            "severity": "critical" if gap_pct > 20 else "warning" if gap_pct > 5 else "info",
            "pct_of_trace": round(gap_pct, 1),
            "evidence": f"{g.gap_count} gaps, {g.total_gap_ns/1e9:.1f}s total idle",
            "top_gap_ns": g.gaps[0].gap_ns if g.gaps else 0,
        })

    # C2: Kernel Launch Overhead
    if l.entries:
        avg_overhead_ms = l.avg_overhead_us / 1000
        categories.append({
            "category": "C2", "title": "Kernel Launch Overhead",
            "severity": "critical" if avg_overhead_ms > 10 else "warning" if avg_overhead_ms > 1 else "info",
            "avg_overhead_ms": round(avg_overhead_ms, 1),
            "max_overhead_ms": round(l.max_overhead_us / 1000, 1),
            "evidence": f"avg {avg_overhead_ms:.1f}ms overhead per launch, {len(l.entries)} samples",
        })

    # C3: Synchronization
    if s.sync_events:
        top_sync = s.sync_events[0]
        categories.append({
            "category": "C3", "title": "Synchronization Cost",
            "severity": "critical" if s.sync_wall_pct > 20 else "warning" if s.sync_wall_pct > 5 else "info",
            "sync_wall_pct": s.sync_wall_pct,
            "top_sync_type": top_sync.sync_type,
            "evidence": f"sync takes {s.sync_wall_pct}% of wall clock, top: {top_sync.sync_type}",
        })

    # C4: Memory Copy
    if m.memcpy_ops:
        total_gb = m.total_bytes / 1e9
        d2h = next((op for op in m.memcpy_ops if op.direction == "D2H"), None)
        categories.append({
            "category": "C4", "title": "Memory Copy",
            "severity": "critical" if d2h and d2h.total_bytes > 1e9 else "warning",
            "total_gb": round(total_gb, 1),
            "d2h_bytes": d2h.total_bytes if d2h else 0,
            "evidence": f"{total_gb:.1f}GB total, {len(m.memcpy_ops)} directions",
        })

    # C5: Compute Inefficiency
    if k.kernels:
        top = k.kernels[0]
        compute_pct = k.total_kernel_ns / trace_ns * 100 if trace_ns > 0 else 0
        categories.append({
            "category": "C5", "title": "Compute Efficiency",
            "severity": "info" if compute_pct > 60 else "warning",
            "compute_pct": round(compute_pct, 1),
            "top_kernel": top.name,
            "evidence": f"GPU compute: {compute_pct:.1f}% of trace, top kernel: {top.name}",
        })

    # C6: Communication (NCCL)
    if n.streams:
        comm_pct = n.total_nccl_ns / trace_ns * 100 if trace_ns > 0 else 0
        categories.append({
            "category": "C6", "title": "NCCL Communication",
            "severity": "critical" if comm_pct > 40 else "warning" if comm_pct > 10 else "info",
            "comm_pct": round(comm_pct, 1),
            "overlap_pct": o.overlap_pct,
            "evidence": f"NCCL: {comm_pct:.1f}% of trace, {o.overlap_pct}% overlapped",
        })

    # C7: Python Pipeline — requires CPU sampling, not available from SQL alone
    categories.append({
        "category": "C7", "title": "Python Pipeline",
        "severity": "info",
        "evidence": "CPU-side analysis requires CPU sampling data or static code review",
    })

    result.categories = categories

    # Build summary
    lines = [f"Profile: {sqlite}"]
    lines.append(f"Trace duration: {trace_ns/1e9:.2f}s")
    lines.append(f"GPU kernel time: {k.total_kernel_ns/1e9:.2f}s")
    for c in categories:
        lines.append(f"  {c['category']} {c['title']}: {c['severity']} — {c.get('evidence', '')[:100]}")
    result.summary = "\n".join(lines)

    return result


CLASSIFY_TOOL = ToolDef(
    name="classify",
    description="Classify an Nsight Systems SQLite profile into C1-C7 bottleneck categories using deterministic SQL analysis",
    parameters={
        "type": "object",
        "properties": {"sqlite": {"type": "string", "description": "Path to .sqlite file"}},
        "required": ["sqlite"],
    },
    fn=classify,
    read_only=True,
)
