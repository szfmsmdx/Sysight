"""nsys_sql.overlap — Compute/NCCL overlap estimation (stream-level approximation)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql._helpers import _open_db, _find_table, _stream_class_query, _stream_span_stats


@dataclass
class OverlapResult:
    compute_only_ns: int = 0
    nccl_only_ns: int = 0
    overlap_ns: int = 0
    total_span_ns: int = 0
    overlap_pct: float = 0.0
    compute_kernels: int = 0
    nccl_kernels: int = 0
    note: str = ""


def overlap(sqlite: str) -> OverlapResult:
    """Estimate compute/NCCL overlap from stream classification (SQL approximation)."""
    with _open_db(sqlite) as (conn, all_tables, has_strings):
        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return OverlapResult(note="No kernel data available")

        try:
            rows = conn.execute(_stream_class_query(conn, kernel_tbl, has_strings)).fetchall()
        except sqlite3.Error as e:
            return OverlapResult(note=f"SQL error: {e}")

        nccl_streams: set[int] = set()
        compute_streams: set[int] = set()
        for row in rows:
            sid = int(row["streamId"])
            if int(row["nccl_count"] or 0) > 0:
                nccl_streams.add(sid)
            else:
                compute_streams.add(sid)

        if not nccl_streams:
            return OverlapResult(note="No NCCL kernels found")

        cs, ce, c_ns, c_count = _stream_span_stats(conn, kernel_tbl, compute_streams)
        ns, ne, n_ns, n_count = _stream_span_stats(conn, kernel_tbl, nccl_streams)

        overlap_start = max(cs, ns)
        overlap_end = min(ce, ne)
        overlap_ns = max(0, overlap_end - overlap_start)

        total_span = max(ce, ne) - min(cs, ns) if (cs and ns) else 0
        nccl_span = ne - ns if ne > ns else 1
        overlap_pct = round(overlap_ns / nccl_span * 100, 1) if nccl_span > 0 else 0.0

        return OverlapResult(
            compute_only_ns=max(0, c_ns - overlap_ns),
            nccl_only_ns=max(0, n_ns - overlap_ns),
            overlap_ns=overlap_ns, total_span_ns=total_span,
            overlap_pct=overlap_pct, compute_kernels=c_count, nccl_kernels=n_count,
            note="Stream-span approximation. Use interval-union for precise analysis.",
        )


OVERLAP_TOOL = ToolDef(
    name="nsys_sql_overlap",
    description="Estimate compute/NCCL communication overlap using stream-level analysis (approximate)",
    parameters={"type": "object", "properties": {"sqlite": {"type": "string"}}, "required": ["sqlite"]},
    fn=overlap, read_only=True,
)
