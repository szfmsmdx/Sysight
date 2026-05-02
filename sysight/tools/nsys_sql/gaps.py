"""nsys_sql.gaps — GPU idle gap (bubble) analysis between consecutive kernels."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql._helpers import _open_db, _find_table, _kernel_name_lookup


@dataclass
class GapInfo:
    stream_id: int
    gap_start_ns: int
    gap_end_ns: int
    gap_ns: int
    before_kernel: str | None = None
    after_kernel: str | None = None


@dataclass
class GapsResult:
    gaps: list[GapInfo] = field(default_factory=list)
    total_gap_ns: int = 0
    gap_count: int = 0


def gaps(sqlite: str, min_gap_ns: int = 1_000_000, limit: int = 20) -> GapsResult:
    """Query GPU idle gaps (bubbles) between consecutive kernels on each stream."""
    result = GapsResult()

    with _open_db(sqlite) as (conn, all_tables, has_strings):
        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return result

        # Collect intervals per stream
        intervals_by_stream: dict[int, list[tuple[int, int]]] = {}
        try:
            for row in conn.execute(
                f"SELECT streamId, start, [end] FROM {kernel_tbl} ORDER BY streamId, start"
            ):
                sid = int(row["streamId"] or 0)
                s, e = int(row["start"]), int(row["end"])
                intervals_by_stream.setdefault(sid, []).append((s, e))
        except sqlite3.Error:
            return result

        # Find gaps in each stream
        all_gaps: list[tuple[int, int, int, int]] = []
        for sid, intervals in intervals_by_stream.items():
            if not intervals:
                continue
            stream_start = min(s for s, _ in intervals)
            stream_end = max(e for _, e in intervals)

            for i in range(len(intervals) - 1):
                gap_start = intervals[i][1]
                gap_end = intervals[i + 1][0]
                dur = gap_end - gap_start
                if dur >= min_gap_ns:
                    all_gaps.append((sid, gap_start, gap_end, dur))

        all_gaps.sort(key=lambda x: x[3], reverse=True)
        result.gap_count = len(all_gaps)

        for sid, gs, ge, dur in all_gaps[:limit]:
            result.total_gap_ns += dur
            before = after = None
            try:
                r = conn.execute(_kernel_name_lookup(conn, kernel_tbl, has_strings,
                    where="k.streamId=? AND k.[end]<=?", order="ORDER BY k.[end] DESC"),
                    (sid, gs)).fetchone()
                if r:
                    before = r[0]
                r = conn.execute(_kernel_name_lookup(conn, kernel_tbl, has_strings,
                    where="k.streamId=? AND k.start>=?", order="ORDER BY k.start ASC"),
                    (sid, ge)).fetchone()
                if r:
                    after = r[0]
            except sqlite3.Error:
                pass
            result.gaps.append(GapInfo(stream_id=sid, gap_start_ns=gs, gap_end_ns=ge,
                                       gap_ns=dur, before_kernel=before, after_kernel=after))

    return result


GAPS_TOOL = ToolDef(
    name="nsys_sql_gaps",
    description="Query GPU idle gaps (bubbles) — periods when no kernel is executing on a stream",
    parameters={
        "type": "object",
        "properties": {
            "sqlite": {"type": "string"},
            "min_gap_ns": {"type": "integer", "default": 1000000},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["sqlite"],
    },
    fn=gaps, read_only=True,
)
