"""Kernel launch pattern skill adapted from nsys-ai."""

from __future__ import annotations

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    trim_clause = ""
    params: list[object] = [target]
    if trim:
        trim_clause = " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        WITH launch_gaps AS (
            SELECT
                k.streamId,
                k.start,
                k.[end],
                (k.[end] - k.start) AS dur_ns,
                LAG(k.[end]) OVER (PARTITION BY k.streamId ORDER BY k.start) AS prev_end
            FROM {prof.schema.kernel_table} k
            WHERE k.deviceId = ?{trim_clause}
        ),
        stream_stats AS (
            SELECT
                streamId,
                COUNT(*) AS kernel_count,
                MIN(start) AS first_start,
                MAX([end]) AS last_end,
                ROUND(SUM(dur_ns) / 1e6, 2) AS total_kernel_ms,
                ROUND(AVG(dur_ns) / 1e3, 1) AS avg_kernel_us,
                ROUND(MAX(CASE WHEN prev_end IS NOT NULL THEN start - prev_end ELSE 0 END) / 1e6, 3) AS max_gap_ms,
                ROUND(AVG(CASE WHEN prev_end IS NOT NULL AND start > prev_end THEN start - prev_end ELSE NULL END) / 1e3, 1) AS avg_gap_us,
                SUM(CASE WHEN prev_end IS NOT NULL AND (start - prev_end) > 1000000 THEN 1 ELSE 0 END) AS sync_stalls
            FROM launch_gaps
            GROUP BY streamId
        )
        SELECT
            streamId,
            kernel_count,
            ROUND((last_end - first_start) / 1e6, 2) AS span_ms,
            total_kernel_ms,
            ROUND(CAST(kernel_count AS REAL) / NULLIF((last_end - first_start) / 1e6, 0), 1) AS dispatch_rate_per_ms,
            ROUND(total_kernel_ms / NULLIF((last_end - first_start) / 1e6, 0) * 100, 1) AS occupancy_pct,
            avg_kernel_us,
            max_gap_ms,
            avg_gap_us,
            sync_stalls
        FROM stream_stats
        ORDER BY kernel_count DESC
        LIMIT 10
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params).fetchall()]


def _format(rows) -> str:
    if not rows:
        return "(No kernel launch data found)"
    lines = [
        "── Kernel Launch Patterns ──",
        f"{'Stream':>7s} {'Kernels':>8s} {'Span(ms)':>10s} {'Rate/ms':>8s} {'Occ%':>6s} {'MaxGap':>10s} {'Stalls':>7s}",
        "─" * 72,
    ]
    for row in rows:
        lines.append(
            f"s{row['streamId']:>5d} {row['kernel_count']:>8d} {row['span_ms']:>10.2f} "
            f"{row['dispatch_rate_per_ms']:>8.1f} {row['occupancy_pct']:>5.1f}% "
            f"{row['max_gap_ms']:>8.3f}ms {row['sync_stalls']:>7d}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="kernel_launch_pattern",
    title="Kernel Launch Pattern Analysis",
    description="Analyzes dispatch rate, inter-launch gaps, and sync-stall density per stream.",
    runner=_run,
    formatter=_format,
)
