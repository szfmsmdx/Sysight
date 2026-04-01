"""CPU-GPU pipeline skill adapted from nsys-ai."""

from __future__ import annotations

from .base import Skill


def _run(prof, device=None, trim=None):
    if not prof.schema.runtime_table or not prof.schema.kernel_table:
        return []
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    trim_clause = ""
    params: list[object] = [target]
    if trim:
        trim_clause = " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        WITH runtime_kernel AS (
            SELECT
                r.globalTid AS cpu_tid,
                r.start AS cpu_dispatch_start,
                r.[end] AS cpu_dispatch_end,
                k.start AS gpu_start,
                k.[end] AS gpu_end,
                (k.start - r.[end]) AS queue_delay_ns,
                (k.[end] - k.start) AS gpu_dur_ns
            FROM {prof.schema.runtime_table} r
            JOIN {prof.schema.kernel_table} k ON r.correlationId = k.correlationId
            WHERE k.deviceId = ?
              AND r.[end] < k.start{trim_clause}
        ),
        per_thread AS (
            SELECT
                cpu_tid,
                COUNT(*) AS dispatches,
                ROUND(AVG(queue_delay_ns) / 1e3, 1) AS avg_queue_delay_us,
                ROUND(MAX(queue_delay_ns) / 1e3, 1) AS max_queue_delay_us,
                ROUND(MIN(queue_delay_ns) / 1e3, 1) AS min_queue_delay_us,
                SUM(CASE WHEN queue_delay_ns > 1000000 THEN 1 ELSE 0 END) AS starvation_events,
                ROUND(AVG(gpu_dur_ns) / 1e3, 1) AS avg_kernel_us
            FROM runtime_kernel
            GROUP BY cpu_tid
        )
        SELECT
            cpu_tid,
            dispatches,
            avg_queue_delay_us,
            min_queue_delay_us,
            max_queue_delay_us,
            starvation_events,
            avg_kernel_us,
            ROUND(CAST(dispatches AS REAL) / (SELECT SUM(dispatches) FROM per_thread) * 100, 1) AS pct_of_dispatches
        FROM per_thread
        ORDER BY dispatches DESC
        LIMIT 10
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params).fetchall()]


def _format(rows) -> str:
    if not rows:
        return "(No CPU-GPU dispatch data — requires Runtime + Kernel correlation)"
    lines = [
        "── CPU-GPU Pipeline Analysis ──",
        f"{'Thread':>12s} {'Dispatches':>11s} {'AvgQueue':>10s} {'MaxQueue':>10s} {'Starve':>7s} {'%Total':>7s}",
        "─" * 69,
    ]
    total_starvation = 0
    for row in rows:
        total_starvation += row["starvation_events"]
        lines.append(
            f"{row['cpu_tid']:>12d} {row['dispatches']:>11d} {row['avg_queue_delay_us']:>8.1f}µs "
            f"{row['max_queue_delay_us']:>8.1f}µs {row['starvation_events']:>7d} {row['pct_of_dispatches']:>6.1f}%"
        )
    lines.append(f"\n  Total GPU starvation events (queue > 1ms): {total_starvation}")
    if total_starvation > 10:
        lines.append(
            "  High starvation count suggests CPU dispatch, Python GIL contention, or explicit syncs are limiting GPU feed."
        )
    return "\n".join(lines)


SKILL = Skill(
    name="cpu_gpu_pipeline",
    title="CPU-GPU Pipeline Analysis",
    description="Measures CPU dispatch lead time, starvation events, and per-thread launch contribution.",
    runner=_run,
    formatter=_format,
)
