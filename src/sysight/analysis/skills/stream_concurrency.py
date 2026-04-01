"""Stream concurrency skill adapted from nsys-ai."""

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
        WITH stream_summary AS (
            SELECT
                k.streamId,
                COUNT(*) AS kernel_count,
                MIN(k.start) AS first_start,
                MAX(k.[end]) AS last_end,
                ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_gpu_ms,
                ROUND(MAX(k.[end] - k.start) / 1e6, 3) AS max_kernel_ms,
                ROUND(AVG(k.[end] - k.start) / 1e3, 1) AS avg_kernel_us
            FROM {prof.schema.kernel_table} k
            WHERE k.deviceId = ?{trim_clause}
            GROUP BY k.streamId
        ),
        global_stats AS (
            SELECT
                COUNT(DISTINCT streamId) AS active_streams,
                SUM(kernel_count) AS total_kernels,
                MIN(first_start) AS global_start,
                MAX(last_end) AS global_end,
                SUM(total_gpu_ms) AS sum_gpu_ms
            FROM stream_summary
        )
        SELECT
            s.streamId,
            s.kernel_count,
            s.total_gpu_ms,
            s.avg_kernel_us,
            s.max_kernel_ms,
            ROUND((s.last_end - s.first_start) / 1e6, 2) AS stream_span_ms,
            ROUND(s.total_gpu_ms / NULLIF((s.last_end - s.first_start) / 1e6, 0) * 100, 1) AS stream_util_pct,
            g.active_streams,
            g.total_kernels,
            ROUND((g.global_end - g.global_start) / 1e6, 2) AS global_span_ms,
            ROUND(g.sum_gpu_ms / NULLIF((g.global_end - g.global_start) / 1e6, 0) * 100, 1) AS sum_util_pct
        FROM stream_summary s, global_stats g
        ORDER BY s.total_gpu_ms DESC
        LIMIT 10
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params).fetchall()]


def _format(rows) -> str:
    if not rows:
        return "(No kernel activity found)"
    head = rows[0]
    lines = [
        "── Stream Concurrency Analysis ──",
        f"  Active streams: {head['active_streams']}",
        f"  Total kernels:  {head['total_kernels']}",
        f"  Global span:    {head['global_span_ms']:.2f}ms",
        f"  Sum GPU time:   {head['sum_util_pct']:.1f}% of span (>100% = true concurrency)",
        "",
        f"{'Stream':>7s} {'Kernels':>8s} {'GPU(ms)':>10s} {'AvgKern':>10s} {'Util%':>7s}",
        "─" * 50,
    ]
    for row in rows:
        lines.append(
            f"s{row['streamId']:>5d} {row['kernel_count']:>8d} "
            f"{row['total_gpu_ms']:>10.2f} {row['avg_kernel_us']:>8.1f}µs "
            f"{row['stream_util_pct']:>6.1f}%"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="stream_concurrency",
    title="Stream Concurrency Analysis",
    description="Analyzes how many GPU streams are active and whether work is serialized on a small subset of streams.",
    runner=_run,
    formatter=_format,
)
