"""CPU thread utilization skill adapted from nsys-ai."""

from __future__ import annotations

from .base import Skill


def _run(prof, device=None, trim=None):
    if "COMPOSITE_EVENTS" not in prof.schema.tables or "ThreadNames" not in prof.schema.tables:
        return []
    sql = """
        SELECT ce.globalTid % 0x1000000 AS tid,
               (
                   SELECT s.value
                   FROM StringIds s
                   WHERE s.id = (
                       SELECT tn.nameId
                       FROM ThreadNames tn
                       WHERE tn.globalTid = ce.globalTid
                       LIMIT 1
                   )
               ) AS thread_name,
               ROUND(100.0 * SUM(ce.cpuCycles) / (
                   SELECT MAX(1, SUM(cpuCycles)) FROM COMPOSITE_EVENTS
               ), 2) AS cpu_pct
        FROM COMPOSITE_EVENTS ce
        GROUP BY ce.globalTid
        ORDER BY cpu_pct DESC
        LIMIT 10
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql).fetchall()]


def _format(rows) -> str:
    if not rows:
        return "(No CPU utilization data found — COMPOSITE_EVENTS table may be missing)"
    lines = [
        "── CPU Thread Utilization ──",
        f"{'TID':>8s}  {'Thread Name':<40s}  {'CPU %':>7s}",
        "─" * 62,
    ]
    for row in rows:
        name = row["thread_name"] or "(unnamed)"
        if len(name) > 38:
            name = name[:35] + "..."
        lines.append(f"{row['tid']:>8d}  {name:<40s}  {row['cpu_pct']:>7.2f}")
    return "\n".join(lines)


SKILL = Skill(
    name="thread_utilization",
    title="CPU Thread Utilization",
    description="Shows CPU utilization by thread to help identify host-side bottlenecks and GIL contention.",
    runner=_run,
    formatter=_format,
)
