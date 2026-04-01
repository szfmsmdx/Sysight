"""Schema inspection skill adapted from nsys-ai."""

from __future__ import annotations

from .base import Skill


def _run(prof, device=None, trim=None):
    sql = """
        SELECT m.name AS table_name,
               p.name AS column_name,
               p.type AS column_type,
               p.pk AS is_pk
        FROM sqlite_master m
        JOIN pragma_table_info(m.name) p
        WHERE m.type = 'table'
          AND m.name NOT LIKE 'sqlite_%'
        ORDER BY m.name, p.cid
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql).fetchall()]


def _format(rows) -> str:
    if not rows:
        return "(No tables found in database)"
    lines = ["── Database Schema ──", ""]
    current_table = None
    for row in rows:
        if row["table_name"] != current_table:
            if current_table is not None:
                lines.append("")
            current_table = row["table_name"]
            lines.append(f"  {current_table}")
            lines.append(f"  {'─' * len(current_table)}")
        pk = " (PK)" if row["is_pk"] else ""
        lines.append(f"    {row['column_name']:<30s}  {(row['column_type'] or ''):<15s}{pk}")
    return "\n".join(lines)


SKILL = Skill(
    name="schema_inspect",
    title="Database Schema Inspector",
    description="Lists all tables and columns available in the Nsight SQLite export.",
    runner=_run,
    formatter=_format,
)
