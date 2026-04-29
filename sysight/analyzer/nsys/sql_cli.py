"""nsys/sql_cli.py — SQLite direct query CLI for agent consumption.

Each command is narrow, read-only, and outputs bounded JSON.
These commands are designed for Codex/agent to directly query nsys SQLite files.

Commands:
  nsys-sql schema <db>          — List tables, columns, and capabilities
  nsys-sql kernels <db>         — Top-N GPU kernels by total time
  nsys-sql gaps <db>            — GPU idle gaps (bubbles) analysis
  nsys-sql sync <db>            — CUDA synchronization events cost
  nsys-sql nvtx <db>            — NVTX range breakdown
  nsys-sql memcpy <db>          — Memory bandwidth analysis
  nsys-sql nccl <db>            — NCCL communication breakdown
  nsys-sql kernel-launch <db>   — Kernel launch overhead (API → GPU gap)
  nsys-sql stream-concurrency <db> — Per-stream concurrency and utilization
  nsys-sql overlap <db>         — Compute / NCCL overlap (pure SQL estimate)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from .extract import inspect_schema, find_gaps, union_ns, intersect_total
from .sql_shared import _COPY_KIND_NAMES, _NCCL_KEYWORDS, _find_table, _get_cols, _kernel_name_expr

logger = logging.getLogger(__name__)


@dataclass
class SqlSchemaResult:
    """Result of nsys-sql schema command."""
    capabilities: list[str]
    tables: dict[str, list[str]]
    gpu_devices: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


@dataclass
class KernelInfo:
    """Single kernel statistics."""
    name: str
    count: int
    total_ns: int
    avg_ns: float
    max_ns: int


@dataclass
class SqlKernelsResult:
    """Result of nsys-sql kernels command."""
    kernels: list[KernelInfo]
    total_kernel_ns: int
    trace_duration_ns: int


@dataclass
class GapInfo:
    """Single idle gap information."""
    stream_id: int
    gap_start_ns: int
    gap_end_ns: int
    gap_ns: int
    before_kernel: str | None = None
    after_kernel: str | None = None


@dataclass
class SqlGapsResult:
    """Result of nsys-sql gaps command."""
    gaps: list[GapInfo]
    total_gap_ns: int
    gap_count: int


@dataclass
class SyncEventInfo:
    """Single synchronization event."""
    sync_type: str
    count: int
    total_ns: int
    avg_ns: float
    max_ns: int


@dataclass
class SqlSyncResult:
    """Result of nsys-sql sync command."""
    sync_events: list[SyncEventInfo]
    total_sync_ns: int
    sync_wall_pct: float


@dataclass
class NvtxRangeInfo:
    """Single NVTX range statistics."""
    text: str
    count: int
    total_ns: int
    avg_ns: float


@dataclass
class SqlNvtxResult:
    """Result of nsys-sql nvtx command."""
    nvtx_ranges: list[NvtxRangeInfo]
    total_nvtx_ns: int


@dataclass
class MemcpyInfo:
    """Memory copy statistics."""
    direction: str
    count: int
    total_bytes: int
    total_ns: int
    avg_bw_gbps: float


@dataclass
class SqlMemcpyResult:
    """Result of nsys-sql memcpy command."""
    memcpy_ops: list[MemcpyInfo]
    total_bytes: int
    total_ns: int


@dataclass
class NcclStreamInfo:
    """NCCL statistics per stream."""
    stream_id: int
    op_count: int
    total_ns: int
    avg_ns: float


@dataclass
class SqlNcclResult:
    """Result of nsys-sql nccl command."""
    streams: list[NcclStreamInfo]
    total_nccl_ns: int
    total_ops: int


def _load_sqlite_tables(conn: sqlite3.Connection) -> tuple[set[str], bool]:
    all_tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    return all_tables, "StringIds" in all_tables


@contextmanager
def _open_sqlite_profile(sqlite_path: str) -> Iterator[tuple[sqlite3.Connection, set[str], bool]]:
    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables, has_strings = _load_sqlite_tables(conn)
        yield conn, all_tables, has_strings


def _has_columns(conn: sqlite3.Connection, table_name: str, *column_names: str) -> bool:
    cols = set(_get_cols(conn, table_name))
    return all(column_name in cols for column_name in column_names)


def _table_time_bounds(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    include_total_ns: bool = False,
) -> tuple[int, int, int]:
    select_total = ", SUM([end] - start)" if include_total_ns else ""
    try:
        row = conn.execute(
            f"SELECT MIN(start), MAX([end]){select_total} FROM {table_name}"
        ).fetchone()
    except sqlite3.Error:
        return 0, 0, 0
    if not row or row[0] is None or row[1] is None:
        return 0, 0, 0
    total_ns = int(row[2] or 0) if include_total_ns else 0
    return int(row[0]), int(row[1]), total_ns


def _stream_span_stats(
    conn: sqlite3.Connection,
    kernel_tbl: str,
    stream_ids: set[int],
) -> tuple[int, int, int, int]:
    if not stream_ids:
        return 0, 0, 0, 0
    placeholders = ",".join("?" * len(stream_ids))
    try:
        row = conn.execute(
            f"""
            SELECT MIN(start), MAX([end]),
                   SUM([end] - start), COUNT(*)
            FROM {kernel_tbl}
            WHERE streamId IN ({placeholders})
            """,
            list(stream_ids),
        ).fetchone()
    except sqlite3.Error:
        return 0, 0, 0, 0
    if not row or row[0] is None or row[1] is None:
        return 0, 0, 0, 0
    return int(row[0]), int(row[1]), int(row[2] or 0), int(row[3] or 0)


def _kernel_name_parts(
    conn: sqlite3.Connection,
    kernel_tbl: str,
    has_strings: bool,
    alias: str = "k",
) -> tuple[str, str]:
    return _kernel_name_expr(conn, kernel_tbl, has_strings, alias=alias)


def _kernel_name_lookup_query(
    conn: sqlite3.Connection,
    kernel_tbl: str,
    has_strings: bool,
    *,
    where_clause: str,
    order_clause: str,
) -> str:
    name_expr, join_clause = _kernel_name_parts(conn, kernel_tbl, has_strings, alias="k")
    return f"""
        SELECT {name_expr} AS kernel_name
        FROM {kernel_tbl} k
        {join_clause}
        WHERE {where_clause}
        {order_clause}
        LIMIT 1
    """


def _kernel_rows_with_names_query(conn: sqlite3.Connection, kernel_tbl: str, has_strings: bool) -> str:
    name_expr, join_clause = _kernel_name_parts(conn, kernel_tbl, has_strings, alias="k")
    return f"""
        SELECT k.streamId,
               k.start,
               k.[end],
               {name_expr} AS kernel_name
        FROM {kernel_tbl} k
        {join_clause}
    """


def _kernel_name_like_any_sql(column_name: str, keywords: tuple[str, ...]) -> str:
    normalized = f"LOWER(COALESCE({column_name}, ''))"
    return " OR ".join(f"{normalized} LIKE '%{keyword}%'" for keyword in keywords)


def _nccl_kernel_name_sql(column_name: str = "kernel_name") -> str:
    return _kernel_name_like_any_sql(column_name, _NCCL_KEYWORDS)


def _stream_classification_query(conn: sqlite3.Connection, kernel_tbl: str, has_strings: bool) -> str:
    nccl_sql = _nccl_kernel_name_sql()
    return f"""
        SELECT streamId,
               SUM(CASE WHEN {nccl_sql} THEN 1 ELSE 0 END) AS nccl_count,
               SUM(CASE WHEN NOT ({nccl_sql}) THEN 1 ELSE 0 END) AS compute_count
        FROM ({_kernel_rows_with_names_query(conn, kernel_tbl, has_strings)})
        GROUP BY streamId
    """


def run_sql_schema(sqlite_path: str) -> SqlSchemaResult:
    """Query SQLite schema and capabilities.

    Args:
        sqlite_path: Path to .sqlite file

    Returns:
        SqlSchemaResult with capabilities, tables, gpu_devices
    """
    schema = inspect_schema(sqlite_path)
    return SqlSchemaResult(
        capabilities=schema.capabilities,
        tables=schema.tables,
        gpu_devices=[asdict(d) for d in schema.gpu_devices],
        warnings=schema.warnings,
    )


def run_sql_kernels(sqlite_path: str, limit: int = 20) -> SqlKernelsResult:
    """Query top-N GPU kernels by total time.

    Args:
        sqlite_path: Path to .sqlite file
        limit: Maximum number of kernels to return

    Returns:
        SqlKernelsResult with kernel list and totals
    """
    kernels: list[KernelInfo] = []
    total_kernel_ns = 0
    trace_duration_ns = 0

    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, has_strings):

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return SqlKernelsResult(kernels=[], total_kernel_ns=0, trace_duration_ns=0)

        name_expr, join_clause = _kernel_name_parts(conn, kernel_tbl, has_strings, alias="k")

        # Get trace duration
        trace_start, trace_end, total_kernel_ns = _table_time_bounds(
            conn, kernel_tbl, include_total_ns=True
        )
        if trace_end > trace_start:
            trace_duration_ns = trace_end - trace_start

        # Get top kernels
        sql = f"""
            SELECT {name_expr} AS kernel_name,
                   COUNT(*) AS invocations,
                   SUM(k.[end] - k.start) AS total_ns,
                   AVG(k.[end] - k.start) AS avg_ns,
                   MAX(k.[end] - k.start) AS max_ns
            FROM {kernel_tbl} k {join_clause}
            GROUP BY kernel_name
            ORDER BY total_ns DESC
            LIMIT {limit}
        """
        try:
            for row in conn.execute(sql).fetchall():
                kernels.append(KernelInfo(
                    name=row["kernel_name"] or "unknown",
                    count=int(row["invocations"]),
                    total_ns=int(row["total_ns"] or 0),
                    avg_ns=float(row["avg_ns"] or 0),
                    max_ns=int(row["max_ns"] or 0),
                ))
        except sqlite3.Error as e:
            logger.debug("run_sql_kernels failed: %s", e)

    return SqlKernelsResult(
        kernels=kernels,
        total_kernel_ns=total_kernel_ns,
        trace_duration_ns=trace_duration_ns,
    )


def run_sql_gaps(sqlite_path: str, min_gap_ns: int = 1_000_000, limit: int = 20) -> SqlGapsResult:
    """Query GPU idle gaps (bubbles).

    Args:
        sqlite_path: Path to .sqlite file
        min_gap_ns: Minimum gap duration in nanoseconds
        limit: Maximum number of gaps to return

    Returns:
        SqlGapsResult with gap list and totals
    """
    gaps: list[GapInfo] = []
    total_gap_ns = 0

    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, has_strings):

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return SqlGapsResult(gaps=[], total_gap_ns=0, gap_count=0)

        # Get all kernel intervals per stream
        interval_sql = f"""
            SELECT streamId, start, [end]
            FROM {kernel_tbl}
            ORDER BY streamId, start
        """
        intervals_by_stream: dict[int, list[tuple[int, int]]] = {}
        trace_start_ns = 0
        trace_end_ns = 0

        try:
            for row in conn.execute(interval_sql):
                sid = int(row["streamId"] or 0)
                s, e = int(row["start"]), int(row["end"])
                if sid not in intervals_by_stream:
                    intervals_by_stream[sid] = []
                intervals_by_stream[sid].append((s, e))
                if trace_start_ns == 0 or s < trace_start_ns:
                    trace_start_ns = s
                if e > trace_end_ns:
                    trace_end_ns = e
        except sqlite3.Error as e:
            logger.debug("run_sql_gaps failed: %s", e)
            return SqlGapsResult(gaps=[], total_gap_ns=0, gap_count=0)

        # Find gaps in each stream
        all_gaps: list[tuple[int, int, int, int]] = []  # (stream_id, start, end, duration)
        for sid, intervals in intervals_by_stream.items():
            if not intervals:
                continue
            # Use this stream's actual range, not global trace range
            stream_start = min(s for s, e in intervals)
            stream_end = max(e for s, e in intervals)
            stream_gaps = find_gaps(intervals, stream_start, stream_end, min_gap_ns)
            for gs, ge in stream_gaps:
                all_gaps.append((sid, gs, ge, ge - gs))

        # Sort by duration and take top
        all_gaps.sort(key=lambda x: x[3], reverse=True)
        top_gaps = all_gaps[:limit]

        # Get kernel names around gaps if available
        for sid, gs, ge, dur in top_gaps:
            before_kernel = None
            after_kernel = None
            total_gap_ns += dur

            try:
                before_row = conn.execute(
                    _kernel_name_lookup_query(
                        conn,
                        kernel_tbl,
                        has_strings,
                        where_clause="k.streamId = ? AND k.[end] <= ?",
                        order_clause="ORDER BY k.[end] DESC",
                    ),
                    (sid, gs),
                ).fetchone()
                if before_row:
                    before_kernel = before_row[0]
                after_row = conn.execute(
                    _kernel_name_lookup_query(
                        conn,
                        kernel_tbl,
                        has_strings,
                        where_clause="k.streamId = ? AND k.start >= ?",
                        order_clause="ORDER BY k.start ASC",
                    ),
                    (sid, ge),
                ).fetchone()
                if after_row:
                    after_kernel = after_row[0]
            except sqlite3.Error:
                pass

            gaps.append(GapInfo(
                stream_id=sid,
                gap_start_ns=gs,
                gap_end_ns=ge,
                gap_ns=dur,
                before_kernel=before_kernel,
                after_kernel=after_kernel,
            ))

    return SqlGapsResult(
        gaps=gaps,
        total_gap_ns=total_gap_ns,
        gap_count=len(all_gaps),
    )


def run_sql_sync(sqlite_path: str) -> SqlSyncResult:
    """Query CUDA synchronization events cost.

    Args:
        sqlite_path: Path to .sqlite file

    Returns:
        SqlSyncResult with sync event list and totals
    """
    sync_events: list[SyncEventInfo] = []
    total_sync_ns = 0
    sync_wall_pct = 0.0

    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, has_strings):

        sync_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
        if not sync_tbl:
            return SqlSyncResult(sync_events=[], total_sync_ns=0, sync_wall_pct=0.0)

        # Get trace duration from kernels
        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        trace_duration_ns = 0
        if kernel_tbl:
            trace_start, trace_end, _ = _table_time_bounds(conn, kernel_tbl)
            if trace_end > trace_start:
                trace_duration_ns = trace_end - trace_start

        # Query sync events - try with enum table first
        sync_enum_tbl = _find_table(all_tables, "ENUM_CUPTI_SYNC_TYPE")
        if sync_enum_tbl and has_strings:
            sql = f"""
                SELECT COALESCE(e.name, 'sync_' || s.syncType) AS sync_name,
                       COUNT(*) AS call_count,
                       SUM(s.[end] - s.start) AS total_ns,
                       AVG(s.[end] - s.start) AS avg_ns,
                       MAX(s.[end] - s.start) AS max_ns
                FROM {sync_tbl} s
                LEFT JOIN {sync_enum_tbl} e ON s.syncType = e.id
                GROUP BY sync_name
                ORDER BY total_ns DESC
            """
        else:
            sql = f"""
                SELECT 'sync_' || CAST(syncType AS TEXT) AS sync_name,
                       COUNT(*) AS call_count,
                       SUM([end] - start) AS total_ns,
                       AVG([end] - start) AS avg_ns,
                       MAX([end] - start) AS max_ns
                FROM {sync_tbl}
                GROUP BY sync_name
                ORDER BY total_ns DESC
            """

        try:
            for row in conn.execute(sql):
                total_ns = int(row["total_ns"] or 0)
                total_sync_ns += total_ns
                sync_events.append(SyncEventInfo(
                    sync_type=row["sync_name"] or "unknown",
                    count=int(row["call_count"]),
                    total_ns=total_ns,
                    avg_ns=float(row["avg_ns"] or 0),
                    max_ns=int(row["max_ns"] or 0),
                ))
        except sqlite3.Error as e:
            logger.debug("run_sql_sync failed: %s", e)

        # Calculate wall-clock percentage using union
        if trace_duration_ns > 0:
            try:
                sync_intervals = []
                for row in conn.execute(
                    f"SELECT start, [end] FROM {sync_tbl} WHERE [end] > start"
                ).fetchall():
                    sync_intervals.append((int(row["start"]), int(row["end"])))
                sync_wall_ns = min(union_ns(sync_intervals), trace_duration_ns)
                sync_wall_pct = round(sync_wall_ns / trace_duration_ns * 100, 1)
            except sqlite3.Error:
                pass

    return SqlSyncResult(
        sync_events=sync_events,
        total_sync_ns=total_sync_ns,
        sync_wall_pct=sync_wall_pct,
    )


def run_sql_nvtx(sqlite_path: str, limit: int = 20) -> SqlNvtxResult:
    """Query NVTX range breakdown.

    Args:
        sqlite_path: Path to .sqlite file
        limit: Maximum number of ranges to return

    Returns:
        SqlNvtxResult with NVTX range list and totals
    """
    nvtx_ranges: list[NvtxRangeInfo] = []
    total_nvtx_ns = 0

    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, has_strings):

        nvtx_tbl = _find_table(all_tables, "NVTX_EVENTS")
        if not nvtx_tbl:
            return SqlNvtxResult(nvtx_ranges=[], total_nvtx_ns=0)
        cols = _get_cols(conn, nvtx_tbl)
        has_textid = "textId" in cols

        if has_strings and has_textid:
            text_expr = "COALESCE(n.text, s.value)"
            join_clause = "LEFT JOIN StringIds s ON n.textId = s.id"
        else:
            text_expr = "n.text"
            join_clause = ""

        sql = f"""
            SELECT {text_expr} AS label,
                   COUNT(*) AS count,
                   SUM(n.[end] - n.start) AS total_ns,
                   AVG(n.[end] - n.start) AS avg_ns
            FROM {nvtx_tbl} n {join_clause}
            WHERE {text_expr} IS NOT NULL AND {text_expr} != '' AND n.[end] > n.start
            GROUP BY label
            ORDER BY total_ns DESC
            LIMIT {limit}
        """

        try:
            for row in conn.execute(sql):
                total_ns = int(row["total_ns"] or 0)
                total_nvtx_ns += total_ns
                nvtx_ranges.append(NvtxRangeInfo(
                    text=row["label"] or "",
                    count=int(row["count"]),
                    total_ns=total_ns,
                    avg_ns=float(row["avg_ns"] or 0),
                ))
        except sqlite3.Error as e:
            logger.debug("run_sql_nvtx failed: %s", e)

    return SqlNvtxResult(nvtx_ranges=nvtx_ranges, total_nvtx_ns=total_nvtx_ns)


def run_sql_memcpy(sqlite_path: str) -> SqlMemcpyResult:
    """Query memory bandwidth analysis.

    Args:
        sqlite_path: Path to .sqlite file

    Returns:
        SqlMemcpyResult with memcpy operations and totals
    """
    memcpy_ops: list[MemcpyInfo] = []
    total_bytes = 0
    total_ns = 0

    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, _):

        memcpy_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMCPY")
        if not memcpy_tbl:
            return SqlMemcpyResult(memcpy_ops=[], total_bytes=0, total_ns=0)

        if not _has_columns(conn, memcpy_tbl, "bytes", "copyKind"):
            return SqlMemcpyResult(memcpy_ops=[], total_bytes=0, total_ns=0)

        sql = f"""
            SELECT copyKind,
                   COUNT(*) AS count,
                   SUM(bytes) AS total_bytes,
                   SUM([end] - start) AS total_ns,
                   CASE WHEN SUM([end] - start) > 0
                        THEN SUM(bytes) / (SUM([end] - start) / 1e9) / 1e9
                        ELSE 0 END AS avg_bw_gbps
            FROM {memcpy_tbl}
            GROUP BY copyKind
            ORDER BY total_bytes DESC
        """

        try:
            for row in conn.execute(sql):
                ops_bytes = int(row["total_bytes"] or 0)
                ops_ns = int(row["total_ns"] or 0)
                total_bytes += ops_bytes
                total_ns += ops_ns
                kind = int(row["copyKind"] or 0)
                memcpy_ops.append(MemcpyInfo(
                    direction=_COPY_KIND_NAMES.get(kind, f"Kind{kind}"),
                    count=int(row["count"]),
                    total_bytes=ops_bytes,
                    total_ns=ops_ns,
                    avg_bw_gbps=float(row["avg_bw_gbps"] or 0),
                ))
        except sqlite3.Error as e:
            logger.debug("run_sql_memcpy failed: %s", e)

    return SqlMemcpyResult(memcpy_ops=memcpy_ops, total_bytes=total_bytes, total_ns=total_ns)


def run_sql_nccl(sqlite_path: str, limit: int = 20) -> SqlNcclResult:
    """Query NCCL communication breakdown.

    Args:
        sqlite_path: Path to .sqlite file
        limit: Maximum number of streams to return

    Returns:
        SqlNcclResult with NCCL streams and totals
    """
    streams: list[NcclStreamInfo] = []
    total_nccl_ns = 0
    total_ops = 0

    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, has_strings):

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return SqlNcclResult(streams=[], total_nccl_ns=0, total_ops=0)

        nccl_sql = _nccl_kernel_name_sql()
        sql = f"""
            SELECT streamId,
                   COUNT(*) AS op_count,
                   SUM([end] - start) AS total_ns,
                   AVG([end] - start) AS avg_ns
            FROM ({_kernel_rows_with_names_query(conn, kernel_tbl, has_strings)})
            WHERE {nccl_sql}
            GROUP BY streamId
            ORDER BY total_ns DESC
            LIMIT {limit}
        """

        try:
            for row in conn.execute(sql):
                ops = int(row["op_count"])
                ns = int(row["total_ns"] or 0)
                total_ops += ops
                total_nccl_ns += ns
                streams.append(NcclStreamInfo(
                    stream_id=int(row["streamId"]),
                    op_count=ops,
                    total_ns=ns,
                    avg_ns=float(row["avg_ns"] or 0),
                ))
        except sqlite3.Error as e:
            logger.debug("run_sql_nccl failed: %s", e)

    return SqlNcclResult(streams=streams, total_nccl_ns=total_nccl_ns, total_ops=total_ops)


# ── Kernel Launch Overhead ─────────────────────────────────────────────────────

@dataclass
class KernelLaunchEntry:
    """Single kernel launch overhead entry."""
    kernel_name: str
    api_ms: float
    kernel_ms: float
    overhead_us: float


@dataclass
class SqlKernelLaunchResult:
    """Result of nsys-sql kernel-launch command."""
    entries: list[KernelLaunchEntry]
    avg_overhead_us: float
    max_overhead_us: float


def run_sql_kernel_launch(sqlite_path: str, limit: int = 20) -> SqlKernelLaunchResult:
    """Query kernel launch overhead (gap between CUDA API call and GPU execution).

    Args:
        sqlite_path: Path to .sqlite file
        limit: Maximum number of entries to return

    Returns:
        SqlKernelLaunchResult with per-kernel launch overhead
    """
    entries: list[KernelLaunchEntry] = []

    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, has_strings):

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        runtime_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")

        if not kernel_tbl or not runtime_tbl:
            return SqlKernelLaunchResult(entries=[], avg_overhead_us=0.0, max_overhead_us=0.0)

        # Check that correlationId exists in both tables
        if not _has_columns(conn, kernel_tbl, "correlationId") or not _has_columns(
            conn, runtime_tbl, "correlationId"
        ):
            return SqlKernelLaunchResult(entries=[], avg_overhead_us=0.0, max_overhead_us=0.0)

        name_expr, kernel_join = _kernel_name_parts(conn, kernel_tbl, has_strings, alias="k")
        sql = f"""
            SELECT {name_expr} AS kernel_name,
                   ROUND((r.[end] - r.start) / 1e6, 3) AS api_ms,
                   ROUND((k.[end] - k.start) / 1e6, 3) AS kernel_ms,
                   ROUND((k.start - r.start) / 1e3, 1) AS overhead_us
            FROM {runtime_tbl} r
            JOIN {kernel_tbl} k ON r.correlationId = k.correlationId
            {kernel_join}
            WHERE k.start >= r.start
            ORDER BY overhead_us DESC
            LIMIT {limit}
        """
        try:
            for row in conn.execute(sql):
                entries.append(KernelLaunchEntry(
                    kernel_name=row["kernel_name"] or "unknown",
                    api_ms=float(row["api_ms"] or 0),
                    kernel_ms=float(row["kernel_ms"] or 0),
                    overhead_us=float(row["overhead_us"] or 0),
                ))
        except sqlite3.Error as e:
            logger.debug("run_sql_kernel_launch failed: %s", e)

    avg_us = sum(e.overhead_us for e in entries) / len(entries) if entries else 0.0
    max_us = max((e.overhead_us for e in entries), default=0.0)
    return SqlKernelLaunchResult(entries=entries, avg_overhead_us=avg_us, max_overhead_us=max_us)


# ── Stream Concurrency ─────────────────────────────────────────────────────────

@dataclass
class StreamStats:
    """Per-stream concurrency statistics."""
    stream_id: int
    kernel_count: int
    total_gpu_ms: float
    avg_kernel_us: float
    stream_span_ms: float
    stream_util_pct: float


@dataclass
class SqlStreamConcurrencyResult:
    """Result of nsys-sql stream-concurrency command."""
    streams: list[StreamStats]
    active_streams: int
    total_kernels: int
    global_span_ms: float
    sum_util_pct: float


def run_sql_stream_concurrency(sqlite_path: str, limit: int = 20) -> SqlStreamConcurrencyResult:
    """Query per-stream concurrency and utilization.

    Args:
        sqlite_path: Path to .sqlite file
        limit: Maximum number of streams to return

    Returns:
        SqlStreamConcurrencyResult with per-stream stats
    """
    streams: list[StreamStats] = []

    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, _):

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return SqlStreamConcurrencyResult(
                streams=[], active_streams=0, total_kernels=0, global_span_ms=0.0, sum_util_pct=0.0
            )

        sql = f"""
            WITH stream_summary AS (
                SELECT
                    k.streamId,
                    COUNT(*) AS kernel_count,
                    MIN(k.start) AS first_start,
                    MAX(k.[end]) AS last_end,
                    ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_gpu_ms,
                    ROUND(AVG(k.[end] - k.start) / 1e3, 1) AS avg_kernel_us
                FROM {kernel_tbl} k
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
                ROUND((s.last_end - s.first_start) / 1e6, 2) AS stream_span_ms,
                ROUND(s.total_gpu_ms / NULLIF((s.last_end - s.first_start) / 1e6, 0) * 100, 1)
                    AS stream_util_pct,
                g.active_streams,
                g.total_kernels,
                ROUND((g.global_end - g.global_start) / 1e6, 2) AS global_span_ms,
                ROUND(g.sum_gpu_ms / NULLIF((g.global_end - g.global_start) / 1e6, 0) * 100, 1)
                    AS sum_util_pct
            FROM stream_summary s, global_stats g
            ORDER BY s.total_gpu_ms DESC
            LIMIT {limit}
        """
        active_streams = 0
        total_kernels = 0
        global_span_ms = 0.0
        sum_util_pct = 0.0
        try:
            rows = conn.execute(sql).fetchall()
            if rows:
                r0 = rows[0]
                active_streams = int(r0["active_streams"] or 0)
                total_kernels = int(r0["total_kernels"] or 0)
                global_span_ms = float(r0["global_span_ms"] or 0)
                sum_util_pct = float(r0["sum_util_pct"] or 0)
            for row in rows:
                streams.append(StreamStats(
                    stream_id=int(row["streamId"]),
                    kernel_count=int(row["kernel_count"]),
                    total_gpu_ms=float(row["total_gpu_ms"] or 0),
                    avg_kernel_us=float(row["avg_kernel_us"] or 0),
                    stream_span_ms=float(row["stream_span_ms"] or 0),
                    stream_util_pct=float(row["stream_util_pct"] or 0),
                ))
        except sqlite3.Error as e:
            logger.debug("run_sql_stream_concurrency failed: %s", e)

    return SqlStreamConcurrencyResult(
        streams=streams,
        active_streams=active_streams,
        total_kernels=total_kernels,
        global_span_ms=global_span_ms,
        sum_util_pct=sum_util_pct,
    )


# ── Compute / NCCL Overlap (pure SQL estimate) ─────────────────────────────────

@dataclass
class SqlOverlapResult:
    """Result of nsys-sql overlap command.

    Note: This is a SQL-level approximation using per-stream kernel assignment.
    For precise interval-union overlap, use nsys-ai overlap_breakdown skill.
    """
    compute_only_ns: int
    nccl_only_ns: int
    overlap_ns: int
    total_span_ns: int
    overlap_pct: float
    compute_kernels: int
    nccl_kernels: int
    note: str


def run_sql_overlap(sqlite_path: str) -> SqlOverlapResult:
    """Estimate compute/NCCL overlap using stream-level SQL analysis.

    Streams that contain NCCL kernels are labeled NCCL streams; others are
    compute streams. Overlap is estimated from concurrent time spans between
    the two stream groups. This is an approximation — for precise interval
    intersection use nsys-ai's overlap_breakdown skill.

    Args:
        sqlite_path: Path to .sqlite file

    Returns:
        SqlOverlapResult with overlap estimates
    """
    with _open_sqlite_profile(sqlite_path) as (conn, all_tables, has_strings):

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")

        if not kernel_tbl:
            return SqlOverlapResult(
                compute_only_ns=0, nccl_only_ns=0, overlap_ns=0,
                total_span_ns=0, overlap_pct=0.0,
                compute_kernels=0, nccl_kernels=0,
                note="No kernel data available",
            )

        # Classify streams into compute vs nccl
        try:
            stream_class_rows = conn.execute(
                _stream_classification_query(conn, kernel_tbl, has_strings)
            ).fetchall()
        except sqlite3.Error as e:
            logger.debug("run_sql_overlap stream classification failed: %s", e)
            return SqlOverlapResult(
                compute_only_ns=0, nccl_only_ns=0, overlap_ns=0,
                total_span_ns=0, overlap_pct=0.0,
                compute_kernels=0, nccl_kernels=0,
                note=f"SQL error: {e}",
            )

        nccl_streams = set()
        compute_streams = set()
        for row in stream_class_rows:
            sid = int(row["streamId"])
            if int(row["nccl_count"] or 0) > 0:
                nccl_streams.add(sid)
            else:
                compute_streams.add(sid)

        if not nccl_streams:
            return SqlOverlapResult(
                compute_only_ns=0, nccl_only_ns=0, overlap_ns=0,
                total_span_ns=0, overlap_pct=0.0,
                compute_kernels=0, nccl_kernels=0,
                note="No NCCL kernels found in this profile",
            )

        c_start, c_end, c_ns, c_count = _stream_span_stats(conn, kernel_tbl, compute_streams)
        n_start, n_end, n_ns, n_count = _stream_span_stats(conn, kernel_tbl, nccl_streams)

        # Overlap = intersection of the two time spans (conservative estimate)
        overlap_start = max(c_start, n_start)
        overlap_end = min(c_end, n_end)
        overlap_ns = max(0, overlap_end - overlap_start)

        total_span_ns = max(c_end, n_end) - min(c_start, n_start) if (c_start and n_start) else 0
        nccl_span = n_end - n_start if n_end > n_start else 1
        overlap_pct = round(overlap_ns / nccl_span * 100, 1) if nccl_span > 0 else 0.0

        # Estimate exclusive time
        compute_only_ns = max(0, c_ns - overlap_ns)
        nccl_only_ns = max(0, n_ns - overlap_ns)

        return SqlOverlapResult(
            compute_only_ns=compute_only_ns,
            nccl_only_ns=nccl_only_ns,
            overlap_ns=overlap_ns,
            total_span_ns=total_span_ns,
            overlap_pct=overlap_pct,
            compute_kernels=c_count,
            nccl_kernels=n_count,
            note=(
                "Stream-span approximation. "
                "Use nsys-ai overlap_breakdown for precise interval-union analysis."
            ),
        )


def _emit_sql_result(result: object) -> None:
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))  # type: ignore[arg-type]


_SQL_RUNNERS: dict[str, tuple] = {
    "schema":             (run_sql_schema,             lambda a: {}),
    "kernels":            (run_sql_kernels,            lambda a: {"limit": a.limit}),
    "gaps":               (run_sql_gaps,               lambda a: {"min_gap_ns": a.min_gap_ns, "limit": a.limit}),
    "sync":               (run_sql_sync,               lambda a: {}),
    "nvtx":               (run_sql_nvtx,               lambda a: {"limit": a.limit}),
    "memcpy":             (run_sql_memcpy,             lambda a: {}),
    "nccl":               (run_sql_nccl,               lambda a: {"limit": a.limit}),
    "kernel-launch":      (run_sql_kernel_launch,      lambda a: {"limit": a.limit}),
    "stream-concurrency": (run_sql_stream_concurrency, lambda a: {"limit": a.limit}),
    "overlap":            (run_sql_overlap,            lambda a: {}),
}


_NSYS_SQL_DESCRIPTION = (
    "直接查询 nsys SQLite 数据库，输出结构化 JSON。\n\n"
    "每个子命令都是只读的，输出有界的 JSON，适合 agent/Codex 直接调用。\n\n"
    "  sysight nsys-sql schema <db>    — 列出表、列和能力\n"
    "  sysight nsys-sql kernels <db>   — Top-N GPU 内核（按总时间）\n"
    "  sysight nsys-sql gaps <db>      — GPU 空闲气泡分析\n"
    "  sysight nsys-sql sync <db>      — CUDA 同步事件代价\n"
    "  sysight nsys-sql nvtx <db>      — NVTX range 分解\n"
    "  sysight nsys-sql memcpy <db>    — 内存带宽分析\n"
    "  sysight nsys-sql nccl <db>      — NCCL 通信分解\n"
)


def _register_nsys_sql_subparsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    sc = sub.add_parser("schema", help="查询 SQLite schema 和能力")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("kernels", help="查询 Top-N GPU 内核")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回内核数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("gaps", help="查询 GPU 空闲气泡")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--min-gap-ns", type=int, default=1000000, help="最小间隙（ns）")
    sc.add_argument("--limit", type=int, default=20, help="返回间隙数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("sync", help="查询 CUDA 同步事件代价")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("nvtx", help="查询 NVTX range 分解")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 range 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("memcpy", help="查询内存带宽分析")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("nccl", help="查询 NCCL 通信分解")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 stream 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("kernel-launch", help="内核启动开销分析（API→GPU 延迟）")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回条目数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("stream-concurrency", help="Stream 并发率分析")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 stream 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    sc = sub.add_parser("overlap", help="Compute/NCCL 重叠估算")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")


def add_nsys_sql_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    sql_parser = subparsers.add_parser(
        "nsys-sql",
        help="直接查询 nsys SQLite 数据库（供 agent 使用）",
        description=_NSYS_SQL_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sql_sub = sql_parser.add_subparsers(dest="sql_cmd")
    _register_nsys_sql_subparsers(sql_sub)


def dispatch_nsys_sql(args: argparse.Namespace) -> bool:
    if getattr(args, "sql_cmd", None) is None:
        return False

    entry = _SQL_RUNNERS.get(args.sql_cmd)
    if entry is None:
        return False

    runner, build_kwargs = entry
    _emit_sql_result(runner(args.sqlite_path, **build_kwargs(args)))
    return True


def main_standalone_nsys_sql(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="sysight nsys-sql",
        description=_NSYS_SQL_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="sql_cmd")
    _register_nsys_sql_subparsers(sub)

    args = p.parse_args(argv)
    if not getattr(args, "sql_cmd", None):
        p.print_help()
        return

    if not dispatch_nsys_sql(args):
        print("错误：未知的 nsys-sql 子命令", file=sys.stderr)
        raise SystemExit(1)
