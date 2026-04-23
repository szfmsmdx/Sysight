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

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from typing import Any

from .extract import inspect_schema, find_gaps, union_ns, intersect_total

logger = logging.getLogger(__name__)

_NCCL_KEYWORDS = ("nccl", "allreduce", "allgather", "reducescatter", "broadcast", "sendrecv", "reduce")
_COPY_KIND_NAMES = {0: "Unknown", 1: "H2D", 2: "D2H", 4: "H2H", 8: "D2D", 10: "P2P"}


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


def _find_table(all_tables: set[str], prefix: str) -> str | None:
    """Find table by exact name or prefix."""
    if prefix in all_tables:
        return prefix
    for t in sorted(all_tables):
        if t.startswith(prefix):
            return t
    return None


def _get_cols(conn: sqlite3.Connection, tbl: str) -> list[str]:
    """Get column names for a table."""
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    except sqlite3.Error:
        return []


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

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return SqlKernelsResult(kernels=[], total_kernel_ns=0, trace_duration_ns=0)

        has_strings = "StringIds" in all_tables
        cols = _get_cols(conn, kernel_tbl)
        has_demangled = "demangledName" in cols

        if has_strings and has_demangled:
            name_expr = "COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS TEXT))"
            join_clause = "LEFT JOIN StringIds s ON k.shortName = s.id LEFT JOIN StringIds d ON k.demangledName = d.id"
        elif has_strings:
            name_expr = "COALESCE(s.value, 'kernel_' || CAST(k.shortName AS TEXT))"
            join_clause = "LEFT JOIN StringIds s ON k.shortName = s.id"
        else:
            name_expr = "CAST(k.shortName AS TEXT)"
            join_clause = ""

        # Get trace duration
        dur_row = conn.execute(
            f"SELECT MIN(start), MAX([end]), SUM([end] - start) FROM {kernel_tbl}"
        ).fetchone()
        if dur_row and dur_row[0] is not None:
            trace_start = int(dur_row[0])
            trace_end = int(dur_row[1])
            trace_duration_ns = trace_end - trace_start
            total_kernel_ns = int(dur_row[2] or 0)

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

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return SqlGapsResult(gaps=[], total_gap_ns=0, gap_count=0)

        has_strings = "StringIds" in all_tables

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

            if has_strings:
                try:
                    # Before kernel
                    before_row = conn.execute(f"""
                        SELECT s.value FROM {kernel_tbl} k
                        JOIN StringIds s ON k.shortName = s.id
                        WHERE k.streamId = {sid} AND k.[end] <= {gs}
                        ORDER BY k.[end] DESC LIMIT 1
                    """).fetchone()
                    if before_row:
                        before_kernel = before_row[0]
                    # After kernel
                    after_row = conn.execute(f"""
                        SELECT s.value FROM {kernel_tbl} k
                        JOIN StringIds s ON k.shortName = s.id
                        WHERE k.streamId = {sid} AND k.start >= {ge}
                        ORDER BY k.start ASC LIMIT 1
                    """).fetchone()
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

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        sync_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
        if not sync_tbl:
            return SqlSyncResult(sync_events=[], total_sync_ns=0, sync_wall_pct=0.0)

        has_strings = "StringIds" in all_tables

        # Get trace duration from kernels
        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        trace_duration_ns = 0
        if kernel_tbl:
            dur_row = conn.execute(
                f"SELECT MIN(start), MAX([end]) FROM {kernel_tbl}"
            ).fetchone()
            if dur_row and dur_row[0] is not None:
                trace_duration_ns = int(dur_row[1]) - int(dur_row[0])

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

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        nvtx_tbl = _find_table(all_tables, "NVTX_EVENTS")
        if not nvtx_tbl:
            return SqlNvtxResult(nvtx_ranges=[], total_nvtx_ns=0)

        has_strings = "StringIds" in all_tables
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

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        memcpy_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMCPY")
        if not memcpy_tbl:
            return SqlMemcpyResult(memcpy_ops=[], total_bytes=0, total_ns=0)

        cols = _get_cols(conn, memcpy_tbl)
        if "bytes" not in cols or "copyKind" not in cols:
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

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return SqlNcclResult(streams=[], total_nccl_ns=0, total_ops=0)

        has_strings = "StringIds" in all_tables
        if not has_strings:
            return SqlNcclResult(streams=[], total_nccl_ns=0, total_ops=0)

        # Build LIKE clauses for NCCL keywords
        like_clauses = " OR ".join(f"LOWER(s.value) LIKE '%{kw}%'" for kw in _NCCL_KEYWORDS)

        sql = f"""
            SELECT k.streamId,
                   COUNT(*) AS op_count,
                   SUM(k.[end] - k.start) AS total_ns,
                   AVG(k.[end] - k.start) AS avg_ns
            FROM {kernel_tbl} k
            JOIN StringIds s ON k.shortName = s.id
            WHERE ({like_clauses})
            GROUP BY k.streamId
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

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        runtime_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
        has_strings = "StringIds" in all_tables

        if not kernel_tbl or not runtime_tbl or not has_strings:
            return SqlKernelLaunchResult(entries=[], avg_overhead_us=0.0, max_overhead_us=0.0)

        # Check that correlationId exists in both tables
        kernel_cols = _get_cols(conn, kernel_tbl)
        runtime_cols = _get_cols(conn, runtime_tbl)
        if "correlationId" not in kernel_cols or "correlationId" not in runtime_cols:
            return SqlKernelLaunchResult(entries=[], avg_overhead_us=0.0, max_overhead_us=0.0)

        sql = f"""
            SELECT s.value AS kernel_name,
                   ROUND((r.[end] - r.start) / 1e6, 3) AS api_ms,
                   ROUND((k.[end] - k.start) / 1e6, 3) AS kernel_ms,
                   ROUND((k.start - r.start) / 1e3, 1) AS overhead_us
            FROM {runtime_tbl} r
            JOIN {kernel_tbl} k ON r.correlationId = k.correlationId
            JOIN StringIds s ON k.shortName = s.id
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

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

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
    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        has_strings = "StringIds" in all_tables

        if not kernel_tbl or not has_strings:
            return SqlOverlapResult(
                compute_only_ns=0, nccl_only_ns=0, overlap_ns=0,
                total_span_ns=0, overlap_pct=0.0,
                compute_kernels=0, nccl_kernels=0,
                note="No kernel or string data available",
            )

        like_clauses = " OR ".join(f"LOWER(s.value) LIKE '%{kw}%'" for kw in _NCCL_KEYWORDS)

        # Classify streams into compute vs nccl
        try:
            stream_class_rows = conn.execute(f"""
                SELECT k.streamId,
                    SUM(CASE WHEN {like_clauses} THEN 1 ELSE 0 END) AS nccl_count,
                    SUM(CASE WHEN NOT ({like_clauses}) THEN 1 ELSE 0 END) AS compute_count
                FROM {kernel_tbl} k
                JOIN StringIds s ON k.shortName = s.id
                GROUP BY k.streamId
            """).fetchall()
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

        def stream_span(stream_ids: set[int]) -> tuple[int, int, int, int]:
            """Return (min_start, max_end, total_ns_sum, kernel_count) for given streams."""
            if not stream_ids:
                return 0, 0, 0, 0
            placeholders = ",".join("?" * len(stream_ids))
            try:
                row = conn.execute(f"""
                    SELECT MIN(start), MAX([end]),
                           SUM([end] - start), COUNT(*)
                    FROM {kernel_tbl}
                    WHERE streamId IN ({placeholders})
                """, list(stream_ids)).fetchone()
                if row and row[0] is not None:
                    return int(row[0]), int(row[1]), int(row[2] or 0), int(row[3] or 0)
            except sqlite3.Error:
                pass
            return 0, 0, 0, 0

        c_start, c_end, c_ns, c_count = stream_span(compute_streams)
        n_start, n_end, n_ns, n_count = stream_span(nccl_streams)

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
