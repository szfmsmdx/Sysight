"""nsys/extract.py — T1/T2/T3 + interval math for Nsight Systems profiles.

Merged from input.py, schema.py, intervals.py for fewer files.

  resolve_profile_input()  T1: .nsys-rep → sqlite path
  inspect_schema()         T2: probe SQLite schema → capabilities
  extract_trace()          T3: sqlite → NsysTrace of TimelineEvents
  union_intervals() etc.   Interval math for bottleneck classification
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .models import GpuDeviceInfo, NsysTrace, SchemaInfo, TimelineEvent, ProfileInput

logger = logging.getLogger(__name__)


# ── T1: Profile input resolution ──────────────────────────────────────────────

def resolve_profile_input(
    profile_path: str,
    sqlite_path: str | None = None,
) -> ProfileInput:
    """Resolve profile path to a usable SQLite file.

    Priority: explicit sqlite_path > sibling .sqlite > direct .sqlite path.
    Returns ProfileInput with action_required=True if no sqlite found.
    """
    original = str(profile_path)
    p = Path(profile_path)

    if sqlite_path is not None:
        sq = Path(sqlite_path)
        if sq.is_file() and sq.stat().st_size > 0:
            return ProfileInput(original_path=original, sqlite_path=str(sq), action_required=False)
        return ProfileInput(original_path=original, sqlite_path=str(sq), action_required=True,
                            reason=f"指定的 sqlite_path 不存在或为空：{sq}")

    if p.suffix.lower() == ".nsys-rep":
        sibling = p.with_suffix(".sqlite")
        if sibling.is_file() and sibling.stat().st_size > 0:
            return ProfileInput(original_path=original, sqlite_path=str(sibling), action_required=False)
        return ProfileInput(
            original_path=original, sqlite_path=None, action_required=True,
            reason=f"需要导出 SQLite，请运行： nsys export -t sqlite {p} -o {sibling} --force-overwrite=true",
        )

    if p.is_file() and p.stat().st_size > 0:
        return ProfileInput(original_path=original, sqlite_path=str(p), action_required=False)

    return ProfileInput(original_path=original, sqlite_path=None, action_required=True,
                        reason=f"Profile 文件不存在或为空：{p}")


# ── T2: Schema inspection ─────────────────────────────────────────────────────

_CAPABILITY_TABLES: dict[str, list[str]] = {
    "cuda_kernel":  ["CUPTI_ACTIVITY_KIND_KERNEL"],
    "cuda_memcpy":  ["CUPTI_ACTIVITY_KIND_MEMCPY"],
    "cuda_memset":  ["CUPTI_ACTIVITY_KIND_MEMSET"],
    "cuda_runtime": ["CUPTI_ACTIVITY_KIND_RUNTIME"],
    "cuda_sync":    ["CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"],
    "nvtx":         ["NVTX_EVENTS"],
    "cpu_sample":   ["COMPOSITE_EVENTS"],
    "cudnn":        ["CUDNN_EVENTS"],
    "osrt":         ["OSRT_API"],
    "mpi":          ["MPI_", "MPIP_"],
    "gpu_metrics":  ["GPU_METRICS", "TARGET_INFO_GPU_METRICS"],
    "file_access":  ["FILE_ACCESS"],
    "string_ids":   ["StringIds"],
}
_REQUIRES_STRING_IDS = {"cuda_kernel", "nvtx", "cpu_sample", "cudnn"}
_SKIP_COLUMN_READ = {"OSRT_CALLCHAINS", "SAMPLING_CALLCHAINS", "NVTX_PAYLOAD_SCHEMAS"}
_KNOWN_PREFIXES = {"CUPTI_", "NVTX_", "OSRT_", "MPI", "StringIds",
                   "META_DATA", "TARGET_INFO", "GPU_METRICS", "FILE_ACCESS",
                   "COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS", "ENUM_",
                   "ANALYSIS_", "CUDA_GPU_", "CUDNN_", "DIAGNOSTIC_",
                   "PROCESSES", "ProcessStreams", "SCHED_", "ThreadNames",
                   "PROFILER_OVERHEAD"}


def inspect_schema(sqlite_path: str) -> SchemaInfo:
    """T2: Probe SQLite schema and return capabilities."""
    warnings: list[str] = []
    tables: dict[str, list[str]] = {}
    table_roles: dict[str, str] = {}
    capabilities: list[str] = []

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        all_tables = {row[0] for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        for tbl in sorted(all_tables):
            if tbl in _SKIP_COLUMN_READ:
                tables[tbl] = []
                continue
            try:
                tables[tbl] = [row[1] for row in cur.execute(f"PRAGMA table_info({tbl})").fetchall()]
            except sqlite3.Error as e:
                warnings.append(f"读取表 {tbl} 列信息失败：{e}")
                tables[tbl] = []

        has_string_ids = "StringIds" in all_tables
        for cap, prefixes in _CAPABILITY_TABLES.items():
            found: str | None = None
            for prefix in prefixes:
                if prefix in all_tables:
                    found = prefix; break
                if prefix.endswith("_"):
                    for tbl in sorted(all_tables):
                        if tbl.startswith(prefix):
                            found = tbl; break
                if found:
                    break
            if found:
                if cap in _REQUIRES_STRING_IDS and not has_string_ids:
                    warnings.append(f"能力 {cap!r} 依赖 StringIds 表（未找到），符号名称解析不可用")
                capabilities.append(cap)
                table_roles[cap] = found

        version_hint = _detect_nsys_version(conn, all_tables)
        gpu_devices = _extract_gpu_devices(conn, tables)

        unknown = [t for t in sorted(all_tables) if not any(t.startswith(p) for p in _KNOWN_PREFIXES)]
        if unknown:
            warnings.append(f"发现未知表（可能是更新版本的 nsys schema）：{unknown}")

    return SchemaInfo(
        tables=tables,
        capabilities=capabilities,
        table_roles=table_roles,
        version_hint=version_hint,
        warnings=warnings,
        gpu_devices=gpu_devices,
    )


def _detect_nsys_version(conn: sqlite3.Connection, all_tables: set[str]) -> str:
    for tbl in sorted(t for t in all_tables if t.startswith("META_DATA")):
        try:
            for row in conn.execute(f"SELECT * FROM {tbl} LIMIT 50").fetchall():
                d = dict(row)
                for k, v in d.items():
                    if v and "Nsight Systems" in str(v):
                        return str(v)
                    if ("version" in str(k).lower() or "exporter" in str(k).lower()) and v:
                        return str(v)
        except sqlite3.Error:
            continue
    return "unknown"


def _extract_gpu_devices(conn: sqlite3.Connection, tables: dict[str, list[str]]) -> list[GpuDeviceInfo]:
    table_name = "TARGET_INFO_GPU"
    if table_name not in tables:
        return []

    cols = set(tables.get(table_name, []))
    select_cols = [
        col for col in (
            "id", "name", "totalMemory", "memoryBandwidth", "smCount",
            "computeMajor", "computeMinor", "busLocation", "chipName",
        )
        if col in cols
    ]
    if not select_cols:
        return []

    try:
        rows = conn.execute(
            f"SELECT {', '.join(select_cols)} FROM {table_name} ORDER BY id"
        ).fetchall()
    except sqlite3.Error:
        return []

    devices: list[GpuDeviceInfo] = []
    for row in rows:
        data = dict(row)
        major = data.get("computeMajor")
        minor = data.get("computeMinor")
        compute = None
        if major is not None and minor is not None:
            compute = f"{int(major)}.{int(minor)}"
        devices.append(GpuDeviceInfo(
            device_id=int(data.get("id", len(devices))),
            name=str(data.get("name") or "unknown GPU"),
            total_memory_bytes=_int_or_none_dict(data, "totalMemory"),
            memory_bandwidth_bytes_per_s=_int_or_none_dict(data, "memoryBandwidth"),
            sm_count=_int_or_none_dict(data, "smCount"),
            compute_capability=compute,
            bus_location=str(data.get("busLocation") or "") or None,
            chip_name=str(data.get("chipName") or "") or None,
        ))
    return devices


# ── Interval math ─────────────────────────────────────────────────────────────

Interval = tuple[int, int]   # (start_ns, end_ns)


def _int_or_none_dict(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def union_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping intervals → sorted non-overlapping list."""
    if not intervals:
        return []
    merged: list[Interval] = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        ps, pe = merged[-1]
        if start <= pe:
            merged[-1] = (ps, max(pe, end))
        else:
            merged.append((start, end))
    return merged


def total_covered(merged: list[Interval]) -> int:
    return sum(e - s for s, e in merged)


def union_ns(intervals: list[Interval]) -> int:
    return total_covered(union_intervals(intervals))


def intersect_intervals(a: list[Interval], b: list[Interval]) -> list[Interval]:
    ma, mb = union_intervals(a), union_intervals(b)
    if not ma or not mb:
        return []
    result: list[Interval] = []
    j = 0
    for a_start, a_end in ma:
        while j < len(mb) and mb[j][1] <= a_start:
            j += 1
        k = j
        while k < len(mb) and mb[k][0] < a_end:
            os, oe = max(a_start, mb[k][0]), min(a_end, mb[k][1])
            if os < oe:
                result.append((os, oe))
            k += 1
    return result


def intersect_total(a: list[Interval], b: list[Interval]) -> int:
    return total_covered(intersect_intervals(a, b))


def find_gaps(
    intervals: list[Interval],
    trace_start_ns: int,
    trace_end_ns: int,
    min_gap_ns: int = 1_000_000,
) -> list[Interval]:
    """Find idle gaps >= min_gap_ns within [trace_start_ns, trace_end_ns]."""
    if not intervals or trace_end_ns <= trace_start_ns:
        if trace_end_ns - trace_start_ns >= min_gap_ns:
            return [(trace_start_ns, trace_end_ns)]
        return []
    gaps: list[Interval] = []
    cursor = trace_start_ns
    for iv_s, iv_e in union_intervals(intervals):
        iv_s, iv_e = max(iv_s, trace_start_ns), min(iv_e, trace_end_ns)
        if iv_e <= cursor:
            continue
        if iv_s > cursor and iv_s - cursor >= min_gap_ns:
            gaps.append((cursor, iv_s))
        cursor = max(cursor, iv_e)
    if cursor < trace_end_ns and trace_end_ns - cursor >= min_gap_ns:
        gaps.append((cursor, trace_end_ns))
    return gaps

# ── NCCL kernel classification (mirrors nsys-ai/overlap.py) ──────────────────

_NCCL_KEYWORDS = frozenset({
    "nccl", "allreduce", "allgather", "reducescatter",
    "broadcast", "sendrecv", "reduce",
})

_SYNC_API_NAMES = frozenset({
    "cudaDeviceSynchronize", "cudaStreamSynchronize",
    "cudaEventSynchronize", "cudaEventQuery",
    "cuStreamSynchronize", "cuDeviceSynchronize",
    "cudaMemcpy", "cudaMemcpyAsync",
    "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol",
    "cudaMemset", "cudaMemsetAsync",
})


def _is_comm_kernel(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _NCCL_KEYWORDS)


# ── Main extractor ────────────────────────────────────────────────────────────


def extract_trace(
    sqlite_path: str,
    schema: SchemaInfo,
    *,
    max_events: int = 0,
) -> NsysTrace:
    """T3: Extract timeline events from a Nsight Systems SQLite file.

    Args:
        sqlite_path:  Path to .sqlite export.
        schema:       Output of inspect_schema() for this file.
        max_events:   已弃用参数，保留以兼容旧调用，设置为 0 表示不限制。
                      详细分析由 classify.py 直接查询 SQLite 完成，无需在内存中
                      加载全量事件。

    Returns NsysTrace with all events and trace-level metadata.
    """
    warnings: list[str] = list(schema.warnings)  # carry forward schema warnings
    events: list[TimelineEvent] = []

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row

        # Determine trace time range from kernel table
        kernel_tbl = schema.table_roles.get("cuda_kernel")
        runtime_tbl = schema.table_roles.get("cuda_runtime")
        duration_ns = 0
        trace_start = 0

        trace_end = 0
        if kernel_tbl:
            row = conn.execute(
                f"SELECT MIN(start), MAX([end]) FROM {kernel_tbl}"
            ).fetchone()
            if row and row[0] is not None:
                trace_start = int(row[0])
                trace_end = int(row[1])
                duration_ns = trace_end - trace_start
        elif runtime_tbl:
            row = conn.execute(
                f"SELECT MIN(start), MAX([end]) FROM {runtime_tbl}"
            ).fetchone()
            if row and row[0] is not None:
                trace_start = int(row[0])
                trace_end = int(row[0]) + (int(row[1]) - int(row[0]))
                duration_ns = trace_end - trace_start

        if duration_ns == 0:
            warnings.append("无法确定 trace 时长（未找到 kernel/runtime 数据）")

        # 提取各类别事件——不设上限。
        # 注意：对于超大 profile，内核数量可能达到数十万乃至数百万。
        # classify.py 会直接查询 SQLite 做统计分析，这里只需加载
        # 足够的区间信息用于瓶颈分类即可。
        # 为了节省内存，对于纯统计目的我们做聚合查询。

        if "cuda_kernel" in schema.capabilities:
            n = _extract_kernels(conn, schema, events, 0, warnings)

        if "cuda_memcpy" in schema.capabilities:
            n = _extract_memcpy(conn, schema, events, 0, warnings)

        if "cuda_memset" in schema.capabilities:
            n = _extract_memset(conn, schema, events, 0, warnings)

        if "cuda_runtime" in schema.capabilities:
            n = _extract_runtime(conn, schema, events, 0, warnings)

        if "cuda_sync" in schema.capabilities:
            n = _extract_sync(conn, schema, events, 0, warnings)

        if "nvtx" in schema.capabilities:
            n = _extract_nvtx(conn, schema, events, 0, warnings)

        if "cpu_sample" in schema.capabilities:
            n = _extract_cpu_samples(conn, schema, events, 0, warnings)

        if "cudnn" in schema.capabilities:
            n = _extract_cudnn(conn, schema, events, 0, warnings)

        if "osrt" in schema.capabilities:
            n = _extract_osrt(conn, schema, events, 0, warnings)

    return NsysTrace(
        tool="nsys",
        profile_path=sqlite_path,
        sqlite_path=sqlite_path,
        duration_ns=duration_ns,
        trace_start_ns=trace_start,
        trace_end_ns=trace_end,
        events=events,
        schema_version=schema.version_hint,
        warnings=warnings,
    )


# ── Per-table extractors ──────────────────────────────────────────────────────

def _has_col(schema: SchemaInfo, tbl: str, col: str) -> bool:
    return col in schema.tables.get(tbl, [])


def _sel_col(
    schema: SchemaInfo, tbl: str, col: str,
    alias: str | None = None, prefix: str = ""
) -> str:
    """Return '{prefix}col [AS alias]' if column exists, else 'NULL AS alias'."""
    label = alias or col
    if _has_col(schema, tbl, col):
        ref = f"{prefix}{col}" if prefix else col
        return f"{ref} AS {label}" if (alias or prefix) else col
    return f"NULL AS {label}"


def _extract_kernels(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    tbl = schema.table_roles["cuda_kernel"]
    has_strings = "string_ids" in schema.capabilities
    name_expr = "s.value AS name" if has_strings else "CAST(k.shortName AS TEXT) AS name"
    join = "JOIN StringIds s ON k.shortName = s.id" if has_strings else ""
    limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""
    sql = f"""
        SELECT k.start, k.[end],
               {_sel_col(schema, tbl, 'deviceId', prefix='k.')},
               {_sel_col(schema, tbl, 'streamId', prefix='k.')},
               {_sel_col(schema, tbl, 'correlationId', prefix='k.')},
               {_sel_col(schema, tbl, 'globalTid', prefix='k.')},
               {name_expr}
        FROM {tbl} k {join}
        ORDER BY k.start {limit_clause}
    """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            name = row["name"] or "unknown_kernel"
            cat = "gpu_comm" if _is_comm_kernel(name) else "gpu_compute"
            dur = int(row["end"]) - int(row["start"])
            events.append(TimelineEvent(
                category=cat, name=name,
                start_ns=int(row["start"]), dur_ns=dur,
                device_id=_int_or_none(row, "deviceId"),
                stream_id=_int_or_none(row, "streamId"),
                correlation_id=_int_or_none(row, "correlationId"),
                global_tid=_int_or_none(row, "globalTid"),
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取 kernel 事件失败（{tbl}）：{e}")
    return count


def _extract_memcpy(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    tbl = schema.table_roles["cuda_memcpy"]
    _KIND_NAMES = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "HtoH"}
    sql = f"""
        SELECT start, [end],
               {_sel_col(schema, tbl, 'deviceId')},
               {_sel_col(schema, tbl, 'streamId')},
               {_sel_col(schema, tbl, 'correlationId')},
               {_sel_col(schema, tbl, 'globalTid')},
               {_sel_col(schema, tbl, 'copyKind')},
               {_sel_col(schema, tbl, 'bytes')}
        FROM {tbl} ORDER BY start
    """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            kind = int(row["copyKind"]) if row["copyKind"] is not None else 0
            dur = int(row["end"]) - int(row["start"])
            size = int(row["bytes"]) if row["bytes"] is not None else 0
            events.append(TimelineEvent(
                category="gpu_memcpy",
                name=f"memcpy_{_KIND_NAMES.get(kind, f'kind{kind}')}",
                start_ns=int(row["start"]), dur_ns=dur,
                device_id=_int_or_none(row, "deviceId"),
                stream_id=_int_or_none(row, "streamId"),
                correlation_id=_int_or_none(row, "correlationId"),
                global_tid=_int_or_none(row, "globalTid"),
                extra={"size_bytes": size, "copy_kind": kind},
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取 memcpy 事件失败（{tbl}）：{e}")
    return count


def _extract_memset(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    tbl = schema.table_roles["cuda_memset"]
    sql = f"""
        SELECT start, [end],
               {_sel_col(schema, tbl, 'deviceId')},
               {_sel_col(schema, tbl, 'streamId')},
               {_sel_col(schema, tbl, 'correlationId')},
               {_sel_col(schema, tbl, 'globalTid')},
               {_sel_col(schema, tbl, 'bytes')}
        FROM {tbl} ORDER BY start
    """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            dur = int(row["end"]) - int(row["start"])
            size = int(row["bytes"]) if row["bytes"] is not None else 0
            events.append(TimelineEvent(
                category="gpu_memcpy", name="memset",
                start_ns=int(row["start"]), dur_ns=dur,
                device_id=_int_or_none(row, "deviceId"),
                stream_id=_int_or_none(row, "streamId"),
                correlation_id=_int_or_none(row, "correlationId"),
                global_tid=_int_or_none(row, "globalTid"),
                extra={"size_bytes": size, "copy_kind": -1},
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取 memset 事件失败（{tbl}）：{e}")
    return count


def _extract_runtime(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    tbl = schema.table_roles["cuda_runtime"]
    has_strings = "string_ids" in schema.capabilities
    name_expr = "s.value AS name" if has_strings else "CAST(r.nameId AS TEXT) AS name"
    join = "JOIN StringIds s ON r.nameId = s.id" if has_strings else ""
    sql = f"""
        SELECT r.start, r.[end],
               {_sel_col(schema, tbl, 'globalTid', prefix='r.')},
               {_sel_col(schema, tbl, 'correlationId', prefix='r.')},
               {name_expr}
        FROM {tbl} r {join}
        ORDER BY r.start
    """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            dur = int(row["end"]) - int(row["start"])
            events.append(TimelineEvent(
                category="cuda_api", name=row["name"] or "cuda_api",
                start_ns=int(row["start"]), dur_ns=dur,
                global_tid=_int_or_none(row, "globalTid"),
                correlation_id=_int_or_none(row, "correlationId"),
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取 CUDA Runtime 事件失败（{tbl}）：{e}")
    return count


def _extract_sync(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    tbl = schema.table_roles["cuda_sync"]
    col_info = schema.tables.get(tbl, [])
    type_col = "type" if "type" in col_info else ("syncType" if "syncType" in col_info else None)
    sync_expr = f"{type_col} AS sync_type" if type_col else "0 AS sync_type"
    sql = f"""
        SELECT start, [end],
               {_sel_col(schema, tbl, 'globalTid')},
               {_sel_col(schema, tbl, 'correlationId')},
               {sync_expr}
        FROM {tbl} ORDER BY start
    """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            dur = int(row["end"]) - int(row["start"])
            events.append(TimelineEvent(
                category="sync_wait", name="cuda_sync",
                start_ns=int(row["start"]), dur_ns=dur,
                global_tid=_int_or_none(row, "globalTid"),
                correlation_id=_int_or_none(row, "correlationId"),
                extra={"sync_type": int(row["sync_type"] or 0)},
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取同步事件失败（{tbl}）：{e}")
    return count


def _extract_nvtx(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    """Extract NVTX ranges → nvtx.  Handles textId schema variant."""
    # Detect textId vs inline text
    tbl = schema.table_roles["nvtx"]
    col_info = schema.tables.get(tbl, [])
    has_textid = "textId" in col_info

    gtid = _sel_col(schema, tbl, "globalTid", prefix="n.")
    if has_textid:
        sql = f"""
            SELECT n.start, n.[end], {gtid},
                   COALESCE(n.text, s.value) AS text
            FROM {tbl} n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE (n.text IS NOT NULL OR s.value IS NOT NULL) AND n.[end] > n.start
            ORDER BY n.start
        """
    else:
        sql = f"""
            SELECT start, [end], {gtid}, text
            FROM {tbl}
            WHERE text IS NOT NULL AND [end] > start
            ORDER BY start
        """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            dur = int(row["end"]) - int(row["start"])
            events.append(TimelineEvent(
                category="nvtx", name=row["text"] or "",
                start_ns=int(row["start"]), dur_ns=dur,
                global_tid=_int_or_none(row, "globalTid"),
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取 NVTX 事件失败（{tbl}）：{e}")
    return count


def _extract_cpu_samples(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    """Extract CPU samples from COMPOSITE_EVENTS → cpu_sample (point events).

    CPU samples are point events (dur_ns=0, is_sample=True) and must NOT
    participate in interval-union calculations.
    """
    tbl = schema.table_roles.get("cpu_sample", "COMPOSITE_EVENTS")

    # COMPOSITE_EVENTS schema: typically has globalTid and timestamp
    col_info = schema.tables.get(tbl, [])
    has_timestamp = "timestamp" in col_info
    has_start = "start" in col_info

    if has_timestamp:
        time_expr = "timestamp"
    elif has_start:
        time_expr = "start"
    else:
        warnings.append("COMPOSITE_EVENTS 表无 timestamp/start 列，跳过 CPU 采样数据")
        return 0

    has_strings = "string_ids" in schema.capabilities
    gtid = _sel_col(schema, tbl, "globalTid")
    sampling_tbl = "SAMPLING_CALLCHAINS"
    tables_present = set(schema.tables.keys())
    if has_strings and sampling_tbl in tables_present:
        # Fetch all stack frames per sample, ordered by stackDepth ascending.
        # stackDepth=0 is the leaf (innermost) frame; higher depths are callers.
        # We group frames per sample using the COMPOSITE_EVENTS timestamp as key.
        #
        # Strategy: fetch all (ts, globalTid, stackDepth, symbol) rows at once,
        # then group by ts in Python.  This avoids N+1 queries.
        sampling_cols = set(schema.tables.get(sampling_tbl, []))
        module_select = "m.value AS module" if "module" in sampling_cols else "NULL AS module"
        module_join = "LEFT JOIN StringIds m ON sc.module = m.id" if "module" in sampling_cols else ""
        sql = f"""
            SELECT ce.{time_expr} AS ts, {gtid},
                   sc.stackDepth AS depth,
                   s.value AS symbol,
                   {module_select}
            FROM {tbl} ce
            JOIN {sampling_tbl} sc ON ce.id = sc.id
            LEFT JOIN StringIds s ON sc.symbol = s.id
            {module_join}
            ORDER BY ce.{time_expr}, sc.stackDepth
        """
    else:
        sql = f"""
            SELECT {time_expr} AS ts, {gtid}, NULL AS depth, NULL AS symbol
            FROM {tbl} ORDER BY {time_expr}
        """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        # Group consecutive rows by (ts, globalTid) to build per-sample callstacks.
        # A sample event is emitted once per unique (ts, globalTid) pair.
        from itertools import groupby
        def _key(r: Any) -> tuple:
            return (int(r["ts"]), r["globalTid"])

        for (ts_val, gtid_val), group_rows in groupby(rows, key=_key):
            frames = []
            structured_frames = []
            leaf_symbol: str | None = None
            for r in group_rows:
                sym = r["symbol"]
                module = r["module"]
                if sym:
                    frames.append(sym)
                    structured_frames.append({"symbol": sym, "module": module})
                    if leaf_symbol is None:
                        leaf_symbol = sym  # depth=0 comes first (ORDER BY depth)
            # Use leaf frame as the event name (matches existing behaviour).
            # Full callstack stored in extra["callstack"] for downstream use.
            name = leaf_symbol or "cpu_sample"
            extra: dict = {}
            if frames:
                extra["callstack"] = frames
                extra["callstack_frames"] = structured_frames
            events.append(TimelineEvent(
                category="cpu_sample", name=name,
                start_ns=ts_val, dur_ns=0, is_sample=True,
                global_tid=gtid_val,
                extra=extra,
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取 CPU 采样事件失败（{tbl}）：{e}")
    return count


def _extract_cudnn(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    """Extract CUDNN_EVENTS as cuda_api point events."""
    tbl = schema.table_roles.get("cudnn", "CUDNN_EVENTS")
    has_strings = "string_ids" in schema.capabilities
    name_expr = "s.value AS name" if has_strings else "CAST(c.nameId AS TEXT) AS name"
    join = "LEFT JOIN StringIds s ON c.nameId = s.id" if has_strings else ""
    gtid = _sel_col(schema, tbl, "globalTid", prefix="c.")
    sql = f"""
        SELECT c.start, c.[end], {gtid}, {name_expr}
        FROM {tbl} c {join}
        ORDER BY c.start
    """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            # CUDNN_EVENTS often has start==end (point events); use dur=0 in that case
            dur = max(0, int(row["end"]) - int(row["start"]))
            events.append(TimelineEvent(
                category="cuda_api", name=row["name"] or "cudnn",
                start_ns=int(row["start"]), dur_ns=dur,
                global_tid=_int_or_none(row, "globalTid"),
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取 CUDNN 事件失败（{tbl}）：{e}")
    return count


def _extract_osrt(
    conn: sqlite3.Connection,
    schema: SchemaInfo,
    events: list[TimelineEvent],
    limit: int,
    warnings: list[str],
) -> int:
    """Extract OS Runtime API calls → osrt."""
    tbl = schema.table_roles.get("osrt", "OSRT_API")
    has_strings = "string_ids" in schema.capabilities

    col_info = schema.tables.get(tbl, [])
    has_name_id = "nameId" in col_info

    gtid = _sel_col(schema, tbl, "globalTid")
    if has_strings and has_name_id:
        sql = f"""
            SELECT o.start, o.[end], {gtid}, s.value AS name
            FROM {tbl} o JOIN StringIds s ON o.nameId = s.id
            ORDER BY o.start
        """
    else:
        sql = f"""
            SELECT start, [end], {gtid}, CAST(nameId AS TEXT) AS name
            FROM {tbl} ORDER BY start
        """
    count = 0
    try:
        rows = conn.execute(sql).fetchall()
        for row in rows:
            dur = int(row["end"]) - int(row["start"])
            events.append(TimelineEvent(
                category="osrt", name=row["name"] or "osrt_api",
                start_ns=int(row["start"]), dur_ns=dur,
                global_tid=_int_or_none(row, "globalTid"),
            ))
            count += 1
    except sqlite3.Error as e:
        warnings.append(f"提取 OSRT 事件失败（{tbl}）：{e}")
    return count


# ── Helpers ───────────────────────────────────────────────────────────────────

def _int_or_none(row: Any, col: str) -> int | None:
    v = row[col]
    return int(v) if v is not None else None
