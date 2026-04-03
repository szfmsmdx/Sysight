"""Low-overhead SQL-backed analysis queries."""

from __future__ import annotations

from collections import defaultdict

from sysight.annotation import EvidenceReport, Finding
from sysight.profile import Profile


def classify_kernel(name: str) -> str:
    """Classify a kernel as compute or NCCL."""
    lowered = (name or "").lower()
    if "nccl" in lowered:
        if "allreduce" in lowered:
            return "nccl_allreduce"
        if "allgather" in lowered:
            return "nccl_allgather"
        if "reducescatter" in lowered:
            return "nccl_reducescatter"
        if "broadcast" in lowered:
            return "nccl_broadcast"
        return "nccl_other"
    return "compute"


def overlap_analysis(prof: Profile, device: int, trim: tuple[int, int] | None = None) -> dict:
    """Measure compute vs NCCL overlap on one GPU."""
    kernels = prof.kernels(device, trim)
    if not kernels:
        return {"error": "no kernels"}

    compute_intervals = []
    nccl_intervals = []
    for kernel in kernels:
        interval = (kernel["start"], kernel["end"])
        if classify_kernel(kernel["name"]).startswith("nccl_"):
            nccl_intervals.append(interval)
        else:
            compute_intervals.append(interval)

    span_start = min(kernel["start"] for kernel in kernels)
    span_end = max(kernel["end"] for kernel in kernels)
    total_ns = span_end - span_start

    compute_merged = _merge_intervals(compute_intervals)
    nccl_merged = _merge_intervals(nccl_intervals)
    compute_ns = _covered_ns(compute_merged)
    nccl_ns = _covered_ns(nccl_merged)
    overlap_ns = _intersection_ns(compute_merged, nccl_merged)
    compute_only_ns = compute_ns - overlap_ns
    nccl_only_ns = nccl_ns - overlap_ns
    idle_ns = total_ns - compute_only_ns - nccl_only_ns - overlap_ns

    return {
        "compute_only_ms": round(compute_only_ns / 1e6, 2),
        "nccl_only_ms": round(nccl_only_ns / 1e6, 2),
        "overlap_ms": round(overlap_ns / 1e6, 2),
        "idle_ms": round(max(idle_ns, 0) / 1e6, 2),
        "total_ms": round(total_ns / 1e6, 2),
        "overlap_pct": round(100 * overlap_ns / nccl_ns, 1) if nccl_ns else 0,
        "compute_kernels": len(compute_intervals),
        "nccl_kernels": len(nccl_intervals),
    }


def nccl_breakdown(prof: Profile, device: int, trim: tuple[int, int] | None = None) -> list[dict]:
    """Aggregate NCCL kernels by collective type."""
    kernels = prof.kernels(device, trim)
    nccl_kernels = [kernel for kernel in kernels if classify_kernel(kernel["name"]).startswith("nccl_")]
    if not nccl_kernels:
        return []

    total_nccl_ns = sum(kernel["end"] - kernel["start"] for kernel in nccl_kernels)
    grouped: dict[str, list[int]] = defaultdict(list)
    names: dict[str, str] = {}
    for kernel in nccl_kernels:
        category = classify_kernel(kernel["name"]).replace("nccl_", "")
        grouped[category].append(kernel["end"] - kernel["start"])
        names.setdefault(category, kernel["name"])

    rows = []
    for category, durations in sorted(grouped.items(), key=lambda item: -sum(item[1])):
        total_ns = sum(durations)
        rows.append(
            {
                "type": category,
                "name": names[category],
                "count": len(durations),
                "total_ms": round(total_ns / 1e6, 2),
                "avg_ms": round(total_ns / len(durations) / 1e6, 3),
                "max_ms": round(max(durations) / 1e6, 3),
                "pct": round(100 * total_ns / total_nccl_ns, 1),
            }
        )
    return rows


def detect_idle_gaps(
    prof: Profile,
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    min_gap_ns: int = 1_000_000,
    limit: int = 20,
) -> dict:
    """Return top idle gaps plus aggregate stats and runtime attribution."""
    if not prof.schema.kernel_table:
        return {"rows": [], "summary": None}

    trim_clause = ""
    params: list[object] = [device]
    if trim:
        trim_clause = " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])

    sql = f"""
        WITH ordered AS (
            SELECT
                k.streamId,
                k.start,
                k.[end],
                s.value AS kernel_name,
                LAG(k.[end]) OVER (PARTITION BY k.deviceId, k.streamId ORDER BY k.start) AS prev_end,
                LAG(s.value) OVER (PARTITION BY k.deviceId, k.streamId ORDER BY k.start) AS prev_kernel
            FROM {prof.schema.kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            WHERE k.deviceId = ?{trim_clause}
        )
        SELECT
            streamId,
            prev_end AS start_ns,
            start AS end_ns,
            (start - prev_end) AS gap_ns,
            prev_kernel AS before_kernel,
            kernel_name AS after_kernel
        FROM ordered
        WHERE prev_end IS NOT NULL AND (start - prev_end) > ?
        ORDER BY gap_ns DESC
        LIMIT ?
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, params + [min_gap_ns, limit]).fetchall()]

    if not rows:
        return {"rows": [], "summary": None}

    summary_sql = f"""
        WITH ordered AS (
            SELECT
                k.streamId,
                k.start,
                k.[end],
                LAG(k.[end]) OVER (PARTITION BY k.deviceId, k.streamId ORDER BY k.start) AS prev_end
            FROM {prof.schema.kernel_table} k
            WHERE k.deviceId = ?{trim_clause}
        )
        SELECT
            COUNT(*) AS gap_count,
            SUM(start - prev_end) AS total_gap_ns,
            SUM(CASE WHEN (start - prev_end) BETWEEN 1000000 AND 5000000 THEN 1 ELSE 0 END) AS gaps_1_5ms,
            SUM(CASE WHEN (start - prev_end) BETWEEN 5000001 AND 50000000 THEN 1 ELSE 0 END) AS gaps_5_50ms,
            SUM(CASE WHEN (start - prev_end) > 50000000 THEN 1 ELSE 0 END) AS gaps_gt50ms
        FROM ordered
        WHERE prev_end IS NOT NULL AND (start - prev_end) > ?
    """
    with prof._lock:
        agg = dict(prof.conn.execute(summary_sql, params + [min_gap_ns]).fetchone())
        time_row = prof.conn.execute(
            f"SELECT MIN(start), MAX([end]) FROM {prof.schema.kernel_table} WHERE deviceId = ?",
            (device,),
        ).fetchone()
        stream_row = prof.conn.execute(
            f"SELECT COUNT(DISTINCT streamId) FROM {prof.schema.kernel_table} WHERE deviceId = ?",
            (device,),
        ).fetchone()

    span_ns = (time_row[1] or 0) - (time_row[0] or 0)
    stream_count = (stream_row[0] or 1) if stream_row else 1
    total_gap_ns = agg.get("total_gap_ns") or 0
    effective_span = max(span_ns * max(stream_count, 1), 1)
    summary = {
        "gap_count": agg.get("gap_count") or 0,
        "total_idle_ms": round(total_gap_ns / 1e6, 2),
        "pct_of_profile": min(round(100 * total_gap_ns / effective_span, 1), 100.0),
        "gaps_1_5ms": agg.get("gaps_1_5ms") or 0,
        "gaps_5_50ms": agg.get("gaps_5_50ms") or 0,
        "gaps_gt50ms": agg.get("gaps_gt50ms") or 0,
    }

    if prof.schema.runtime_table:
        for gap in rows[: min(5, len(rows))]:
            attribution = runtime_activity_in_window(prof, gap["start_ns"], gap["end_ns"])
            gap["attribution"] = attribution

    return {"rows": rows, "summary": summary}


def runtime_activity_in_window(prof: Profile, start_ns: int, end_ns: int) -> dict:
    """Summarize CUDA runtime activity overlapping a time window."""
    if not prof.schema.runtime_table:
        return {}
    sql = f"""
        SELECT
            s.value AS api_name,
            COUNT(*) AS call_count,
            SUM(r.[end] - r.start) AS total_ns
        FROM {prof.schema.runtime_table} r
        JOIN StringIds s ON r.nameId = s.id
        WHERE r.start < ? AND r.[end] > ?
        GROUP BY s.value
        ORDER BY total_ns DESC
        LIMIT 5
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, (end_ns, start_ns)).fetchall()]
    names = [row["api_name"] for row in rows]
    category = "cpu_stall"
    description = "No CUDA API activity - possible DataLoader, GIL, or I/O wait"
    for name in names:
        if "cudaDeviceSynchronize" in name or "cudaStreamSynchronize" in name or "cudaEventSynchronize" in name:
            category = "synchronization"
            description = "Explicit synchronization stall"
            break
        if "cudaMemcpy" in name or "cudaMemset" in name:
            category = "memory_transfer"
            description = "Blocked on memory transfer"
            break
        if "cudaLaunchKernel" in name:
            category = "kernel_launch"
            description = "Kernel launch overhead"
            break
    return {
        "category": category,
        "description": description,
        "top_apis": [
            {"name": row["api_name"], "total_ms": round((row["total_ns"] or 0) / 1e6, 2)}
            for row in rows[:3]
        ],
    }


def memory_transfer_summary(
    prof: Profile, device: int | None = None, trim: tuple[int, int] | None = None
) -> list[dict]:
    """Break down memcpy by direction."""
    if not prof.schema.memcpy_table:
        return []
    trim_clause = ""
    params: list[object] = []
    if device is not None:
        trim_clause += " AND k.deviceId = ?"
        params.append(device)
    if trim:
        trim_clause += " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        SELECT
            k.copyKind,
            COUNT(*) AS count,
            ROUND(SUM(k.bytes) / 1e6, 2) AS total_mb,
            ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms
        FROM {prof.schema.memcpy_table} k
        WHERE 1=1{trim_clause}
        GROUP BY k.copyKind
        ORDER BY total_ms DESC
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params).fetchall()]


def memory_bandwidth_summary(
    prof: Profile, device: int | None = None, trim: tuple[int, int] | None = None
) -> list[dict]:
    """Compute bandwidth statistics for memcpy operations."""
    if not prof.schema.memcpy_table:
        return []
    trim_clause = ""
    params: list[object] = []
    if device is not None:
        trim_clause += " AND k.deviceId = ?"
        params.append(device)
    if trim:
        trim_clause += " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        WITH ranked AS (
            SELECT copyKind, bytes, (k.[end] - k.start) AS dur_ns
            FROM {prof.schema.memcpy_table} k
            WHERE 1=1{trim_clause}
        )
        SELECT
            copyKind,
            COUNT(*) AS op_count,
            ROUND(SUM(bytes) / 1e6, 2) AS total_mb,
            ROUND(AVG(bytes) / 1e3, 1) AS avg_kb,
            ROUND(SUM(dur_ns) / 1e6, 2) AS total_dur_ms,
            CASE
                WHEN SUM(dur_ns) > 0
                THEN ROUND(SUM(bytes) / (SUM(dur_ns) / 1e9) / 1e9, 2)
                ELSE 0
            END AS avg_bandwidth_gbps,
            COALESCE(
                ROUND(
                    MAX(CASE WHEN dur_ns > 0 THEN bytes / (dur_ns / 1e9) / 1e9 END),
                    2
                ),
                0
            ) AS peak_bandwidth_gbps
        FROM ranked
        GROUP BY copyKind
        ORDER BY total_mb DESC
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params).fetchall()]


def top_kernel_summary(
    prof: Profile,
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    limit: int = 15,
) -> list[dict]:
    """Return the heaviest kernels ranked by total execution time."""
    trim_clause = ""
    params: list[object] = [device]
    if trim:
        trim_clause = " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        SELECT
            COALESCE(d.value, s.value) AS kernel_name,
            COUNT(*) AS invocations,
            ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms,
            ROUND(AVG(k.[end] - k.start) / 1e6, 3) AS avg_ms,
            ROUND(MIN(k.[end] - k.start) / 1e6, 3) AS min_ms,
            ROUND(MAX(k.[end] - k.start) / 1e6, 3) AS max_ms
        FROM {prof.schema.kernel_table} k
        JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN StringIds d ON k.demangledName = d.id
        WHERE k.deviceId = ?{trim_clause}
        GROUP BY COALESCE(d.value, s.value)
        ORDER BY total_ms DESC
        LIMIT ?
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params + [limit]).fetchall()]


def h2d_distribution(
    prof: Profile, device: int | None = None, trim: tuple[int, int] | None = None
) -> dict:
    """Analyze whether H2D copies are concentrated at startup or spread out."""
    if not prof.schema.memcpy_table:
        return {"rows": [], "pattern": None}
    trim_clause = ""
    params: list[object] = []
    if device is not None:
        trim_clause += " AND k.deviceId = ?"
        params.append(device)
    if trim:
        trim_clause += " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        WITH baseline AS (
            SELECT MIN(k.start) AS min_start
            FROM {prof.schema.memcpy_table} k
            WHERE k.copyKind = 1{trim_clause}
        )
        SELECT
            CAST((k.start - b.min_start) / 1000000000.0 AS INT) AS second,
            COUNT(*) AS ops,
            SUM(k.bytes) / 1e6 AS total_mb,
            COALESCE(SUM(k.bytes) / NULLIF(SUM(k.[end] - k.start), 0), 0) AS avg_gbps
        FROM {prof.schema.memcpy_table} k
        CROSS JOIN baseline b
        WHERE k.copyKind = 1{trim_clause}
        GROUP BY 1
        ORDER BY 1
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, params + params).fetchall()]
    if not rows:
        return {"rows": [], "pattern": None}

    total_mb = sum(row["total_mb"] for row in rows)
    first_2s_mb = sum(row["total_mb"] for row in rows if row["second"] <= 1)
    sorted_mb = sorted(row["total_mb"] for row in rows)
    median_mb = sorted_mb[len(sorted_mb) // 2] if sorted_mb else 0

    pattern = {"type": "unknown", "detail": "Too few data points to classify."}
    if total_mb > 0 and first_2s_mb / total_mb > 0.8:
        pattern = {
            "type": "init_heavy",
            "detail": (
                f"H2D concentrated in the first 2 seconds ({first_2s_mb:.1f}/{total_mb:.1f} MB)."
            ),
        }
    elif median_mb > 0:
        spikes = [row for row in rows if row["total_mb"] > 3 * median_mb]
        if spikes:
            seconds = ", ".join(str(row["second"]) for row in spikes[:5])
            pattern = {
                "type": "spike",
                "detail": f"H2D spikes detected at second(s) [{seconds}] compared with median {median_mb:.1f} MB.",
            }
        elif len(rows) >= 3:
            pattern = {
                "type": "spread_out",
                "detail": f"H2D transfers spread across {len(rows)} seconds ({total_mb:.1f} MB total).",
            }
    return {"rows": rows, "pattern": pattern}


def kernel_launch_overhead(
    prof: Profile,
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    limit: int = 20,
) -> list[dict]:
    """Measure runtime call to kernel start overhead."""
    if not prof.schema.runtime_table:
        return []
    trim_clause = ""
    params: list[object] = []
    if trim:
        trim_clause += " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        SELECT
            s.value AS kernel_name,
            k.streamId AS stream,
            r.start AS runtime_start_ns,
            r.[end] AS runtime_end_ns,
            k.start AS kernel_start_ns,
            k.[end] AS kernel_end_ns,
            ROUND((r.[end] - r.start) / 1e6, 3) AS api_ms,
            ROUND((k.[end] - k.start) / 1e6, 3) AS kernel_ms,
            ROUND((k.start - r.start) / 1e3, 1) AS overhead_us
        FROM {prof.schema.runtime_table} r
        JOIN {prof.schema.kernel_table} k ON r.correlationId = k.correlationId
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.deviceId = ?{trim_clause}
        ORDER BY overhead_us DESC
        LIMIT ?
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, [device] + params + [limit]).fetchall()]


def nccl_anomalies(
    prof: Profile,
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    threshold: float = 3.0,
    limit: int = 20,
) -> list[dict]:
    """Find NCCL outlier collectives."""
    trim_clause = ""
    params: list[object] = [device]
    if trim:
        trim_clause = " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        WITH nccl_ops AS (
            SELECT
                s.value AS name,
                k.correlationId,
                k.streamId,
                k.start,
                k.[end],
                (k.[end] - k.start) AS dur_ns
            FROM {prof.schema.kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            WHERE k.deviceId = ? AND (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%'){trim_clause}
        ),
        op_stats AS (
            SELECT
                CASE
                    WHEN name LIKE '%AllReduce%' THEN 'AllReduce'
                    WHEN name LIKE '%AllGather%' THEN 'AllGather'
                    WHEN name LIKE '%ReduceScatter%' THEN 'ReduceScatter'
                    WHEN name LIKE '%Broadcast%' THEN 'Broadcast'
                    WHEN name LIKE '%AllToAll%' THEN 'AllToAll'
                    WHEN name LIKE '%Reduce%' THEN 'Reduce'
                    ELSE 'Other'
                END AS op_type,
                name,
                dur_ns,
                start,
                correlationId,
                streamId
            FROM nccl_ops
        ),
        op_avg AS (
            SELECT op_type, AVG(dur_ns) AS avg_dur_ns, COUNT(*) AS total_count
            FROM op_stats
            GROUP BY op_type
        )
        SELECT
            o.op_type,
            o.name,
            ROUND(o.dur_ns / 1e6, 3) AS dur_ms,
            ROUND(a.avg_dur_ns / 1e6, 3) AS avg_ms,
            ROUND(CAST(o.dur_ns AS REAL) / NULLIF(a.avg_dur_ns, 0), 1) AS ratio_to_avg,
            o.start,
            o.streamId,
            a.total_count
        FROM op_stats o
        JOIN op_avg a ON o.op_type = a.op_type
        WHERE o.dur_ns > a.avg_dur_ns * ?
        ORDER BY o.dur_ns DESC
        LIMIT ?
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params + [threshold, limit]).fetchall()]


def sync_api_summary(
    prof: Profile, device: int | None = None, trim: tuple[int, int] | None = None
) -> dict | None:
    """Summarize blocking synchronization APIs."""
    if not prof.schema.runtime_table:
        return None
    with prof._lock:
        names = prof.conn.execute(
            """
            SELECT id, value FROM StringIds
            WHERE value LIKE 'cudaDeviceSynchronize%'
               OR value LIKE 'cudaStreamSynchronize%'
               OR value LIKE 'cudaEventSynchronize%'
               OR value LIKE 'cudaStreamWaitEvent%'
            """
        ).fetchall()
        if not names:
            return None
        name_ids = [str(row[0]) for row in names]
        thread_ids: list[str] = []
        if device is not None:
            thread_rows = prof.conn.execute(
                f"""
                SELECT DISTINCT r.globalTid
                FROM {prof.schema.runtime_table} r
                JOIN {prof.schema.kernel_table} k ON r.correlationId = k.correlationId
                WHERE k.deviceId = ?
                """,
                (device,),
            ).fetchall()
            thread_ids = [str(row[0]) for row in thread_rows if row[0] is not None]
            if not thread_ids:
                return None

        where_parts = [f"nameId IN ({','.join(name_ids)})"]
        params: list[object] = []
        if thread_ids:
            where_parts.append(f"globalTid IN ({','.join(thread_ids)})")
        if trim:
            where_parts.append("[end] > ? AND start < ?")
            params.extend([trim[0], trim[1]])
        grouped = prof.conn.execute(
            f"""
            SELECT nameId, COUNT(*) AS call_count, SUM([end] - start) AS total_ns
            FROM {prof.schema.runtime_table}
            WHERE {' AND '.join(where_parts)}
            GROUP BY nameId
            """,
            params,
        ).fetchall()
        if not grouped:
            return None
        interval_rows = prof.conn.execute(
            f"""
            SELECT start, [end]
            FROM {prof.schema.runtime_table}
            WHERE {' AND '.join(where_parts)}
            """,
            params,
        ).fetchall()
        kernel_total_sql = f"SELECT MIN(start), MAX([end]) FROM {prof.schema.kernel_table} WHERE 1=1"
        kernel_total_params: list[object] = []
        if device is not None:
            kernel_total_sql += " AND deviceId = ?"
            kernel_total_params.append(device)
        if trim:
            kernel_total_sql += " AND [end] > ? AND start < ?"
            kernel_total_params.extend([trim[0], trim[1]])
        kernel_total = prof.conn.execute(kernel_total_sql, kernel_total_params).fetchone()

    id_to_name = {row[0]: row[1] for row in names}
    total_sync_ns = sum(row[2] or 0 for row in grouped)
    merged_sync_ns = _covered_ns(
        _merge_intervals(
            [
                (int(row[0]), int(row[1]))
                for row in interval_rows
                if row[0] is not None and row[1] is not None and row[1] > row[0]
            ]
        )
    )
    total_gpu_ns = (
        max(int(kernel_total[1] or 0) - int(kernel_total[0] or 0), 0)
        if kernel_total
        else 0
    )
    api_names = sorted({id_to_name[row[0]].split("_v")[0] for row in grouped})
    return {
        "call_count": sum(row[1] for row in grouped),
        "total_ms": round(total_sync_ns / 1e6, 2),
        "pct_of_gpu_time": round(100 * merged_sync_ns / total_gpu_ns, 1) if total_gpu_ns else 0,
        "api_names": api_names,
    }


def pageable_memcpy_summary(
    prof: Profile, device: int | None = None, trim: tuple[int, int] | None = None
) -> dict | None:
    """Detect pageable memory usage in memcpy operations."""
    if not prof.schema.memcpy_table:
        return None
    sql = f"""
        SELECT COUNT(*) AS pageable_count,
               COALESCE(SUM(bytes), 0) AS total_bytes,
               COALESCE(SUM([end] - start), 0) AS total_ns
        FROM {prof.schema.memcpy_table}
        WHERE (srcKind = 1 OR dstKind = 1)
    """
    params: list[object] = []
    if device is not None:
        sql += " AND deviceId = ?"
        params.append(device)
    if trim:
        sql += " AND [end] > ? AND start < ?"
        params.extend([trim[0], trim[1]])
    with prof._lock:
        row = prof.conn.execute(sql, params).fetchone()
    if not row or row[0] == 0:
        return None
    return {
        "count": row[0],
        "total_mb": round((row[1] or 0) / 1e6, 1),
        "total_ms": round((row[2] or 0) / 1e6, 1),
    }


def top_kernel_instances(
    prof: Profile,
    device: int,
    kernel_name: str,
    trim: tuple[int, int] | None = None,
    *,
    limit: int = 3,
) -> list[dict]:
    """Return the longest individual instances for a given kernel name."""
    trim_clause = ""
    params: list[object] = [device, kernel_name]
    if trim:
        trim_clause = " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        SELECT
            k.start AS start_ns,
            k.[end] AS end_ns,
            k.streamId AS stream,
            ROUND((k.[end] - k.start) / 1e6, 3) AS duration_ms
        FROM {prof.schema.kernel_table} k
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.deviceId = ? AND s.value = ?{trim_clause}
        ORDER BY (k.[end] - k.start) DESC
        LIMIT ?
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params + [limit]).fetchall()]


def large_h2d_transfers(
    prof: Profile, device: int, trim: tuple[int, int] | None = None, *, limit: int = 5
) -> list[dict]:
    """Return the largest H2D transfers for one GPU."""
    if not prof.schema.memcpy_table:
        return []
    trim_clause = ""
    params: list[object] = [device]
    if trim:
        trim_clause = " AND start >= ? AND [end] <= ?"
        params.extend([trim[0], trim[1]])
    sql = f"""
        SELECT
            start AS start_ns,
            [end] AS end_ns,
            streamId AS stream,
            ROUND(bytes / 1e6, 2) AS size_mb,
            ROUND(([end] - start) / 1e6, 3) AS duration_ms
        FROM {prof.schema.memcpy_table}
        WHERE deviceId = ? AND copyKind = 1{trim_clause}
        ORDER BY bytes DESC
        LIMIT ?
    """
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params + [limit]).fetchall()]


def _make_issue_window(
    kind: str,
    start_ns: int,
    end_ns: int,
    *,
    stream: int | None = None,
    label: str = "",
    evidence: str = "",
    attribution: dict | None = None,
    before_kernel: str = "",
    after_kernel: str = "",
) -> dict:
    return {
        "kind": kind,
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "stream": stream,
        "duration_ms": round((end_ns - start_ns) / 1e6, 3),
        "label": label,
        "evidence": evidence,
        "attribution": attribution or {},
        "before_kernel": before_kernel,
        "after_kernel": after_kernel,
    }


def _idle_gap_windows(data: dict, *, limit: int = 2, category: str | None = None) -> list[dict]:
    rows = data.get("idle_gaps", {}).get("rows", [])
    if category:
        rows = [row for row in rows if (row.get("attribution") or {}).get("category") == category]
    windows = []
    for row in rows[:limit]:
        windows.append(
            _make_issue_window(
                "idle_gap",
                int(row["start_ns"]),
                int(row["end_ns"]),
                stream=int(row["streamId"]),
                label=f"GPU gap {row['gap_ns'] / 1e6:.2f}ms",
                evidence=(row.get("attribution") or {}).get("description", "GPU idle gap"),
                attribution=row.get("attribution") or {},
                before_kernel=row.get("before_kernel") or "",
                after_kernel=row.get("after_kernel") or "",
            )
        )
    return windows


def _h2d_windows(data: dict, *, limit: int = 2) -> list[dict]:
    windows = []
    for row in (data.get("large_h2d") or [])[:limit]:
        windows.append(
            _make_issue_window(
                "h2d_transfer",
                int(row["start_ns"]),
                int(row["end_ns"]),
                stream=int(row["stream"]),
                label=f"H2D {row['size_mb']:.1f}MB",
                evidence=f"H2D copy took {row['duration_ms']:.2f}ms",
                attribution={"category": "memory_transfer", "description": "Large H2D transfer"},
            )
        )
    return windows


def _nccl_windows(data: dict, *, limit: int = 2) -> list[dict]:
    windows = []
    for row in (data.get("nccl_anomalies") or [])[:limit]:
        dur_ns = int(float(row["dur_ms"]) * 1e6)
        start_ns = int(row["start"])
        windows.append(
            _make_issue_window(
                "nccl_anomaly",
                start_ns,
                start_ns + dur_ns,
                stream=int(row["streamId"]),
                label=f"{row['op_type']} {row['dur_ms']:.2f}ms",
                evidence=f"{row['name']} is {row['ratio_to_avg']}x slower than its average",
                attribution={"category": "communication", "description": "Long NCCL collective"},
            )
        )
    return windows


def _launch_windows(data: dict, *, limit: int = 2) -> list[dict]:
    windows = []
    rows = sorted(
        (data.get("launch_overhead") or []),
        key=lambda row: float(row.get("overhead_us") or 0),
        reverse=True,
    )
    for row in rows[:limit]:
        start_ns = int(row["runtime_start_ns"])
        end_ns = int(row["kernel_start_ns"])
        if end_ns <= start_ns:
            continue
        windows.append(
            _make_issue_window(
                "launch_overhead",
                start_ns,
                end_ns,
                stream=int(row["stream"]),
                label=f"Launch overhead {row['overhead_us']:.1f}us",
                evidence=f"{row['kernel_name']} launch overhead exceeds kernel duration",
                attribution={"category": "kernel_launch", "description": "Large launch overhead window"},
            )
        )
    return windows


def _slow_iteration_windows(data: dict, *, limit: int = 1) -> list[dict]:
    summary = data.get("iterations_summary")
    if not summary:
        return []
    windows = []
    for row in summary.get("slow_iterations", [])[:limit]:
        start_ns = row.get("gpu_start_ns")
        end_ns = row.get("gpu_end_ns")
        if start_ns is None or end_ns is None:
            continue
        windows.append(
            _make_issue_window(
                "slow_iteration",
                int(start_ns),
                int(end_ns),
                label=f"Iteration {row['iteration']} {row['duration_ms']:.1f}ms",
                evidence=f"Slow iteration above median {summary['median_ms']:.1f}ms",
                attribution={"category": "iteration", "description": "Slow iteration window"},
            )
        )
    return windows


def match_root_causes(data: dict) -> list[dict]:
    """Turn structured query results into actionable root cause hypotheses."""
    findings: list[dict] = []

    gap_summary = data["idle_gaps"]["summary"]
    if gap_summary and gap_summary["gap_count"] >= 3 and gap_summary["pct_of_profile"] >= 5:
        findings.append(
            {
                "pattern": "GPU Bubbles (Pipeline Stalls)",
                "severity": "warning",
                "evidence": (
                    f"检测到 {gap_summary['gap_count']} 个超过 1ms 的 GPU idle gap，"
                    f"总计闲置 {gap_summary['total_idle_ms']:.1f}ms（占 profile 时间跨度 {gap_summary['pct_of_profile']}%）"
                ),
                "recommendation": (
                    "删除代码中显式的 cudaDeviceSynchronize / cudaStreamSynchronize 调用，"
                    "改用 CUDA event 做依赖同步，或引入 CUDA Graphs 消除启动开销。"
                ),
                "score": round(float(gap_summary["total_idle_ms"]), 2),
                "windows": _idle_gap_windows(data, limit=2),
            }
        )

    overlap = data["overlap"]
    if "error" not in overlap:
        nccl_total = overlap["nccl_only_ms"] + overlap["overlap_ms"]
        if nccl_total > 0 and overlap["overlap_pct"] < 30:
            findings.append(
                {
                    "pattern": "Low Compute/NCCL Overlap",
                    "severity": "critical",
                    "evidence": (
                        f"NCCL 与计算重叠仅 {overlap['overlap_pct']}%，"
                        f"纯通信时间 {overlap['nccl_only_ms']:.1f}ms（占总跨度 {overlap['total_ms']:.1f}ms）。"
                    ),
                    "recommendation": (
                        "确保 NCCL 通信运行在独立的 stream 上，避免在 AllReduce 后立即调用 host 同步；"
                        "可在 backward 后手动重叠下一层的 forward 与当前层的 AllReduce。"
                    ),
                    "score": round(float(nccl_total) + max(0.0, 30.0 - float(overlap["overlap_pct"])), 2),
                    "windows": _nccl_windows(data, limit=2),
                }
            )

    bandwidth_rows = data["memory_bandwidth"]
    h2d_row = next((row for row in bandwidth_rows if row["copyKind"] == 1), None)
    if h2d_row and h2d_row["total_dur_ms"] > 50:
        findings.append(
            {
                "pattern": "Excessive H2D Transfers",
                "severity": "warning",
                "evidence": (
                    f"H2D 传输总耗时 {h2d_row['total_dur_ms']:.1f}ms，{h2d_row['total_mb']:.1f}MB，"
                    f"{h2d_row['op_count']} 次操作，平均带宽 {h2d_row['avg_bandwidth_gbps']:.1f} GB/s"
                ),
                "recommendation": (
                    "DataLoader 开启 pin_memory=True，tensor 转设备改用 .to(device, non_blocking=True)，"
                    "并提前预取 batch 以隐藏 H2D 延迟。"
                ),
                "score": round(float(h2d_row["total_dur_ms"]), 2),
                "windows": _h2d_windows(data, limit=2),
            }
        )

    h2d_pattern = data["h2d_distribution"]["pattern"]
    if h2d_pattern and h2d_pattern["type"] == "spread_out":
        findings.append(
            {
                "pattern": "Continuous H2D Transfers",
                "severity": "warning",
                "evidence": (
                    f"H2D 常驻热路径：{h2d_pattern['detail']}"
                ),
                "recommendation": (
                    "排查热路径上是否每个 step 都在从 host 拉取数据，或在训练循环内频繁调用 .cpu() / .item()。"
                ),
                "score": round(float(h2d_row["total_dur_ms"]) * 0.9, 2) if h2d_row else 0.0,
                "windows": _h2d_windows(data, limit=2),
            }
        )

    launch_rows = data["launch_overhead"]
    high_overhead = [
        row for row in launch_rows if row["kernel_ms"] > 0 and row["overhead_us"] > row["kernel_ms"] * 1000
    ]
    if len(high_overhead) >= 5:
        findings.append(
            {
                "pattern": "Small Kernel Overhead",
                "severity": "warning",
                "evidence": f"检测到 {len(high_overhead)} 个 kernel 的启动开销远超过 kernel 执行时间（小 kernel 问题）。",
                "recommendation": (
                    "使用 torch.compile() 或 CUDA Graphs 融合小 kernel，减少启动开销；"
                    "也可考虑算子融合降低 dispatch 压力。"
                ),
                "score": round(float(len(high_overhead) * 10), 2),
                "windows": _launch_windows(data, limit=2),
            }
        )

    sync_summary = data["sync_summary"]
    if sync_summary and sync_summary["total_ms"] >= 1.0 and sync_summary["pct_of_gpu_time"] >= 2.0:
        findings.append(
            {
                "pattern": "Excessive Synchronization",
                "severity": "warning",
                "evidence": (
                    f"{sync_summary['call_count']} 次同步 API 调用，总耗时 {sync_summary['total_ms']:.1f}ms（占 GPU kernel 时间 {sync_summary['pct_of_gpu_time']:.1f}%），"
                    f"涉及接口：{', '.join(sync_summary['api_names'])}"
                ),
                "recommendation": (
                    "从训练循环中删除 .item() / .cpu() 调用，用 CUDA event 替代全局 device sync；"
                    "重点排查 backward 阶段是否有隐式同步点。"
                ),
                "score": round(float(sync_summary["total_ms"]), 2),
                "windows": _idle_gap_windows(data, limit=2, category="synchronization"),
            }
        )

    pageable = data["pageable_memcpy"]
    if pageable:
        findings.append(
            {
                "pattern": "Pageable Memory in Memcpy",
                "severity": "warning",
                "evidence": (
                    f"{pageable['count']} 次内存拷贝使用了可分页内存：{pageable['total_mb']:.1f}MB，耗时 {pageable['total_ms']:.1f}ms。"
                ),
                "recommendation": (
                    "将 host 侧 buffer 改为 pinned memory，使异步 H2D 拷贝能与计算并行进行。"
                ),
                "score": round(float(pageable["total_ms"]), 2),
                "windows": _h2d_windows(data, limit=2),
            }
        )

    top_kernel = data["target_summary"]["top_kernels"][0] if data["target_summary"]["top_kernels"] else None
    if top_kernel and "nccl" in top_kernel["name"].lower() and top_kernel["pct"] >= 15:
        findings.append(
            {
                "pattern": "NCCL Hotspot",
                "severity": "warning",
                "evidence": (
                    f"NCCL 热点 kernel `{top_kernel['name']}` 占目标 GPU 计算时间 {top_kernel['pct']}%（{top_kernel['total_ms']:.1f}ms）。"
                ),
                "recommendation": (
                    "检查 DDP gradient bucket 大小（bucket_cap_mb）、通信调度顺序，"
                    "以及当前模型并行度对该任务是否过高。"
                ),
                "score": round(float(top_kernel["total_ms"]), 2),
                "windows": _nccl_windows(data, limit=2),
            }
        )

    if "error" not in overlap:
        compute_ms = overlap["compute_only_ms"]
        nccl_total = overlap["nccl_only_ms"] + overlap["overlap_ms"]
        if compute_ms > 0 and nccl_total > 0:
            ratio = compute_ms / nccl_total
            if ratio < 0.5:
                findings.append(
                    {
                        "pattern": "Compute-Communication Imbalance",
                        "severity": "critical",
                    "evidence": (
                        f"计算/通信时间比 = {ratio:.2f}；"
                        f"纯计算 {compute_ms:.1f}ms，纯 NCCL {nccl_total:.1f}ms。"
                    ),
                    "recommendation": (
                        "重新评估张量并行度和 gradient bucket 大小，判断该任务是否处于通信瓶颈；"
                        "可尝试减小并行度或引入计算/通信流水线。"
                    ),
                        "score": round(float(nccl_total) * 1.5, 2),
                        "windows": _nccl_windows(data, limit=2),
                    }
                )

    nvtx_regions = [
        row for row in data.get("nvtx_regions", [])
        if row and "error" not in row
    ]
    total_nccl_ms = sum(row.get("nccl_ms", 0.0) for row in nvtx_regions)
    if total_nccl_ms > 0:
        heaviest_nccl = max(nvtx_regions, key=lambda row: row.get("nccl_ms", 0.0))
        heaviest_nccl_ms = heaviest_nccl.get("nccl_ms", 0.0)
        heaviest_pct = 100 * heaviest_nccl_ms / total_nccl_ms if total_nccl_ms else 0.0
        if heaviest_nccl_ms > 0 and heaviest_pct >= 40:
            findings.append(
                {
                    "pattern": "Layer NCCL Hotspot",
                    "severity": "warning",
                    "evidence": (
                        f"NVTX 区域 '{heaviest_nccl.get('nvtx_path') or heaviest_nccl.get('nvtx_region', '?')}' "
                        f"占总 NCCL 时间 {heaviest_pct:.0f}%（{heaviest_nccl_ms:.1f}ms / {total_nccl_ms:.1f}ms）。"
                    ),
                    "recommendation": (
                        "优先排查该层/阶段的 AllReduce 时序；调整 gradient bucket 分组，"
                        "或将该区域的通信与下一层计算重叠。"
                    ),
                    "score": round(float(heaviest_nccl_ms), 2),
                    "windows": _nccl_windows(data, limit=2),
                }
            )

    compute_regions = [row for row in nvtx_regions if row.get("compute_ms", 0.0) > 0.01]
    if len(compute_regions) >= 2:
        heaviest = max(compute_regions, key=lambda row: row["compute_ms"])
        lightest = min(compute_regions, key=lambda row: row["compute_ms"])
        if lightest["compute_ms"] > 0:
            ratio = heaviest["compute_ms"] / lightest["compute_ms"]
            if ratio >= 3.0:
                findings.append(
                    {
                        "pattern": "Pipeline Imbalance",
                        "severity": "warning",
                    "evidence": (
                        f"NVTX 区域间计算时间相差 {ratio:.1f}倍；"
                        f"最重：'{heaviest.get('nvtx_path') or heaviest.get('nvtx_region', '?')}'（{heaviest['compute_ms']:.1f}ms），"
                        f"最轻：'{lightest.get('nvtx_path') or lightest.get('nvtx_region', '?')}'（{lightest['compute_ms']:.1f}ms）。"
                    ),
                "recommendation": (
                    "优先排查最重的 NVTX 区域是否存在 kernel 组合不合理或 stage 负载不均，"
                    "再考虑细粒度的 kernel 级优化。"
                ),
                        "score": round(float(heaviest["compute_ms"]), 2),
                        "windows": [],
                    }
                )

    iteration_meta = data.get("iterations_summary")
    if iteration_meta and iteration_meta["count"] >= 2 and iteration_meta["slow_iterations"]:
        slow = iteration_meta["slow_iterations"][0]
        findings.append(
            {
                "pattern": "Iteration Variance",
                "severity": "warning",
                "evidence": (
                    f"Iteration {slow['iteration']} 耗时 {slow['duration_ms']:.1f}ms，"
                    f"远超中位数 {iteration_meta['median_ms']:.1f}ms。"
                ),
                "recommendation": (
                    "将慢 iteration 与中位数对比，排查 host 侧输入时序、同步点和各层 NVTX 时间分解。"
                ),
                "score": round(float(slow["duration_ms"] - iteration_meta["median_ms"]), 2),
                "windows": _slow_iteration_windows(data, limit=1),
            }
        )

    if not findings:
        findings.append(
            {
                "pattern": "未检测到明显反模式",
                "severity": "info",
                "evidence": "轻量检查未发现显著的反模式信号。",
                "recommendation": "下一步：使用 Nsight Compute（ncu）对热点 kernel 做更深的分析，或补充 NVTX marker 提升归因精度。",
                "score": 0.0,
                "windows": [],
            }
        )
    return findings


def build_evidence_report(data: dict) -> EvidenceReport:
    """Build a small evidence report for downstream tooling."""
    target = data["target_device"]
    findings: list[Finding] = []

    for row in data["idle_gaps"]["rows"][:5]:
        findings.append(
            Finding(
                type="highlight",
                label=f"GPU Idle Gap ({row['gap_ns'] / 1e6:.2f}ms)",
                start_ns=int(row["start_ns"]),
                end_ns=int(row["end_ns"]),
                stream=int(row["streamId"]),
                gpu_id=target,
                severity="warning",
                note=(row.get("attribution") or {}).get("description", "Large idle gap between kernels."),
            )
        )

    for row in data["nccl_anomalies"][:3]:
        findings.append(
            Finding(
                type="highlight",
                label=f"Long NCCL ({row['dur_ms']:.2f}ms)",
                start_ns=int(row["start"]),
                end_ns=int(row["start"] + row["dur_ms"] * 1e6),
                stream=int(row["streamId"]),
                gpu_id=target,
                severity="critical",
                note=f"{row['op_type']} is {row['ratio_to_avg']}x slower than its average duration.",
            )
        )

    top_kernel = data["target_summary"]["top_kernels"][0] if data["target_summary"]["top_kernels"] else None
    if top_kernel:
        for row in top_kernel_instances(data["profile"], target, top_kernel["name"], data["trim"])[:3]:
            findings.append(
                Finding(
                    type="highlight",
                    label=f"Hotspot: {top_kernel['name']}",
                    start_ns=int(row["start_ns"]),
                    end_ns=int(row["end_ns"]),
                    stream=int(row["stream"]),
                    gpu_id=target,
                    severity="info",
                    note=f"One of the longest instances of the top aggregated hotspot ({row['duration_ms']:.2f}ms).",
                )
            )

    for row in large_h2d_transfers(data["profile"], target, data["trim"])[:5]:
        findings.append(
            Finding(
                type="highlight",
                label=f"Large H2D Transfer ({row['size_mb']:.1f}MB)",
                start_ns=int(row["start_ns"]),
                end_ns=int(row["end_ns"]),
                stream=int(row["stream"]),
                gpu_id=target,
                severity="warning",
                note=f"H2D copy took {row['duration_ms']:.2f}ms.",
            )
        )

    if data["idle_gaps"]["summary"]:
        summary = data["idle_gaps"]["summary"]
        findings.append(
            Finding(
                type="region",
                label=f"GPU Idle Summary ({summary['pct_of_profile']}% of profile)",
                start_ns=int(data["target_range"][0]),
                end_ns=int(data["target_range"][1]),
                gpu_id=target,
                severity="info",
                note=(
                    f"{summary['gap_count']} gaps, {summary['total_idle_ms']:.1f}ms idle across "
                    f"the selected range."
                ),
            )
        )

    return EvidenceReport(
        title="Sysight lightweight analysis",
        profile_path=getattr(data["profile"], "path", ""),
        findings=findings,
    )


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _covered_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_ns(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    total = 0
    j = 0
    for start_a, end_a in left:
        while j < len(right) and right[j][1] <= start_a:
            j += 1
        k = j
        while k < len(right) and right[k][0] < end_a:
            total += max(0, min(end_a, right[k][1]) - max(start_a, right[k][0]))
            if right[k][1] >= end_a:
                break
            k += 1
    return total


# ---------------------------------------------------------------------------
# Tensor Core utilisation
# ---------------------------------------------------------------------------

# Kernel name patterns that strongly indicate Tensor Core usage.
# Sources: cuBLAS HGEMM/WMMA kernels, FlashAttention, cuDNN conv, Triton matmul.
_TENSOR_CORE_PATTERNS = (
    # cuBLAS HMMA / IMMA
    "h100_xmma_", "s16816", "s884", "hgemm", "sgemm_128x128",
    "volta_h884", "turing_h1688", "ampere_s16816",
    # FlashAttention (Triton / CUDA)
    "_attn_fwd", "_attn_bwd", "flash_fwd", "flash_bwd",
    # cuDNN convolution NHWC Tensor Core
    "nhwc_", "cudnn::detail::implicit",
    # Generic WMMA / mma patterns
    "wmma", "mma_", "_mma",
    # Triton matmul (heuristic: _kernel + matmul-like block sizes)
    "matmul_kernel",
    # cutlass
    "cutlass::gemm", "cutlass::conv",
)


def tensor_core_summary(
    prof: "Profile",
    device: int,
    trim: tuple[int, int] | None = None,
) -> dict:
    """Estimate Tensor Core kernel share by name-pattern matching.

    Returns a dict with:
    - tc_kernel_count  : number of TC-matching kernel invocations
    - total_count      : total kernel invocations on this device/trim
    - tc_ms            : total duration of TC kernels (ms)
    - total_ms         : total duration of all kernels (ms)
    - tc_time_pct      : tc_ms / total_ms * 100
    - tc_count_pct     : tc_kernel_count / total_count * 100
    - top_tc_kernels   : top-5 TC kernels by total_ms [{name, count, total_ms, pct}]
    - non_tc_ms        : total_ms - tc_ms
    """
    trim_clause = ""
    params: list[object] = [device]
    if trim:
        trim_clause = " AND k.start >= ? AND k.[end] <= ?"
        params.extend(trim)

    sql = f"""
        SELECT
            COALESCE(d.value, s.value) AS kernel_name,
            COUNT(*) AS invocations,
            ROUND(SUM(k.[end] - k.start) / 1e6, 3) AS total_ms
        FROM {prof.schema.kernel_table} k
        JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN StringIds d ON k.demangledName = d.id
        WHERE k.deviceId = ?{trim_clause}
        GROUP BY COALESCE(d.value, s.value)
        ORDER BY total_ms DESC
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, params).fetchall()]

    if not rows:
        return {
            "tc_kernel_count": 0, "total_count": 0,
            "tc_ms": 0.0, "total_ms": 0.0,
            "tc_time_pct": 0.0, "tc_count_pct": 0.0,
            "top_tc_kernels": [], "non_tc_ms": 0.0,
        }

    total_ms = sum(r["total_ms"] for r in rows)
    total_count = sum(r["invocations"] for r in rows)

    def _is_tc(name: str) -> bool:
        lower = (name or "").lower()
        return any(pat in lower for pat in _TENSOR_CORE_PATTERNS)

    tc_rows = [r for r in rows if _is_tc(r["kernel_name"])]
    tc_ms = sum(r["total_ms"] for r in tc_rows)
    tc_count = sum(r["invocations"] for r in tc_rows)

    top_tc = sorted(tc_rows, key=lambda r: -r["total_ms"])[:5]
    top_tc_out = [
        {
            "name": r["kernel_name"],
            "count": r["invocations"],
            "total_ms": r["total_ms"],
            "pct": round(100 * r["total_ms"] / total_ms, 1) if total_ms else 0.0,
        }
        for r in top_tc
    ]

    return {
        "tc_kernel_count": tc_count,
        "total_count": total_count,
        "tc_ms": round(tc_ms, 2),
        "total_ms": round(total_ms, 2),
        "tc_time_pct": round(100 * tc_ms / total_ms, 1) if total_ms else 0.0,
        "tc_count_pct": round(100 * tc_count / total_count, 1) if total_count else 0.0,
        "top_tc_kernels": top_tc_out,
        "non_tc_ms": round(total_ms - tc_ms, 2),
    }


# ---------------------------------------------------------------------------
# Kernel Grid / Block dimension summary
# ---------------------------------------------------------------------------

def kernel_grid_block_summary(
    prof: "Profile",
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    limit: int = 15,
) -> list[dict]:
    """Return per-kernel aggregated Grid/Block dimensions and resource usage.

    Each row contains:
    - kernel_name
    - invocations
    - total_ms
    - avg_grid_x/y/z, avg_block_x/y/z
    - threads_per_block  (blockX * blockY * blockZ)
    - total_threads      (gridX * gridY * gridZ * threads_per_block, per-call avg)
    - registers_per_thread
    - static_smem_bytes  (staticSharedMemory)
    - dynamic_smem_bytes (dynamicSharedMemory, averaged)
    - local_mem_per_thread
    """
    trim_clause = ""
    params: list[object] = [device]
    if trim:
        trim_clause = " AND k.start >= ? AND k.[end] <= ?"
        params.extend(trim)

    sql = f"""
        SELECT
            COALESCE(d.value, s.value) AS kernel_name,
            COUNT(*) AS invocations,
            ROUND(SUM(k.[end] - k.start) / 1e6, 3) AS total_ms,
            ROUND(AVG(k.[end] - k.start) / 1e6, 4) AS avg_ms,
            ROUND(AVG(k.gridX), 1)  AS avg_grid_x,
            ROUND(AVG(k.gridY), 1)  AS avg_grid_y,
            ROUND(AVG(k.gridZ), 1)  AS avg_grid_z,
            ROUND(AVG(k.blockX), 1) AS avg_block_x,
            ROUND(AVG(k.blockY), 1) AS avg_block_y,
            ROUND(AVG(k.blockZ), 1) AS avg_block_z,
            ROUND(AVG(k.blockX * k.blockY * k.blockZ), 0) AS threads_per_block,
            ROUND(AVG(k.gridX * k.gridY * k.gridZ * k.blockX * k.blockY * k.blockZ), 0) AS total_threads,
            ROUND(AVG(k.registersPerThread), 1) AS registers_per_thread,
            ROUND(AVG(k.staticSharedMemory), 0) AS static_smem_bytes,
            ROUND(AVG(k.dynamicSharedMemory), 0) AS dynamic_smem_bytes,
            ROUND(AVG(k.localMemoryPerThread), 0) AS local_mem_per_thread
        FROM {prof.schema.kernel_table} k
        JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN StringIds d ON k.demangledName = d.id
        WHERE k.deviceId = ?{trim_clause}
          AND k.blockX IS NOT NULL
          AND k.gridX IS NOT NULL
        GROUP BY COALESCE(d.value, s.value)
        ORDER BY total_ms DESC
        LIMIT ?
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, params + [limit]).fetchall()]

    # Add a convenience field: block occupancy hint
    # A100 has 1024 max threads/block and 2048 resident threads/SM
    # We flag kernels with very small blocks (<= 64 threads) as potentially low-occupancy
    for row in rows:
        tpb = int(row.get("threads_per_block") or 0)
        row["threads_per_block"] = tpb
        row["low_occupancy_hint"] = tpb > 0 and tpb <= 64

    return rows
