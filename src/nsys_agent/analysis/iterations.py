"""Iteration timing helpers adapted from nsys-ai."""

from __future__ import annotations

import statistics

from nsys_agent.profile import Profile


def _find_primary_thread(prof: Profile, device: int) -> int:
    """Return the CPU thread with the most kernel launches for one GPU."""
    if not prof.schema.runtime_table:
        return 0
    sql = f"""
        SELECT r.globalTid, COUNT(*) AS launch_count
        FROM {prof.schema.runtime_table} r
        JOIN {prof.schema.kernel_table} k ON r.correlationId = k.correlationId
        WHERE k.deviceId = ?
        GROUP BY r.globalTid
        ORDER BY launch_count DESC
    """
    with prof._lock:
        row = prof.conn.execute(sql, (device,)).fetchone()
    return int(row[0]) if row else 0


def detect_iterations(
    prof: Profile,
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    marker: str = "sample_0",
) -> list[dict]:
    """Detect iterations from NVTX markers with a heuristic fallback."""
    if not prof.schema.runtime_table:
        return []

    primary_tid = _find_primary_thread(prof, device)
    if primary_tid == 0:
        return []

    time_range = trim or prof.meta.time_range
    iterations = _nvtx_iterations(prof, primary_tid, time_range, marker)
    if not iterations:
        iterations = _heuristic_iterations(prof, device, primary_tid, time_range)
    if not iterations:
        return []

    kernel_map = prof.kernel_map(device)
    runtime_sql = f"""
        SELECT start, [end], correlationId
        FROM {prof.schema.runtime_table}
        WHERE globalTid = ?
        ORDER BY start
    """
    with prof._lock:
        runtime_rows = prof.conn.execute(runtime_sql, (primary_tid,)).fetchall()

    results = []
    for index, it in enumerate(iterations):
        cpu_start = int(it["start"])
        cpu_end = int(it["end"])
        kernels = []
        for row in runtime_rows:
            if row["start"] > cpu_end:
                break
            if row["start"] >= cpu_start and row["end"] <= cpu_end:
                kernel = kernel_map.get(row["correlationId"])
                if kernel:
                    kernels.append(kernel)
        if not kernels:
            continue

        gpu_start = min(kernel["start"] for kernel in kernels)
        gpu_end = max(kernel["end"] for kernel in kernels)
        compute_ns = sum(kernel["end"] - kernel["start"] for kernel in kernels)
        nccl_count = sum(1 for kernel in kernels if "nccl" in (kernel["name"] or "").lower())
        results.append(
            {
                "iteration": index,
                "text": it.get("text") or "",
                "gpu_start_s": round(gpu_start / 1e9, 4),
                "gpu_end_s": round(gpu_end / 1e9, 4),
                "duration_ms": round((gpu_end - gpu_start) / 1e6, 2),
                "compute_ms": round(compute_ns / 1e6, 2),
                "kernel_count": len(kernels),
                "nccl_count": nccl_count,
            }
        )
    return results


def iteration_summary(rows: list[dict]) -> dict | None:
    """Return aggregate iteration statistics."""
    if not rows:
        return None
    durations = [row["duration_ms"] for row in rows]
    avg_ms = sum(durations) / len(durations)
    median_ms = statistics.median(durations)
    slow_rows = [row for row in rows if median_ms > 0 and row["duration_ms"] > 1.5 * median_ms]
    return {
        "count": len(rows),
        "avg_ms": round(avg_ms, 2),
        "median_ms": round(median_ms, 2),
        "min_ms": round(min(durations), 2),
        "max_ms": round(max(durations), 2),
        "slow_iterations": slow_rows,
        "heuristic": any((row.get("text") or "").startswith("heuristic_") for row in rows),
    }


def _nvtx_iterations(
    prof: Profile,
    primary_tid: int,
    time_range: tuple[int, int],
    marker: str,
) -> list[dict]:
    if not prof.schema.nvtx_table:
        return []

    if prof._nvtx_has_text_id:
        sql = f"""
            SELECT COALESCE(n.text, s.value) AS text, n.start, n.[end]
            FROM {prof.schema.nvtx_table} n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
              AND COALESCE(n.text, s.value) LIKE ?
              AND n.[end] > n.start
              AND n.globalTid = ?
              AND n.start >= ? AND n.start <= ?
            ORDER BY n.start
        """
    else:
        sql = f"""
            SELECT text, start, [end]
            FROM {prof.schema.nvtx_table}
            WHERE text LIKE ?
              AND [end] > start
              AND globalTid = ?
              AND start >= ? AND start <= ?
            ORDER BY start
        """
    with prof._lock:
        rows = prof.conn.execute(
            sql,
            (f"%{marker}%", primary_tid, time_range[0], time_range[1]),
        ).fetchall()

    iterations = []
    last_end = 0
    for row in rows:
        if row["start"] >= last_end:
            iterations.append({"text": row["text"] or "", "start": row["start"], "end": row["end"]})
            last_end = row["end"]
    return iterations


def _heuristic_iterations(
    prof: Profile,
    device: int,
    primary_tid: int,
    time_range: tuple[int, int],
) -> list[dict]:
    kernel_map = prof.kernel_map(device)
    if not kernel_map:
        return []

    sql = f"""
        SELECT start, [end], correlationId
        FROM {prof.schema.runtime_table}
        WHERE globalTid = ?
        ORDER BY start
    """
    with prof._lock:
        runtime_rows = prof.conn.execute(sql, (primary_tid,)).fetchall()

    kernel_entries = []
    for row in runtime_rows:
        kernel = kernel_map.get(row["correlationId"])
        if not kernel:
            continue
        if kernel["end"] < time_range[0] or kernel["start"] > time_range[1]:
            continue
        kernel_entries.append(
            {
                "kernel": kernel,
                "rt_start": row["start"],
                "rt_end": row["end"],
            }
        )
    kernel_entries.sort(key=lambda entry: entry["kernel"]["start"])
    if not kernel_entries:
        return []

    boundaries = [kernel_entries[0]["rt_start"]]
    last_end = kernel_entries[0]["kernel"]["end"]
    for entry in kernel_entries[1:]:
        kernel = entry["kernel"]
        if kernel["start"] - last_end > 2_000_000:
            boundaries.append(entry["rt_start"])
        last_end = max(last_end, kernel["end"])
    boundaries.append(kernel_entries[-1]["rt_end"])

    return [
        {
            "text": f"heuristic_step_{index}",
            "start": boundaries[index],
            "end": boundaries[index + 1],
        }
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] > boundaries[index]
    ]
