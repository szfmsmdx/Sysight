"""Lightweight NVTX attribution helpers adapted from nsys-ai."""

from __future__ import annotations

from collections import defaultdict

from nsys_agent.analysis.queries import classify_kernel
from nsys_agent.profile import Profile


def attribute_kernels_to_nvtx(
    prof: Profile,
    device: int | None = None,
    trim: tuple[int, int] | None = None,
) -> list[dict]:
    """Attribute GPU kernels to their enclosing NVTX ranges via sort-merge."""
    if not prof.schema.runtime_table or not prof.schema.nvtx_table:
        return []

    trim_clause = ""
    params: list[object] = []
    if device is not None:
        trim_clause += " AND k.deviceId = ?"
        params.append(device)
    if trim:
        trim_clause += " AND k.start >= ? AND k.[end] <= ?"
        params.extend([trim[0], trim[1]])

    kernel_sql = f"""
        SELECT
            r.globalTid,
            r.start AS runtime_start,
            r.[end] AS runtime_end,
            k.start AS kernel_start,
            k.[end] AS kernel_end,
            k.deviceId,
            k.streamId,
            k.shortName
        FROM {prof.schema.kernel_table} k
        JOIN {prof.schema.runtime_table} r ON r.correlationId = k.correlationId
        WHERE 1=1{trim_clause}
        ORDER BY r.globalTid, r.start
    """
    with prof._lock:
        kernel_rows = prof.conn.execute(kernel_sql, params).fetchall()
    if not kernel_rows:
        return []

    if prof._nvtx_has_text_id:
        nvtx_sql = f"""
            SELECT
                n.globalTid,
                n.start,
                n.[end],
                COALESCE(n.text, s.value) AS text
            FROM {prof.schema.nvtx_table} n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
              AND n.[end] > n.start
            ORDER BY n.globalTid, n.start
        """
    else:
        nvtx_sql = f"""
            SELECT globalTid, start, [end], text
            FROM {prof.schema.nvtx_table}
            WHERE text IS NOT NULL AND [end] > start
            ORDER BY globalTid, start
        """
    with prof._lock:
        nvtx_rows = prof.conn.execute(nvtx_sql).fetchall()

    short_name_ids = sorted({row["shortName"] for row in kernel_rows if row["shortName"] is not None})
    if short_name_ids:
        placeholders = ",".join("?" for _ in short_name_ids)
        with prof._lock:
            string_rows = prof.conn.execute(
                f"SELECT id, value FROM StringIds WHERE id IN ({placeholders})",
                short_name_ids,
            ).fetchall()
        name_map = {row[0]: row[1] for row in string_rows}
    else:
        name_map = {}

    nvtx_by_tid: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for row in nvtx_rows:
        nvtx_by_tid[row["globalTid"]].append((row["start"], row["end"], row["text"] or ""))

    kernels_by_tid: dict[int, list[dict]] = defaultdict(list)
    for row in kernel_rows:
        kernels_by_tid[row["globalTid"]].append(
            {
                "runtime_start": row["runtime_start"],
                "runtime_end": row["runtime_end"],
                "kernel_start": row["kernel_start"],
                "kernel_end": row["kernel_end"],
                "device_id": row["deviceId"],
                "stream": row["streamId"],
                "kernel_name": name_map.get(row["shortName"], f"kernel_{row['shortName']}"),
            }
        )

    results: list[dict] = []
    for tid, kernel_list in kernels_by_tid.items():
        nvtx_list = nvtx_by_tid.get(tid)
        if not nvtx_list:
            continue

        nvtx_idx = 0
        open_stack: list[tuple[int, int, str]] = []
        for kernel in kernel_list:
            runtime_start = int(kernel["runtime_start"])
            runtime_end = int(kernel["runtime_end"])

            while open_stack and open_stack[-1][1] < runtime_start:
                open_stack.pop()

            while nvtx_idx < len(nvtx_list) and nvtx_list[nvtx_idx][0] <= runtime_start:
                if nvtx_list[nvtx_idx][1] >= runtime_start:
                    open_stack.append(nvtx_list[nvtx_idx])
                nvtx_idx += 1

            best_idx = -1
            for index in range(len(open_stack) - 1, -1, -1):
                start_ns, end_ns, _ = open_stack[index]
                if start_ns <= runtime_start and end_ns >= runtime_end:
                    best_idx = index
                    break
            if best_idx < 0:
                continue

            enclosing = [
                entry
                for entry in open_stack[: best_idx + 1]
                if entry[0] <= runtime_start and entry[1] >= runtime_end
            ]
            results.append(
                {
                    "nvtx_text": enclosing[-1][2],
                    "nvtx_depth": len(enclosing) - 1,
                    "nvtx_path": " > ".join(entry[2] for entry in enclosing if entry[2]),
                    "kernel_name": kernel["kernel_name"],
                    "k_start": int(kernel["kernel_start"]),
                    "k_end": int(kernel["kernel_end"]),
                    "k_dur_ns": int(kernel["kernel_end"] - kernel["kernel_start"]),
                    "device_id": int(kernel["device_id"]),
                    "stream": int(kernel["stream"]),
                    "global_tid": int(tid),
                }
            )
    return results


def nvtx_layer_breakdown(
    prof: Profile,
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    limit: int = 20,
    depth: int | None = None,
) -> list[dict]:
    """Aggregate GPU time by NVTX region for one GPU."""
    rows = attribute_kernels_to_nvtx(prof, device=device, trim=trim)
    if not rows:
        return []

    if depth is not None:
        if depth < 0:
            return [{"error": "Invalid depth < 0 requested"}]
        rows = [row for row in rows if row.get("nvtx_depth") == depth]
        if not rows:
            return []

    groups: dict[str, dict] = defaultdict(
        lambda: {
            "total_ns": 0,
            "compute_ns": 0,
            "nccl_ns": 0,
            "kernel_count": 0,
            "max_ns": 0,
            "nvtx_region": "",
            "nvtx_depth": -1,
            "nvtx_path": "",
        }
    )

    for row in rows:
        label = row.get("nvtx_path") or row.get("nvtx_text") or "(unnamed)"
        stats = groups[label]
        dur_ns = int(row["k_dur_ns"])
        stats["total_ns"] += dur_ns
        stats["kernel_count"] += 1
        stats["max_ns"] = max(stats["max_ns"], dur_ns)
        if classify_kernel(row["kernel_name"]).startswith("nccl_"):
            stats["nccl_ns"] += dur_ns
        else:
            stats["compute_ns"] += dur_ns
        if stats["nvtx_depth"] < 0:
            stats["nvtx_region"] = row.get("nvtx_text") or ""
            stats["nvtx_depth"] = int(row.get("nvtx_depth") or 0)
            stats["nvtx_path"] = row.get("nvtx_path") or label

    result = []
    for stats in groups.values():
        total_ns = stats["total_ns"]
        kernel_count = stats["kernel_count"]
        result.append(
            {
                "nvtx_region": stats["nvtx_region"],
                "nvtx_depth": stats["nvtx_depth"],
                "nvtx_path": stats["nvtx_path"],
                "kernel_count": kernel_count,
                "total_gpu_ms": round(total_ns / 1e6, 2),
                "compute_ms": round(stats["compute_ns"] / 1e6, 2),
                "nccl_ms": round(stats["nccl_ns"] / 1e6, 2),
                "nccl_pct": round(100 * stats["nccl_ns"] / total_ns, 1) if total_ns else 0.0,
                "avg_kernel_ms": round(total_ns / kernel_count / 1e6, 3) if kernel_count else 0.0,
                "max_kernel_ms": round(stats["max_ns"] / 1e6, 3),
            }
        )

    result.sort(key=lambda item: (-item["total_gpu_ms"], item["nvtx_path"]))
    return result[:limit]


def nvtx_kernel_map(
    prof: Profile,
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    limit: int = 50,
) -> list[dict]:
    """Return the earliest NVTX to kernel mappings for one GPU."""
    rows = attribute_kernels_to_nvtx(prof, device=device, trim=trim)
    rows.sort(key=lambda row: row["k_start"])
    return [
        {
            "nvtx_text": row["nvtx_text"],
            "nvtx_path": row["nvtx_path"],
            "kernel_name": row["kernel_name"],
            "start_ms": round(row["k_start"] / 1e6, 3),
            "end_ms": round(row["k_end"] / 1e6, 3),
            "stream": row["stream"],
        }
        for row in rows[:limit]
    ]
