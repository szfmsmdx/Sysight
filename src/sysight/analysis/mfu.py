"""Lightweight FLOPs and region-level MFU helpers."""

from __future__ import annotations

import re

from sysight.profile import Profile

GPU_PEAK_TFLOPS: dict[str, float] = {
    "B200": 2250.0,
    "B100": 1750.0,
    "GB200": 2250.0,
    "RTX 5090": 838.0,
    "RTX 5080": 453.0,
    "RTX 5070 Ti": 370.0,
    "RTX 5070": 246.0,
    "H200": 989.0,
    "H100 SXM": 989.0,
    "H100 80GB HBM3": 989.0,
    "H100 PCIe": 756.0,
    "H100 NVL": 835.0,
    "H800": 989.0,
    "L40S": 362.0,
    "L40": 181.0,
    "L4": 121.0,
    "RTX 6000 Ada": 364.0,
    "RTX 5880 Ada": 305.0,
    "RTX 5000 Ada": 200.0,
    "RTX 4500 Ada": 160.0,
    "RTX 4000 Ada": 102.0,
    "RTX 4090": 330.0,
    "RTX 4080 SUPER": 204.0,
    "RTX 4080": 204.0,
    "RTX 4070 Ti SUPER": 184.0,
    "RTX 4070 Ti": 184.0,
    "RTX 4070 SUPER": 175.0,
    "RTX 4070": 147.0,
    "A100-SXM4-80GB": 312.0,
    "A100-SXM4-40GB": 312.0,
    "A100-PCIE-80GB": 312.0,
    "A100-PCIE-40GB": 312.0,
    "A100 80GB": 312.0,
    "A100 40GB": 312.0,
    "A100X": 312.0,
    "A800": 312.0,
    "A40": 150.0,
    "A30": 165.0,
    "A10G": 125.0,
    "A10": 125.0,
    "A16": 16.9,
    "A2": 36.0,
    "RTX A6000": 310.0,
    "RTX A5500": 256.0,
    "RTX A5000": 222.0,
    "RTX A4500": 185.0,
    "RTX A4000": 153.0,
    "RTX 3090 Ti": 160.0,
    "RTX 3090": 142.0,
    "RTX 3080 Ti": 136.0,
    "RTX 3080": 119.0,
    "RTX 3070 Ti": 87.0,
    "RTX 3070": 81.0,
    "T4": 65.0,
    "RTX 2080 Ti": 53.8,
    "V100 SXM2": 125.0,
    "V100S PCIe": 130.0,
    "V100 PCIe": 112.0,
    "V100": 112.0,
}

_VALID_OPERATIONS = {
    "attention",
    "qkv_proj",
    "output_proj",
    "mlp",
    "full_layer",
    "full_model",
    "linear",
}


def get_peak_tflops(gpu_name: str) -> dict:
    """Look up peak dense BF16/FP16 tensor-core TFLOPS from a GPU name."""
    if not (gpu_name or "").strip():
        return {"gpu_name": "", "error": "No GPU name provided"}
    name = gpu_name.strip()
    normalized = re.sub(r"\s+", " ", name.replace("-", " ").replace("NVIDIA", "").strip())
    for key in sorted(GPU_PEAK_TFLOPS, key=len, reverse=True):
        if key.replace("-", " ") in normalized:
            return {"gpu_name": name, "peak_tflops": GPU_PEAK_TFLOPS[key]}
    return {"gpu_name": name, "error": f"Unknown GPU '{name}'; provide peak_tflops manually"}


def compute_theoretical_flops(
    operation: str,
    *,
    hidden_dim: int = 0,
    seq_len: int = 0,
    num_layers: int = 1,
    ffn_dim: int | None = None,
    batch_size: int = 1,
    multiplier: int = 1,
    m_dim: int = 0,
    n_dim: int = 0,
    k_dim: int = 0,
) -> dict:
    """Compute exact theoretical FLOPs for common transformer operations."""
    op = (operation or "").lower().strip()
    if op not in _VALID_OPERATIONS:
        return {
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": f"operation must be one of {sorted(_VALID_OPERATIONS)}, got '{operation}'.",
            }
        }

    if op == "linear":
        if m_dim <= 0 or n_dim <= 0 or k_dim <= 0:
            return {
                "error": {
                    "code": "INVALID_ARGUMENT",
                    "message": "m_dim, n_dim, and k_dim must all be positive for 'linear'.",
                }
            }
        base = 2 * int(m_dim) * int(n_dim) * int(k_dim)
        total = base * max(int(batch_size), 1) * max(int(multiplier), 1)
        return {
            "operation": op,
            "theoretical_flops": total,
            "formula": f"2 * {m_dim} * {n_dim} * {k_dim} * batch({batch_size}) * mul({multiplier})",
            "breakdown": {
                "per_op": base,
                "batch_size": max(int(batch_size), 1),
                "multiplier": max(int(multiplier), 1),
            },
        }

    hidden = int(hidden_dim)
    seq = int(seq_len)
    layers = max(int(num_layers), 1)
    batch = max(int(batch_size), 1)
    mul = max(int(multiplier), 1)
    ffn = int(ffn_dim) if ffn_dim is not None else 4 * hidden
    if hidden <= 0 or seq <= 0:
        return {
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "hidden_dim and seq_len must be positive.",
            }
        }

    components: dict[str, int] = {}
    if op in {"attention", "full_layer", "full_model"}:
        components["attention"] = 4 * seq * seq * hidden
    if op in {"qkv_proj", "full_layer", "full_model"}:
        components["qkv_proj"] = 6 * seq * hidden * hidden
    if op in {"output_proj", "full_layer", "full_model"}:
        components["output_proj"] = 2 * seq * hidden * hidden
    if op in {"mlp", "full_layer", "full_model"}:
        components["mlp"] = 4 * seq * hidden * ffn

    per_layer = sum(components.values())
    total = per_layer * layers * batch * mul
    parts = " + ".join(f"{name}={value:.4e}" for name, value in components.items())
    return {
        "operation": op,
        "theoretical_flops": total,
        "formula": f"({parts}) * L({layers}) * batch({batch}) * mul({mul})",
        "breakdown": {
            "per_layer_flops": per_layer,
            "components": components,
            "num_layers": layers,
            "batch_size": batch,
            "multiplier": mul,
        },
    }


def format_theoretical_flops(result: dict) -> str:
    """Format a theoretical FLOPs result."""
    if "error" in result:
        error = result["error"]
        return f"(FLOPs error: {error.get('code', '?')}: {error.get('message', '')})"
    lines = ["── Theoretical FLOPs ──"]
    lines.append(f"  Operation: {result.get('operation', '?')}")
    lines.append(f"  Total FLOPs: {result.get('theoretical_flops', 0):,.0f}")
    lines.append(f"  Formula: {result.get('formula', '?')}")
    breakdown = result.get("breakdown", {})
    if isinstance(breakdown, dict):
        components = breakdown.get("components")
        for key, value in breakdown.items():
            if key == "components":
                continue
            lines.append(f"    {key}: {value:,.0f}" if isinstance(value, (int, float)) else f"    {key}: {value}")
        if isinstance(components, dict) and components:
            lines.append("    components:")
            for key, value in components.items():
                lines.append(f"      {key}: {value:,.0f}")
    return "\n".join(lines)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _resolve_string_ids(prof: Profile, pattern: str, match_mode: str = "contains") -> dict[int, str]:
    if match_mode == "exact":
        sql = "SELECT id, value FROM StringIds WHERE value = ?"
        params: list[object] = [pattern]
    elif match_mode == "startswith":
        sql = "SELECT id, value FROM StringIds WHERE value LIKE ? ESCAPE '\\'"
        params = [f"{_escape_like(pattern)}%"]
    else:
        sql = "SELECT id, value FROM StringIds WHERE value LIKE ? ESCAPE '\\'"
        params = [f"%{_escape_like(pattern)}%"]
    with prof._lock:
        return {int(row[0]): str(row[1]) for row in prof.conn.execute(sql, params).fetchall()}


def find_kernels_by_name(
    prof: Profile,
    name: str,
    *,
    match_mode: str = "contains",
    device: int | None = None,
    trim: tuple[int, int] | None = None,
) -> list[dict]:
    """Find kernels whose short names match a pattern."""
    id_map = _resolve_string_ids(prof, name, match_mode=match_mode)
    if not id_map:
        return []
    placeholders = ",".join("?" for _ in id_map)
    params: list[object] = list(id_map)
    filters = [f"k.shortName IN ({placeholders})", "k.[end] > k.start"]
    if device is not None:
        filters.append("k.deviceId = ?")
        params.append(device)
    if trim:
        filters.append("k.start >= ? AND k.[end] <= ?")
        params.extend([trim[0], trim[1]])
    sql = f"""
        SELECT k.shortName, k.start, k.[end], k.deviceId, k.streamId
        FROM {prof.schema.kernel_table} k
        WHERE {' AND '.join(filters)}
        ORDER BY k.start
    """
    with prof._lock:
        rows = prof.conn.execute(sql, params).fetchall()
    return [
        {
            "name": id_map.get(int(row["shortName"]), ""),
            "start_ns": int(row["start"]),
            "end_ns": int(row["end"]),
            "duration_ns": int(row["end"] - row["start"]),
            "device_id": int(row["deviceId"]) if row["deviceId"] is not None else None,
            "stream_id": int(row["streamId"]) if row["streamId"] is not None else None,
        }
        for row in rows
        if int(row["end"] - row["start"]) > 0
    ]


def find_nvtx_ranges(prof: Profile, name: str, *, match_mode: str = "contains") -> list[dict]:
    """Find NVTX ranges whose resolved text matches a pattern."""
    if not prof.schema.nvtx_table or not name:
        return []
    if match_mode == "exact":
        comparator = "= ?"
        value = name
    elif match_mode == "startswith":
        comparator = "LIKE ? ESCAPE '\\'"
        value = f"{_escape_like(name)}%"
    else:
        comparator = "LIKE ? ESCAPE '\\'"
        value = f"%{_escape_like(name)}%"
    if prof._nvtx_has_text_id:
        text_expr = "COALESCE(n.text, s.value)"
        sql = f"""
            SELECT {text_expr} AS text, n.start, n.[end], n.globalTid
            FROM {prof.schema.nvtx_table} n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE ({text_expr}) IS NOT NULL
              AND n.[end] > n.start
              AND {text_expr} {comparator}
            ORDER BY n.start
        """
    else:
        sql = f"""
            SELECT n.text AS text, n.start, n.[end], n.globalTid
            FROM {prof.schema.nvtx_table} n
            WHERE n.text IS NOT NULL
              AND n.[end] > n.start
              AND n.text {comparator}
            ORDER BY n.start
        """
    with prof._lock:
        rows = prof.conn.execute(sql, (value,)).fetchall()
    return [
        {
            "text": str(row["text"]),
            "start_ns": int(row["start"]),
            "end_ns": int(row["end"]),
            "global_tid": int(row["globalTid"]) if row["globalTid"] is not None else None,
            "duration_ns": int(row["end"] - row["start"]),
        }
        for row in rows
        if row["text"] is not None and int(row["end"] - row["start"]) > 0
    ]


def _select_occurrence(matches: list[dict], occurrence_index: int) -> dict:
    if not matches:
        return {"error": {"code": "NVTX_NOT_FOUND", "message": "No NVTX range matched the requested name."}}
    if occurrence_index <= 0:
        return {"error": {"code": "INVALID_ARGUMENT", "message": "occurrence_index must be >= 1."}}
    if occurrence_index > len(matches):
        return {
            "error": {
                "code": "NVTX_OCCURRENCE_OUT_OF_RANGE",
                "message": f"Requested occurrence_index {occurrence_index}, but only {len(matches)} matches were found.",
            }
        }
    chosen = dict(matches[occurrence_index - 1])
    chosen["occurrence_index"] = occurrence_index
    return chosen


def get_region_kernels(
    prof: Profile,
    *,
    nvtx_start_ns: int,
    nvtx_end_ns: int,
    global_tid: int | None,
    device: int | None,
) -> list[dict]:
    """Attribute kernels to an NVTX region via runtime correlation IDs."""
    if not prof.schema.runtime_table or not prof.schema.kernel_table:
        return []
    filters = ["r.start >= ?", "r.[end] <= ?"]
    params: list[object] = [int(nvtx_start_ns), int(nvtx_end_ns)]
    if global_tid is not None:
        filters.append("r.globalTid = ?")
        params.append(int(global_tid))
    if device is not None:
        filters.append("k.deviceId = ?")
        params.append(int(device))
    sql = f"""
        SELECT
            r.correlationId,
            k.deviceId,
            k.streamId,
            k.start,
            k.[end],
            s.value AS kernel_name
        FROM {prof.schema.runtime_table} r
        JOIN {prof.schema.kernel_table} k ON r.correlationId = k.correlationId
        LEFT JOIN StringIds s ON k.shortName = s.id
        WHERE {' AND '.join(filters)}
        ORDER BY k.start
    """
    with prof._lock:
        rows = prof.conn.execute(sql, params).fetchall()
    return [
        {
            "correlation_id": int(row["correlationId"]),
            "device_id": int(row["deviceId"]) if row["deviceId"] is not None else None,
            "stream_id": int(row["streamId"]) if row["streamId"] is not None else None,
            "start_ns": int(row["start"]),
            "end_ns": int(row["end"]),
            "duration_ns": int(row["end"] - row["start"]),
            "name": str(row["kernel_name"]) if row["kernel_name"] is not None else "",
        }
        for row in rows
        if int(row["end"] - row["start"]) > 0
    ]


def _merge_intervals(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_start, cur_end = intervals[0]
    for start_ns, end_ns in intervals[1:]:
        if start_ns <= cur_end:
            cur_end = max(cur_end, end_ns)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start_ns, end_ns
    total += cur_end - cur_start
    return total


def summarize_region_kernel_times(kernels: list[dict]) -> dict:
    if not kernels:
        return {
            "kernel_count": 0,
            "kernel_sum_ns": 0,
            "kernel_union_ns": 0,
            "device_ids": [],
            "stream_ids": [],
        }
    kernel_sum_ns = sum(int(kernel["duration_ns"]) for kernel in kernels)
    kernel_union_ns = _merge_intervals(
        [(int(kernel["start_ns"]), int(kernel["end_ns"])) for kernel in kernels]
    )
    return {
        "kernel_count": len(kernels),
        "kernel_sum_ns": int(kernel_sum_ns),
        "kernel_union_ns": int(kernel_union_ns),
        "device_ids": sorted(
            {int(kernel["device_id"]) for kernel in kernels if kernel.get("device_id") is not None}
        ),
        "stream_ids": sorted(
            {int(kernel["stream_id"]) for kernel in kernels if kernel.get("stream_id") is not None}
        ),
    }


def _resolve_peak_tflops(prof: Profile, device: int | None, explicit_peak_tflops: float | None) -> dict:
    if explicit_peak_tflops is not None:
        peak = float(explicit_peak_tflops)
        if peak <= 0:
            return {
                "error": {
                    "code": "INVALID_ARGUMENT",
                    "message": "peak_tflops must be positive when provided explicitly.",
                }
            }
        return {"peak_tflops": peak, "source": "explicit"}
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else None)
    gpu_name = prof.meta.gpu_info.get(target).name if target is not None and target in prof.meta.gpu_info else ""
    info = get_peak_tflops(gpu_name)
    if "error" in info:
        return {
            "error": {
                "code": "GPU_PEAK_UNKNOWN",
                "message": info["error"],
            }
        }
    return {"peak_tflops": float(info["peak_tflops"]), "source": "profile", "gpu_name": gpu_name}


def compute_region_mfu(
    prof: Profile,
    name: str,
    theoretical_flops: float,
    *,
    source: str = "nvtx",
    peak_tflops: float | None = None,
    num_gpus: int = 1,
    occurrence_index: int = 1,
    device: int | None = None,
    match_mode: str = "contains",
) -> dict:
    """Compute MFU for a named NVTX region or kernel pattern."""
    if not name:
        return {"error": {"code": "INVALID_ARGUMENT", "message": "name must be a non-empty string."}}
    if float(theoretical_flops) <= 0:
        return {
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "theoretical_flops must be positive.",
            }
        }
    if source not in {"nvtx", "kernel"}:
        return {
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "source must be 'nvtx' or 'kernel'.",
            }
        }

    if source == "kernel":
        kernels = find_kernels_by_name(prof, name, match_mode=match_mode, device=device)
        if not kernels:
            return {
                "error": {
                    "code": "KERNEL_NOT_FOUND",
                    "message": f"No kernels matching '{name}' were found.",
                }
            }
        summary = summarize_region_kernel_times(kernels)
        wall_time_ns = max(kernel["end_ns"] for kernel in kernels) - min(
            kernel["start_ns"] for kernel in kernels
        )
        matched_name = kernels[0]["name"]
    else:
        matches = find_nvtx_ranges(prof, name, match_mode=match_mode)
        chosen = _select_occurrence(matches, int(occurrence_index))
        if "error" in chosen:
            return chosen
        wall_time_ns = int(chosen["duration_ns"])
        kernels = get_region_kernels(
            prof,
            nvtx_start_ns=int(chosen["start_ns"]),
            nvtx_end_ns=int(chosen["end_ns"]),
            global_tid=chosen.get("global_tid"),
            device=device,
        )
        summary = summarize_region_kernel_times(kernels)
        if summary["kernel_count"] == 0:
            return {
                "error": {
                    "code": "NO_KERNELS_IN_REGION",
                    "message": "No kernels were attributed to the selected NVTX region.",
                }
            }
        matched_name = str(chosen.get("text") or name)

    peak_info = _resolve_peak_tflops(prof, device, peak_tflops)
    if "error" in peak_info:
        return peak_info
    peak_per_gpu = float(peak_info["peak_tflops"])
    effective_num_gpus = max(int(num_gpus), 1)
    effective_peak = peak_per_gpu * effective_num_gpus
    wall_time_s = wall_time_ns / 1e9
    kernel_sum_s = summary["kernel_sum_ns"] / 1e9
    kernel_union_s = summary["kernel_union_ns"] / 1e9

    def achieved(time_s: float) -> float:
        return (float(theoretical_flops) / time_s) / 1e12 if time_s > 0 else 0.0

    achieved_wall = achieved(wall_time_s)
    achieved_sum = achieved(kernel_sum_s)
    achieved_union = achieved(kernel_union_s)

    return {
        "source": source,
        "name": name,
        "matched_name": matched_name,
        "match_mode": match_mode,
        "occurrence_index": int(occurrence_index) if source == "nvtx" else None,
        "device": device,
        "num_gpus": effective_num_gpus,
        "peak_tflops_per_gpu": peak_per_gpu,
        "effective_peak_tflops": effective_peak,
        "kernel_count": summary["kernel_count"],
        "device_ids": summary["device_ids"],
        "stream_ids": summary["stream_ids"],
        "timing": {
            "wall_time_ms": round(wall_time_s * 1e3, 3),
            "kernel_sum_ms": round(kernel_sum_s * 1e3, 3),
            "kernel_union_ms": round(kernel_union_s * 1e3, 3),
        },
        "mfu": {
            "peak_tflops": effective_peak,
            "achieved_tflops_wall": round(achieved_wall, 2),
            "achieved_tflops_kernel_sum": round(achieved_sum, 2),
            "achieved_tflops_kernel_union": round(achieved_union, 2),
            "mfu_pct_wall": round(100.0 * achieved_wall / effective_peak, 2) if effective_peak else 0.0,
            "mfu_pct_kernel_sum": round(100.0 * achieved_sum / effective_peak, 2) if effective_peak else 0.0,
            "mfu_pct_kernel_union": round(100.0 * achieved_union / effective_peak, 2) if effective_peak else 0.0,
        },
        "theoretical_flops": float(theoretical_flops),
    }


def format_region_mfu(result: dict) -> str:
    """Format a region MFU result."""
    if "error" in result:
        error = result["error"]
        return f"(MFU error: {error.get('code', '?')}: {error.get('message', '')})"
    timing = result.get("timing", {})
    mfu = result.get("mfu", {})
    lines = ["── Region MFU ──"]
    lines.append(f"  Region: {result.get('matched_name', '?')}")
    lines.append(f"  Source: {result.get('source', '?')}")
    if result.get("device_ids"):
        lines.append(f"  Devices: {result['device_ids']}")
    if result.get("stream_ids"):
        lines.append(f"  Streams: {result['stream_ids']}")
    lines.append(f"  Kernel count: {result.get('kernel_count', 0)}")
    if timing:
        lines.append(f"  Wall time:    {timing.get('wall_time_ms', 0.0):.2f}ms")
        lines.append(f"  Kernel sum:   {timing.get('kernel_sum_ms', 0.0):.2f}ms")
        lines.append(f"  Kernel union: {timing.get('kernel_union_ms', 0.0):.2f}ms")
    if mfu:
        lines.append(f"  Peak:         {mfu.get('peak_tflops', 0.0):.1f} TFLOPS")
        lines.append(
            f"  Achieved:     wall={mfu.get('achieved_tflops_wall', 0.0):.2f} | "
            f"sum={mfu.get('achieved_tflops_kernel_sum', 0.0):.2f} | "
            f"union={mfu.get('achieved_tflops_kernel_union', 0.0):.2f} TFLOPS"
        )
        lines.append(
            f"  MFU:          wall={mfu.get('mfu_pct_wall', 0.0):.2f}% | "
            f"sum={mfu.get('mfu_pct_kernel_sum', 0.0):.2f}% | "
            f"union={mfu.get('mfu_pct_kernel_union', 0.0):.2f}%"
        )
    return "\n".join(lines)
