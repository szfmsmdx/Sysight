"""Root-cause pattern checks for deep Nsight Systems SQL analysis."""

from __future__ import annotations

import logging
import sqlite3

from .models import NsysFinding
from .sql_shared import _NCCL_KEYWORDS, _get_cols

logger = logging.getLogger(__name__)


def _sql_root_cause_analysis(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    kernel_tbl: str,
    runtime_tbl: str | None,
    memcpy_tbl: str | None,
    memset_tbl: str | None,
    sync_tbl: str | None,
    nvtx_tbl: str | None,
    has_strings: bool,
    total_ns: int,
) -> None:
    """程序化检测已知 GPU 性能反模式。"""
    del sync_tbl, nvtx_tbl, total_ns
    patterns: list[dict[str, str]] = []

    if runtime_tbl and has_strings:
        try:
            sync_name_rows = conn.execute(
                """
                SELECT id, value FROM StringIds
                WHERE value LIKE 'cudaDeviceSynchronize%'
                   OR value LIKE 'cudaStreamSynchronize%'
                   OR value LIKE 'cudaEventSynchronize%'
                   OR value LIKE 'cudaStreamWaitEvent%'
                """
            ).fetchall()
            if sync_name_rows:
                ph = ",".join(str(row[0]) for row in sync_name_rows)
                sync_stat = conn.execute(
                    f"""
                    SELECT COUNT(*) AS call_count, COALESCE(SUM([end] - start), 0) AS total_ns
                    FROM {runtime_tbl}
                    WHERE nameId IN ({ph})
                    """
                ).fetchone()
                if sync_stat and sync_stat[0]:
                    total_gpu_ns_row = conn.execute(
                        f"SELECT COALESCE(SUM([end] - start), 0) FROM {kernel_tbl}"
                    ).fetchone()
                    total_gpu_ns = int(total_gpu_ns_row[0] or 0) if total_gpu_ns_row else 0
                    sync_ns = int(sync_stat[1] or 0)
                    sync_ms = sync_ns / 1e6
                    sync_pct = (sync_ns / total_gpu_ns * 100) if total_gpu_ns > 0 else 100.0
                    if sync_ms >= 1.0 and sync_pct >= 2.0:
                        api_names = ", ".join(sorted({row[1].split("_v")[0] for row in sync_name_rows}))
                        patterns.append({
                            "pattern": "过度同步（Excessive Synchronization）",
                            "severity": "warning",
                            "evidence": (
                                f"{sync_stat[0]} 次同步调用共 {sync_ms:.1f}ms"
                                f"（GPU 时间的 {sync_pct:.1f}%）。API：{api_names}"
                            ),
                            "recommendation": (
                                "从训练循环中移除 .item()/.cpu() 隐式同步。"
                                "用 torch.cuda.set_sync_debug_mode(1) 定位隐式同步。"
                                "用 CUDA events 替换 cudaDeviceSynchronize。"
                            ),
                        })
        except sqlite3.Error as exc:
            logger.debug("_sql_root_cause (sync_api): %s", exc)

    if runtime_tbl and memcpy_tbl and has_strings:
        try:
            sync_memcpy_names = conn.execute(
                """
                SELECT id FROM StringIds
                WHERE value LIKE 'cudaMemcpy%'
                  AND value NOT LIKE 'cudaMemcpyAsync%'
                """
            ).fetchall()
            if sync_memcpy_names:
                ph = ",".join(str(row[0]) for row in sync_memcpy_names)
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS cnt,
                           COALESCE(SUM(m.bytes), 0) AS total_bytes,
                           COALESCE(SUM(m.[end] - m.start), 0) AS total_ns
                    FROM {runtime_tbl} r
                    JOIN {memcpy_tbl} m ON r.correlationId = m.correlationId
                    WHERE r.nameId IN ({ph})
                    """
                ).fetchone()
                if row and row[0]:
                    patterns.append({
                        "pattern": "同步 Memcpy（Synchronous Memcpy）",
                        "severity": "warning",
                        "evidence": (
                            f"{row[0]} 次同步 cudaMemcpy：{int(row[1] or 0)/1e6:.1f}MB，"
                            f"{int(row[2] or 0)/1e6:.1f}ms。这些调用阻塞 host 线程。"
                        ),
                        "recommendation": (
                            "改用 cudaMemcpyAsync + pinned memory。"
                            "DataLoader 使用 pin_memory=True，张量 .to(device) 加 non_blocking=True。"
                        ),
                    })
        except sqlite3.Error as exc:
            logger.debug("_sql_root_cause (sync_memcpy): %s", exc)

    if memcpy_tbl:
        try:
            cols = _get_cols(conn, memcpy_tbl)
            if "srcKind" in cols and "dstKind" in cols:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS cnt,
                           COALESCE(SUM(bytes), 0) AS total_bytes,
                           COALESCE(SUM([end] - start), 0) AS total_ns
                    FROM {memcpy_tbl}
                    WHERE srcKind = 1 OR dstKind = 1
                    """
                ).fetchone()
                if row and row[0]:
                    patterns.append({
                        "pattern": "Pageable Memory 异步拷贝（实为同步）",
                        "severity": "warning",
                        "evidence": (
                            f"{row[0]} 次 memcpy 使用 pageable（非 pinned）内存："
                            f"{int(row[1] or 0)/1e6:.1f}MB，{int(row[2] or 0)/1e6:.1f}ms。"
                            "Pageable memory 导致 async memcpy 静默降级为同步。"
                        ),
                        "recommendation": (
                            "使用 cudaMallocHost() / DataLoader pin_memory=True。"
                            "Pinned memory 才能实现真正的异步 H2D 重叠。"
                        ),
                    })
        except sqlite3.Error as exc:
            logger.debug("_sql_root_cause (pageable): %s", exc)

    if runtime_tbl and memset_tbl and has_strings:
        try:
            sync_memset_names = conn.execute(
                """
                SELECT id FROM StringIds
                WHERE value LIKE 'cudaMemset%'
                  AND value NOT LIKE 'cudaMemsetAsync%'
                """
            ).fetchall()
            if sync_memset_names:
                ph = ",".join(str(row[0]) for row in sync_memset_names)
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS cnt,
                           COALESCE(SUM(ms.[end] - ms.start), 0) AS total_ns
                    FROM {runtime_tbl} r
                    JOIN {memset_tbl} ms ON r.correlationId = ms.correlationId
                    WHERE r.nameId IN ({ph})
                    """
                ).fetchone()
                if row and row[0]:
                    patterns.append({
                        "pattern": "同步 Memset（Synchronous Memset）",
                        "severity": "info",
                        "evidence": f"{row[0]} 次同步 cudaMemset，{int(row[1] or 0)/1e6:.2f}ms 阻塞 host 线程。",
                        "recommendation": "改用 cudaMemsetAsync 在对应 stream 上执行。",
                    })
        except sqlite3.Error as exc:
            logger.debug("_sql_root_cause (sync_memset): %s", exc)

    if has_strings:
        try:
            like_clauses = " OR ".join(
                f"LOWER(s.value) LIKE '%{keyword}%'" for keyword in _NCCL_KEYWORDS
            )
            same_stream_rows = conn.execute(
                f"""
                SELECT k.streamId,
                    SUM(CASE WHEN {like_clauses} THEN 1 ELSE 0 END) AS nccl_count,
                    SUM(CASE WHEN NOT ({like_clauses}) THEN 1 ELSE 0 END) AS compute_count
                FROM {kernel_tbl} k
                JOIN StringIds s ON k.shortName = s.id
                GROUP BY k.streamId
                HAVING nccl_count > 0 AND compute_count > 0
                """
            ).fetchall()
            if same_stream_rows:
                streams_list = [str(row[0]) for row in same_stream_rows]
                patterns.append({
                    "pattern": "NCCL 序列化——同 Stream（Same-Stream Serialization）",
                    "severity": "critical",
                    "evidence": (
                        f"Stream [{', '.join(streams_list[:5])}] 上 NCCL 与计算 kernel 共存，"
                        "被 CUDA stream 顺序强制序列化。"
                    ),
                    "recommendation": (
                        "将 AllReduce 移到独立的 NCCL stream。"
                        "PyTorch DDP：检查 find_unused_parameters=True 是否强制同步。"
                        "使用 bucket_cap_mb 调整 gradient bucketing 粒度。"
                    ),
                })
        except sqlite3.Error as exc:
            logger.debug("_sql_root_cause (nccl same_stream): %s", exc)

    if runtime_tbl and has_strings:
        try:
            sync_names_for_nccl = conn.execute(
                """
                SELECT id FROM StringIds
                WHERE value LIKE 'cudaStreamSynchronize%'
                   OR value LIKE 'cudaDeviceSynchronize%'
                """
            ).fetchall()
            if sync_names_for_nccl:
                like_clauses_nccl = " OR ".join(
                    f"LOWER(sn.value) LIKE '%{keyword}%'" for keyword in _NCCL_KEYWORDS
                )
                ph = ",".join(str(row[0]) for row in sync_names_for_nccl)
                found = conn.execute(
                    f"""
                    SELECT 1 FROM {kernel_tbl} k
                    JOIN StringIds sn ON k.shortName = sn.id
                    WHERE ({like_clauses_nccl})
                      AND EXISTS (
                          SELECT 1 FROM {runtime_tbl} r
                          WHERE r.nameId IN ({ph})
                            AND r.start >= k.[end]
                            AND r.start <= k.[end] + 1000000
                      )
                    LIMIT 1
                    """
                ).fetchone()
                if found:
                    patterns.append({
                        "pattern": "NCCL 后立即同步（Sync-After-NCCL）",
                        "severity": "warning",
                        "evidence": (
                            "NCCL kernel 完成后 1ms 内检测到 "
                            "cudaStreamSynchronize / cudaDeviceSynchronize 调用。"
                        ),
                        "recommendation": (
                            "移除 NCCL 操作后的显式同步。"
                            "通信/计算传输使用 non_blocking=True。"
                        ),
                    })
        except sqlite3.Error as exc:
            logger.debug("_sql_root_cause (sync_after_nccl): %s", exc)

    if not patterns:
        return

    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    patterns.sort(key=lambda pattern: severity_rank.get(pattern["severity"], 3))

    evidence_lines: list[str] = []
    for pattern in patterns:
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(pattern["severity"], "⚪")
        evidence_lines.append(f"{icon} {pattern['pattern']}：{pattern['evidence']}")
        evidence_lines.append(f"   建议：{pattern['recommendation']}")

    findings.append(NsysFinding(
        category="sql_root_cause_analysis",
        severity=patterns[0]["severity"],
        title=f"根因分析：{len(patterns)} 个反模式（最严重：{patterns[0]['severity']}）",
        description=(
            f"程序化检测发现 {len(patterns)} 个已知 GPU 性能反模式："
            + "；".join(pattern["pattern"] for pattern in patterns[:3])
            + "。"
        ),
        evidence=evidence_lines[:12],
        related_hotspots=[pattern["pattern"] for pattern in patterns[:5]],
        next_step=(
            "按 severity 顺序逐一修复。CRITICAL 级（同 stream 序列化）通常收益最大。"
            "每次修复后重新 profile 验证效果。"
        ),
    ))
