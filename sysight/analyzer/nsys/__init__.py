"""nsys — profile analyzer for NVIDIA Nsight Systems."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from .classify import classify_bottlenecks
from .extract import extract_trace, inspect_schema, resolve_profile_input
from .localization import (
    has_cli_investigator,
    register_cli_investigator,
    run_code_localization,
)
from .models import (
    BottleneckSummary,
    EvidenceWindow,
    LocalizationResult,
    NsysAnalysisRequest,
    NsysDiag,
    NsysFinding,
    NsysTrace,
    ProfileInput,
    SampleHotspot,
    SchemaInfo,
    TimelineEvent,
)
from .render import render_nsys_profile_report
from .stacks import build_cpu_hotspots
from .windows import extract_evidence_windows

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_nsys",
    "prepare_analysis_request",
    "register_cli_investigator",
    "inspect_schema",
    "extract_trace",
    "classify_bottlenecks",
    "extract_evidence_windows",
    "NsysDiag",
    "NsysAnalysisRequest",
    "NsysFinding",
    "LocalizationResult",
    "EvidenceWindow",
    "BottleneckSummary",
    "SampleHotspot",
    "NsysTrace",
    "ProfileInput",
    "SchemaInfo",
    "TimelineEvent",
]



def analyze_nsys(request: NsysAnalysisRequest) -> NsysDiag:
    """Main entry point for nsys profile analysis."""
    try:
        request = prepare_analysis_request(request)
    except ValueError as exc:
        message = str(exc)
        return _make_diag(
            status="action_required",
            profile_path=_request_input_path(request),
            sqlite_path=request.sqlite_path,
            required_action=message,
            summary=f"无法继续分析：{message}",
        )

    request_path = _request_input_path(request)
    prof_input = resolve_profile_input(request_path, sqlite_path=request.sqlite_path)
    if prof_input.action_required or prof_input.sqlite_path is None:
        return _make_diag(
            status="action_required",
            profile_path=request_path,
            sqlite_path=prof_input.sqlite_path,
            required_action=prof_input.reason or (
                f"需要导出 SQLite 文件：请运行 `nsys export -t sqlite {request_path}` 生成 .sqlite 文件后重试。"
            ),
            summary="无法继续分析：缺少 SQLite 导出文件。",
        )

    sqlite_path = prof_input.sqlite_path
    try:
        schema = inspect_schema(sqlite_path)
    except Exception as exc:
        logger.error("Schema inspection failed: %s", exc)
        return _make_diag(
            status="error",
            profile_path=request_path,
            sqlite_path=sqlite_path,
            warnings=[f"Schema 检查错误：{exc}"],
            summary=f"读取 SQLite Schema 失败：{exc}",
        )

    if "cuda_kernel" not in schema.capabilities:
        return _make_diag(
            status="error",
            profile_path=request_path,
            sqlite_path=sqlite_path,
            required_action=(
                "Profile 中未包含 GPU 内核数据。"
                "请重新采集并启用 CUDA 内核追踪："
                "nsys profile --trace=cuda,nvtx,osrt ..."
            ),
            warnings=schema.warnings,
            summary="未在 Profile 中找到 GPU 内核数据。",
            gpu_devices=schema.gpu_devices,
        )

    if request.emit_progress_info:
        _emit_status("info", f"读取 SQLite schema: {sqlite_path}")
    try:
        trace = extract_trace(sqlite_path, schema)
    except Exception as exc:
        logger.error("Trace extraction failed: %s", exc)
        return _make_diag(
            status="error",
            profile_path=request_path,
            sqlite_path=sqlite_path,
            warnings=schema.warnings + [f"Trace 提取错误：{exc}"],
            summary=f"提取 Trace 失败：{exc}",
            gpu_devices=schema.gpu_devices,
        )

    if request.emit_progress_info:
        _emit_status("info", f"提取 trace 事件 {len(trace.events)} 条，开始瓶颈分析")
    try:
        bottleneck_summary, findings = classify_bottlenecks(
            trace,
            include_deep_sql=request.include_deep_sql,
        )
    except Exception as exc:
        logger.error("Bottleneck classification failed: %s", exc)
        bottleneck_summary = None
        findings = []
        trace.warnings.append(f"分类分析错误：{exc}")

    if request.emit_progress_info:
        progress_label = "生成证据窗口与 CPU 热点" if request.include_evidence_windows else "生成 CPU 热点"
        _emit_status("info", progress_label)
    windows: list[EvidenceWindow] = []
    if request.include_evidence_windows:
        windows = extract_evidence_windows(
            trace,
            findings,
            top_n=request.top_windows_per_finding,
        )
    hotspots = _build_cpu_hotspots(trace, request.top_hotspots)
    summary = _build_summary(bottleneck_summary, findings, trace)
    warnings = _merge_warnings(schema.warnings, trace.warnings)

    profile_diag = _make_diag(
        status="ok",
        profile_path=request_path,
        sqlite_path=sqlite_path,
        bottlenecks=bottleneck_summary,
        findings=findings,
        hotspots=hotspots,
        warnings=warnings,
        summary=summary,
        gpu_devices=schema.gpu_devices,
        windows=windows,
        localization=None,
    )

    localization = None
    if request.run_localization:
        if request.emit_progress_info:
            _emit_status("info", "生成 Codex 调查提示并启动代码调查")
        localization = run_code_localization(
            request,
            summary=summary,
            findings=findings,
            windows=windows,
            sqlite_path=sqlite_path,
            bottleneck_summary=bottleneck_summary,
            hotspots=hotspots,
            profile_report_text=render_nsys_profile_report(profile_diag, verbose=True),
        )
        if localization.status == "error" and localization.error:
            warnings = _merge_warnings(warnings, [f"Codex 调查失败：{localization.error}"])

    if request.emit_progress_info and localization is not None:
        _emit_status("info", "Codex 调查结果已回填")
    return _make_diag(
        status="ok",
        profile_path=request_path,
        sqlite_path=sqlite_path,
        bottlenecks=bottleneck_summary,
        findings=findings,
        hotspots=hotspots,
        warnings=warnings,
        summary=summary,
        gpu_devices=schema.gpu_devices,
        windows=windows,
        localization=localization,
    )



def prepare_analysis_request(request: NsysAnalysisRequest) -> NsysAnalysisRequest:
    """Normalize request fields and validate simple preconditions."""
    profile_path = _normalize_optional_path(request.profile_path)
    sqlite_path = _normalize_optional_path(request.sqlite_path)
    repo_root = _normalize_optional_path(request.repo_root)
    run_localization = request.run_localization or bool(request.localization_backend)
    localization_backend = request.localization_backend or ("codex" if run_localization else None)
    include_deep_sql = request.include_deep_sql or run_localization
    include_evidence_windows = request.include_evidence_windows or run_localization

    if not profile_path and not sqlite_path:
        raise ValueError("缺少分析文件：请提供 profile_path 或 sqlite_path。")

    if run_localization:
        if not repo_root:
            raise ValueError("Codex 调查需要 repo_root 才能启动。")
        if not Path(repo_root).is_dir():
            raise ValueError(f"repo_root 不存在或不是目录：{repo_root}")
        if not localization_backend:
            raise ValueError("缺少 localization_backend。")
        if not has_cli_investigator(localization_backend):
            raise ValueError(f"未注册的 localization backend：{localization_backend}")
        if localization_backend == "codex" and shutil.which("codex") is None:
            raise ValueError("未找到 codex CLI，请先确认 `codex` 已安装并可执行。")

    return NsysAnalysisRequest(
        profile_path=profile_path,
        sqlite_path=sqlite_path,
        repo_root=repo_root,
        top_hotspots=request.top_hotspots,
        top_windows_per_finding=request.top_windows_per_finding,
        run_localization=run_localization,
        localization_backend=localization_backend,
        localization_model=request.localization_model,
        emit_progress_info=request.emit_progress_info,
        include_deep_sql=include_deep_sql,
        include_evidence_windows=include_evidence_windows,
    )



def _request_input_path(request: NsysAnalysisRequest) -> str:
    if request.profile_path:
        return str(Path(request.profile_path))
    if request.sqlite_path:
        return str(Path(request.sqlite_path))
    return ""



def _normalize_optional_path(value: str | None) -> str | None:
    if value is None or not value:
        return None
    return str(Path(value).expanduser())



def _emit_status(level: str, message: str) -> None:
    print(f"[{level}] {message}", file=sys.stderr, flush=True)



def _merge_warnings(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for warning in group:
            if warning in seen:
                continue
            seen.add(warning)
            merged.append(warning)
    return merged





def _make_diag(
    *,
    status: str,
    profile_path: str,
    sqlite_path: str | None,
    required_action: str | None = None,
    bottlenecks: BottleneckSummary | None = None,
    findings: list[NsysFinding] | None = None,
    hotspots: list[SampleHotspot] | None = None,
    warnings: list[str] | None = None,
    summary: str = "",
    gpu_devices: list | None = None,
    windows: list[EvidenceWindow] | None = None,
    localization: LocalizationResult | None = None,
) -> NsysDiag:
    return NsysDiag(
        status=status,
        profile_path=profile_path,
        sqlite_path=sqlite_path,
        required_action=required_action,
        bottlenecks=bottlenecks,
        findings=findings or [],
        hotspots=hotspots or [],
        warnings=warnings or [],
        summary=summary,
        gpu_devices=gpu_devices or [],
        windows=windows or [],
        localization=localization,
    )



def _build_cpu_hotspots(trace: NsysTrace, top_n: int = 20) -> list[SampleHotspot]:
    """Compatibility wrapper for existing tests and callers."""
    return build_cpu_hotspots(trace, top_n=top_n)



def _build_summary(
    bottlenecks: BottleneckSummary | None,
    findings: list[NsysFinding],
    trace: NsysTrace,
) -> str:
    lines: list[str] = []
    if bottlenecks:
        idle_pct = bottlenecks.gpu_idle_ns / bottlenecks.total_ns * 100 if bottlenecks.total_ns else 0
        lines.append(
            f"Trace 时长：{bottlenecks.total_ns / 1e6:.1f}ms，"
            f"GPU 活跃：{bottlenecks.gpu_active_ns / 1e6:.1f}ms（{100 - idle_pct:.1f}%），"
            f"GPU 空闲：{bottlenecks.gpu_idle_ns / 1e6:.1f}ms（{idle_pct:.1f}%）。"
        )
        if bottlenecks.labels:
            top = bottlenecks.labels[0]
            cat_zh = {
                "gpu_compute": "GPU 计算",
                "gpu_comm": "GPU 通信（NCCL）",
                "gpu_memcpy": "GPU 内存拷贝",
                "gpu_idle": "GPU 空闲",
                "sync_wait": "同步等待",
            }.get(top.category, top.category)
            lines.append(f"主要瓶颈：{cat_zh}（占 trace {top.pct_of_trace * 100:.1f}%）。")

    crit = [finding for finding in findings if finding.severity == "critical"]
    warn = [finding for finding in findings if finding.severity == "warning"]
    if crit:
        lines.append(f"严重问题 {len(crit)} 条：" + "；".join(finding.title for finding in crit[:3]))
    if warn:
        lines.append(f"警告 {len(warn)} 条：" + "；".join(finding.title for finding in warn[:3]))
    if not lines:
        lines.append(f"Trace {trace.profile_path}：已提取 {len(trace.events)} 个事件。")
    return " ".join(lines)

