"""cli — Command-line interface for sysight.analyzer.

Dispatches to repo analysis (repo.py) and nsys profile analysis (nsys/).
All rendering / serialisation helpers live here so repo.py stays pure.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from .analyzer import (
    analyze_repo, render_summary, render_trace,
    discover_repo, scan_repo, build_dag,
    search_symbols, impact_radius, trace_from,
)
from .nsys import analyze_nsys
from .nsys.models import NsysAnalysisRequest, NsysDiag
from .nsys.render import render_nsys_terminal
from .nsys.sql_cli import (
    run_sql_schema, run_sql_kernels, run_sql_gaps,
    run_sql_sync, run_sql_nvtx, run_sql_memcpy, run_sql_nccl,
    run_sql_kernel_launch, run_sql_stream_concurrency, run_sql_overlap,
)
from .scanner_cli import (
    run_search, run_lookup,
    run_callers, run_callees, run_impact, run_trace,
    run_callsites, run_callsite_context, to_json,
)


def _nsys_diag_to_dict(diag: NsysDiag) -> dict:
    """JSON-serialisable dict for NsysDiag."""

    d: dict = {
        "status": diag.status,
        "profile_path": diag.profile_path,
        "sqlite_path": diag.sqlite_path,
        "required_action": diag.required_action,
        "summary": diag.summary,
        "warnings": diag.warnings,
        "investigation": None,
    }

    if diag.bottlenecks:
        b = diag.bottlenecks
        d["bottlenecks"] = {
            "total_ns": b.total_ns,
            "gpu_active_ns": b.gpu_active_ns,
            "gpu_idle_ns": b.gpu_idle_ns,
            "labels": [
                {
                    "category": lb.category,
                    "active_ns": lb.active_ns,
                    "pct_of_trace": lb.pct_of_trace,
                    "pct_of_gpu_active": lb.pct_of_gpu_active,
                    "evidence": lb.evidence,
                }
                for lb in b.labels
            ],
            "top_events": [
                {
                    "name": ev.name,
                    "category": ev.category,
                    "count": ev.count,
                    "total_ns": ev.total_ns,
                    "max_ns": ev.max_ns,
                    "avg_ns": ev.avg_ns,
                    "inclusive_pct": ev.inclusive_pct,
                }
                for ev in b.top_events
            ],
        }
    else:
        d["bottlenecks"] = None

    d["gpu_devices"] = [
        {
            "device_id": dev.device_id,
            "name": dev.name,
            "total_memory_bytes": dev.total_memory_bytes,
            "memory_bandwidth_bytes_per_s": dev.memory_bandwidth_bytes_per_s,
            "sm_count": dev.sm_count,
            "compute_capability": dev.compute_capability,
            "bus_location": dev.bus_location,
            "chip_name": dev.chip_name,
        }
        for dev in diag.gpu_devices
    ]

    d["findings"] = [
        {
            "category": f.category,
            "severity": f.severity,
            "title": f.title,
            "description": f.description,
            "time_range_ns": list(f.time_range_ns) if f.time_range_ns else None,
            "device_id": f.device_id,
            "rank": f.rank,
            "evidence": f.evidence,
            "related_events": f.related_events,
            "related_hotspots": f.related_hotspots,
            "next_step": f.next_step,
        }
        for f in diag.findings
    ]

    d["hotspots"] = [
        {
            "symbol": h.frame.symbol,
            "summary": h.frame.raw,
            "source_file": h.frame.source_file,
            "source_line": h.frame.source_line,
            "count": h.count,
            "pct": h.pct,
            "event_window_ns": list(h.event_window_ns) if h.event_window_ns else None,
            "callstack": h.callstack,
            "coarse_location": h.coarse_location,
        }
        for h in diag.hotspots
    ]

    d["windows"] = [
        {
            "problem_id": w.problem_id,
            "category": w.category,
            "start_ns": w.start_ns,
            "end_ns": w.end_ns,
            "duration_ns": w.duration_ns,
            "device_id": w.device_id,
            "stream_id": w.stream_id,
            "correlation_id": w.correlation_id,
            "event_name": w.event_name,
            "event_category": w.event_category,
            "before_event": w.before_event,
            "after_event": w.after_event,
            "runtime_api": w.runtime_api,
            "nvtx_labels": w.nvtx_labels,
            "coarse_location": w.coarse_location,
            "callstack_summaries": w.callstack_summaries,
            "sample_callstack": w.sample_callstack,
            "first_user_python_frame": w.first_user_python_frame,
            "actionable_chain": w.actionable_chain,
            "actionable_leaf_reason": w.actionable_leaf_reason,
            "why_not_actionable": w.why_not_actionable,
            "window_rank_in_iter": w.window_rank_in_iter,
            "kernel_constraints": w.kernel_constraints,
        }
        for w in diag.windows
    ]

    if diag.investigation is not None:
        d["investigation"] = {
            "backend": diag.investigation.backend,
            "status": diag.investigation.status,
            "output": diag.investigation.output,
            "error": diag.investigation.error,
            "command": diag.investigation.command,
            "prompt": diag.investigation.prompt,
            "output_path": diag.investigation.output_path,
            "pid": diag.investigation.pid,
            "summary": diag.investigation.summary,
            "artifact_dir": diag.investigation.artifact_dir,
            "prompt_path": diag.investigation.prompt_path,
            "stdout_path": diag.investigation.stdout_path,
            "stderr_path": diag.investigation.stderr_path,
            "questions": [
                {
                    "question_id": question.question_id,
                    "problem_id": question.problem_id,
                    "category": question.category,
                    "title": question.title,
                    "file_path": question.file_path,
                    "line": question.line,
                    "function": question.function,
                    "rationale": question.rationale,
                    "suggestion": question.suggestion,
                    "status": question.status,
                    "window_ids": question.window_ids,
                }
                for question in diag.investigation.questions
            ],
            "anchors": [
                {
                    "window_id": anchor.window_id,
                    "problem_id": anchor.problem_id,
                    "category": anchor.category,
                    "event_name": anchor.event_name,
                    "file_path": anchor.file_path,
                    "line": anchor.line,
                    "function": anchor.function,
                    "rationale": anchor.rationale,
                    "suggestion": anchor.suggestion,
                    "status": anchor.status,
                }
                for anchor in diag.investigation.anchors
            ],
        }

    return d


def _sanitize_nsys_artifact_stem(path: str | None) -> str:
    raw = Path(path or "nsys-profile").stem or "nsys-profile"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return cleaned or "nsys-profile"


def _default_nsys_json_path(diag: NsysDiag) -> Path:
    artifact_dir = Path.cwd() / ".sysight" / "nsys"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_nsys_artifact_stem(diag.sqlite_path or diag.profile_path)
    return artifact_dir / f"{stem}.json"


def _write_nsys_json_artifact(diag: NsysDiag) -> Path:
    output_path = _default_nsys_json_path(diag)
    output_path.write_text(
        json.dumps(_nsys_diag_to_dict(diag), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _infer_repo_root(profile_path: str) -> str | None:
    """推断 repo_root：优先找 profile 文件旁的 workspace/ 目录，
    否则找 cwd 下的 workspace/ 目录。"""
    profile_p = Path(profile_path).resolve()
    # Check sibling workspace/ relative to profile file
    sibling = profile_p.parent.parent / "workspace"
    if sibling.is_dir():
        return str(sibling)
    # Check cwd/workspace
    cwd_workspace = Path.cwd() / "workspace"
    if cwd_workspace.is_dir():
        return str(cwd_workspace)
    return None


def _infer_sqlite_path(profile: str) -> str | None:
    """给定路径（可以是 .nsys-rep 或目录），尝试找到可用的 .sqlite 文件。
    优先找 profile 同名 .sqlite；如果 profile 本身就是 .sqlite，返回 None（保持原逻辑）。"""
    p = Path(profile)
    # Already a sqlite file – let resolve_profile_input handle it
    if p.suffix.lower() == ".sqlite":
        return None
    # .nsys-rep → look for sibling .sqlite
    if p.suffix.lower() == ".nsys-rep":
        sibling = p.with_suffix(".sqlite")
        if sibling.is_file() and sibling.stat().st_size > 0:
            return str(sibling)
        return None  # will trigger export hint via resolve_profile_input
    # If it's a directory (e.g. profile/ passed as a hint), return None
    return None


def _add_nsys_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("profile", help=".nsys-rep 或 .sqlite 文件路径")
    parser.add_argument(
        "--sqlite", metavar="FILE",
        help="显式指定 .sqlite 路径（覆盖自动检测）",
    )
    parser.add_argument(
        "--repo-root", metavar="DIR",
        help="Stage 6 调查使用的 repo 根目录；未指定时自动推断 workspace/ 目录",
    )
    parser.add_argument(
        "--no-codex", action="store_true",
        help="快速模式：不运行 Codex，仅输出统计分析结果",
    )
    parser.add_argument(
        "--codex-model", metavar="MODEL",
        help="可选：指定 codex model",
    )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _emit_nsys(args: argparse.Namespace, default_repo_root: str | None = None) -> None:
    # --no-codex = fast mode (no Codex); otherwise Codex is on by default
    no_codex = getattr(args, "no_codex", False)
    # repo_root: explicit > default_repo_root > auto-infer (only when Codex wanted)
    explicit_repo_root = getattr(args, "repo_root", None) or default_repo_root
    repo_root = explicit_repo_root or (
        _infer_repo_root(args.profile) if not no_codex else None
    )

    # sqlite_path: explicit --sqlite > auto-infer sibling .sqlite for .nsys-rep
    explicit_sqlite = getattr(args, "sqlite", None)
    if explicit_sqlite is None:
        explicit_sqlite = _infer_sqlite_path(args.profile)

    use_codex = bool(repo_root) and not no_codex
    include_evidence_windows = use_codex
    req = NsysAnalysisRequest(
        profile_path=args.profile,
        sqlite_path=explicit_sqlite,
        repo_root=repo_root,
        top_hotspots=getattr(args, "top_hotspots", 20),
        top_windows_per_finding=getattr(args, "top_windows_per_finding", 3),
        run_investigation=use_codex,
        investigation_backend="codex" if use_codex else None,
        investigation_model=getattr(args, "codex_model", None),
        emit_stage_info=True,
        include_deep_sql=True,
        include_evidence_windows=include_evidence_windows,
    )
    diag = analyze_nsys(req)
    if args.json:
        print(json.dumps(_nsys_diag_to_dict(diag), indent=2, ensure_ascii=False))
    else:
        report = render_nsys_terminal(diag, verbose=use_codex)
        print(report)
        if use_codex:
            # Only write JSON artifact when full Codex run is complete
            json_path = _write_nsys_json_artifact(diag)
            print()
            print(f"结构化 JSON: {json_path}")


# ── nsys-sql CLI helpers ────────────────────────────────────────────────────────

def _emit_nsys_sql_schema(args: argparse.Namespace) -> None:
    """Emit nsys-sql schema result as JSON."""
    from dataclasses import asdict
    result = run_sql_schema(args.sqlite_path)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_kernels(args: argparse.Namespace) -> None:
    """Emit nsys-sql kernels result as JSON."""
    from dataclasses import asdict
    result = run_sql_kernels(args.sqlite_path, limit=args.limit)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_gaps(args: argparse.Namespace) -> None:
    """Emit nsys-sql gaps result as JSON."""
    from dataclasses import asdict
    result = run_sql_gaps(args.sqlite_path, min_gap_ns=args.min_gap_ns, limit=args.limit)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_sync(args: argparse.Namespace) -> None:
    """Emit nsys-sql sync result as JSON."""
    from dataclasses import asdict
    result = run_sql_sync(args.sqlite_path)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_nvtx(args: argparse.Namespace) -> None:
    """Emit nsys-sql nvtx result as JSON."""
    from dataclasses import asdict
    result = run_sql_nvtx(args.sqlite_path, limit=args.limit)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_memcpy(args: argparse.Namespace) -> None:
    """Emit nsys-sql memcpy result as JSON."""
    from dataclasses import asdict
    result = run_sql_memcpy(args.sqlite_path)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_nccl(args: argparse.Namespace) -> None:
    """Emit nsys-sql nccl result as JSON."""
    from dataclasses import asdict
    result = run_sql_nccl(args.sqlite_path, limit=args.limit)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_kernel_launch(args: argparse.Namespace) -> None:
    """Emit nsys-sql kernel-launch result as JSON."""
    from dataclasses import asdict
    result = run_sql_kernel_launch(args.sqlite_path, limit=args.limit)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_stream_concurrency(args: argparse.Namespace) -> None:
    """Emit nsys-sql stream-concurrency result as JSON."""
    from dataclasses import asdict
    result = run_sql_stream_concurrency(args.sqlite_path, limit=args.limit)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _emit_nsys_sql_overlap(args: argparse.Namespace) -> None:
    """Emit nsys-sql overlap result as JSON."""
    from dataclasses import asdict
    result = run_sql_overlap(args.sqlite_path)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


def _add_nsys_sql_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add nsys-sql subcommands to the given subparsers."""
    sql_parser = subparsers.add_parser(
        "nsys-sql",
        help="直接查询 nsys SQLite 数据库（供 agent 使用）",
        description=(
            "直接查询 nsys SQLite 数据库，输出结构化 JSON。\n\n"
            "每个子命令都是只读的，输出有界的 JSON，适合 agent/Codex 直接调用。\n\n"
            "  sysight nsys-sql schema <db>    — 列出表、列和能力\n"
            "  sysight nsys-sql kernels <db>   — Top-N GPU 内核（按总时间）\n"
            "  sysight nsys-sql gaps <db>      — GPU 空闲气泡分析\n"
            "  sysight nsys-sql sync <db>      — CUDA 同步事件代价\n"
            "  sysight nsys-sql nvtx <db>      — NVTX range 分解\n"
            "  sysight nsys-sql memcpy <db>    — 内存带宽分析\n"
            "  sysight nsys-sql nccl <db>      — NCCL 通信分解\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sql_sub = sql_parser.add_subparsers(dest="sql_cmd")

    # schema
    sc = sql_sub.add_parser("schema", help="查询 SQLite schema 和能力")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # kernels
    sc = sql_sub.add_parser("kernels", help="查询 Top-N GPU 内核")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回内核数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # gaps
    sc = sql_sub.add_parser("gaps", help="查询 GPU 空闲气泡")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--min-gap-ns", type=int, default=1000000, help="最小间隙（ns）")
    sc.add_argument("--limit", type=int, default=20, help="返回间隙数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # sync
    sc = sql_sub.add_parser("sync", help="查询 CUDA 同步事件代价")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # nvtx
    sc = sql_sub.add_parser("nvtx", help="查询 NVTX range 分解")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 range 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # memcpy
    sc = sql_sub.add_parser("memcpy", help="查询内存带宽分析")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # nccl
    sc = sql_sub.add_parser("nccl", help="查询 NCCL 通信分解")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 stream 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # kernel-launch
    sc = sql_sub.add_parser("kernel-launch", help="内核启动开销分析（API→GPU 延迟）")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回条目数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # stream-concurrency
    sc = sql_sub.add_parser("stream-concurrency", help="Stream 并发率分析")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 stream 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # overlap
    sc = sql_sub.add_parser("overlap", help="Compute/NCCL 重叠估算")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")


def _dispatch_nsys_sql(args: argparse.Namespace) -> bool:
    """Dispatch nsys-sql subcommand. Returns True if handled."""
    if getattr(args, "sql_cmd", None) is None:
        return False

    sql_cmd = args.sql_cmd
    if sql_cmd == "schema":
        _emit_nsys_sql_schema(args)
    elif sql_cmd == "kernels":
        _emit_nsys_sql_kernels(args)
    elif sql_cmd == "gaps":
        _emit_nsys_sql_gaps(args)
    elif sql_cmd == "sync":
        _emit_nsys_sql_sync(args)
    elif sql_cmd == "nvtx":
        _emit_nsys_sql_nvtx(args)
    elif sql_cmd == "memcpy":
        _emit_nsys_sql_memcpy(args)
    elif sql_cmd == "nccl":
        _emit_nsys_sql_nccl(args)
    elif sql_cmd == "kernel-launch":
        _emit_nsys_sql_kernel_launch(args)
    elif sql_cmd == "stream-concurrency":
        _emit_nsys_sql_stream_concurrency(args)
    elif sql_cmd == "overlap":
        _emit_nsys_sql_overlap(args)
    else:
        return False
    return True


def _standalone_nsys_argv(argv: list[str]) -> list[str] | None:
    """Return args for top-level `sysight nsys ...`, preserving simple globals."""
    prefix: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "nsys":
            return prefix + argv[idx + 1:]
        if token == "nsys-sql":
            # nsys-sql is handled separately as a top-level command
            return None
        if token in ("--json", "--verbose", "-v"):
            prefix.append(token)
            idx += 1
            continue
        return None
    return None


def _main_standalone_nsys(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="sysight nsys",
        description="分析 Nsight Systems profile",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="开启调试日志")
    p.add_argument("--json", action="store_true", help="输出 JSON 格式")
    _add_nsys_args(p)

    args = p.parse_args(argv)
    _configure_logging(args.verbose)
    _emit_nsys(args)


def _main_standalone_nsys_sql(argv: list[str]) -> None:
    """Handle standalone `sysight nsys-sql ...` command."""
    p = argparse.ArgumentParser(
        prog="sysight nsys-sql",
        description=(
            "直接查询 nsys SQLite 数据库，输出结构化 JSON。\n\n"
            "每个子命令都是只读的，输出有界的 JSON，适合 agent/Codex 直接调用。\n\n"
            "  sysight nsys-sql schema <db>    — 列出表、列和能力\n"
            "  sysight nsys-sql kernels <db>   — Top-N GPU 内核（按总时间）\n"
            "  sysight nsys-sql gaps <db>      — GPU 空闲气泡分析\n"
            "  sysight nsys-sql sync <db>      — CUDA 同步事件代价\n"
            "  sysight nsys-sql nvtx <db>      — NVTX range 分解\n"
            "  sysight nsys-sql memcpy <db>    — 内存带宽分析\n"
            "  sysight nsys-sql nccl <db>      — NCCL 通信分解\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="sql_cmd")

    # schema
    sc = sub.add_parser("schema", help="查询 SQLite schema 和能力")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # kernels
    sc = sub.add_parser("kernels", help="查询 Top-N GPU 内核")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回内核数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # gaps
    sc = sub.add_parser("gaps", help="查询 GPU 空闲气泡")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--min-gap-ns", type=int, default=1000000, help="最小间隙（ns）")
    sc.add_argument("--limit", type=int, default=20, help="返回间隙数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # sync
    sc = sub.add_parser("sync", help="查询 CUDA 同步事件代价")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # nvtx
    sc = sub.add_parser("nvtx", help="查询 NVTX range 分解")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 range 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # memcpy
    sc = sub.add_parser("memcpy", help="查询内存带宽分析")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # nccl
    sc = sub.add_parser("nccl", help="查询 NCCL 通信分解")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 stream 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # kernel-launch
    sc = sub.add_parser("kernel-launch", help="内核启动开销分析（API→GPU 延迟）")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回条目数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # stream-concurrency
    sc = sub.add_parser("stream-concurrency", help="Stream 并发率分析")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--limit", type=int, default=20, help="返回 stream 数量上限")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    # overlap
    sc = sub.add_parser("overlap", help="Compute/NCCL 重叠估算")
    sc.add_argument("sqlite_path", help=".sqlite 文件路径")
    sc.add_argument("--json", action="store_true", help="输出 JSON 格式（默认）")

    args = p.parse_args(argv)
    if not getattr(args, "sql_cmd", None):
        p.print_help()
        return

    if not _dispatch_nsys_sql(args):
        print("错误：未知的 nsys-sql 子命令", file=sys.stderr)
        sys.exit(1)


# ── nsys windows CLI ──────────────────────────────────────────────────────────

def _main_standalone_nsys_windows(argv: list[str]) -> None:
    """Handle standalone `sysight nsys windows <profile>` command.

    Extracts and renders evidence windows as plain terminal text.
    Designed for ad-hoc inspection; no JSON output.
    """
    p = argparse.ArgumentParser(
        prog="sysight nsys windows",
        description=(
            "从 nsys profile 提取证据窗口并以终端格式输出。\n"
            "证据窗口是耗时最长的感兴趣事件（GPU 内核、同步、memcpy 等），\n"
            "带有 NVTX 上下文、粗定位和调用栈摘要。"
        ),
    )
    p.add_argument("profile_path", help=".nsys-rep 或 .sqlite 文件路径")
    p.add_argument("--top", type=int, default=3, help="每个 finding 最多提取几个窗口（默认 3）")
    p.add_argument("--verbose", "-v", action="store_true", help="显示完整调用栈")

    args = p.parse_args(argv)
    _configure_logging(args.verbose)

    from .nsys import analyze_nsys
    from .nsys.models import NsysAnalysisRequest
    from .nsys.windows import extract_evidence_windows

    req = NsysAnalysisRequest(
        profile_path=args.profile_path if not args.profile_path.endswith(".sqlite") else None,
        sqlite_path=args.profile_path if args.profile_path.endswith(".sqlite") else None,
        include_deep_sql=False,
        include_evidence_windows=False,  # we do it manually below
        emit_stage_info=True,
        top_windows_per_finding=args.top,
    )
    diag = analyze_nsys(req)

    if diag.status not in ("ok",):
        print(f"分析失败：{diag.status} — {diag.summary}", file=sys.stderr)
        sys.exit(1)

    # Extract windows manually using the findings from the diag
    from .nsys.extract import extract_trace, inspect_schema

    # Re-use sqlite path resolved by analyze_nsys
    sqlite_path = diag.sqlite_path
    if not sqlite_path:
        print("无法确定 SQLite 路径", file=sys.stderr)
        sys.exit(1)

    schema = inspect_schema(sqlite_path)
    trace = extract_trace(sqlite_path, schema)
    windows = extract_evidence_windows(trace, diag.findings, top_n=args.top)

    if not windows:
        print("未提取到证据窗口（无有效 finding 或 trace 数据）")
        return

    print(f"\n证据窗口  共 {len(windows)} 个\n{'─' * 80}")
    for idx, w in enumerate(windows, start=1):
        dur_ms = w.duration_ns / 1e6
        print(f"\nW{idx}  [{w.category}] {w.event_name}")
        print(f"    时间：{w.start_ns/1e6:.3f} – {w.end_ns/1e6:.3f} ms  耗时：{dur_ms:.3f} ms")
        if w.device_id is not None:
            print(f"    设备：GPU {w.device_id}  Stream {w.stream_id}")
        if w.nvtx_labels:
            print(f"    NVTX：{' > '.join(w.nvtx_labels)}")
        if w.coarse_location:
            print(f"    粗定位：{w.coarse_location}")
        if w.runtime_api:
            print(f"    Runtime API：{w.runtime_api}")
        if args.verbose and w.callstack_summaries:
            print("    调用栈：")
            for frame in w.callstack_summaries[:8]:
                print(f"      {frame}")
        if w.first_user_python_frame:
            print(f"    首个用户 Python 帧：{w.first_user_python_frame}")
    print(f"\n{'─' * 80}")


# ── scanner CLI helpers ────────────────────────────────────────────────────────

def _emit_scanner_search(args: argparse.Namespace) -> None:
    """Emit scanner search result as JSON."""
    result = run_search(args.repo_root, args.query, limit=args.limit)
    print(to_json(result))


def _emit_scanner_lookup(args: argparse.Namespace) -> None:
    """Emit scanner lookup result as JSON."""
    result = run_lookup(
        args.repo_root,
        file=getattr(args, "file", None),
        symbol=getattr(args, "symbol", None),
        line=getattr(args, "line", None),
        include_context=not getattr(args, "no_context", False),
    )
    print(to_json(result))


def _emit_scanner_callers(args: argparse.Namespace) -> None:
    """Emit scanner callers result as JSON."""
    result = run_callers(args.repo_root, args.symbol, limit=args.limit)
    print(to_json(result))


def _emit_scanner_callees(args: argparse.Namespace) -> None:
    """Emit scanner callees result as JSON."""
    result = run_callees(args.repo_root, args.symbol, limit=args.limit)
    print(to_json(result))


def _emit_scanner_impact(args: argparse.Namespace) -> None:
    """Emit scanner impact result as JSON."""
    result = run_impact(
        args.repo_root,
        args.files,
        max_depth=getattr(args, "depth", 5),
    )
    print(to_json(result))


def _emit_scanner_trace(args: argparse.Namespace) -> None:
    """Emit scanner trace result as JSON."""
    result = run_trace(
        args.repo_root,
        args.target,
        symbol=getattr(args, "symbol", None),
        max_depth=getattr(args, "depth", 8),
    )
    print(to_json(result))


def _emit_scanner_callsites(args: argparse.Namespace) -> None:
    """Emit scanner callsites result as JSON."""
    result = run_callsites(
        args.repo_root,
        call_name=getattr(args, "call", None),
        file_filter=getattr(args, "file", None),
        finding_type=getattr(args, "finding_type", None),
        limit=getattr(args, "limit", 50),
    )
    print(to_json(result))


def _emit_scanner_callsite_context(args: argparse.Namespace) -> None:
    """Emit scanner callsite context result as JSON."""
    result = run_callsite_context(args.repo_root, args.callsite_id)
    if result is None:
        print(json.dumps({"error": f"Callsite not found: {args.callsite_id}"}))
    else:
        print(to_json(result))


def _add_scanner_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add scanner subcommands to the given subparsers."""
    scanner_parser = subparsers.add_parser(
        "scanner",
        help="静态代码分析工具（供 agent 使用）",
        description=(
            "静态代码分析 CLI，输出结构化 JSON，适合 agent/Codex 直接调用。\n\n"
            "每个子命令都是只读的，输出有界的 JSON。\n\n"
            "  sysight scanner search <repo> <query>        — 符号/文件搜索\n"
            "  sysight scanner lookup <repo> [opts]         — 精确定位\n"
            "  sysight scanner callers <repo> <symbol>      — 查找调用者\n"
            "  sysight scanner callees <repo> <symbol>      — 查找被调用者\n"
            "  sysight scanner impact <repo> <files...>     — 影响范围分析\n"
            "  sysight scanner trace <repo> <target>        — 调用链追踪\n"
            "  sysight scanner callsites <repo> [opts]      — 调用点搜索\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scanner_sub = scanner_parser.add_subparsers(dest="scanner_cmd")

    # search
    sc = scanner_sub.add_parser("search", help="符号/文件搜索")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("query", help="搜索关键词")
    sc.add_argument("--limit", type=int, default=20, help="返回结果上限")

    # lookup
    sc = scanner_sub.add_parser("lookup", help="精确定位（文件+行号 或 符号）")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("--file", "-f", help="文件路径")
    sc.add_argument("--symbol", "-s", help="符号名")
    sc.add_argument("--line", "-l", type=int, help="行号")
    sc.add_argument("--no-context", action="store_true", help="不返回源码上下文")

    # callers
    sc = scanner_sub.add_parser("callers", help="查找符号的所有调用者")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("symbol", help="符号名（qualified_name）")
    sc.add_argument("--limit", type=int, default=20, help="返回结果上限")

    # callees
    sc = scanner_sub.add_parser("callees", help="查找符号调用的所有函数")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("symbol", help="符号名（qualified_name）")
    sc.add_argument("--limit", type=int, default=20, help="返回结果上限")

    # impact
    sc = scanner_sub.add_parser("impact", help="影响范围分析")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("files", nargs="+", help="变更的文件")
    sc.add_argument("--depth", type=int, default=5, help="最大追踪深度")

    # trace
    sc = scanner_sub.add_parser("trace", help="调用链追踪")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("target", help="目标文件或符号")
    sc.add_argument("--symbol", "-s", help="指定目标文件中的函数名")
    sc.add_argument("--depth", type=int, default=8, help="最大追踪深度")

    # callsites
    sc = scanner_sub.add_parser("callsites", help="调用点搜索")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("--call", "-c", help="调用函数名（如 to, cuda, synchronize）")
    sc.add_argument("--file", "-f", help="限定文件")
    sc.add_argument("--finding-type", help="发现类型（自动填充搜索参数）")
    sc.add_argument("--limit", type=int, default=50, help="返回结果上限")

    # callsite-context
    sc = scanner_sub.add_parser("callsite-context", help="获取调用点上下文")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("callsite_id", help="调用点 ID")


def _dispatch_scanner(args: argparse.Namespace) -> bool:
    """Dispatch scanner subcommand. Returns True if handled."""
    if getattr(args, "scanner_cmd", None) is None:
        return False

    scanner_cmd = args.scanner_cmd
    if scanner_cmd == "search":
        _emit_scanner_search(args)
    elif scanner_cmd == "lookup":
        _emit_scanner_lookup(args)
    elif scanner_cmd == "callers":
        _emit_scanner_callers(args)
    elif scanner_cmd == "callees":
        _emit_scanner_callees(args)
    elif scanner_cmd == "impact":
        _emit_scanner_impact(args)
    elif scanner_cmd == "trace":
        _emit_scanner_trace(args)
    elif scanner_cmd == "callsites":
        _emit_scanner_callsites(args)
    elif scanner_cmd == "callsite-context":
        _emit_scanner_callsite_context(args)
    else:
        return False
    return True


def _main_standalone_scanner(argv: list[str]) -> None:
    """Handle standalone `sysight scanner ...` command."""
    p = argparse.ArgumentParser(
        prog="sysight scanner",
        description=(
            "静态代码分析 CLI，输出结构化 JSON，适合 agent/Codex 直接调用。\n\n"
            "每个子命令都是只读的，输出有界的 JSON。\n\n"
            "  sysight scanner search <repo> <query>        — 符号/文件搜索\n"
            "  sysight scanner lookup <repo> [opts]         — 精确定位\n"
            "  sysight scanner callers <repo> <symbol>      — 查找调用者\n"
            "  sysight scanner callees <repo> <symbol>      — 查找被调用者\n"
            "  sysight scanner impact <repo> <files...>     — 影响范围分析\n"
            "  sysight scanner trace <repo> <target>        — 调用链追踪\n"
            "  sysight scanner callsites <repo> [opts]      — 调用点搜索\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="scanner_cmd")

    # search
    sc = sub.add_parser("search", help="符号/文件搜索")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("query", help="搜索关键词")
    sc.add_argument("--limit", type=int, default=20, help="返回结果上限")

    # lookup
    sc = sub.add_parser("lookup", help="精确定位（文件+行号 或 符号）")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("--file", "-f", help="文件路径")
    sc.add_argument("--symbol", "-s", help="符号名")
    sc.add_argument("--line", "-l", type=int, help="行号")
    sc.add_argument("--no-context", action="store_true", help="不返回源码上下文")

    # callers
    sc = sub.add_parser("callers", help="查找符号的所有调用者")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("symbol", help="符号名（qualified_name）")
    sc.add_argument("--limit", type=int, default=20, help="返回结果上限")

    # callees
    sc = sub.add_parser("callees", help="查找符号调用的所有函数")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("symbol", help="符号名（qualified_name）")
    sc.add_argument("--limit", type=int, default=20, help="返回结果上限")

    # impact
    sc = sub.add_parser("impact", help="影响范围分析")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("files", nargs="+", help="变更的文件")
    sc.add_argument("--depth", type=int, default=5, help="最大追踪深度")

    # trace
    sc = sub.add_parser("trace", help="调用链追踪")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("target", help="目标文件或符号")
    sc.add_argument("--symbol", "-s", help="指定目标文件中的函数名")
    sc.add_argument("--depth", type=int, default=8, help="最大追踪深度")

    # callsites
    sc = sub.add_parser("callsites", help="调用点搜索")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("--call", "-c", help="调用函数名（如 to, cuda, synchronize）")
    sc.add_argument("--file", "-f", help="限定文件")
    sc.add_argument("--finding-type", help="发现类型（自动填充搜索参数）")
    sc.add_argument("--limit", type=int, default=50, help="返回结果上限")

    # callsite-context
    sc = sub.add_parser("callsite-context", help="获取调用点上下文")
    sc.add_argument("repo_root", help="仓库根目录")
    sc.add_argument("callsite_id", help="调用点 ID")

    args = p.parse_args(argv)
    if not getattr(args, "scanner_cmd", None):
        p.print_help()
        return

    if not _dispatch_scanner(args):
        print("错误：未知的 scanner 子命令", file=sys.stderr)
        sys.exit(1)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    # Check for scanner as top-level command first
    if argv_list and argv_list[0] == "scanner":
        _main_standalone_scanner(argv_list[1:])
        return

    # Check for nsys-sql as top-level command
    if argv_list and argv_list[0] == "nsys-sql":
        _main_standalone_nsys_sql(argv_list[1:])
        return

    # Check for nsys windows as top-level command: `sysight nsys windows <profile>`
    if len(argv_list) >= 2 and argv_list[0] == "nsys" and argv_list[1] == "windows":
        _main_standalone_nsys_windows(argv_list[2:])
        return

    standalone_nsys = _standalone_nsys_argv(argv_list)
    if standalone_nsys is not None:
        _main_standalone_nsys(standalone_nsys)
        return

    p = argparse.ArgumentParser(
        prog="sysight",
        description=(
            "仓库静态分析 + nsys 性能分析工具\n\n"
            "  sysight scanner <cmd> <repo> [opts]       静态代码分析（供 agent 使用）\n"
            "  sysight nsys <profile>                    独立 nsys profile 分析\n"
            "  sysight nsys-sql <cmd> <db>               直接查询 nsys SQLite（供 agent 使用）\n"
            "  sysight <repo>                             全量扫描分析\n"
            "  sysight <repo> search <query>              搜索符号/文件\n"
            "  sysight <repo> impact <files...>           影响范围分析\n"
            "  sysight <repo> trace <file-or-sym>         调用链追踪\n"
            "  sysight <repo> manifest                    路径清单（仅 Stage 1）\n"
            "  sysight <repo> nsys <profile>              兼容旧用法"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("repo_path", help="仓库路径（当前目录用 '.'）")
    p.add_argument("--verbose", "-v", action="store_true", help="开启调试日志")
    p.add_argument("--json", action="store_true", help="输出 JSON 格式")
    p.add_argument("--top", type=int, default=10, help="显示 Top N 入口点")
    p.add_argument("--depth", type=int, default=8, help="最大调用链深度")

    sub = p.add_subparsers(dest="subcmd")

    sc = sub.add_parser("search", help="按关键词搜索符号/文件")
    sc.add_argument("query")
    sc.add_argument("--limit", type=int, default=20)

    sc = sub.add_parser("impact", help="计算文件变更的影响范围")
    sc.add_argument("files", nargs="+", help="变更的文件路径（相对仓库根）")

    sc = sub.add_parser("trace", help="从文件或符号追踪调用链")
    sc.add_argument("target", help="文件路径（支持模糊匹配）或符号名")
    sc.add_argument("--symbol", "-s", help="指定目标文件中的函数名")

    sub.add_parser("manifest", help="Stage-1 路径清单，不读文件内容")

    sc = sub.add_parser(
        "nsys",
        help="分析 Nsight Systems 性能文件",
    )
    _add_nsys_args(sc)

    # Add nsys-sql as a subcommand under sysight
    _add_nsys_sql_subparser(sub)

    # Add scanner as a subcommand under sysight
    _add_scanner_subparser(sub)

    args = p.parse_args(argv_list)

    _configure_logging(args.verbose)

    # Handle nsys-sql subcommand
    if args.subcmd == "nsys-sql":
        if _dispatch_nsys_sql(args):
            return
        else:
            print("错误：未指定 nsys-sql 子命令。使用 --help 查看帮助。", file=sys.stderr)
            sys.exit(1)

    # Handle scanner subcommand
    if args.subcmd == "scanner":
        if _dispatch_scanner(args):
            return
        else:
            print("错误：未指定 scanner 子命令。使用 --help 查看帮助。", file=sys.stderr)
            sys.exit(1)

    if args.subcmd == "manifest":
        m = discover_repo(Path(args.repo_path).resolve())
        if args.json:
            print(json.dumps(asdict(m), indent=2))
        else:
            print(f"仓库：{m.repo_root}")
            print(f"文件数：{len(m.files)}  跳过目录：{m.ignored_dir_count}")
            print(f"语言：{m.languages}")
            print(f"GPU 候选文件：{len(m.candidate_gpu_files)}")
            print(f"入口候选文件：{len(m.candidate_entry_files)}")
            for w in m.warnings:
                print(f"  注意：{w}")

    elif args.subcmd == "search":
        files, _ = scan_repo(Path(args.repo_path).resolve())
        results = search_symbols(files, args.query, limit=args.limit)
        if args.json:
            print(json.dumps([asdict(r) for r in results], indent=2))
        else:
            for r in results:
                print(f"  [{r.kind}] {r.symbol}  行={r.line}  得分={r.score:.1f}")

    elif args.subcmd == "impact":
        files, _ = scan_repo(Path(args.repo_path).resolve())
        dag = build_dag(files)
        ir = impact_radius(files, dag, args.files, max_depth=args.depth)
        if args.json:
            print(json.dumps(asdict(ir), indent=2))
        else:
            print(f"起始文件：{ir.seed_files}")
            print(f"影响范围（{len(ir.impacted_files)} 个文件）：")
            for f in ir.impacted_files:
                print(f"  深度={ir.depth_map.get(f, '?')}  {f}")

    elif args.subcmd == "trace":
        files, _ = scan_repo(Path(args.repo_path).resolve())
        dag = build_dag(files)
        chains = trace_from(files, dag, args.target,
                            symbol=getattr(args, "symbol", None),
                            max_depth=args.depth)
        if not chains:
            print(f"未找到：{args.target!r}")
            print("建议：先用 'search' 子命令确认文件/符号名称")
        elif args.json:
            print(json.dumps([asdict(c) for c in chains], indent=2))
        else:
            print(render_trace(chains, args.target))

    elif args.subcmd == "nsys":
        _emit_nsys(args, default_repo_root=str(Path(args.repo_path).resolve()))

    else:
        result = analyze_repo(args.repo_path, top_n=args.top, max_chain_depth=args.depth)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(render_summary(result))


if __name__ == "__main__":
    main()
