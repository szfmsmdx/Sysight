"""cli — Command-line interface for sysight.analyzer.

Dispatches to nsys profile analysis (nsys/).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from .nsys import analyze_nsys
from .nsys.models import NsysAnalysisRequest, NsysDiag
from .nsys.render import render_nsys_terminal
from .nsys.sql_cli import (
    run_sql_schema, run_sql_kernels, run_sql_gaps,
    run_sql_sync, run_sql_nvtx, run_sql_memcpy, run_sql_nccl,
    run_sql_kernel_launch, run_sql_stream_concurrency, run_sql_overlap,
)
from .scanner.fs import list_files
from .scanner.search import search as scanner_search
from .scanner.reader import read_file as scanner_read_file
from .scanner.callsites import find_callsites
from .scanner.symbols import list_symbols, find_callers, find_callees, trace_symbol
from .scanner.variants import find_variants


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

def _emit_sql_result(result: object) -> None:
    from dataclasses import asdict
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))  # type: ignore[arg-type]


# Maps sql_cmd → (runner, extra_kwargs_from_args)
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


def _register_nsys_sql_subparsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register all nsys-sql sub-sub-commands onto *sub* (a subparsers object)."""
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


def _add_nsys_sql_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add nsys-sql subcommands to the given subparsers."""
    sql_parser = subparsers.add_parser(
        "nsys-sql",
        help="直接查询 nsys SQLite 数据库（供 agent 使用）",
        description=_NSYS_SQL_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sql_sub = sql_parser.add_subparsers(dest="sql_cmd")
    _register_nsys_sql_subparsers(sql_sub)


def _dispatch_nsys_sql(args: argparse.Namespace) -> bool:
    """Dispatch nsys-sql subcommand. Returns True if handled."""
    if getattr(args, "sql_cmd", None) is None:
        return False

    sql_cmd = args.sql_cmd
    entry = _SQL_RUNNERS.get(sql_cmd)
    if entry is not None:
        runner, build_kwargs = entry
        _emit_sql_result(runner(args.sqlite_path, **build_kwargs(args)))
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
        description=_NSYS_SQL_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="sql_cmd")
    _register_nsys_sql_subparsers(sub)

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


# ── Scanner CLI ───────────────────────────────────────────────────────────────

def _add_scanner_subparser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add `sysight scanner` subcommand with all sub-sub-commands."""
    sc = sub.add_parser(
        "scanner",
        help="静态 repo 代码分析工具集",
        description=(
            "静态 repo 代码分析工具集（只读，不执行目标代码）\n\n"
            "  sysight scanner files   <repo>                文件列举\n"
            "  sysight scanner search  <repo> <query>        全文搜索\n"
            "  sysight scanner read    <repo> <file>         读取文件\n"
            "  sysight scanner callsites <repo> --call <sym> 调用点定位\n"
            "  sysight scanner symbols <repo> --file <f>     符号列表\n"
            "  sysight scanner callers <repo> <sym>          查调用者\n"
            "  sysight scanner callees <repo> --file <f> --symbol <s>  查被调用\n"
            "  sysight scanner trace   <repo> <sym>          调用链追踪\n"
            "  sysight scanner variants <repo>               variant 映射"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ssub = sc.add_subparsers(dest="scanner_cmd")

    # files
    p_files = ssub.add_parser("files", help="列出 repo 中的文件")
    p_files.add_argument("repo", help="repo 根目录")
    p_files.add_argument("--ext", help="只列此扩展名（如 py）")
    p_files.add_argument("--pattern", help="glob 过滤（如 '*/data/*'）")
    p_files.add_argument("--max", type=int, default=2000, dest="max_results")
    p_files.add_argument("--json", action="store_true", help="JSON 输出")

    # search
    p_search = ssub.add_parser("search", help="全文搜索")
    p_search.add_argument("repo", help="repo 根目录")
    p_search.add_argument("query", help="搜索关键字或正则")
    p_search.add_argument("--ext", help="只搜此扩展名")
    p_search.add_argument("--fixed", action="store_true", help="字面字符串（非正则）")
    p_search.add_argument("--ignore-case", "-i", action="store_true")
    p_search.add_argument("--max", type=int, default=500, dest="max_results")
    p_search.add_argument("--json", action="store_true")

    # read
    p_read = ssub.add_parser("read", help="读取文件（带行号）")
    p_read.add_argument("repo", help="repo 根目录")
    p_read.add_argument("file", help="相对于 repo 的文件路径")
    p_read.add_argument("--start", type=int, help="起始行（含）")
    p_read.add_argument("--end", type=int, help="结束行（含）")
    p_read.add_argument("--around", type=int, help="中心行，配合 --context 使用")
    p_read.add_argument("--context", type=int, default=10, help="around 上下文行数（默认 10）")
    p_read.add_argument("--json", action="store_true")

    # callsites
    p_cs = ssub.add_parser("callsites", help="查找符号的所有调用点")
    p_cs.add_argument("repo", help="repo 根目录")
    p_cs.add_argument("--call", required=True, dest="symbol", metavar="SYMBOL")
    p_cs.add_argument("--file", dest="file_filter", help="只在此文件中搜索")
    p_cs.add_argument("--ext", help="只搜此扩展名")
    p_cs.add_argument("--max", type=int, default=300, dest="max_results")
    p_cs.add_argument("--json", action="store_true")

    # symbols
    p_sym = ssub.add_parser("symbols", help="列出文件中的所有符号定义")
    p_sym.add_argument("repo", help="repo 根目录")
    p_sym.add_argument("--file", required=True, dest="file", metavar="FILE")
    p_sym.add_argument("--symbol", dest="symbol", help="只显示此符号的详情")
    p_sym.add_argument("--json", action="store_true")

    # callers
    p_callers = ssub.add_parser("callers", help="查找调用某符号的所有位置")
    p_callers.add_argument("repo", help="repo 根目录")
    p_callers.add_argument("symbol", help="目标符号名")
    p_callers.add_argument("--json", action="store_true")

    # callees
    p_callees = ssub.add_parser("callees", help="查找符号内部调用了哪些函数")
    p_callees.add_argument("repo", help="repo 根目录")
    p_callees.add_argument("--file", required=True, dest="file", metavar="FILE")
    p_callees.add_argument("--symbol", required=True, dest="symbol")
    p_callees.add_argument("--json", action="store_true")

    # trace
    p_trace = ssub.add_parser("trace", help="浅层调用链追踪")
    p_trace.add_argument("repo", help="repo 根目录")
    p_trace.add_argument("symbol", help="起始符号名")
    p_trace.add_argument("--depth", type=int, default=2, help="追踪深度（默认 2）")
    p_trace.add_argument("--json", action="store_true")

    # variants
    p_var = ssub.add_parser("variants", help="Variant/Factory 映射解析")
    p_var.add_argument("repo", help="repo 根目录")
    p_var.add_argument("--key", help="只显示此 key 对应的映射")
    p_var.add_argument("--file", dest="file_filter", help="只搜此文件")
    p_var.add_argument("--max", type=int, default=500, dest="max_results")
    p_var.add_argument("--json", action="store_true")


def _dispatch_scanner(args: argparse.Namespace) -> None:
    """Dispatch scanner sub-commands."""
    cmd = getattr(args, "scanner_cmd", None)
    if cmd is None:
        print(
            "scanner 命令：files / search / read / callsites / symbols / "
            "callers / callees / trace / variants\n"
            "使用 `sysight scanner <cmd> --help` 查看详情",
            file=sys.stderr,
        )
        return

    use_json = getattr(args, "json", False)

    if cmd == "files":
        result = list_files(args.repo, ext=getattr(args, "ext", None),
                            pattern=getattr(args, "pattern", None),
                            max_results=args.max_results)
        if use_json:
            print(json.dumps({
                "repo": result.repo, "total": result.total,
                "files": [{"path": f.path, "ext": f.ext, "size_bytes": f.size_bytes}
                          for f in result.files],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.repo}  ({result.total} files)")
            for f in result.files:
                print(f"  {f.path}  [{f.ext}  {f.size_bytes}B]")

    elif cmd == "search":
        result = scanner_search(
            args.repo, args.query,
            ext=getattr(args, "ext", None),
            fixed=args.fixed,
            case_sensitive=not args.ignore_case,
            max_results=args.max_results,
        )
        if use_json:
            print(json.dumps({
                "repo": result.repo, "query": result.query,
                "total_matches": result.total_matches,
                "matches": [{"path": m.path, "line": m.line, "column": m.column, "text": m.text}
                            for m in result.matches],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.total_matches} matches for {result.query!r} in {result.repo}")
            for m in result.matches:
                print(f"  {m.path}:{m.line}:{m.column}  {m.text}")

    elif cmd == "read":
        result = scanner_read_file(
            args.repo, args.file,
            start=getattr(args, "start", None),
            end=getattr(args, "end", None),
            around=getattr(args, "around", None),
            context=args.context,
        )
        if result.error:
            print(f"错误：{result.error}", file=sys.stderr)
            sys.exit(1)
        if use_json:
            print(json.dumps({
                "repo": result.repo, "path": result.path,
                "total_lines": result.total_lines,
                "shown_start": result.shown_start, "shown_end": result.shown_end,
                "lines": [{"line": ln.line, "text": ln.text} for ln in result.lines],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.path}  (lines {result.shown_start}-{result.shown_end} / {result.total_lines})")
            for ln in result.lines:
                print(f"{ln.line:6d}  {ln.text}")

    elif cmd == "callsites":
        result = find_callsites(
            args.repo, args.symbol,
            file_filter=getattr(args, "file_filter", None),
            ext=getattr(args, "ext", None),
            max_results=args.max_results,
        )
        if use_json:
            print(json.dumps({
                "repo": result.repo, "symbol": result.symbol, "total": result.total,
                "sites": [{"path": s.path, "line": s.line, "enclosing": s.enclosing, "source": s.source}
                          for s in result.sites],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.total} call-sites for {result.symbol!r} in {result.repo}")
            for s in result.sites:
                enc = f"  [{s.enclosing}]" if s.enclosing else ""
                print(f"  {s.path}:{s.line}{enc}  {s.source}")

    elif cmd == "symbols":
        result = list_symbols(args.repo, args.file)
        if result.error:
            print(f"错误：{result.error}", file=sys.stderr)
            sys.exit(1)
        filter_sym = getattr(args, "symbol", None)
        syms = result.symbols
        if filter_sym:
            syms = [s for s in syms if s.name == filter_sym]
        if use_json:
            print(json.dumps({
                "repo": result.repo, "file": result.file,
                "symbols": [
                    {"name": s.name, "kind": s.kind, "file": s.file,
                     "line": s.line, "end_line": s.end_line,
                     "signature": s.signature, "docstring": s.docstring}
                    for s in syms
                ],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.file}  ({len(syms)} symbols)")
            for s in syms:
                doc = f"  # {s.docstring}" if s.docstring else ""
                print(f"  L{s.line:4d}  [{s.kind:14s}]  {s.name}{doc}")

    elif cmd == "callers":
        result = find_callers(args.repo, args.symbol)
        if use_json:
            print(json.dumps({
                "repo": result.repo, "symbol": result.symbol, "total": result.total,
                "sites": result.sites,
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.total} callers of {result.symbol!r} in {result.repo}")
            for s in result.sites:
                enc = f"  [{s['enclosing']}]" if s.get("enclosing") else ""
                print(f"  {s['path']}:{s['line']}{enc}  {s['source']}")

    elif cmd == "callees":
        result = find_callees(args.repo, args.file, args.symbol)
        if use_json:
            print(json.dumps({
                "repo": result.repo, "symbol": result.symbol,
                "file": result.file, "callees": result.callees,
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# callees of {result.symbol!r} in {result.file}")
            for name in result.callees:
                print(f"  {name}")

    elif cmd == "trace":
        result = trace_symbol(args.repo, args.symbol, max_depth=args.depth)
        if use_json:
            print(json.dumps({
                "repo": result.repo, "root_symbol": result.root_symbol,
                "chain": result.chain,
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# trace {result.root_symbol!r}  (depth={args.depth})")
            for entry in result.chain:
                ext_mark = " [external]" if entry.get("external") else ""
                loc = f"  {entry['file']}:{entry['line']}" if entry.get("file") else ""
                print(f"  {entry['symbol']}{ext_mark}{loc}")
                for callee in entry.get("callees", []):
                    print(f"    → {callee}")

    elif cmd == "variants":
        result = find_variants(
            args.repo,
            key=getattr(args, "key", None),
            file_filter=getattr(args, "file_filter", None),
            max_results=args.max_results,
        )
        if use_json:
            print(json.dumps({
                "repo": result.repo, "total": result.total,
                "entries": [
                    {"key": e.key, "target": e.target, "file": e.file,
                     "line": e.line, "kind": e.kind, "context": e.context}
                    for e in result.entries
                ],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.total} variant mappings in {result.repo}")
            for e in result.entries:
                print(f"  [{e.kind:12s}]  {e.key!r:30s} → {e.target}  ({e.file}:{e.line})")

    else:
        print(f"未知 scanner 子命令：{cmd}", file=sys.stderr)
        sys.exit(1)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    # Check for nsys-sql as top-level command
    if argv_list and argv_list[0] == "nsys-sql":
        _main_standalone_nsys_sql(argv_list[1:])
        return

    # Check for nsys windows as top-level command: `sysight nsys windows <profile>`
    if len(argv_list) >= 2 and argv_list[0] == "nsys" and argv_list[1] == "windows":
        _main_standalone_nsys_windows(argv_list[2:])
        return

    # Check for scanner as top-level command
    if argv_list and argv_list[0] == "scanner":
        # Parse as `sysight scanner <cmd> ...` with full argparse
        pass  # falls through to main argparse below

    standalone_nsys = _standalone_nsys_argv(argv_list)
    if standalone_nsys is not None:
        _main_standalone_nsys(standalone_nsys)
        return

    p = argparse.ArgumentParser(
        prog="sysight",
        description=(
            "nsys 性能分析工具 + 静态代码分析工具集\n\n"
            "  sysight nsys <profile>           nsys profile 分析\n"
            "  sysight nsys-sql <cmd> <db>      直接查询 nsys SQLite\n"
            "  sysight scanner <cmd> <repo>     静态 repo 代码分析"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--verbose", "-v", action="store_true", help="开启调试日志")

    sub = p.add_subparsers(dest="subcmd")

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

    elif args.subcmd == "nsys":
        _emit_nsys(args)

    elif args.subcmd == "scanner":
        _dispatch_scanner(args)

    else:
        p.print_help()


if __name__ == "__main__":
    main()
