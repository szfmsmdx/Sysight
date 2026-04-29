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
from .memory.memory_cli import add_memory_subparser, dispatch_memory
from .nsys.sql_cli import add_nsys_sql_subparser, dispatch_nsys_sql, main_standalone_nsys_sql
from .scanner.scanner_cli import add_scanner_subparser, dispatch_scanner


def _nsys_diag_to_dict(diag: NsysDiag) -> dict:
    """JSON-serialisable dict for NsysDiag."""

    d: dict = {
        "status": diag.status,
        "profile_path": diag.profile_path,
        "sqlite_path": diag.sqlite_path,
        "required_action": diag.required_action,
        "summary": diag.summary,
        "warnings": diag.warnings,
        "localization": None,
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

    if diag.localization is not None:
        d["localization"] = {
            "backend": diag.localization.backend,
            "status": diag.localization.status,
            "output": diag.localization.output,
            "error": diag.localization.error,
            "command": diag.localization.command,
            "prompt": diag.localization.prompt,
            "output_path": diag.localization.output_path,
            "pid": diag.localization.pid,
            "summary": diag.localization.summary,
            "artifact_dir": diag.localization.artifact_dir,
            "prompt_path": diag.localization.prompt_path,
            "stdout_path": diag.localization.stdout_path,
            "stderr_path": diag.localization.stderr_path,
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
                for question in diag.localization.questions
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
                for anchor in diag.localization.anchors
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
        help="代码定位使用的 repo 根目录；未指定时自动推断 workspace/ 目录",
    )
    parser.add_argument(
        "--no-codex", action="store_true",
        help="快速模式：不运行 Codex，仅输出统计分析结果",
    )
    parser.add_argument(
        "--codex-model", metavar="MODEL",
        help="可选：指定 codex model",
    )
    parser.add_argument(
        "--memory-namespace", metavar="NS",
        help="Memory namespace (e.g. bench/case_1) for workspace isolation",
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
        run_localization=use_codex,
        localization_backend="codex" if use_codex else None,
        localization_model=getattr(args, "codex_model", None),
        emit_progress_info=True,
        include_deep_sql=True,
        include_evidence_windows=include_evidence_windows,
        memory_namespace=getattr(args, "memory_namespace", None),
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
        emit_progress_info=True,
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


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    # Check for nsys-sql as top-level command
    if argv_list and argv_list[0] == "nsys-sql":
        main_standalone_nsys_sql(argv_list[1:])
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
            "  sysight scanner <cmd> <repo>     静态 repo 代码分析\n"
            "  sysight memory <cmd>             运行态 memory 读写"
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
    add_nsys_sql_subparser(sub)

    # Add scanner as a subcommand under sysight
    add_scanner_subparser(sub)

    # Add memory as a subcommand under sysight
    add_memory_subparser(sub)

    args = p.parse_args(argv_list)

    _configure_logging(args.verbose)

    # Handle nsys-sql subcommand
    if args.subcmd == "nsys-sql":
        if dispatch_nsys_sql(args):
            return
        else:
            print("错误：未指定 nsys-sql 子命令。使用 --help 查看帮助。", file=sys.stderr)
            sys.exit(1)

    elif args.subcmd == "nsys":
        _emit_nsys(args)

    elif args.subcmd == "scanner":
        dispatch_scanner(args)

    elif args.subcmd == "memory":
        if dispatch_memory(args):
            return
        print("错误：未指定 memory 子命令。使用 --help 查看帮助。", file=sys.stderr)
        sys.exit(1)

    else:
        p.print_help()


if __name__ == "__main__":
    main()
