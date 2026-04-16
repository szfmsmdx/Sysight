"""cli — Command-line interface for sysight.analyzer.

Dispatches to repo analysis (repo.py) and nsys profile analysis (nsys/).
All rendering / serialisation helpers live here so repo.py stays pure.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .analyzer import (
    analyze_repo, render_summary, render_trace,
    discover_repo, scan_repo, build_dag,
    search_symbols, impact_radius, trace_from,
)
from .nsys import analyze_nsys
from .nsys.models import NsysAnalysisRequest, NsysDiag


# ── Nsys render helpers ───────────────────────────────────────────────────────

def _render_nsys_diag(diag: NsysDiag) -> str:
    """Human-readable text output for NsysDiag."""

    lines: list[str] = []
    lines.append(f"文件：{diag.profile_path}")
    if diag.sqlite_path:
        lines.append(f"SQLite：{diag.sqlite_path}")
    _STATUS_ZH = {"ok": "分析完成", "error": "分析失败", "action_required": "需要手动操作"}
    lines.append(f"状态：{_STATUS_ZH.get(diag.status, diag.status)}")
    lines.append("")

    if diag.status == "action_required":
        lines.append(f"请执行：{diag.required_action}")
        return "\n".join(lines)

    if diag.status == "error":
        for w in diag.repo_warnings:
            lines.append(f"错误：{w}")
        return "\n".join(lines)

    # Bottleneck summary
    if diag.bottlenecks:
        b = diag.bottlenecks
        total_ms = b.total_ns / 1e6
        active_ms = b.gpu_active_ns / 1e6
        idle_ms = b.gpu_idle_ns / 1e6
        lines.append(
            f"Trace 时长：{total_ms:.1f}ms  |  "
            f"GPU 活跃：{active_ms:.1f}ms ({active_ms / total_ms * 100:.1f}%)  |  "
            f"GPU 空闲：{idle_ms:.1f}ms ({idle_ms / total_ms * 100:.1f}%)"
        )
        lines.append("")
        lines.append("瓶颈分布：")
        for lb in b.labels:
            pct_gpu = (
                f"  (GPU 活跃时间的 {lb.pct_of_gpu_active * 100:.1f}%)"
                if lb.pct_of_gpu_active is not None else ""
            )
            lines.append(
                f"  {lb.category:<22}  {lb.pct_of_trace * 100:5.1f}% of trace"
                f"  {lb.active_ns / 1e6:8.1f}ms{pct_gpu}"
            )
        lines.append("")
        if b.top_events:
            lines.append(f"Top {len(b.top_events)} 热点事件：")
            for ev in b.top_events[:10]:
                lines.append(
                    f"  [{ev.category}] {ev.name[:60]:<60}  "
                    f"{ev.total_ns / 1e6:8.1f}ms  ×{ev.count}"
                )
            lines.append("")

    # Findings
    if diag.findings:
        sev_icon = {"critical": "✖", "warning": "⚠", "info": "ℹ"}
        sev_zh = {"critical": "严重", "warning": "警告", "info": "提示"}
        lines.append(f"分析结果（{len(diag.findings)} 条）：")
        for f in diag.findings:
            icon = sev_icon.get(f.severity, "·")
            label = sev_zh.get(f.severity, f.severity.upper())
            lines.append(f"  {icon} [{label}] {f.title}")
            if f.evidence:
                for ev in f.evidence[:3]:
                    lines.append(f"      {ev}")
            if f.next_step:
                lines.append(f"    → {f.next_step[:120]}")
        lines.append("")

    # Hotspots
    if diag.hotspots:
        lines.append(f"代码热点（已映射到仓库，共 {len(diag.hotspots)} 个）：")
        for h in diag.hotspots[:10]:
            mapped = h.repo_file or "(未在仓库中找到)"
            fn = f"  [{h.function}]" if h.function else ""
            lines.append(
                f"  {h.sample.pct * 100:5.1f}%  {h.sample.frame.symbol or '?':<40}"
                f"  → {mapped}{fn}  置信度={h.match_confidence:.2f}"
            )
        lines.append("")

    # Warnings
    for w in diag.repo_warnings:
        lines.append(f"注意：{w}")

    if diag.summary:
        lines.append("")
        lines.append(f"摘要：{diag.summary}")

    return "\n".join(lines)


def _nsys_diag_to_dict(diag: NsysDiag) -> dict:
    """JSON-serialisable dict for NsysDiag."""

    d: dict = {
        "status": diag.status,
        "profile_path": diag.profile_path,
        "sqlite_path": diag.sqlite_path,
        "required_action": diag.required_action,
        "summary": diag.summary,
        "repo_warnings": diag.repo_warnings,
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
                    "confidence": lb.confidence,
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

    d["findings"] = [
        {
            "category": f.category,
            "severity": f.severity,
            "confidence": f.confidence,
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
            "symbol": h.sample.frame.symbol,
            "source_file": h.sample.frame.source_file,
            "source_line": h.sample.frame.source_line,
            "count": h.sample.count,
            "pct": h.sample.pct,
            "repo_file": h.repo_file,
            "function": h.function,
            "match_confidence": h.match_confidence,
            "match_reason": h.match_reason,
            "callers": h.callers,
            "callees": h.callees,
        }
        for h in diag.hotspots
    ]

    d["evidence_links"] = [
        {
            "bottleneck_category": el.bottleneck_category,
            "event_name": el.event_name,
            "hotspot_function": el.hotspot_function,
            "hotspot_file": el.hotspot_file,
            "link_type": el.link_type,
            "confidence": el.confidence,
            "reason": el.reason,
        }
        for el in diag.evidence_links
    ]

    return d


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="sysight",
        description=(
            "仓库静态分析 + nsys 性能分析工具\n\n"
            "  sysight <repo>                             全量扫描分析\n"
            "  sysight <repo> search <query>              搜索符号/文件\n"
            "  sysight <repo> impact <files...>           影响范围分析\n"
            "  sysight <repo> trace <file-or-sym>         调用链追踪\n"
            "  sysight <repo> manifest                    路径清单（仅 Stage 1）\n"
            "  sysight <repo> nsys <profile>              nsys 性能文件分析\n"
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
        help="分析 Nsight Systems 性能文件并映射热点到源码",
    )
    sc.add_argument("profile", help=".nsys-rep 或 .sqlite 文件路径")
    sc.add_argument(
        "--sqlite", metavar="FILE",
        help="显式指定 .sqlite 路径（覆盖自动检测）",
    )
    sc.add_argument(
        "--no-repo", action="store_true",
        help="跳过仓库映射，仅做瓶颈分析（更快）",
    )
    sc.add_argument(
        "--scope", choices=("targeted", "full"), default="targeted",
        help="仓库扫描模式：targeted（默认）或 full",
    )
    sc.add_argument(
        "--top-hotspots", type=int, default=20,
        help="映射到源码的 Top N 热点（默认 20）",
    )

    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

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
        req = NsysAnalysisRequest(
            repo_root=str(Path(args.repo_path).resolve()),
            profile_path=args.profile,
            sqlite_path=getattr(args, "sqlite", None),
            top_hotspots=getattr(args, "top_hotspots", 20),
            repo_scope_mode=getattr(args, "scope", "targeted"),
            include_repo_context=not getattr(args, "no_repo", False),
        )
        diag = analyze_nsys(req)
        if args.json:
            print(json.dumps(_nsys_diag_to_dict(diag), indent=2))
        else:
            print(_render_nsys_diag(diag))

    else:
        result = analyze_repo(args.repo_path, top_n=args.top, max_chain_depth=args.depth)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(render_summary(result))
