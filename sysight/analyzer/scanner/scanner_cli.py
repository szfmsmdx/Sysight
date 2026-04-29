"""CLI helpers for `sysight scanner` subcommands."""

from __future__ import annotations

import argparse
import json
import sys

from .callsites import find_callsites
from .fs import list_files
from .reader import read_file as scanner_read_file
from .search import search as scanner_search
from .symbols import find_callees, find_callers, list_symbols, trace_symbol
from .variants import find_variants

_SCANNER_DESCRIPTION = (
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
)


def add_scanner_subparser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add `sysight scanner` subcommand with all sub-sub-commands."""
    sc = sub.add_parser(
        "scanner",
        help="静态 repo 代码分析工具集",
        description=_SCANNER_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ssub = sc.add_subparsers(dest="scanner_cmd")

    p_files = ssub.add_parser("files", help="列出 repo 中的文件")
    p_files.add_argument("repo", help="repo 根目录")
    p_files.add_argument("--ext", help="只列此扩展名（如 py）")
    p_files.add_argument("--pattern", help="glob 过滤（如 '*/data/*'）")
    p_files.add_argument("--max", type=int, default=2000, dest="max_results")
    p_files.add_argument("--json", action="store_true", help="JSON 输出")

    p_search = ssub.add_parser("search", help="全文搜索")
    p_search.add_argument("repo", help="repo 根目录")
    p_search.add_argument("query", help="搜索关键字或正则")
    p_search.add_argument("--ext", help="只搜此扩展名")
    p_search.add_argument("--fixed", action="store_true", help="字面字符串（非正则）")
    p_search.add_argument("--ignore-case", "-i", action="store_true")
    p_search.add_argument("--max", type=int, default=500, dest="max_results")
    p_search.add_argument("--json", action="store_true")

    p_read = ssub.add_parser("read", help="读取文件（带行号）")
    p_read.add_argument("repo", help="repo 根目录")
    p_read.add_argument("file", help="相对于 repo 的文件路径")
    p_read.add_argument("--start", type=int, help="起始行（含）")
    p_read.add_argument("--end", type=int, help="结束行（含）")
    p_read.add_argument("--around", type=int, help="中心行，配合 --context 使用")
    p_read.add_argument("--context", type=int, default=10, help="around 上下文行数（默认 10）")
    p_read.add_argument("--json", action="store_true")

    p_cs = ssub.add_parser("callsites", help="查找符号的所有调用点")
    p_cs.add_argument("repo", help="repo 根目录")
    p_cs.add_argument("--call", required=True, dest="symbol", metavar="SYMBOL")
    p_cs.add_argument("--file", dest="file_filter", help="只在此文件中搜索")
    p_cs.add_argument("--ext", help="只搜此扩展名")
    p_cs.add_argument("--max", type=int, default=300, dest="max_results")
    p_cs.add_argument("--json", action="store_true")

    p_sym = ssub.add_parser("symbols", help="列出文件中的所有符号定义")
    p_sym.add_argument("repo", help="repo 根目录")
    p_sym.add_argument("--file", required=True, dest="file", metavar="FILE")
    p_sym.add_argument("--symbol", dest="symbol", help="只显示此符号的详情")
    p_sym.add_argument("--json", action="store_true")

    p_callers = ssub.add_parser("callers", help="查找调用某符号的所有位置")
    p_callers.add_argument("repo", help="repo 根目录")
    p_callers.add_argument("symbol", help="目标符号名")
    p_callers.add_argument("--json", action="store_true")

    p_callees = ssub.add_parser("callees", help="查找符号内部调用了哪些函数")
    p_callees.add_argument("repo", help="repo 根目录")
    p_callees.add_argument("--file", required=True, dest="file", metavar="FILE")
    p_callees.add_argument("--symbol", required=True, dest="symbol")
    p_callees.add_argument("--json", action="store_true")

    p_trace = ssub.add_parser("trace", help="浅层调用链追踪")
    p_trace.add_argument("repo", help="repo 根目录")
    p_trace.add_argument("symbol", help="起始符号名")
    p_trace.add_argument("--depth", type=int, default=2, help="追踪深度（默认 2）")
    p_trace.add_argument("--json", action="store_true")

    p_var = ssub.add_parser("variants", help="Variant/Factory 映射解析")
    p_var.add_argument("repo", help="repo 根目录")
    p_var.add_argument("--key", help="只显示此 key 对应的映射")
    p_var.add_argument("--file", dest="file_filter", help="只搜此文件")
    p_var.add_argument("--max", type=int, default=500, dest="max_results")
    p_var.add_argument("--json", action="store_true")


def dispatch_scanner(args: argparse.Namespace) -> None:
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
        result = list_files(
            args.repo,
            ext=getattr(args, "ext", None),
            pattern=getattr(args, "pattern", None),
            max_results=args.max_results,
        )
        if use_json:
            print(json.dumps({
                "repo": result.repo,
                "total": result.total,
                "files": [
                    {"path": f.path, "ext": f.ext, "size_bytes": f.size_bytes}
                    for f in result.files
                ],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.repo}  ({result.total} files)")
            for f in result.files:
                print(f"  {f.path}  [{f.ext}  {f.size_bytes}B]")

    elif cmd == "search":
        result = scanner_search(
            args.repo,
            args.query,
            ext=getattr(args, "ext", None),
            fixed=args.fixed,
            case_sensitive=not args.ignore_case,
            max_results=args.max_results,
        )
        if use_json:
            print(json.dumps({
                "repo": result.repo,
                "query": result.query,
                "total_matches": result.total_matches,
                "matches": [
                    {"path": m.path, "line": m.line, "column": m.column, "text": m.text}
                    for m in result.matches
                ],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.total_matches} matches for {result.query!r} in {result.repo}")
            for m in result.matches:
                print(f"  {m.path}:{m.line}:{m.column}  {m.text}")

    elif cmd == "read":
        result = scanner_read_file(
            args.repo,
            args.file,
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
                "repo": result.repo,
                "path": result.path,
                "total_lines": result.total_lines,
                "shown_start": result.shown_start,
                "shown_end": result.shown_end,
                "lines": [{"line": ln.line, "text": ln.text} for ln in result.lines],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.path}  (lines {result.shown_start}-{result.shown_end} / {result.total_lines})")
            for ln in result.lines:
                print(f"{ln.line:6d}  {ln.text}")

    elif cmd == "callsites":
        result = find_callsites(
            args.repo,
            args.symbol,
            file_filter=getattr(args, "file_filter", None),
            ext=getattr(args, "ext", None),
            max_results=args.max_results,
        )
        if use_json:
            print(json.dumps({
                "repo": result.repo,
                "symbol": result.symbol,
                "total": result.total,
                "sites": [
                    {"path": s.path, "line": s.line, "enclosing": s.enclosing, "source": s.source}
                    for s in result.sites
                ],
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
        syms = result.symbols
        filter_sym = getattr(args, "symbol", None)
        if filter_sym:
            syms = [s for s in syms if s.name == filter_sym]
        if use_json:
            print(json.dumps({
                "repo": result.repo,
                "file": result.file,
                "symbols": [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "file": s.file,
                        "line": s.line,
                        "end_line": s.end_line,
                        "signature": s.signature,
                        "docstring": s.docstring,
                    }
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
                "repo": result.repo,
                "symbol": result.symbol,
                "total": result.total,
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
                "repo": result.repo,
                "symbol": result.symbol,
                "file": result.file,
                "callees": result.callees,
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# callees of {result.symbol!r} in {result.file}")
            for name in result.callees:
                print(f"  {name}")

    elif cmd == "trace":
        result = trace_symbol(args.repo, args.symbol, max_depth=args.depth)
        if use_json:
            print(json.dumps({
                "repo": result.repo,
                "root_symbol": result.root_symbol,
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
                "repo": result.repo,
                "total": result.total,
                "entries": [
                    {
                        "key": e.key,
                        "target": e.target,
                        "file": e.file,
                        "line": e.line,
                        "kind": e.kind,
                        "context": e.context,
                    }
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
