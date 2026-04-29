"""CLI helpers for `sysight memory` subcommands."""

from __future__ import annotations

import argparse
import json
import sys

from .store import build_memory_brief, default_memory_root, read_memory_file, search_memory, write_memory_file

_MEMORY_DESCRIPTION = (
    "运行态 memory 工具（默认根目录：.sysight/memory）\n\n"
    "  sysight memory search <query>          搜索 memory 内容\n"
    "  sysight memory read <path>             读取 memory 文件\n"
    "  sysight memory write <path>            写入或追加 memory 文件"
)


def add_memory_subparser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    mem = sub.add_parser(
        "memory",
        help="运行态 memory 读写工具",
        description=_MEMORY_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    msub = mem.add_subparsers(dest="memory_cmd")

    p_search = msub.add_parser("search", help="搜索 memory 内容")
    p_search.add_argument("query", help="搜索关键字或正则")
    p_search.add_argument("--root", default=str(default_memory_root()), help="memory 根目录")
    p_search.add_argument("--scope", choices=["workspace", "experience", "all"], default="all")
    p_search.add_argument("--fixed", action="store_true", help="字面搜索")
    p_search.add_argument("--ignore-case", "-i", action="store_true")
    p_search.add_argument("--namespace", default=None, help="workspace namespace (e.g. bench/case_1)")
    p_search.add_argument("--max", type=int, default=200, dest="max_results")
    p_search.add_argument("--json", action="store_true")

    p_read = msub.add_parser("read", help="读取 memory 文件")
    p_read.add_argument("path", help="memory 相对路径，或 workspace / experience")
    p_read.add_argument("--root", default=str(default_memory_root()), help="memory 根目录")
    p_read.add_argument("--start", type=int)
    p_read.add_argument("--end", type=int)
    p_read.add_argument("--around", type=int)
    p_read.add_argument("--context", type=int, default=10)
    p_read.add_argument("--json", action="store_true")

    p_write = msub.add_parser("write", help="写入或追加 memory 文件")
    p_write.add_argument("path", help="memory 相对路径，或 workspace / experience")
    p_write.add_argument("content", nargs="?", default=None, help="直接写入内容")
    p_write.add_argument("--root", default=str(default_memory_root()), help="memory 根目录")
    p_write.add_argument("--append", action="store_true", help="以追加模式写入")
    p_write.add_argument("--stdin", action="store_true", help="从 stdin 读取内容")
    p_write.add_argument("--json", action="store_true")

    p_brief = msub.add_parser("brief", help="输出 memory brief 用于 prompt 注入")
    p_brief.add_argument("--root", default=str(default_memory_root()), help="memory 根目录")
    p_brief.add_argument("--repo-root", default=None, help="repo 根目录（用于推导 namespace）")
    p_brief.add_argument("--namespace", default=None, help="workspace namespace (e.g. bench/case_1)")
    p_brief.add_argument("--json", action="store_true")


def dispatch_memory(args: argparse.Namespace) -> bool:
    cmd = getattr(args, "memory_cmd", None)
    if cmd is None:
        print("memory 命令：search / read / write", file=sys.stderr)
        return False

    use_json = getattr(args, "json", False)

    if cmd == "search":
        scope = None if args.scope == "all" else args.scope
        ns = getattr(args, "namespace", None)
        result = search_memory(
            args.root,
            args.query,
            scope=scope,
            fixed=args.fixed,
            case_sensitive=not args.ignore_case,
            max_results=args.max_results,
            namespace=ns,
        )
        if use_json:
            print(json.dumps({
                "root": result.root,
                "query": result.query,
                "total_matches": result.total_matches,
                "matches": [
                    {"path": m.path, "line": m.line, "column": m.column, "text": m.text}
                    for m in result.matches
                ],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"# {result.total_matches} matches for {result.query!r} in {result.root}")
            for m in result.matches:
                print(f"  {m.path}:{m.line}:{m.column}  {m.text}")
        return True

    if cmd == "read":
        result = read_memory_file(
            args.root,
            args.path,
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
                "root": result.root,
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
        return True

    if cmd == "brief":
        brief = build_memory_brief(
            args.root,
            repo_root=getattr(args, "repo_root", None),
            namespace=getattr(args, "namespace", None),
        )
        if use_json:
            print(json.dumps({"brief": brief}, ensure_ascii=False, indent=2))
        else:
            print(brief)
        return True

    if cmd == "write":
        content = args.content
        if args.stdin:
            content = sys.stdin.read()
        if content is None:
            print("错误：write 需要内容参数或 --stdin", file=sys.stderr)
            sys.exit(1)
        result = write_memory_file(args.root, args.path, content, append=args.append)
        if result.error:
            print(f"错误：{result.error}", file=sys.stderr)
            sys.exit(1)
        if use_json:
            print(json.dumps({
                "root": result.root,
                "path": result.path,
                "abs_path": result.abs_path,
                "bytes_written": result.bytes_written,
                "append": result.append,
            }, ensure_ascii=False, indent=2))
        else:
            mode = "append" if result.append else "replace"
            print(f"# wrote {result.bytes_written} bytes to {result.path} ({mode})")
        return True

    return False
