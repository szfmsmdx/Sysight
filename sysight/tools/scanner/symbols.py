"""symbols — list symbols in a file, and find callers/callees.

Commands:
  scanner symbols <repo> --file src/ops/dispatch.py
      list all top-level def/class in that file

  scanner symbols <repo> --file src/ops/dispatch.py --symbol dispatch_experts
      show signature + first docstring line + callers within file

  scanner callers <repo> <symbol>
      files+lines that call <symbol>  (delegates to callsites)

  scanner callees <repo> --file src/ops/dispatch.py --symbol dispatch_experts
      symbols called *by* dispatch_experts  (text-based heuristic)

  scanner trace <repo> <symbol>
      shallow call-chain: symbol → direct callees (1 level)
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .callers import find_callsites
from .files import _walk
from .search import _TEXT_EXTS


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SymbolDef:
    name: str
    kind: str       # "function" | "async_function" | "class" | "method"
    file: str       # relative to repo root
    line: int       # 1-based, line of def/class keyword
    end_line: int   # best-effort: last line of body (0 if unknown)
    signature: str  # first line of def/class stripped
    docstring: str  # first non-empty docstring line, "" if none


@dataclass
class SymbolsResult:
    repo: str
    file: str
    symbols: list[SymbolDef] = field(default_factory=list)
    error: str | None = None


@dataclass
class CallersResult:
    repo: str
    symbol: str
    total: int
    sites: list[dict] = field(default_factory=list)   # re-use CallSite fields


@dataclass
class CalleesResult:
    repo: str
    symbol: str
    file: str
    callees: list[str] = field(default_factory=list)  # bare names called inside body


@dataclass
class TraceResult:
    repo: str
    root_symbol: str
    chain: list[dict] = field(default_factory=list)
    # each entry: {symbol, file, line, callees: [str]}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_symbols_ast(root: Path, rel: str) -> SymbolsResult:
    """Parse with AST for precise symbol extraction."""
    p = root / rel
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(p))
    except (OSError, SyntaxError) as e:
        return SymbolsResult(repo=str(root), file=rel, error=str(e))

    lines = src.splitlines()
    syms: list[SymbolDef] = []

    def _docstring(node) -> str:
        if (
            isinstance(node.body, list) and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            doc = node.body[0].value.value.strip()
            return doc.splitlines()[0] if doc else ""
        return ""

    def _end_line(node) -> int:
        return getattr(node, "end_lineno", 0) or 0

    def _visit(node, depth: int = 0) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            if depth > 0:
                kind = "method"
            sig = lines[node.lineno - 1].rstrip() if node.lineno <= len(lines) else ""
            syms.append(SymbolDef(
                name=node.name, kind=kind, file=rel,
                line=node.lineno, end_line=_end_line(node),
                signature=sig, docstring=_docstring(node),
            ))
            for child in ast.iter_child_nodes(node):
                _visit(child, depth + 1)
        elif isinstance(node, ast.ClassDef):
            sig = lines[node.lineno - 1].rstrip() if node.lineno <= len(lines) else ""
            syms.append(SymbolDef(
                name=node.name, kind="class", file=rel,
                line=node.lineno, end_line=_end_line(node),
                signature=sig, docstring=_docstring(node),
            ))
            for child in ast.iter_child_nodes(node):
                _visit(child, depth + 1)

    for node in ast.iter_child_nodes(tree):
        _visit(node, 0)

    return SymbolsResult(repo=str(root), file=rel, symbols=syms)


def _callees_in_body(root: Path, rel: str, symbol: str) -> list[str]:
    """Return bare names of all calls made inside *symbol*'s body (AST walk)."""
    p = root / rel
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(p))
    except (OSError, SyntaxError):
        return []

    # Find the target function node
    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                target_node = node
                break

    if target_node is None:
        return []

    called: list[str] = []
    for node in ast.walk(target_node):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.append(func.id)
            elif isinstance(func, ast.Attribute):
                called.append(func.attr)

    # deduplicate, preserve order
    seen: set[str] = set()
    result = []
    for name in called:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _find_def_in_repo(root: Path, symbol: str) -> tuple[str, int] | None:
    """Search all .py files for the first def/async def/class named *symbol*.

    Returns (rel_path, line_number) or None.
    """
    for p in _walk(root):
        if p.suffix.lower() != ".py":
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(p))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol:
                    rel = str(p.relative_to(root))
                    return rel, node.lineno
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def list_symbols(repo: str, file: str) -> SymbolsResult:
    """List all top-level and nested symbols defined in *file*."""
    root = Path(repo).resolve()
    rel = file
    if Path(file).is_absolute():
        try:
            rel = str(Path(file).relative_to(root))
        except ValueError:
            pass
    return _parse_symbols_ast(root, rel)


def find_callers(repo: str, symbol: str) -> CallersResult:
    """Find all callers of *symbol* across the repo (delegates to callsites)."""
    cs = find_callsites(repo, symbol)
    return CallersResult(
        repo=cs.repo, symbol=symbol, total=cs.total,
        sites=[
            {"path": s.path, "line": s.line, "enclosing": s.enclosing, "source": s.source}
            for s in cs.sites
        ],
    )


def find_callees(repo: str, file: str, symbol: str) -> CalleesResult:
    """Find all symbols called inside *symbol* in *file*."""
    root = Path(repo).resolve()
    rel = file
    if Path(file).is_absolute():
        try:
            rel = str(Path(file).relative_to(root))
        except ValueError:
            pass
    callees = _callees_in_body(root, rel, symbol)
    return CalleesResult(repo=str(root), symbol=symbol, file=rel, callees=callees)


def trace_symbol(repo: str, symbol: str, max_depth: int = 2) -> TraceResult:
    """Shallow call-chain: find *symbol* definition then its direct callees.

    max_depth=1 → just the direct callees of root symbol.
    max_depth=2 → callees of callees (one more level).
    Caps at max_depth to avoid explosion.
    """
    root = Path(repo).resolve()
    chain: list[dict] = []
    visited: set[str] = set()

    def _expand(sym: str, depth: int) -> None:
        if depth > max_depth or sym in visited:
            return
        visited.add(sym)

        # Find where this symbol is defined
        loc = _find_def_in_repo(root, sym)
        if loc is None:
            # Symbol not found in repo — record as external
            chain.append({"symbol": sym, "file": None, "line": None, "callees": [], "external": True})
            return

        rel, lineno = loc
        callees = _callees_in_body(root, rel, sym)

        chain.append({
            "symbol": sym,
            "file": rel,
            "line": lineno,
            "callees": callees,
            "external": False,
        })

        for callee in callees:
            _expand(callee, depth + 1)

    _expand(symbol, 0)
    return TraceResult(repo=str(root), root_symbol=symbol, chain=chain)

# ── ToolDefs ───────────────────────────────────────────────────────────────
from sysight.tools.registry import ToolDef  # noqa: E402

SYMBOLS_TOOL = ToolDef(
    name="scanner_symbols",
    description="List all top-level function/class definitions in a file (AST-based)",
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Path to repo root"},
            "file": {"type": "string", "description": "File path relative to repo root"},
        },
        "required": ["repo", "file"],
    },
    fn=list_symbols,
    read_only=True,
)

CALLERS_SYMBOL_TOOL = ToolDef(
    name="scanner_symbol_callers",
    description="Find all files and lines that call a given symbol (text-based)",
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Path to repo root"},
            "symbol": {"type": "string", "description": "Symbol name to find callers of"},
        },
        "required": ["repo", "symbol"],
    },
    fn=find_callers,
    read_only=True,
)

CALLEES_TOOL = ToolDef(
    name="scanner_callees",
    description="Find symbols called by a given function (text-based heuristic)",
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Path to repo root"},
            "file": {"type": "string", "description": "File path relative to repo root"},
            "symbol": {"type": "string", "description": "Symbol name to find callees of"},
        },
        "required": ["repo", "file", "symbol"],
    },
    fn=find_callees,
    read_only=True,
)

TRACE_TOOL = ToolDef(
    name="scanner_trace",
    description="Trace call-chain from a symbol (1 level deep by default)",
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Path to repo root"},
            "symbol": {"type": "string", "description": "Starting symbol name"},
            "max_depth": {"type": "integer", "default": 2, "description": "Maximum depth of the call chain"},
        },
        "required": ["repo", "symbol"],
    },
    fn=trace_symbol,
    read_only=True,
)
