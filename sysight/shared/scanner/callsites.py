"""callsites — find all call-sites of a symbol across a repo.

A "call-site" is any line that contains a call to a given name, e.g.:
  - function call:   foo(...)  or  obj.foo(...)
  - method access:   obj.foo   (e.g. passed as callback)

Commands:
  scanner callsites <repo> --call foo
  scanner callsites <repo> --call dispatch_experts
  scanner callsites <repo> --call topk --file src/routing/planner.py
  scanner callsites <repo> --call item  --ext py

Strategy: pure text-based static search (no AST import to avoid
running target-repo code). We search for the call pattern
`<name>(` or `.<name>(` and also attribute access `.<name>` on its own line.
This is intentionally conservative — may have false positives on comments
but will not miss real call-sites.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .fs import _walk
from .search import _TEXT_EXTS


@dataclass
class CallSite:
    path: str       # relative to repo root
    line: int       # 1-based
    enclosing: str  # best-effort enclosing function/class name, "" if unknown
    source: str     # full source line (stripped)


@dataclass
class CallSitesResult:
    repo: str
    symbol: str
    total: int
    sites: list[CallSite] = field(default_factory=list)


# Matches def/class definitions to track enclosing scope
_DEF_RE = re.compile(r"^(\s*)(?:def|class|async\s+def)\s+(\w+)")


def _build_scope_index(lines: list[str]) -> list[str]:
    """Return a list of enclosing scope name for each line (1-indexed by offset+1).

    Uses indentation heuristic: when we enter a new def/class we record it;
    when indentation drops we pop back.
    """
    scopes: list[tuple[int, str]] = []   # (indent_level, name)
    result: list[str] = []
    for line in lines:
        m = _DEF_RE.match(line)
        if m:
            indent = len(m.group(1))
            name = m.group(2)
            # pop scopes that are at same or deeper indent
            while scopes and scopes[-1][0] >= indent:
                scopes.pop()
            scopes.append((indent, name))
        result.append(scopes[-1][1] if scopes else "")
    return result


def find_callsites(
    repo: str,
    symbol: str,
    file_filter: str | None = None,
    ext: str | None = None,
    max_results: int = 300,
) -> CallSitesResult:
    """Find all call-sites of *symbol* in the repo.

    Args:
        repo: repo root.
        symbol: function/method name to search for (bare name, no module).
        file_filter: restrict to this relative file path (exact).
        ext: restrict to this extension, e.g. "py".
        max_results: safety cap.
    """
    root = Path(repo).resolve()
    norm_ext = ("." + ext.lstrip(".")).lower() if ext else None

    # Pattern: optional leading dot or whitespace, then `symbol(` or `symbol (`
    # Also matches attribute accesses `obj.symbol` but requires at least a dot
    # or the symbol appearing at word boundary followed by '('
    call_re = re.compile(
        r"(?:^|[^\w])(?:\.)?"
        + re.escape(symbol)
        + r"(?:\s*\(|\s*$|[,\s\)])"
    )

    sites: list[CallSite] = []

    for p in _walk(root):
        if p.suffix.lower() not in _TEXT_EXTS:
            continue
        if norm_ext and p.suffix.lower() != norm_ext:
            continue
        rel = str(p.relative_to(root))
        if file_filter and rel != file_filter:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lines = text.splitlines()
        scope_index = _build_scope_index(lines)

        for lineno, line in enumerate(lines, 1):
            # skip comment-only lines
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if call_re.search(line):
                sites.append(CallSite(
                    path=rel,
                    line=lineno,
                    enclosing=scope_index[lineno - 1],
                    source=stripped,
                ))
                if len(sites) >= max_results:
                    return CallSitesResult(
                        repo=str(root), symbol=symbol,
                        total=max_results, sites=sites,
                    )

    return CallSitesResult(
        repo=str(root), symbol=symbol,
        total=len(sites), sites=sites,
    )
