"""search — keyword / pattern search across repo files.

Commands:
  scanner search <repo> <keyword>           full-text search, returns file+line+snippet
  scanner search <repo> <keyword> --ext py  limit to .py files
  scanner search <repo> <keyword> --fixed   treat as literal string (no regex)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .files import _walk


@dataclass
class SearchMatch:
    path: str        # relative to repo root
    line: int        # 1-based
    column: int      # 1-based, start of match
    text: str        # full line content (stripped)


@dataclass
class SearchResult:
    repo: str
    query: str
    total_matches: int
    matches: list[SearchMatch] = field(default_factory=list)


_TEXT_EXTS = {
    ".py", ".pyx", ".pxd",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".java", ".kt", ".scala", ".go", ".rs",
    ".js", ".ts", ".jsx", ".tsx",
    ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".conf",
    ".sh", ".bash", ".zsh", ".fish",
    ".md", ".txt", ".rst",
    ".sql",
}


def search(
    repo: str,
    query: str,
    ext: str | None = None,
    fixed: bool = False,
    case_sensitive: bool = True,
    max_results: int = 500,
) -> SearchResult:
    """Search for *query* in all text files under *repo*.

    Args:
        repo: repo root path.
        query: regex pattern (or literal if fixed=True).
        ext: restrict to this extension, e.g. "py".
        fixed: if True, treat query as literal string.
        case_sensitive: default True.
        max_results: safety cap.
    """
    root = Path(repo).resolve()
    norm_ext = ("." + ext.lstrip(".")).lower() if ext else None

    flags = 0 if case_sensitive else re.IGNORECASE
    if fixed:
        pattern = re.compile(re.escape(query), flags)
    else:
        try:
            pattern = re.compile(query, flags)
        except re.error:
            pattern = re.compile(re.escape(query), flags)

    matches: list[SearchMatch] = []

    for p in _walk(root):
        if p.suffix.lower() not in _TEXT_EXTS:
            continue
        if norm_ext and p.suffix.lower() != norm_ext:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            m = pattern.search(line)
            if m:
                matches.append(SearchMatch(
                    path=str(p.relative_to(root)),
                    line=lineno,
                    column=m.start() + 1,
                    text=line.rstrip(),
                ))
                if len(matches) >= max_results:
                    return SearchResult(
                        repo=str(root), query=query,
                        total_matches=max_results, matches=matches,
                    )

    return SearchResult(
        repo=str(root), query=query,
        total_matches=len(matches), matches=matches,
    )

# ── ToolDef ────────────────────────────────────────────────────────────────
from sysight.tools.registry import ToolDef  # noqa: E402

SEARCH_TOOL = ToolDef(
    name="scanner_search",
    description="Full-text search in repo source files. Returns file path, line number, and matching line text",
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Path to repo root"},
            "query": {"type": "string", "description": "Search term or regex pattern"},
            "ext": {"type": "string", "description": "Restrict to file extension, e.g. 'py'"},
            "fixed": {"type": "boolean", "default": False, "description": "Treat query as literal string, not regex"},
            "case_sensitive": {"type": "boolean", "default": True},
            "max_results": {"type": "integer", "default": 500},
        },
        "required": ["repo", "query"],
    },
    fn=search,
    read_only=True,
)
