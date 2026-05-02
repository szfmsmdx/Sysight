"""fs — file discovery inside a repo root.

Commands exposed via CLI:
  scanner files <repo>                     list all tracked files
  scanner files <repo> --ext py            filter by extension
  scanner files <repo> --pattern "*/data/*" glob filter
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


# Files / dirs to always skip (mirrors common .gitignore patterns)
_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "*.egg-info", ".eggs",
}


def _is_skip_dir(name: str) -> bool:
    for pat in _SKIP_DIRS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


@dataclass
class FileEntry:
    path: str          # relative to repo root
    ext: str           # lower-case extension including dot, e.g. ".py"
    size_bytes: int


@dataclass
class FilesResult:
    repo: str
    total: int
    files: list[FileEntry] = field(default_factory=list)


def list_files(
    repo: str,
    ext: str | None = None,
    pattern: str | None = None,
    max_results: int = 2000,
) -> FilesResult:
    """Return all files under repo, optionally filtered.

    Args:
        repo: absolute or relative path to repo root.
        ext: file extension filter, e.g. "py" or ".py".
        pattern: glob pattern matched against the *relative* path, e.g. "*/data/*".
        max_results: safety cap to avoid huge output.
    """
    root = Path(repo).resolve()
    norm_ext = ("." + ext.lstrip(".")).lower() if ext else None

    entries: list[FileEntry] = []

    for p in _walk(root):
        rel = p.relative_to(root)
        rel_str = str(rel)

        # ext filter
        if norm_ext and p.suffix.lower() != norm_ext:
            continue
        # glob pattern filter
        if pattern and not fnmatch.fnmatch(rel_str, pattern):
            continue

        entries.append(FileEntry(
            path=rel_str,
            ext=p.suffix.lower(),
            size_bytes=p.stat().st_size,
        ))
        if len(entries) >= max_results:
            break

    entries.sort(key=lambda e: e.path)
    return FilesResult(repo=str(root), total=len(entries), files=entries)


def _walk(root: Path):
    """Yield all files under root, skipping ignored dirs."""
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            items = sorted(cur.iterdir())
        except PermissionError:
            continue
        for item in items:
            if item.is_dir():
                if not _is_skip_dir(item.name):
                    stack.append(item)
            elif item.is_file():
                yield item

# ── ToolDef ────────────────────────────────────────────────────────────────
from sysight.tools.registry import ToolDef  # noqa: E402

FILES_TOOL = ToolDef(
    name="scanner_files",
    description="List all tracked files in a repo, optionally filtered by extension or glob pattern",
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Path to repo root"},
            "ext": {"type": "string", "description": "File extension filter, e.g. 'py'"},
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '*/data/*'"},
            "max_results": {"type": "integer", "default": 2000},
        },
        "required": ["repo"],
    },
    fn=list_files,
    read_only=True,
)
