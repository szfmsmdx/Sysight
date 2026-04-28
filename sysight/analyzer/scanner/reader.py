"""reader — read a file with line numbers, optionally sliced.

Commands:
  scanner read <repo> <file>                  read whole file with line numbers
  scanner read <repo> <file> --start 10       from line 10
  scanner read <repo> <file> --start 10 --end 30
  scanner read <repo> <file> --around 42 --context 5   ±5 lines around line 42
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LineEntry:
    line: int   # 1-based
    text: str   # raw line content (without trailing newline)


@dataclass
class ReadResult:
    repo: str
    path: str           # relative to repo root
    abs_path: str
    total_lines: int
    shown_start: int    # first line shown (1-based)
    shown_end: int      # last line shown (1-based)
    lines: list[LineEntry] = field(default_factory=list)
    error: str | None = None


def read_file(
    repo: str,
    path: str,
    start: int | None = None,
    end: int | None = None,
    around: int | None = None,
    context: int = 10,
) -> ReadResult:
    """Read a file with explicit line numbers.

    Args:
        repo: repo root.
        path: path relative to repo root (or absolute inside root).
        start: first line to include (1-based). None = beginning.
        end: last line to include (1-based, inclusive). None = end of file.
        around: if given, show [around-context .. around+context] (overrides start/end).
        context: lines on each side when using *around*.
    """
    root = Path(repo).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()

    # Security: keep inside repo root
    try:
        target.relative_to(root)
    except ValueError:
        return ReadResult(
            repo=str(root), path=path, abs_path=str(target),
            total_lines=0, shown_start=0, shown_end=0,
            error=f"path is outside repo root: {target}",
        )

    if not target.is_file():
        return ReadResult(
            repo=str(root), path=path, abs_path=str(target),
            total_lines=0, shown_start=0, shown_end=0,
            error=f"file not found: {target}",
        )

    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ReadResult(
            repo=str(root), path=path, abs_path=str(target),
            total_lines=0, shown_start=0, shown_end=0,
            error=str(e),
        )

    all_lines = raw.splitlines()
    total = len(all_lines)

    if around is not None:
        lo = max(1, around - context)
        hi = min(total, around + context)
    else:
        lo = start if start is not None else 1
        hi = end if end is not None else total

    lo = max(1, lo)
    hi = min(total, hi)

    rel = str(target.relative_to(root))
    lines = [
        LineEntry(line=i, text=all_lines[i - 1])
        for i in range(lo, hi + 1)
    ]
    return ReadResult(
        repo=str(root), path=rel, abs_path=str(target),
        total_lines=total, shown_start=lo, shown_end=hi,
        lines=lines,
    )
