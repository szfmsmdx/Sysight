"""Terminal text helpers for nsys report rendering."""

from __future__ import annotations

import unicodedata
from typing import Sequence


def display_width(text: object) -> int:
    """Return terminal cell width for plain text."""
    width = 0
    for ch in str(text):
        if unicodedata.combining(ch):
            continue
        if unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
    return width


def clip_display(text: object, width: int) -> str:
    """Clip text without exceeding the requested terminal cell width."""
    clipped: list[str] = []
    used = 0
    for ch in str(text):
        ch_width = display_width(ch)
        if used + ch_width > width:
            break
        clipped.append(ch)
        used += ch_width
    return "".join(clipped)


def pad_display(text: object, width: int) -> str:
    """Left-pad text to a terminal cell width."""
    value = str(text)
    return value + " " * max(0, width - display_width(value))


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    col_widths: Sequence[int] | None = None,
) -> list[str]:
    """Render a fixed-width table using terminal display widths."""
    if not rows:
        return []

    if col_widths is None:
        widths = [display_width(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], display_width(cell))
    else:
        widths = list(col_widths)

    sep = "  "
    lines = [
        sep.join(pad_display(clip_display(h, w), w) for h, w in zip(headers, widths)),
        sep.join("─" * w for w in widths),
    ]
    for row in rows:
        cells = list(row[:len(widths)])
        while len(cells) < len(widths):
            cells.append("")
        lines.append(
            sep.join(pad_display(clip_display(cell, w), w) for cell, w in zip(cells, widths))
        )
    return lines
