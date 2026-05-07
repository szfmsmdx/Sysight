"""Reference parsing and staleness propagation for wiki pages.

Parses markdown content for internal links and citations,
then tracks staleness when referenced pages are modified.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

_WIKI_LINK_RE = re.compile(r"(?<!!)\[(?:[^\]]*)\]\(([^)]+)\)")
_CITATION_RE = re.compile(r"\[\^\d+\]:\s*(.+)$", re.MULTILINE)


def parse_references(content: str) -> tuple[list[str], list[str]]:
    """Parse markdown content for wiki links and citation references.

    Returns (links_to, cites) where:
      - links_to: target paths from [text](path.md) patterns
      - cites: source filenames from [^1]: source.pdf patterns
    """
    links: list[str] = []
    for match in _WIKI_LINK_RE.finditer(content):
        href = match.group(1)
        if href.startswith(("http://", "https://", "#", "mailto:", "data:")):
            continue
        # Strip anchor fragments
        if "#" in href:
            href = href.split("#")[0]
        if href:
            links.append(href)

    citations: list[str] = []
    for match in _CITATION_RE.finditer(content):
        raw = match.group(1).strip().lstrip("*").rstrip("*")
        # Extract just the filename from e.g. "[paper.pdf](paper.pdf), p.3"
        link_match = re.match(r"\[([^\]]+)\]\([^)]*\)", raw)
        if link_match:
            raw = link_match.group(1)
        filename = raw.split(",")[0].strip()
        if filename:
            citations.append(filename)

    return links, citations


def propagate_staleness(db: sqlite3.Connection, target_path: str) -> None:
    """Mark all pages that reference target_path as stale."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.execute(
        "UPDATE wiki_links SET stale_since = ? WHERE target_path = ?",
        (now, target_path),
    )
    db.commit()


def find_stale_pages(db: sqlite3.Connection) -> list[str]:
    """Return distinct source paths that have stale links."""
    rows = db.execute(
        "SELECT DISTINCT source_path FROM wiki_links WHERE stale_since IS NOT NULL"
    ).fetchall()
    return [r[0] for r in rows]


def find_uncited_sources(db: sqlite3.Connection) -> list[str]:
    """Return cited sources that are not linked to by any wiki page."""
    rows = db.execute("""
        SELECT DISTINCT target_path FROM wiki_links
        WHERE link_type = 'cites'
          AND target_path NOT IN (
            SELECT DISTINCT target_path FROM wiki_links WHERE link_type = 'links_to'
          )
    """).fetchall()
    return [r[0] for r in rows]
