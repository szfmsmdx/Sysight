"""SQLite FTS index for wiki content search.

TODO: Stage 3 — full FTS5 implementation. Currently uses file-system grep fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SearchResult:
    path: str
    title: str = ""
    snippet: str = ""
    score: float = 0.0


class FTSIndex:
    """Full-text search over wiki pages."""

    def __init__(self, wiki_dir: str | Path):
        self._wiki_dir = Path(wiki_dir)

    def search(self, query: str, namespace: str | None = None, limit: int = 10) -> list[SearchResult]:
        """Search wiki pages for query terms. Fallback: file-system grep."""
        results: list[SearchResult] = []
        search_dir = self._wiki_dir
        if namespace:
            search_dir = search_dir / "workspaces" / namespace
            if not search_dir.is_dir():
                return results

        terms = query.lower().split()
        for md in sorted(search_dir.rglob("*.md")):
            if md.name == "INDEX.md":
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = text
            title = ""
            if text.startswith("---"):
                end = text.find("---", 3)
                if end != -1:
                    body = text[end + 3:]
                    for line in text[3:end].split("\n"):
                        if line.startswith("title:"):
                            title = line.split(":", 1)[1].strip()

            score = sum(body.lower().count(t) for t in terms)
            if score > 0:
                rel = md.relative_to(self._wiki_dir)
                snippet = self._extract_snippet(body, terms[0] if terms else "")
                results.append(SearchResult(
                    path=str(rel),
                    title=title,
                    snippet=snippet,
                    score=float(score),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    @staticmethod
    def _extract_snippet(text: str, term: str, context: int = 60) -> str:
        idx = text.lower().find(term.lower())
        if idx == -1:
            return text[:context * 2].replace("\n", " ")
        start = max(0, idx - context)
        end = min(len(text), idx + len(term) + context)
        return text[start:end].replace("\n", " ").strip()

    def rebuild(self) -> None:
        """Rebuild the FTS index. No-op in fallback mode."""
        pass
