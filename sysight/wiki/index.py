"""SQLite FTS5 index for wiki content search.

Chunks wiki pages into sections, stores them in SQLite with FTS5
for fast full-text search with snippet extraction.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

_CHUNK_TOKENS = 512
_OVERLAP_TOKENS = 128
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class SearchResult:
    path: str
    title: str = ""
    snippet: str = ""
    section_title: str = ""
    score: float = 0.0


class FTSIndex:
    """Full-text search over wiki pages via SQLite FTS5."""

    def __init__(self, db_path: str | Path, wiki_dir: str | Path | None = None):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._wiki_dir = Path(wiki_dir) if wiki_dir else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def search(self, query: str, namespace: str | None = None, limit: int = 10) -> list[SearchResult]:
        """Search wiki pages via FTS5. Returns results with snippet and section info."""
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            conditions = ["wiki_fts MATCH ?"]
            params: list = [query]

            if namespace:
                conditions.append("dc.page_path LIKE ?")
                params.append(f"workspaces/{namespace}/%")

            where = " AND ".join(conditions)
            sql = f"""
                SELECT dc.page_path, dc.section_title, dc.content,
                       rank AS score
                FROM wiki_chunks dc
                JOIN wiki_fts fts ON dc.rowid = fts.rowid
                WHERE {where}
                ORDER BY rank
                LIMIT ?
            """
            params.append(limit)

            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []

            results: list[SearchResult] = []
            for row in rows:
                snippet = self._extract_snippet(row["content"], query)
                results.append(SearchResult(
                    path=row["page_path"],
                    snippet=snippet,
                    section_title=row["section_title"] or "",
                    score=float(row["score"]),
                ))
            return results

    def chunk_page(self, page_path: str, content: str) -> None:
        """Split page content into chunks and store in wiki_chunks.

        Chunks by markdown sections at ~512 tokens with ~128 token overlap.
        FTS5 triggers keep the index in sync automatically.
        """
        sections = self._split_sections(content)
        chunks: list[dict] = []
        chunk_index = 0

        for section_title, section_text in sections:
            tokens = _tokenize(section_text)
            if len(tokens) <= _CHUNK_TOKENS:
                chunks.append({
                    "id": f"{page_path}:{chunk_index}",
                    "page_path": page_path,
                    "chunk_index": chunk_index,
                    "section_title": section_title,
                    "content": section_text,
                    "token_count": len(tokens),
                })
                chunk_index += 1
            else:
                start = 0
                while start < len(tokens):
                    end = min(start + _CHUNK_TOKENS, len(tokens))
                    chunk_tokens = tokens[start:end]
                    chunk_text = " ".join(chunk_tokens)
                    chunks.append({
                        "id": f"{page_path}:{chunk_index}",
                        "page_path": page_path,
                        "chunk_index": chunk_index,
                        "section_title": section_title,
                        "content": chunk_text,
                        "token_count": len(chunk_tokens),
                    })
                    chunk_index += 1
                    if end >= len(tokens):
                        break
                    start = end - _OVERLAP_TOKENS

        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM wiki_chunks WHERE page_path = ?", (page_path,))
            for c in chunks:
                conn.execute(
                    "INSERT INTO wiki_chunks (id, page_path, chunk_index, section_title, content, token_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (c["id"], c["page_path"], c["chunk_index"],
                     c["section_title"], c["content"], c["token_count"]),
                )
            conn.commit()

    def rebuild(self) -> None:
        """Rebuild the FTS index from all wiki pages on disk."""
        if not self._wiki_dir or not self._wiki_dir.is_dir():
            return

        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM wiki_chunks")
            conn.commit()

        for md in sorted(self._wiki_dir.rglob("*.md")):
            if md.name == "INDEX.md":
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = text
            if text.startswith("---"):
                end = text.find("---", 3)
                if end != -1:
                    body = text[end + 3:]
            rel = md.relative_to(self._wiki_dir).as_posix()
            self.chunk_page(rel, body)

    @staticmethod
    def _extract_snippet(content: str, query: str, context: int = 60) -> str:
        """Extract a context window around the first matching query term."""
        first_term = query.split()[0] if query else ""
        if not first_term:
            return content[:context * 2].replace("\n", " ")
        idx = content.lower().find(first_term.lower())
        if idx == -1:
            return content[:context * 2].replace("\n", " ")
        start = max(0, idx - context)
        end = min(len(content), idx + len(first_term) + context)
        return content[start:end].replace("\n", " ").strip()

    @staticmethod
    def _split_sections(content: str) -> list[tuple[str, str]]:
        """Split markdown content into (section_title, text) pairs."""
        matches = list(_HEADER_RE.finditer(content))
        if not matches:
            return [("", content)]

        sections: list[tuple[str, str]] = []
        # Content before the first header
        if matches[0].start() > 0:
            sections.append(("", content[:matches[0].start()].strip()))

        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            text = content[start:end].strip()
            if text:
                sections.append((title, text))

        return sections


def _tokenize(text: str) -> list[str]:
    """Split text into approximate tokens by whitespace."""
    return text.split()
