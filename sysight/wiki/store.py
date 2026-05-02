"""Wiki storage — Repository pattern over `.sysight/memory/wiki/`.

Parent-only write operations. Child agents read via memory.search / memory.read tools.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

_ALIAS_TO_FILE = {"workspace": "workspace.md", "experience": "experience.md"}
_TEXT_EXTS = {".md", ".txt", ".json", ".yml", ".yaml", ".log"}
_SEPARATOR = "\n---\n"

_CATEGORY_SIGNAL_MAP: dict[str, str] = {
    "C1": "C1-host-scheduling",
    "C2": "C2-kernel-launch-overhead",
    "C3": "C3-synchronization",
    "C4": "C4-memory-copy",
    "C5": "C5-compute-inefficiency",
    "C6": "C6-communication",
    "C7": "C7-python-pipeline",
}


# ── Repository class ──────────────────────────────────────────────────────────

class WikiRepository:
    """CRUD operations on the Sysight wiki store. Parent-only writes."""

    def __init__(self, root: str | Path | None = None):
        self._root = Path(root).resolve() if root else (Path.cwd() / ".sysight" / "memory").resolve()

    @property
    def root(self) -> Path:
        return self._root

    def read_page(self, path: str) -> str | None:
        """Read a wiki page by path. Returns body text or None."""
        target = self._resolve_path(path)
        if not target.exists():
            return None
        return target.read_text(encoding="utf-8")

    def write_page(self, path: str, content: str, title: str = "",
                   category: str | None = None, tags: list[str] | None = None,
                   scope: str = "workspace", source_run: str = "") -> Path:
        """Write (create or replace) a wiki page with YAML frontmatter."""
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = self._build_frontmatter(title, category, tags or [], scope, source_run)
        target.write_text(frontmatter + "\n" + content, encoding="utf-8")
        if "workspaces/" in path or "experiences/" in path:
            self._update_index()
        if category and category in _CATEGORY_SIGNAL_MAP:
            self._update_signal(category, path, title)
        return target

    def append_worklog(self, namespace: str, entry: str) -> Path:
        """Append an entry to the workspace worklog."""
        path = f"workspaces/{namespace}/worklog.md"
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        timestamp = _now_iso()
        with target.open("a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp}\n\n{entry}\n")
        return target

    def list_experiences(self, category: str | None = None) -> list[dict]:
        """List experience pages, optionally filtered by category."""
        exp_dir = self._root / "wiki" / "experiences"
        if not exp_dir.is_dir():
            return []
        results = []
        for p in sorted(exp_dir.glob("*.md")):
            info = self._parse_frontmatter(p)
            if category is None or info.get("category") == category:
                info["path"] = str(p.relative_to(self._root))
                results.append(info)
        return results

    def workspace_namespace(self, repo_root: str | None = None, namespace: str | None = None) -> str:
        """Derive namespace from repo root or explicit namespace."""
        if namespace:
            return namespace
        if repo_root:
            name = Path(repo_root).resolve().name
            return re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        return "default"

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> Path:
        path = _ALIAS_TO_FILE.get(path, path)
        if not path.startswith("wiki/"):
            path = "wiki/" + path
        target = (self._root / path).resolve()
        if not str(target).startswith(str(self._root)):
            raise ValueError(f"Path {path!r} escapes memory root")
        return target

    def _build_frontmatter(self, title: str, category: str | None,
                           tags: list[str], scope: str, source_run: str) -> str:
        lines = ["---"]
        if title:
            lines.append(f"title: {title}")
        if category:
            lines.append(f"category: {category}")
        if tags:
            lines.append(f"tags: [{', '.join(tags)}]")
        lines.append(f"scope: {scope}")
        if source_run:
            lines.append(f"source_run: {source_run}")
        lines.append(f"created: {_now_iso()}")
        lines.append("---")
        return "\n".join(lines)

    def _parse_frontmatter(self, path: Path) -> dict:
        """Parse YAML frontmatter from a wiki page."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return {}
        if not text.startswith("---"):
            return {}
        end = text.find("---", 3)
        if end == -1:
            return {}
        info = {}
        for line in text[3:end].strip().split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()
        return info

    def _update_index(self) -> None:
        """Regenerate INDEX.md from wiki pages."""
        wiki_dir = self._root / "wiki"
        if not wiki_dir.is_dir():
            return
        entries = []
        for md in sorted(wiki_dir.rglob("*.md")):
            if md.name == "INDEX.md":
                continue
            rel = md.relative_to(wiki_dir)
            info = self._parse_frontmatter(md)
            title = info.get("title", rel.stem)
            entries.append(f"- [{title}]({rel})")
        index_path = wiki_dir / "INDEX.md"
        index_path.write_text("# Wiki Index\n\n" + "\n".join(entries) + "\n", encoding="utf-8")

    def _update_signal(self, category: str, ref_path: str, title: str) -> None:
        """Append a reference to the category signal page."""
        signal_name = _CATEGORY_SIGNAL_MAP.get(category)
        if not signal_name:
            return
        signal_path = self._root / "wiki" / "signals" / f"{signal_name}.md"
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        entry = f"- [{title}](../{ref_path})\n"
        if signal_path.exists():
            content = signal_path.read_text(encoding="utf-8")
            if entry.strip() not in content:
                signal_path.write_text(content + entry, encoding="utf-8")
        else:
            signal_path.write_text(f"# {signal_name}\n\n{entry}", encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
