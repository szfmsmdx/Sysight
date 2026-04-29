"""Storage helpers for runtime `.sysight/memory` files.

Supports:
- flat workspace.md / experience.md (legacy)
- wiki layering: wiki/workspaces/<ns>/, wiki/experiences/<slug>.md, wiki/signals/<Cn>.md
- raw run trace: raw/runs/<run_id>/
- graph meta: backlinks, references, stale markers in INDEX.md
- namespace isolation for benchmark cases
- memory brief generation for prompt injection
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

_ALIAS_TO_FILE = {
    "workspace": "workspace.md",
    "experience": "experience.md",
}

_TEXT_EXTS = {".md", ".txt", ".json", ".yml", ".yaml", ".log"}

_CATEGORY_SIGNAL_MAP: dict[str, str] = {
    "C1": "C1-host-scheduling",
    "C2": "C2-kernel-launch-overhead",
    "C3": "C3-synchronization",
    "C4": "C4-memory-copy",
    "C5": "C5-compute-inefficiency",
    "C6": "C6-communication",
    "C7": "C7-python-pipeline",
}

_SEPARATOR = "\n---\n"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LineEntry:
    line: int
    text: str


@dataclass
class MemoryReadResult:
    root: str
    path: str
    abs_path: str
    total_lines: int
    shown_start: int
    shown_end: int
    lines: list[LineEntry] = field(default_factory=list)
    error: str | None = None


@dataclass
class MemorySearchMatch:
    path: str
    line: int
    column: int
    text: str


@dataclass
class MemorySearchResult:
    root: str
    query: str
    total_matches: int
    matches: list[MemorySearchMatch] = field(default_factory=list)


@dataclass
class MemoryWriteResult:
    root: str
    path: str
    abs_path: str
    bytes_written: int
    append: bool
    error: str | None = None


# ── Root / path helpers ───────────────────────────────────────────────────────

def default_memory_root() -> Path:
    return (Path.cwd() / ".sysight" / "memory").resolve()


def _resolve_root(root: str | Path | None) -> Path:
    return Path(root).resolve() if root is not None else default_memory_root()


def _normalize_target_path(path: str) -> str:
    return _ALIAS_TO_FILE.get(path, path)


def resolve_memory_path(root: str | Path | None, path: str) -> tuple[Path, str]:
    base = _resolve_root(root)
    target = Path(_normalize_target_path(path))
    if target.is_absolute():
        resolved = target.resolve()
    else:
        resolved = (base / target).resolve()
    return base, str(target).replace("\\", "/"), resolved


def _ensure_inside_root(base: Path, target: Path) -> str | None:
    try:
        target.relative_to(base)
    except ValueError:
        return f"path is outside memory root: {target}"
    return None


# ── Namespace ─────────────────────────────────────────────────────────────────

def workspace_namespace(*, repo_root: str | None = None, namespace: str | None = None) -> str:
    """Return a stable namespace string for wiki/workspaces/<ns>.

    If *namespace* is given directly (e.g. ``bench/case_1``), use it as-is
    (the ``/`` creates a sub-directory tree under wiki/workspaces/).
    Otherwise derive from *repo_root* via a short hash so that different
    repos get isolated wiki trees.
    """
    if namespace:
        return namespace
    if repo_root:
        digest = hashlib.sha256(repo_root.encode()).hexdigest()[:12]
        return f"repo-{digest}"
    return "default"


# ── Slug helper ───────────────────────────────────────────────────────────────

def _title_to_slug(title: str) -> str:
    """Convert an experience title to a filesystem slug."""
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


# ── Read ──────────────────────────────────────────────────────────────────────

def read_memory_file(
    root: str | Path | None,
    path: str,
    start: int | None = None,
    end: int | None = None,
    around: int | None = None,
    context: int = 10,
) -> MemoryReadResult:
    base, rel_path, target = resolve_memory_path(root, path)
    error = _ensure_inside_root(base, target)
    if error:
        return MemoryReadResult(str(base), rel_path, str(target), 0, 0, 0, error=error)
    if not target.is_file():
        return MemoryReadResult(str(base), rel_path, str(target), 0, 0, 0, error=f"file not found: {target}")
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return MemoryReadResult(str(base), rel_path, str(target), 0, 0, 0, error=str(exc))

    all_lines = raw.splitlines()
    total = len(all_lines)
    if around is not None:
        lo = max(1, around - context)
        hi = min(total, around + context)
    else:
        lo = max(1, start if start is not None else 1)
        hi = min(total, end if end is not None else total)
    lines = [LineEntry(line=i, text=all_lines[i - 1]) for i in range(lo, hi + 1)] if total else []
    shown_start = lo if lines else 0
    shown_end = hi if lines else 0
    return MemoryReadResult(str(base), str(target.relative_to(base)), str(target), total, shown_start, shown_end, lines)


# ── Search ────────────────────────────────────────────────────────────────────

def _iter_search_targets(base: Path, scope: str | None, namespace: str | None = None):
    """Yield file paths to search, optionally constrained by scope and namespace."""
    # If namespace is given, constrain workspace scope to that namespace dir
    if scope == "workspace" and namespace:
        ns = workspace_namespace(namespace=namespace)
        ns_dir = base / "wiki" / "workspaces" / ns
        if ns_dir.is_dir():
            for p in sorted(ns_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in _TEXT_EXTS:
                    yield p
            return
        # fallback: also try flat workspace.md for backwards compat
        flat = base / "workspace.md"
        if flat.is_file():
            yield flat
        return

    if scope in _ALIAS_TO_FILE:
        target = base / _ALIAS_TO_FILE[scope]
        if target.is_file():
            yield target
        return
    if not base.exists():
        return
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in _TEXT_EXTS:
            yield path


def search_memory(
    root: str | Path | None,
    query: str,
    scope: str | None = None,
    fixed: bool = False,
    case_sensitive: bool = True,
    max_results: int = 200,
    namespace: str | None = None,
) -> MemorySearchResult:
    base = _resolve_root(root)
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags) if fixed else None
    if pattern is None:
        try:
            pattern = re.compile(query, flags)
        except re.error:
            pattern = re.compile(re.escape(query), flags)

    matches: list[MemorySearchMatch] = []
    for path in _iter_search_targets(base, scope, namespace=namespace):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            found = pattern.search(line)
            if not found:
                continue
            matches.append(
                MemorySearchMatch(
                    path=str(path.relative_to(base)),
                    line=lineno,
                    column=found.start() + 1,
                    text=line.rstrip(),
                )
            )
            if len(matches) >= max_results:
                return MemorySearchResult(str(base), query, len(matches), matches)
    return MemorySearchResult(str(base), query, len(matches), matches)


# ── Write ─────────────────────────────────────────────────────────────────────

def write_memory_file(
    root: str | Path | None,
    path: str,
    content: str,
    *,
    append: bool = False,
) -> MemoryWriteResult:
    base, rel_path, target = resolve_memory_path(root, path)
    error = _ensure_inside_root(base, target)
    if error:
        return MemoryWriteResult(str(base), rel_path, str(target), 0, append, error=error)
    try:
        base.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if append and target.exists():
            existing = target.read_text(encoding="utf-8", errors="replace")
        target.write_text(existing + content if append else content, encoding="utf-8")
    except OSError as exc:
        return MemoryWriteResult(str(base), rel_path, str(target), 0, append, error=str(exc))
    return MemoryWriteResult(
        root=str(base),
        path=str(target.relative_to(base)),
        abs_path=str(target),
        bytes_written=len(content.encode("utf-8")),
        append=append,
    )


# ── Wiki page writers ─────────────────────────────────────────────────────────

def _write_workspace_page(base: Path, ns: str, content: str, category: str | None = None) -> None:
    """Write or append content into wiki/workspaces/<ns>/overview.md."""
    page = base / "wiki" / "workspaces" / ns / "overview.md"
    page.parent.mkdir(parents=True, exist_ok=True)

    section_label = f" ({category})" if category else ""
    header = f"## Workspace{section_label}\n"
    block = header + content.rstrip() + "\n"

    if page.exists():
        existing = page.read_text(encoding="utf-8", errors="replace")
        # dedup: skip if exact content already present
        if content.strip() and content.strip() in existing:
            return
        page.write_text(existing.rstrip() + _SEPARATOR + block, encoding="utf-8")
    else:
        front_matter = (
            f"# Workspace: {ns}\n\n"
            f"namespace: {ns}\n"
            f"created: {_now_iso()}\n\n"
        )
        page.write_text(front_matter + block, encoding="utf-8")


def _write_experience_page(
    base: Path,
    title: str,
    content: str,
    *,
    category: str | None = None,
    tags: list[str] | None = None,
    raw_run_manifest: str | None = None,
) -> None:
    """Write an experience wiki page at wiki/experiences/<slug>.md.

    If the page already exists with identical content, skip (dedup).
    Otherwise append a revision with timestamp.
    """
    slug = _title_to_slug(title)
    page = base / "wiki" / "experiences" / f"{slug}.md"
    page.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata header
    meta_lines = [
        f"title: {title}",
        f"category: {category or 'unknown'}",
        f"tags: {', '.join(tags) if tags else ''}",
        f"updated: {_now_iso()}",
    ]
    if raw_run_manifest:
        meta_lines.append(f"raw_run: {raw_run_manifest}")
    meta = "\n".join(f"- {l}" for l in meta_lines if l)

    body = content.rstrip() + "\n"

    if page.exists():
        existing = page.read_text(encoding="utf-8", errors="replace")
        # exact dedup
        if body.strip() and body.strip() in existing:
            return
        # append revision
        revision = (
            f"{_SEPARATOR}"
            f"## Revision {_now_iso()}\n\n"
            f"{meta}\n\n"
            f"{body}"
        )
        page.write_text(existing.rstrip() + "\n" + revision, encoding="utf-8")
    else:
        front = (
            f"# {title}\n\n"
            f"{meta}\n\n"
            f"{body}"
        )
        page.write_text(front, encoding="utf-8")

    # Update signal page backlink
    if category and category in _CATEGORY_SIGNAL_MAP:
        _update_signal_page(base, category, f"wiki/experiences/{slug}.md", title)


def _update_signal_page(base: Path, category: str, experience_ref: str, title: str) -> None:
    """Append a backlink entry to wiki/signals/<Cn>-<name>.md."""
    signal_key = _CATEGORY_SIGNAL_MAP.get(category)
    if not signal_key:
        return
    page = base / "wiki" / "signals" / f"{signal_key}.md"
    page.parent.mkdir(parents=True, exist_ok=True)

    entry = f"- [{title}]({experience_ref})\n"

    if page.exists():
        existing = page.read_text(encoding="utf-8", errors="replace")
        if experience_ref in existing:
            return
        page.write_text(existing.rstrip() + "\n" + entry, encoding="utf-8")
    else:
        label = signal_key.split("-", 1)[1].replace("-", " ").title()
        front = (
            f"# Signal: {label}\n\n"
            f"category: {category}\n\n"
            f"## References\n\n"
            f"{entry}"
        )
        page.write_text(front, encoding="utf-8")


def _update_index(base: Path) -> None:
    """Rebuild wiki/INDEX.md with page listing and stale-page markers.

    A page is stale if it hasn't been updated in >30 days (heuristic).
    """
    wiki = base / "wiki"
    if not wiki.is_dir():
        return

    index = wiki / "INDEX.md"
    lines: list[str] = [
        "# Memory Index\n",
        f"updated: {_now_iso()}\n",
    ]

    # Collect pages
    pages: list[Path] = sorted(wiki.rglob("*.md"))
    pages = [p for p in pages if p.name != "INDEX.md"]

    # Workspace pages
    lines.append("## Workspaces\n")
    for p in pages:
        rel = str(p.relative_to(wiki)).replace("\\", "/")
        if rel.startswith("workspaces/"):
            lines.append(f"- [{rel}]({rel})")
    lines.append("")

    # Experience pages
    lines.append("## Experiences\n")
    for p in pages:
        rel = str(p.relative_to(wiki)).replace("\\", "/")
        if rel.startswith("experiences/"):
            lines.append(f"- [{rel}]({rel})")
    lines.append("")

    # Signal pages
    lines.append("## Signals\n")
    for p in pages:
        rel = str(p.relative_to(wiki)).replace("\\", "/")
        if rel.startswith("signals/"):
            lines.append(f"- [{rel}]({rel})")
    lines.append("")

    # Stale Pages (simple heuristic: check mtime > 30 days)
    lines.append("## Stale Pages\n")
    stale_count = 0
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    for p in pages:
        try:
            age_days = (now_ts - p.stat().st_mtime) / 86400
        except OSError:
            continue
        if age_days > 30:
            rel = str(p.relative_to(wiki)).replace("\\", "/")
            lines.append(f"- {rel} (last updated {age_days:.0f} days ago)")
            stale_count += 1
    if stale_count == 0:
        lines.append("- (none)")

    index.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Apply memory updates (central write-back) ────────────────────────────────

def apply_memory_updates(
    root: str | Path | None,
    updates: list[dict[str, object]],
    *,
    repo_root: str | None = None,
    namespace: str | None = None,
    raw_run: dict[str, object] | None = None,
) -> None:
    """Apply structured memory_updates to wiki pages.

    Each update dict may contain:
      - path: "workspace" | "experience" | relative path
      - content: text to write
      - append: bool (default True for workspace/experience)
      - title: str (for experience pages → slug)
      - category: "C1"-"C7" (for signal page backlinks)
      - tags: list[str]
    """
    base = _resolve_root(root)
    ns = workspace_namespace(repo_root=repo_root, namespace=namespace)
    raw_manifest = raw_run.get("manifest_path") if raw_run and isinstance(raw_run, dict) else None

    # Write raw run manifest if provided
    if raw_run and isinstance(raw_run, dict):
        run_id = raw_run.get("run_id")
        manifest_path_str = raw_run.get("manifest_path")
        if run_id and manifest_path_str:
            manifest_file = base / manifest_path_str
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                import json as _json
                manifest_file.write_text(
                    _json.dumps(raw_run, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except OSError:
                pass

    wiki_touched = False

    for upd in updates:
        if not isinstance(upd, dict):
            continue
        path = str(upd.get("path") or "").strip()
        content = str(upd.get("content") or "").strip()
        if not path or not content:
            continue

        append = bool(upd.get("append", True))
        title = str(upd.get("title") or "").strip()
        category = str(upd.get("category") or "").strip()
        tags = upd.get("tags")
        if not isinstance(tags, list):
            tags = []

        if path == "workspace":
            # Write to wiki namespace page AND flat legacy file
            _write_workspace_page(base, ns, content, category=category or None)
            write_memory_file(str(base), "workspace.md", _SEPARATOR + content, append=True)
            wiki_touched = True

        elif path == "experience":
            if title:
                _write_experience_page(
                    base, title, content,
                    category=category or None,
                    tags=tags,
                    raw_run_manifest=raw_manifest,
                )
            else:
                # No title → fall back to flat file
                write_memory_file(str(base), "experience.md", _SEPARATOR + content, append=True)
            wiki_touched = True

        else:
            # Arbitrary relative path
            write_memory_file(str(base), path, content, append=append)

    if wiki_touched:
        _update_index(base)


# ── Memory Brief ──────────────────────────────────────────────────────────────

def build_memory_brief(
    root: str | Path | None,
    *,
    repo_root: str | None = None,
    namespace: str | None = None,
) -> str:
    """Build a short MEMORY_BRIEF string for prompt injection.

    Contains:
    - namespace / workspace path
    - top signal pages that exist
    - recent experience titles (up to 5)
    - raw run manifest if available
    - CLI usage instructions
    """
    base = _resolve_root(root)
    ns = workspace_namespace(repo_root=repo_root, namespace=namespace)
    wiki = base / "wiki"

    lines: list[str] = [
        f"namespace: {ns}",
        f"workspace_path: wiki/workspaces/{ns}/overview.md",
    ]

    # Signal pages
    signal_dir = wiki / "signals"
    if signal_dir.is_dir():
        for sp in sorted(signal_dir.glob("*.md")):
            rel = f"wiki/signals/{sp.name}"
            lines.append(f"signal: {rel}")

    # Recent experience titles (up to 5)
    exp_dir = wiki / "experiences"
    if exp_dir.is_dir():
        exps = sorted(exp_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        for ep in exps[:5]:
            rel = f"wiki/experiences/{ep.name}"
            # extract title from first line
            try:
                first = ep.read_text(encoding="utf-8", errors="replace").splitlines()[0]
                title = first.lstrip("# ").strip()
            except (OSError, IndexError):
                title = ep.stem
            lines.append(f"experience: {title} → {rel}")

    # Raw run
    raw_dir = base / "raw" / "runs"
    if raw_dir.is_dir():
        run_dirs = sorted(raw_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_dirs:
            latest = run_dirs[0]
            mf = latest / "manifest.json"
            lines.append(f"latest_raw_run: raw/runs/{latest.name}/manifest.json" if mf.exists() else f"latest_raw_run: raw/runs/{latest.name}/")

    lines.append("")
    lines.append("查询工具：")
    lines.append("  python3 -m sysight.analyzer.cli memory search <query> --root .sysight/memory")
    lines.append("  python3 -m sysight.analyzer.cli memory read <path> --root .sysight/memory")

    return "\n".join(lines)


# ── Utility ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
