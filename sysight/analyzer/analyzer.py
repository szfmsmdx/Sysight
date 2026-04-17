"""analyzer — static source-code analysis core.

Scope-aware three-stage pipeline for Python and C/C++/CUDA repos:
  Stage 1 — discover_repo()     : path walk, no file reads
  Stage 2 — scan_repo(scope=)   : targeted parse of relevant files
  Stage 3 — get_repo_context()  : bounded source snippets for optimizer

To add a language scanner: subclass BaseScanner in scanners/<lang>.py,
then append to SCANNERS below.
"""

from __future__ import annotations

import difflib
import logging
import os
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .scanners import (
    BaseScanner, CallSiteFacts, FileFacts, FunctionFacts,
    PythonScanner, CppScanner,
)
from .scanners.base import NON_ENTRY_NAMES, should_ignore

logger = logging.getLogger(__name__)

# ── Scanner registry ─────────────────────────────────────────────────────────
# To add a language: append its class here.  No other changes needed.

SCANNERS: list[type[BaseScanner]] = [
    PythonScanner,
    CppScanner,
]

# Extensions that indicate GPU/ML relevance
_GPU_EXTENSIONS = frozenset({".cu", ".cuh"})
_ML_EXTENSIONS = frozenset({".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"})
_GPU_FILE_HINTS = frozenset({
    "kernel", "cuda", "gpu", "triton", "nccl", "nvtx",
    "train", "trainer", "infer", "inference",
})

# ── Scope & Budget data structures ───────────────────────────────────────────

@dataclass
class RepoScope:
    """Controls which files are scanned.  Default mode is 'targeted'."""
    mode: str = "targeted"              # "targeted" | "entry" | "full"
    seed_files: list[str] = field(default_factory=list)     # exact relative paths
    seed_symbols: list[str] = field(default_factory=list)   # function/kernel names
    seed_kernels: list[str] = field(default_factory=list)   # GPU kernel names
    include_extensions: set[str] = field(
        default_factory=lambda: {".py", ".cpp", ".cc", ".cxx", ".c",
                                 ".h", ".hpp", ".cu", ".cuh"}
    )
    follow_imports_depth: int = 1
    max_files: int = 500
    max_file_bytes: int = 512_000
    include_gpu_related: bool = True
    include_runfiles: bool = False


@dataclass
class ContextBudget:
    """Limits how much context the optimizer receives."""
    max_files: int = 30
    max_functions: int = 80
    max_lines_per_function: int = 80
    max_total_chars: int = 80_000


@dataclass
class RepoManifest:
    """Stage-1 output: path metadata only, no source content."""
    repo_root: str
    files: list[str]                        # all discoverable source paths (relative)
    languages: dict[str, int]               # ext -> file count
    candidate_gpu_files: list[str]          # .cu/.cuh or GPU-hinted filenames
    candidate_entry_files: list[str]        # training/inference entry-point hints
    ignored_dir_count: int
    warnings: list[str]


# ── New index & lookup data structures ───────────────────────────────────────

@dataclass
class RepoIndex:
    """Pre-built lookup tables for fast mapping queries."""
    files: dict[str, FileFacts]
    # basename (e.g. "train.py")  →  [full_relative_path, ...]
    by_basename: dict[str, list[str]]
    # short symbol name  →  [FunctionFacts, ...]  (may span multiple files)
    by_symbol: dict[str, list[FunctionFacts]]
    # qualified_name  →  FunctionFacts  (unique)
    by_qualified: dict[str, FunctionFacts]
    # qualified_name  →  relative path
    symbol_to_file: dict[str, str]


@dataclass
class FileMatch:
    path: str
    confidence: float           # 0.0–1.0
    reason: str                 # "exact" | "suffix" | "basename" | "fuzzy"
    alternatives: list[str]


@dataclass
class LocationMatch:
    repo_file: str | None
    function: FunctionFacts | None
    line: int | None
    confidence: float
    reason: str                 # "file_line" | "symbol" | "basename_symbol" | "none"
    alternatives: list[str]


# ── Existing data models (unchanged) ─────────────────────────────────────────

@dataclass
class EntryPoint:
    path: str
    module_name: str
    mode: str
    score: int
    reasons: list[str]
    start_symbols: list[str]


@dataclass
class CallStep:
    from_symbol: str
    to_symbol: str
    to_file: str
    depth: int


@dataclass
class CallChain:
    entry_path: str
    entry_symbol: str
    visited_files: list[str]
    visited_symbols: list[str]
    steps: list[CallStep]
    truncated: bool


@dataclass
class HubNode:
    path: str
    symbol: str
    in_degree: int
    out_degree: int
    total_degree: int


@dataclass
class SearchResult:
    path: str
    symbol: str
    kind: str       # "function" | "file"
    line: int
    score: float


@dataclass
class ImpactResult:
    seed_files: list[str]
    impacted_files: list[str]
    impacted_symbols: list[str]
    depth_map: dict[str, int]
    truncated: bool


@dataclass
class AnalysisResult:
    repo_root: str
    source_files: int
    languages: dict[str, int]   # lang -> file count
    entry_points: list[EntryPoint]
    call_chains: list[CallChain]
    hub_nodes: list[HubNode]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ── File-level DAG ────────────────────────────────────────────────────────────

@dataclass
class FileDAG:
    """Directed file-import graph.  Edge A->B means A imports B.
    Files with in-degree 0 are candidate entry points.
    """
    edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    nodes: set[str] = field(default_factory=set)

    def add_node(self, p: str) -> None:
        self.nodes.add(p)

    def add_edge(self, a: str, b: str) -> None:
        self.nodes.update((a, b))
        self.edges[a].add(b)

    def in_degrees(self) -> dict[str, int]:
        d: dict[str, int] = {n: 0 for n in self.nodes}
        for targets in self.edges.values():
            for t in targets:
                d[t] = d.get(t, 0) + 1
        return d

    def zero_indegree(self) -> set[str]:
        return {n for n, v in self.in_degrees().items() if v == 0}

    def reachable_from(self, start: str, max_depth: int = 10) -> list[tuple[str, int]]:
        visited: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        result: list[tuple[str, int]] = []
        while queue:
            node, depth = queue.popleft()
            if node in visited:
                continue
            visited[node] = depth
            result.append((node, depth))
            if depth < max_depth:
                for child in sorted(self.edges.get(node, [])):
                    if child not in visited:
                        queue.append((child, depth + 1))
        return result


# ── Stage 1: Manifest-only discovery ─────────────────────────────────────────

def discover_repo(repo_root: Path) -> RepoManifest:
    """Enumerate source paths without reading file contents.

    Cheap operation: suitable for large repos to determine scope before any
    file reads. Returns a RepoManifest with path-level metadata only.
    """
    root = repo_root.resolve()
    all_scanner_exts: set[str] = set()
    for cls in SCANNERS:
        all_scanner_exts.update(cls.EXTENSIONS)

    files: list[str] = []
    ext_counts: dict[str, int] = defaultdict(int)
    candidate_gpu: list[str] = []
    candidate_entry: list[str] = []
    ignored_dirs: set[str] = set()
    warnings: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        # Prune ignored directories in-place
        filtered: list[str] = []
        for d in dirnames:
            child = dir_path / d
            if should_ignore(root, child):
                ignored_dirs.add(str(child.relative_to(root)))
            else:
                filtered.append(d)
        dirnames[:] = filtered

        for fname in filenames:
            fpath = dir_path / fname
            if should_ignore(root, fpath):
                continue
            suffix = fpath.suffix.lower()
            if suffix not in all_scanner_exts:
                continue
            try:
                rel = str(fpath.relative_to(root))
            except ValueError:
                continue

            files.append(rel)
            ext_counts[suffix] += 1

            # Candidate GPU files: .cu/.cuh or GPU-hinted basename
            stem_lower = fpath.stem.lower()
            if suffix in _GPU_EXTENSIONS or any(h in stem_lower for h in _GPU_FILE_HINTS):
                candidate_gpu.append(rel)

            # Candidate entry files: training/inference hints
            if any(h in stem_lower for h in {
                "train", "trainer", "pretrain", "finetune",
                "infer", "inference", "predict", "main", "run",
            }):
                candidate_entry.append(rel)

    # Convert ext counts to language distribution
    ext_to_lang = {
        ".py": "python",
        ".c": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp",
        ".h": "cpp", ".hh": "cpp", ".hpp": "cpp", ".hxx": "cpp",
        ".cu": "cuda", ".cuh": "cuda",
    }
    lang_counts: dict[str, int] = defaultdict(int)
    for ext, cnt in ext_counts.items():
        lang = ext_to_lang.get(ext, ext.lstrip("."))
        lang_counts[lang] += cnt

    return RepoManifest(
        repo_root=str(root),
        files=sorted(files),
        languages=dict(sorted(lang_counts.items())),
        candidate_gpu_files=sorted(candidate_gpu),
        candidate_entry_files=sorted(candidate_entry),
        ignored_dir_count=len(ignored_dirs),
        warnings=warnings,
    )


# ── Stage 2: Scope-aware file parsing ────────────────────────────────────────

def _resolve_scope_files(
    repo_root: Path,
    scope: RepoScope,
    manifest: RepoManifest | None = None,
) -> set[str]:
    """Determine which relative paths to parse given a RepoScope."""
    root = repo_root.resolve()
    all_files: list[str] = (
        manifest.files
        if manifest
        else [
            str(p.relative_to(root))
            for cls in SCANNERS
            for p in root.rglob("*")
            if p.suffix.lower() in cls.EXTENSIONS
            and not should_ignore(root, p, include_runfiles=scope.include_runfiles)
        ]
    )

    if scope.mode == "full":
        selected = set(all_files)
        if scope.max_files and len(selected) > scope.max_files:
            logger.warning(
                "full scan: %d files exceed max_files=%d; truncating",
                len(selected), scope.max_files,
            )
            selected = set(sorted(selected)[:scope.max_files])
        return selected

    # Build helper maps (computed once)
    all_files_set = set(all_files)
    basename_to_paths: dict[str, list[str]] = defaultdict(list)
    for rel in all_files:
        basename_to_paths[Path(rel).name].append(rel)

    selected: set[str] = set()

    # 1. Exact seed_files + suffix fallback
    for sf in scope.seed_files:
        if sf in all_files_set:
            selected.add(sf)
        else:
            sf_norm = sf.lstrip("/").replace("\\", "/")
            for rel in all_files:
                if rel.replace("\\", "/").endswith(sf_norm):
                    selected.add(rel)
                    break

    # 2. seed_files by basename
    for sf in scope.seed_files:
        bn = Path(sf).name
        for rel in basename_to_paths.get(bn, []):
            selected.add(rel)

    # 3. seed_symbols / seed_kernels: add all files whose basename contains hint
    for sym in scope.seed_symbols + scope.seed_kernels:
        sym_lower = sym.lower()
        for rel in all_files:
            stem = Path(rel).stem.lower()
            if sym_lower in stem or stem in sym_lower:
                selected.add(rel)

    # 4. GPU-related files (if include_gpu_related)
    if scope.include_gpu_related:
        for rel in all_files:
            p = Path(rel)
            if p.suffix.lower() in _GPU_EXTENSIONS:
                selected.add(rel)
            elif any(h in p.stem.lower() for h in _GPU_FILE_HINTS):
                selected.add(rel)

    # 5. Same-directory neighbors of already-selected files
    if scope.mode == "targeted":
        dirs_of_selected = {str(Path(rel).parent) for rel in selected}
        for rel in all_files:
            if str(Path(rel).parent) in dirs_of_selected:
                selected.add(rel)

    # 6. Apply extension filter
    selected = {
        rel for rel in selected
        if Path(rel).suffix.lower() in scope.include_extensions
    }

    # 7. Apply file size limit
    oversized: list[str] = []
    for rel in list(selected):
        try:
            size = (root / rel).stat().st_size
            if size > scope.max_file_bytes:
                oversized.append(rel)
                selected.discard(rel)
        except OSError:
            pass
    if oversized:
        logger.debug("scope: skipped %d oversized files", len(oversized))

    # 8. Cap total files
    if len(selected) > scope.max_files:
        logger.warning(
            "scope: %d files selected, capping to max_files=%d",
            len(selected), scope.max_files,
        )
        selected = set(sorted(selected)[:scope.max_files])

    return selected


def scan_repo(
    repo_root: Path,
    scope: RepoScope | None = None,
    manifest: RepoManifest | None = None,
) -> tuple[dict[str, FileFacts], list[str]]:
    """Parse selected source files and return (files, warnings).

    scope=None  → full scan of all languages (backward-compatible).
    scope set   → targeted parse; only files in the resolved scope are read.
    """
    if scope is None:
        # Backward-compatible full scan
        all_files: dict[str, FileFacts] = {}
        warnings: list[str] = []
        for cls in SCANNERS:
            scanner = cls(repo_root)
            all_files.update(scanner.scan())
            warnings.extend(scanner.warnings)
        return all_files, warnings

    selected = _resolve_scope_files(repo_root, scope, manifest)
    logger.debug("scan_repo: scope=%s, selected=%d files", scope.mode, len(selected))

    # Build ext → scanner-class map
    ext_to_cls: dict[str, type[BaseScanner]] = {
        ext: cls for cls in SCANNERS for ext in cls.EXTENSIONS
    }

    # Group selected relative paths by scanner class → convert to absolute Paths
    cls_to_abspaths: dict[type[BaseScanner], list[Path]] = defaultdict(list)
    for rel in selected:
        cls = ext_to_cls.get(Path(rel).suffix.lower())
        if cls:
            cls_to_abspaths[cls].append(repo_root / rel)

    # Use scan_paths() for truly targeted parsing — no rglob, no post-filter.
    all_parsed: dict[str, FileFacts] = {}
    warnings: list[str] = []
    for cls, abs_paths in cls_to_abspaths.items():
        scanner = cls(repo_root, include_runfiles=scope.include_runfiles)
        all_parsed.update(scanner.scan_paths(abs_paths))
        warnings.extend(scanner.warnings)

    return all_parsed, warnings


# ── RepoIndex construction ────────────────────────────────────────────────────

def build_repo_index(files: dict[str, FileFacts]) -> RepoIndex:
    """Build lookup tables from parsed FileFacts.

    Intentionally omits by_suffix to avoid memory amplification on large repos.
    Suffix-match logic is done on-the-fly in fuzzy_file_match() using by_basename.
    """
    by_basename: dict[str, list[str]] = defaultdict(list)
    by_symbol: dict[str, list[FunctionFacts]] = defaultdict(list)
    by_qualified: dict[str, FunctionFacts] = {}
    symbol_to_file: dict[str, str] = {}

    for path, facts in files.items():
        by_basename[Path(path).name].append(path)
        for fn in facts.functions.values():
            by_symbol[fn.name].append(fn)
            by_qualified[fn.qualified_name] = fn
            symbol_to_file[fn.qualified_name] = path

    return RepoIndex(
        files=files,
        by_basename=dict(by_basename),
        by_symbol=dict(by_symbol),
        by_qualified=by_qualified,
        symbol_to_file=symbol_to_file,
    )


def build_dag(files: dict[str, FileFacts]) -> FileDAG:
    dag = FileDAG()
    for facts in files.values():
        dag.add_node(facts.path)
        for b in facts.imports.values():
            if b.target_file and b.target_file in files:
                dag.add_edge(facts.path, b.target_file)
    logger.debug("DAG: %d nodes, %d edges", len(dag.nodes),
                 sum(len(v) for v in dag.edges.values()))
    return dag


def lang_distribution(files: dict[str, FileFacts]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for f in files.values():
        counts[f.language] += 1
    return dict(sorted(counts.items()))


# ── Fuzzy file match ──────────────────────────────────────────────────────────

def fuzzy_file_match(
    path_or_basename: str,
    index: RepoIndex,
) -> FileMatch | None:
    """Find the best matching file in the index.

    Priority:
      1. Repo-relative exact match
      2. Normalized path-suffix match (handle absolute nsys paths)
      3. Basename unique match
      4. Basename + difflib fuzzy match
      5. Returns None if nothing found with confidence > 0
    """
    if not path_or_basename:
        return None

    query = path_or_basename.replace("\\", "/")
    all_paths = list(index.files.keys())

    # 1. Exact match
    if query in index.files:
        return FileMatch(path=query, confidence=1.0, reason="exact", alternatives=[])

    # Normalize query: strip leading slash and common prefixes
    q_norm = query.lstrip("/")

    # 2. Longest-suffix match: try progressively shorter path suffixes from the query.
    # This handles nsys absolute paths like /home/user/project/model/layers.py where
    # the repo contains model/layers.py.  A k-part suffix match beats a mere basename.
    query_parts = Path(query).parts
    max_k = min(len(query_parts), 8)
    for k in range(max_k, 0, -1):
        suffix = "/".join(query_parts[-k:])
        hits = [p for p in all_paths if p.replace("\\", "/").endswith(suffix)]
        if not hits:
            continue
        if len(hits) == 1:
            conf = 0.95 if k >= 2 else 0.85
            return FileMatch(path=hits[0], confidence=conf, reason="suffix",
                             alternatives=[])
        # Multiple hits — pick shortest (most specific) with others as alternatives
        hits.sort(key=len)
        conf = 0.80 if k >= 2 else 0.70
        return FileMatch(path=hits[0], confidence=conf, reason="suffix",
                         alternatives=hits[1:5])

    # 3. Basename unique match
    basename = Path(query).name
    bn_hits = index.by_basename.get(basename, [])
    if len(bn_hits) == 1:
        return FileMatch(path=bn_hits[0], confidence=0.80, reason="basename",
                         alternatives=[])
    if len(bn_hits) > 1:
        # Among basename candidates, pick the one whose parent path suffix best
        # matches the query directory portion
        query_dir = str(Path(query).parent).replace("\\", "/")
        scored = sorted(
            bn_hits,
            key=lambda p: difflib.SequenceMatcher(
                None, query_dir, str(Path(p).parent).replace("\\", "/")
            ).ratio(),
            reverse=True,
        )
        best_score = difflib.SequenceMatcher(
            None, query_dir,
            str(Path(scored[0]).parent).replace("\\", "/"),
        ).ratio()
        return FileMatch(
            path=scored[0],
            confidence=max(0.5, 0.75 * best_score),
            reason="basename",
            alternatives=scored[1:5],
        )

    # 4. difflib fuzzy across all paths
    close = difflib.get_close_matches(q_norm, all_paths, n=5, cutoff=0.4)
    if close:
        return FileMatch(path=close[0], confidence=0.4, reason="fuzzy",
                         alternatives=close[1:])

    return None


# ── File-line and symbol lookup ───────────────────────────────────────────────

def lookup_by_file_line(
    source_file: str,
    line: int,
    index: RepoIndex,
) -> LocationMatch:
    """Locate the function that contains (source_file, line).

    Requires FunctionFacts.end_line to be populated; falls back to nearest
    function start if end_line is missing.
    """
    fm = fuzzy_file_match(source_file, index)
    if fm is None or fm.confidence < 0.3:
        return LocationMatch(
            repo_file=None, function=None, line=line, confidence=0.0,
            reason="none", alternatives=[],
        )

    facts = index.files.get(fm.path)
    if facts is None:
        return LocationMatch(
            repo_file=fm.path, function=None, line=line, confidence=0.3,
            reason="none", alternatives=fm.alternatives,
        )

    best_fn: FunctionFacts | None = None
    # Priority 1: exact range match with end_line
    for fn in facts.functions.values():
        if fn.end_line is not None:
            if fn.line <= line <= fn.end_line:
                if best_fn is None or (fn.end_line - fn.line) < (best_fn.end_line - best_fn.line):  # type: ignore[operator]
                    best_fn = fn
    if best_fn is not None:
        return LocationMatch(
            repo_file=fm.path,
            function=best_fn,
            line=line,
            confidence=min(1.0, fm.confidence),
            reason="file_line",
            alternatives=fm.alternatives,
        )

    # Priority 2: nearest function start (for scanners without end_line)
    candidates = sorted(
        (fn for fn in facts.functions.values() if fn.line <= line),
        key=lambda fn: line - fn.line,
    )
    if candidates:
        best_fn = candidates[0]
        return LocationMatch(
            repo_file=fm.path,
            function=best_fn,
            line=line,
            confidence=min(0.7, fm.confidence * 0.85),
            reason="file_line",
            alternatives=fm.alternatives,
        )

    # File matched but no function contains the line
    return LocationMatch(
        repo_file=fm.path, function=None, line=line,
        confidence=fm.confidence * 0.6,
        reason="file_line", alternatives=fm.alternatives,
    )


def lookup_by_symbol(
    symbol: str,
    index: RepoIndex,
    source_file: str | None = None,
) -> list[LocationMatch]:
    """Find functions matching symbol name.

    Priority:
      1. Exact qualified_name match
      2. Short name exact match
      3. Short name substring match
    Optionally filtered by source_file hint.
    """
    results: list[LocationMatch] = []

    # 1. Exact qualified_name
    if symbol in index.by_qualified:
        fn = index.by_qualified[symbol]
        path = index.symbol_to_file[symbol]
        results.append(LocationMatch(
            repo_file=path, function=fn, line=fn.line,
            confidence=0.95, reason="symbol", alternatives=[],
        ))

    if results:
        return results

    # 2. Short name exact
    short_hits = index.by_symbol.get(symbol, [])
    # Also try demangled / last component (e.g. "MyClass::forward" → "forward")
    if not short_hits and "::" in symbol:
        short_hits = index.by_symbol.get(symbol.split("::")[-1], [])

    # Filter by source_file hint if provided
    if source_file and short_hits:
        fm = fuzzy_file_match(source_file, index)
        if fm and fm.confidence >= 0.5:
            file_hits = [fn for fn in short_hits if index.symbol_to_file.get(fn.qualified_name) == fm.path]
            if file_hits:
                short_hits = file_hits

    for fn in short_hits:
        path = index.symbol_to_file.get(fn.qualified_name, "")
        results.append(LocationMatch(
            repo_file=path, function=fn, line=fn.line,
            confidence=0.70, reason="symbol", alternatives=[],
        ))

    if results:
        return results

    # 3. Substring / fuzzy name match
    sym_lower = symbol.lower()
    for name, fns in index.by_symbol.items():
        if sym_lower in name.lower() or name.lower() in sym_lower:
            for fn in fns:
                path = index.symbol_to_file.get(fn.qualified_name, "")
                results.append(LocationMatch(
                    repo_file=path, function=fn, line=fn.line,
                    confidence=0.4, reason="symbol", alternatives=[],
                ))

    return results[:20]  # cap output


# ── Caller / callee lookup ────────────────────────────────────────────────────

def callers_of(
    qualified_symbol: str,
    files: dict[str, FileFacts],
    limit: int = 20,
) -> list[str]:
    """Return qualified names of functions that call qualified_symbol.

    Matches both exact qualified names and the short name component.
    For class methods (e.g. ``model.py::Attention.forward``) we match both
    the full short name (``Attention.forward``) and the leaf name (``forward``)
    so that ``self.forward()`` and ``module.forward()`` call sites are found.
    """
    # short_full: e.g. "Attention.forward"
    short_full = qualified_symbol.split("::")[-1]
    # short_leaf: e.g. "forward"  (last dotted component)
    short_leaf = short_full.split(".")[-1]

    results: list[str] = []
    seen: set[str] = set()
    for facts in files.values():
        for fn in facts.functions.values():
            if fn.qualified_name == qualified_symbol:
                continue
            for call in fn.calls:
                call_short = call.split("::")[-1].split(".")[-1]
                if (
                    call == qualified_symbol
                    or call.split("::")[-1] == short_full
                    or call_short == short_leaf
                    or call_short == short_full
                ):
                    if fn.qualified_name not in seen:
                        seen.add(fn.qualified_name)
                        results.append(fn.qualified_name)
                    break
            if len(results) >= limit:
                return results
    return results


def callees_of(
    qualified_symbol: str,
    files: dict[str, FileFacts],
    limit: int = 20,
) -> list[str]:
    """Return qualified names of functions called by qualified_symbol."""
    # Find the function
    target_fn: FunctionFacts | None = None
    for facts in files.values():
        if qualified_symbol in facts.functions:
            target_fn = facts.functions[qualified_symbol]
            break
        for fn in facts.functions.values():
            if fn.qualified_name == qualified_symbol:
                target_fn = fn
                break
        if target_fn:
            break

    if target_fn is None:
        return []

    # Build a flat symbol lookup once for efficiency
    name_to_qualified: dict[str, str] = {}
    qualified_set: set[str] = set()
    for facts in files.values():
        for fn in facts.functions.values():
            name_to_qualified.setdefault(fn.name, fn.qualified_name)
            qualified_set.add(fn.qualified_name)

    results: list[str] = []
    seen: set[str] = set()
    for call in target_fn.calls:
        call_short = call.split("::")[-1].split(".")[-1]
        resolved = (
            call if call in qualified_set
            else name_to_qualified.get(call_short)
        )
        if resolved and resolved not in seen:
            seen.add(resolved)
            results.append(resolved)
            if len(results) >= limit:
                break

    return results


# ── Context bundle (Stage 3) ──────────────────────────────────────────────────

@dataclass
class SourceSnippet:
    """Extracted source code for a single function."""
    repo_file: str
    function_name: str
    start_line: int
    end_line: int | None
    source: str


@dataclass
class RepoContextBundle:
    """Bounded source context for optimizer consumption."""
    snippets: list[SourceSnippet]
    total_chars: int
    truncated: bool
    warnings: list[str]


def get_repo_context(
    repo_root: str,
    targets: list[str],         # list of qualified_name or "file::symbol"
    budget: ContextBudget | None = None,
    files: dict[str, FileFacts] | None = None,
    index: RepoIndex | None = None,
) -> RepoContextBundle:
    """Extract bounded source snippets for the given target symbols.

    Args:
        repo_root: Repository root path.
        targets:   List of qualified_name strings to extract.
        budget:    Context size limits (defaults to ContextBudget()).
        files:     Pre-parsed FileFacts (if available, skips re-parse).
        index:     Pre-built RepoIndex (if available).
    """
    if budget is None:
        budget = ContextBudget()

    root = Path(repo_root)
    if files is None:
        files, _ = scan_repo(root)
    if index is None:
        index = build_repo_index(files)

    snippets: list[SourceSnippet] = []
    total_chars = 0
    warnings: list[str] = []
    seen_fns: set[str] = set()

    for target in targets:
        if len(snippets) >= budget.max_functions:
            warnings.append(f"budget.max_functions={budget.max_functions} reached")
            break

        matches = lookup_by_symbol(target, index)
        if not matches:
            warnings.append(f"No match for target: {target}")
            continue

        for m in matches[:3]:  # up to 3 candidates per target
            if not m.function or not m.repo_file:
                continue
            fn = m.function
            if fn.qualified_name in seen_fns:
                continue
            seen_fns.add(fn.qualified_name)

            # Read source file
            src_path = root / m.repo_file
            try:
                src_lines = src_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                warnings.append(f"Cannot read {m.repo_file}")
                continue

            start = max(0, fn.line - 1)
            end = (
                min(fn.end_line, len(src_lines))
                if fn.end_line
                else min(start + budget.max_lines_per_function, len(src_lines))
            )
            # Cap lines per function
            if end - start > budget.max_lines_per_function:
                end = start + budget.max_lines_per_function

            snippet_text = "\n".join(src_lines[start:end])
            if total_chars + len(snippet_text) > budget.max_total_chars:
                warnings.append(f"budget.max_total_chars={budget.max_total_chars} reached")
                return RepoContextBundle(snippets=snippets, total_chars=total_chars,
                                         truncated=True, warnings=warnings)

            snippets.append(SourceSnippet(
                repo_file=m.repo_file,
                function_name=fn.qualified_name,
                start_line=fn.line,
                end_line=fn.end_line,
                source=snippet_text,
            ))
            total_chars += len(snippet_text)

    return RepoContextBundle(
        snippets=snippets, total_chars=total_chars,
        truncated=False, warnings=warnings,
    )


# ── Entry-point detector ──────────────────────────────────────────────────────

class EntryPointDetector:
    def detect(self, files: dict[str, FileFacts], dag: FileDAG | None = None) -> list[EntryPoint]:
        zero = dag.zero_indegree() if dag else set()
        indeg = dag.in_degrees() if dag else {}

        entries: list[EntryPoint] = []
        for facts in files.values():
            if Path(facts.path).name in NON_ENTRY_NAMES:
                logger.debug("skip infra file: %s", facts.path)
                continue

            base = max(facts.training_score, facts.inference_score, facts.generic_score)
            d = indeg.get(facts.path, 0)
            bonus = 2 if facts.path in zero else (1 if d <= 1 else 0)
            score = base + bonus
            if score < 3:
                continue

            if (facts.training_score >= facts.inference_score
                    and facts.training_score >= facts.generic_score):
                mode = "training"
            elif facts.inference_score >= facts.generic_score:
                mode = "inference"
            else:
                mode = "generic"

            reasons = list(facts.notes)
            if bonus:
                reasons.append(f"dag:indegree={d}")

            starts = (
                facts.main_guard_calls
                or (["main"] if "main" in facts.functions else facts.top_level_calls[:5])
            )
            entries.append(EntryPoint(
                path=facts.path, module_name=facts.module_name,
                mode=mode, score=score, reasons=reasons, start_symbols=starts,
            ))

        entries.sort(key=lambda e: (-e.score, e.path))
        return entries


# ── Call-chain tracer ─────────────────────────────────────────────────────────

class CallChainTracer:
    def __init__(self, files: dict[str, FileFacts], dag: FileDAG | None = None) -> None:
        self.files = files
        self.dag = dag
        # (module_name, fn_name) -> FunctionFacts — for same-file local calls
        self._sym: dict[tuple[str, str], FunctionFacts] = {
            (facts.module_name, fn.name): fn
            for facts in files.values()
            for fn in facts.functions.values()
        }

    def trace(self, entry: EntryPoint, max_depth: int = 8, max_steps: int = 200) -> CallChain:
        v_files: set[str] = {entry.path}
        v_syms: set[str] = set()
        steps: list[CallStep] = []
        seen: set[tuple[str, str]] = set()
        trunc = False

        queue: list[tuple[str, str, int]] = [(entry.path, s, 0) for s in entry.start_symbols]
        if not queue and "main" in self.files[entry.path].functions:
            queue.append((entry.path, "main", 0))

        while queue:
            if len(steps) >= max_steps:
                trunc = True; break
            cur_path, sym, depth = queue.pop(0)
            if depth > max_depth:
                continue
            res = self._resolve_sym(cur_path, sym)
            if not res:
                continue
            r_path, fn = res
            if fn.qualified_name in v_syms:
                continue
            v_syms.add(fn.qualified_name)
            v_files.add(r_path)

            for call in fn.calls:
                child = self._resolve_call(self.files[r_path], call)
                if not child:
                    continue
                c_path, c_fn = child
                edge = (fn.qualified_name, c_fn.qualified_name)
                if edge in seen:
                    continue
                seen.add(edge)
                v_files.add(c_path)
                steps.append(CallStep(from_symbol=fn.qualified_name,
                                      to_symbol=c_fn.qualified_name,
                                      to_file=c_path, depth=depth + 1))
                if c_fn.qualified_name not in v_syms:
                    queue.append((c_path, c_fn.name, depth + 1))

        return CallChain(
            entry_path=entry.path,
            entry_symbol=entry.start_symbols[0] if entry.start_symbols else "main",
            visited_files=sorted(v_files),
            visited_symbols=sorted(v_syms),
            steps=steps, truncated=trunc,
        )

    def _resolve_sym(self, path: str, name: str) -> tuple[str, FunctionFacts] | None:
        facts = self.files.get(path)
        if not facts:
            return None
        if name in facts.functions:
            return path, facts.functions[name]
        for key, fn in facts.functions.items():
            if fn.name == name:
                return path, fn
        binding = facts.imports.get(name)
        if binding and binding.target_file:
            t = self.files.get(binding.target_file)
            sym = binding.target_symbol or name
            if t and sym in t.functions:
                return binding.target_file, t.functions[sym]
        return None

    def _resolve_call(self, facts: FileFacts, call: str) -> tuple[str, FunctionFacts] | None:
        sep = "::" if "::" in call else "."
        parts = call.split(sep)
        base, fn_name = parts[0], parts[-1]

        binding = facts.imports.get(base)
        if binding and binding.target_file:
            t = self.files.get(binding.target_file)
            if t and fn_name in t.functions:
                return binding.target_file, t.functions[fn_name]

        if sep == "::" and len(parts) >= 2:
            cur_mod = facts.module_name
            candidate_mod = "crate." + ".".join(parts[:-1])
            for rel, f2 in self.files.items():
                if f2.module_name == candidate_mod and fn_name in f2.functions:
                    return rel, f2.functions[fn_name]
            if cur_mod and cur_mod != "crate":
                parent = ".".join(cur_mod.split(".")[:-1])
                candidate_mod2 = (parent + "." if parent else "") + ".".join(parts[:-1])
                for rel, f2 in self.files.items():
                    if f2.module_name == candidate_mod2 and fn_name in f2.functions:
                        return rel, f2.functions[fn_name]

        fn = self._sym.get((facts.module_name, fn_name))
        if fn:
            return facts.path, fn
        return None


# ── Hub detection ─────────────────────────────────────────────────────────────

def find_hubs(
    files: dict[str, FileFacts],
    dag: FileDAG,
    top_n: int = 15,
) -> list[HubNode]:
    in_deg: dict[str, int] = {}
    out_deg: dict[str, int] = {}

    for facts in files.values():
        if Path(facts.path).name in NON_ENTRY_NAMES:
            continue
        for fn in facts.functions.values():
            qn = fn.qualified_name
            out_deg.setdefault(qn, 0)
            in_deg.setdefault(qn, 0)
            for call in fn.calls:
                sep = "::" if "::" in call else "."
                fn_name = call.split(sep)[-1]
                out_deg[qn] = out_deg.get(qn, 0) + 1
                in_deg[fn_name] = in_deg.get(fn_name, 0) + 1

    scored: list[HubNode] = []
    for facts in files.values():
        if Path(facts.path).name in NON_ENTRY_NAMES:
            continue
        for fn in facts.functions.values():
            qn = fn.qualified_name
            ind = in_deg.get(qn, 0) + in_deg.get(fn.name, 0)
            outd = out_deg.get(qn, 0)
            total = ind + outd
            if total > 0:
                scored.append(HubNode(path=facts.path, symbol=qn,
                                      in_degree=ind, out_degree=outd, total_degree=total))

    scored.sort(key=lambda h: (-h.total_degree, h.symbol))
    return scored[:top_n]


# ── Symbol / file search ──────────────────────────────────────────────────────

def search_symbols(
    files: dict[str, FileFacts],
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    q = query.lower().strip()
    if not q:
        return []
    results: list[SearchResult] = []
    for facts in files.values():
        path_lower = facts.path.lower()
        if q in path_lower:
            results.append(SearchResult(
                path=facts.path, symbol=facts.path, kind="file", line=0,
                score=2.0 if path_lower.endswith(q) else 1.5,
            ))
        for fn in facts.functions.values():
            name_lower = fn.name.lower()
            if name_lower == q:
                score = 3.0
            elif name_lower.startswith(q):
                score = 2.0
            elif q in name_lower:
                score = 1.0
            else:
                continue
            results.append(SearchResult(
                path=facts.path, symbol=fn.qualified_name,
                kind="function", line=fn.line, score=score,
            ))
    results.sort(key=lambda r: (-r.score, r.symbol))
    return results[:limit]


# ── Impact radius ─────────────────────────────────────────────────────────────

def impact_radius(
    files: dict[str, FileFacts],
    dag: FileDAG,
    changed_files: list[str],
    max_depth: int = 5,
    max_nodes: int = 200,
) -> ImpactResult:
    reverse: dict[str, set[str]] = {}
    for src, targets in dag.edges.items():
        for tgt in targets:
            reverse.setdefault(tgt, set()).add(src)

    seeds = [f for f in changed_files if f in files]
    depth_map: dict[str, int] = {f: 0 for f in seeds}
    queue: deque[tuple[str, int]] = deque((f, 0) for f in seeds)
    impacted: set[str] = set(seeds)

    while queue:
        if len(impacted) >= max_nodes:
            break
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for importer in reverse.get(node, []):
            if importer not in impacted:
                impacted.add(importer)
                depth_map[importer] = depth + 1
                queue.append((importer, depth + 1))

    impacted_syms: list[str] = []
    for path in sorted(impacted - set(seeds)):
        facts = files.get(path)
        if facts:
            impacted_syms.extend(fn.qualified_name for fn in facts.functions.values())

    return ImpactResult(
        seed_files=seeds,
        impacted_files=sorted(impacted - set(seeds)),
        impacted_symbols=impacted_syms,
        depth_map=depth_map,
        truncated=len(impacted) >= max_nodes,
    )


# ── Trace from file / symbol ──────────────────────────────────────────────────

def trace_from(
    files: dict[str, FileFacts],
    dag: FileDAG,
    target: str,
    symbol: str | None = None,
    max_depth: int = 8,
    max_steps: int = 300,
) -> list[CallChain]:
    tracer = CallChainTracer(files, dag)
    matched: list[str] = []

    if target in files:
        matched.append(target)
    if not matched:
        tl = target.lower().replace("\\", "/")
        matched = [p for p in files if p.lower().replace("\\", "/").endswith(tl)]
    if not matched:
        matched = [p for p, f in files.items() if target in f.functions]

    if not matched:
        logger.warning("trace_from: no file or symbol matched %r", target)
        return []

    chains: list[CallChain] = []
    for path in matched:
        facts = files[path]
        if symbol:
            if symbol in facts.functions:
                seeds = [symbol]
            else:
                seeds = [k for k in facts.functions if k == symbol or k.endswith(f".{symbol}")]
        else:
            guard_seeds = [s for s in (facts.main_guard_calls or []) if s in facts.functions]
            top_level = [k for k in facts.functions if "." not in k]
            methods = [k for k in facts.functions if "." in k]
            seeds = guard_seeds or top_level or methods

        if not seeds:
            logger.warning("trace_from: no traceable symbols in %s", path)
            continue

        for sym in seeds:
            ep = EntryPoint(path=path, module_name=facts.module_name,
                            mode="generic", score=0, reasons=["trace_from"],
                            start_symbols=[sym])
            chains.append(tracer.trace(ep, max_depth=max_depth, max_steps=max_steps))

    return chains


# ── High-level API ────────────────────────────────────────────────────────────

def analyze_repo(
    repo_root: str | Path,
    top_n: int = 10,
    max_chain_depth: int = 8,
) -> AnalysisResult:
    root = Path(repo_root).resolve()
    files, warnings = scan_repo(root)  # full scan for repo-only mode
    dag = build_dag(files)
    entries = EntryPointDetector().detect(files, dag)[:top_n]
    tracer = CallChainTracer(files, dag)
    chains = [tracer.trace(e, max_depth=max_chain_depth) for e in entries]
    hubs = find_hubs(files, dag, top_n=10)
    notes = warnings + [
        "Static analysis only: dynamic dispatch and config-driven imports may be missed.",
        "Supports Python, Rust, C/C++, Java, Go.",
    ]
    return AnalysisResult(
        repo_root=str(root),
        source_files=len(files),
        languages=lang_distribution(files),
        entry_points=entries,
        call_chains=chains,
        hub_nodes=hubs,
        notes=notes,
    )


# ── Render helpers ────────────────────────────────────────────────────────────

def render_summary(result: AnalysisResult) -> str:
    lines: list[str] = [
        f"Repo: {result.repo_root}",
        f"Source files analyzed: {result.source_files}",
    ]
    if result.languages:
        lang_str = "  ".join(f"{k}:{v}" for k, v in result.languages.items())
        lines.append(f"  Languages: {lang_str}")
    lines.append("")

    lines.append(f"Likely entry points: {len(result.entry_points)} found")
    for i, ep in enumerate(result.entry_points, 1):
        lines.append(f"  {i}. {ep.path} [{ep.mode}] score={ep.score}")
        lines.append(f"     reasons: {', '.join(ep.reasons[:6])}")
        lines.append(f"     start:   {', '.join(ep.start_symbols[:6])}")
    lines.append("")

    if result.hub_nodes:
        lines.append(f"Hub nodes (top {len(result.hub_nodes)}):")
        for h in result.hub_nodes:
            lines.append(f"  {h.symbol}  in={h.in_degree} out={h.out_degree} total={h.total_degree}")
        lines.append("")

    lines.append("Call chain preview:")
    for chain in result.call_chains:
        trunc = " [truncated]" if chain.truncated else ""
        lines.append(
            f"  - {chain.entry_path}: {len(chain.visited_files)} files, "
            f"{len(chain.visited_symbols)} symbols, {len(chain.steps)} steps{trunc}"
        )
        for step in chain.steps[:10]:
            ind = "  " * step.depth
            lines.append(f"    {ind}d{step.depth}: {step.from_symbol} -> {step.to_symbol}")
    lines.append("")

    for note in result.notes:
        lines.append(f"Notes: {note}")
    return "\n".join(lines)


def render_trace(chains: list[CallChain], target: str) -> str:
    lines = [f"Trace from: {target}", f"Chains: {len(chains)}"]
    for chain in chains:
        trunc = " [truncated]" if chain.truncated else ""
        lines += [
            "",
            f"  [{chain.entry_symbol}] {chain.entry_path}"
            f"  ({len(chain.visited_files)} files, "
            f"{len(chain.visited_symbols)} symbols, {len(chain.steps)} steps){trunc}",
        ]
        for step in chain.steps:
            ind = "  " * step.depth
            lines.append(f"    {ind}d{step.depth}: {step.from_symbol} -> {step.to_symbol}")
            if step.to_file != chain.entry_path:
                lines.append(f"    {ind}       @ {step.to_file}")
    return "\n".join(lines)


# ── Callsite index & search ───────────────────────────────────────────────────

# File/function name keywords that indicate training-loop relevance.
# Used by derive_analysis_scope to select priority files.
_SCOPE_PATTERNS: dict[str, dict] = {
    # Canonical category names from classify.py
    "gpu_memcpy_hotspot": {
        "file_keywords": {"train", "trainer", "data", "loader", "dataset",
                          "collate", "forward", "step", "memory", "prefetch"},
        "func_keywords": {"train_step", "forward", "collate_fn", "step",
                          "__getitem__", "on_train_batch", "to_device",
                          "prefetch_pin_memory"},
        "call_names": ["to", "cuda", "copy_", "pin_memory"],
        "call_keywords": ["device", "cuda", "non_blocking"],
    },
    "gpu_idle": {
        "file_keywords": {"train", "trainer", "data", "loader", "schedule"},
        "func_keywords": {"train_step", "train_iter", "train_one_iter",
                          "forward", "step", "__iter__"},
        "call_names": ["synchronize", "barrier", "wait"],
        "call_keywords": [],
    },
    "gpu_comm_hotspot": {
        "file_keywords": {"train", "trainer", "ddp", "dist", "parallel"},
        "func_keywords": {"backward", "train_step", "train_iter", "iter_impl",
                          "reduce", "all_reduce"},
        "call_names": ["all_reduce", "barrier", "backward", "reduce_scatter",
                       "allgather", "no_sync"],
        "call_keywords": ["async_op"],
    },
    "sync_wait": {
        "file_keywords": {"train", "trainer", "model", "forward"},
        "func_keywords": {"train_step", "train_iter", "train_one_iter",
                          "iter_impl", "forward", "step"},
        "call_names": ["synchronize", "item", "wait", "cpu"],
        "call_keywords": [],
    },
    # Aliases / derived categories for forward-compat
    "gpu_memcpy_h2d": {
        "file_keywords": {"train", "trainer", "data", "loader", "dataset",
                          "collate", "forward", "step", "memory", "prefetch"},
        "func_keywords": {"train_step", "forward", "collate_fn", "step",
                          "__getitem__", "on_train_batch", "to_device",
                          "prefetch_pin_memory"},
        "call_names": ["to", "cuda", "copy_", "pin_memory"],
        "call_keywords": ["device", "cuda", "non_blocking"],
    },
    "gpu_memcpy_d2h": {
        "file_keywords": {"train", "trainer", "log", "metric", "eval"},
        "func_keywords": {"train_step", "forward", "log", "evaluate"},
        "call_names": ["cpu", "numpy", "item", "tolist"],
        "call_keywords": [],
    },
}
_DEFAULT_SCOPE_PATTERN = {
    "file_keywords": {"train", "trainer"},
    "func_keywords": set(),
    "call_names": [],
    "call_keywords": [],
}


@dataclass
class AnalysisScope:
    """Finding-aware scan scope.  Drives callsite search — not a full-repo scan."""
    finding_type: str
    priority_file_patterns: list[str]       # file keyword hints
    priority_func_patterns: list[str]       # function keyword hints
    call_names: list[str]                   # call names to search
    call_keywords: list[str]                # keyword args to match
    seed_files: list[str]                   # initial candidate files
    selected_files: list[str]               # files included in this scope
    excluded_patterns: list[str]
    reason: list[str]                       # audit: why these files were chosen
    max_files: int = 50
    max_candidates: int = 100


def derive_analysis_scope(
    finding_type: str,
    files: dict[str, FileFacts],
    max_files: int = 50,
) -> AnalysisScope:
    """Build a finding-aware AnalysisScope from the scanned file set.

    Selects files whose names or module paths match the finding's priority
    keywords.  Falls back to all Python files if nothing matches.
    """
    pattern = _SCOPE_PATTERNS.get(finding_type, _DEFAULT_SCOPE_PATTERN)
    file_kw: set[str] = pattern["file_keywords"]
    call_names: list[str] = pattern["call_names"]
    call_keywords: list[str] = pattern["call_keywords"]

    selected: list[str] = []
    reason: list[str] = []

    for rel, facts in files.items():
        if facts.language != "python":
            continue
        parts_for_match = [
            part for part in Path(rel.replace("\\", "/")).parts
            if not part.lower().endswith(".runfiles")
        ]
        low = "/".join(parts_for_match).lower()
        parts = set(low.replace("/", " ").replace("_", " ").replace(".", " ").split())
        if parts & file_kw:
            selected.append(rel)
            reason.append(f"keyword match: {rel}")

    # Fallback: include all Python files up to max_files
    if not selected:
        selected = [r for r, f in files.items() if f.language == "python"]
        reason.append("fallback: all python files")

    selected = selected[:max_files]

    return AnalysisScope(
        finding_type=finding_type,
        priority_file_patterns=sorted(file_kw),
        priority_func_patterns=sorted(pattern["func_keywords"]),
        call_names=call_names,
        call_keywords=call_keywords,
        seed_files=selected[:],
        selected_files=selected,
        excluded_patterns=[],
        reason=reason,
        max_files=max_files,
        max_candidates=100,
    )


@dataclass
class CallSiteCandidate:
    """A scored callsite returned by search_calls."""
    id: str
    path: str
    line: int
    call_name: str
    full_call_name: str | None
    receiver: str | None
    keywords: dict[str, str]
    loop_depth: int
    enclosing_function: str | None
    source_line: str
    score: float


def build_callsite_index(
    files: dict[str, FileFacts],
) -> dict[str, list[CallSiteFacts]]:
    """Build call_name → [CallSiteFacts, ...] index from all scanned files."""
    index: dict[str, list[CallSiteFacts]] = defaultdict(list)
    for facts in files.values():
        for cs in facts.callsites:
            index[cs.call_name].append(cs)
    return dict(index)


def search_calls(
    index: dict[str, list[CallSiteFacts]],
    scope: AnalysisScope,
    names: list[str] | None = None,
    keywords: list[str] | None = None,
    limit: int = 50,
) -> list[CallSiteCandidate]:
    """Search the callsite index and return ranked candidates.

    Scoring (higher = more relevant):
      +3  file is in scope.selected_files
      +2  loop_depth >= 1
      +1  per matching keyword in call.keywords
    """
    search_names = names if names is not None else scope.call_names
    search_kw = set(keywords) if keywords else set(scope.call_keywords)
    scope_set = set(scope.selected_files)
    func_patterns = [p.lower() for p in scope.priority_func_patterns]

    # Training-critical function name fragments — these are semantically
    # "inside the training loop" even if their AST loop_depth=0, because
    # the outer for-loop calls them.  Give a significant score boost.
    _TRAIN_ITER_FRAGMENTS = frozenset({
        "train_one_iter", "train_iter_impl", "iter_impl", "train_iter",
        "train_step", "train_batch", "forward_backward", "step",
    })

    def _collect(scope_only: bool) -> list[CallSiteCandidate]:
        candidates: list[CallSiteCandidate] = []
        for name in search_names:
            for cs in index.get(name, []):
                in_scope = cs.path in scope_set
                if scope_only and scope_set and not in_scope:
                    continue
                score = 0.0
                if in_scope:
                    score += 3.0

                # loop_depth: still useful for list comprehensions, explicit loops
                # inside a function — but cap the bonus to avoid over-weighting
                # syntax-deep trivial loops.
                if cs.loop_depth >= 1:
                    score += min(2.0, cs.loop_depth * 1.0)

                # Enclosing-function semantic boost:
                # 1. Named func_patterns match (from scope config)
                # 2. Training-iteration fragments — these ARE the hot path
                fn_name = (cs.enclosing_function or "").lower()
                if fn_name:
                    if any(p in fn_name for p in func_patterns):
                        score += 2.0
                    # Extra boost for known training-iteration function patterns
                    if any(frag in fn_name for frag in _TRAIN_ITER_FRAGMENTS):
                        score += 3.0

                # Keyword argument matches
                call_text = " ".join(
                    cs.args_repr
                    + list(cs.keywords.values())
                    + [cs.source_line, cs.full_call_name or cs.call_name]
                ).lower()
                for kw in search_kw:
                    if kw in cs.keywords:
                        score += 1.0
                    if kw.lower() in call_text:
                        score += 1.0
                candidates.append(CallSiteCandidate(
                    id=cs.id,
                    path=cs.path,
                    line=cs.line,
                    call_name=cs.call_name,
                    full_call_name=cs.full_call_name,
                    receiver=cs.receiver,
                    keywords=cs.keywords,
                    loop_depth=cs.loop_depth,
                    enclosing_function=cs.enclosing_function,
                    source_line=cs.source_line,
                    score=score,
                ))
        return candidates

    candidates = _collect(scope_only=True)
    if not candidates:
        candidates = _collect(scope_only=False)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:limit]


# ── Callsite context ──────────────────────────────────────────────────────────

@dataclass
class CallsiteContext:
    """Source context around a callsite, for LLM consumption."""
    callsite_id: str
    path: str
    line: int
    enclosing_function: str | None
    source_snippet: str         # lines of the enclosing function body
    callers: list[str]          # qualified names of functions that call enclosing_function


def get_callsite_context(
    callsite_id: str,
    files: dict[str, FileFacts],
    repo_root: str | Path | None = None,
) -> CallsiteContext | None:
    """Return source context for a callsite id.

    Returns the enclosing function's source lines (not the whole file)
    plus callers of that function within the scanned files.
    Returns None if the id is not found.

    repo_root is needed to read the actual source file for the snippet.
    If not provided, falls back to the single callsite source_line.
    """
    # Parse id: "<path>:<line>:<col>:<call_name>"
    parts = callsite_id.split(":")
    if len(parts) < 4:
        return None
    cs_path = parts[0]
    try:
        cs_line = int(parts[1])
    except ValueError:
        return None

    facts = files.get(cs_path)
    if facts is None:
        return None

    # Find the callsite in the file's callsite list
    cs = next((c for c in facts.callsites if c.id == callsite_id), None)
    if cs is None:
        return None

    # Find the enclosing function in FileFacts.functions
    enclosing_fn_facts = None
    if cs.enclosing_function:
        short_key = cs.enclosing_function.split("::")[-1] if "::" in cs.enclosing_function else cs.enclosing_function
        enclosing_fn_facts = facts.functions.get(short_key)

    # Build source snippet: enclosing function body, or single callsite line as fallback
    snippet_lines: list[str] = []
    if enclosing_fn_facts and enclosing_fn_facts.end_line and repo_root:
        try:
            full_path = Path(repo_root) / cs_path
            src_lines = full_path.read_text(encoding="utf-8").splitlines()
            start = max(enclosing_fn_facts.line - 1, 0)
            end = min(enclosing_fn_facts.end_line, len(src_lines))
            snippet_lines = src_lines[start:end]
        except (OSError, UnicodeDecodeError):
            pass

    if not snippet_lines:
        snippet_lines = [cs.source_line]

    # Find callers: functions whose calls list contains the enclosing function's short name
    callers: list[str] = []
    if enclosing_fn_facts:
        target_short = enclosing_fn_facts.name
        for _, f in files.items():
            for _, fn in f.functions.items():
                if target_short in fn.calls:
                    callers.append(fn.qualified_name)

    return CallsiteContext(
        callsite_id=callsite_id,
        path=cs_path,
        line=cs_line,
        enclosing_function=cs.enclosing_function,
        source_snippet="\n".join(snippet_lines),
        callers=callers[:20],
    )
