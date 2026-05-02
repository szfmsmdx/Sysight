"""WARMUP stage — per-repo, one-time exploration → RepoSetup.

Runs once per repo. Discovers entry points, metrics, test commands.
Stores result in workspace wiki for ANALYZE/OPTIMIZE to reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sysight.types.repo_setup import RepoSetup


@dataclass
class WarmupResult:
    repo: str = ""
    repo_setup: RepoSetup = field(default_factory=RepoSetup)
    errors: list[str] = field(default_factory=list)


def run_warmup(
    repo: str,
    registry,
    provider,
    knowledge=None,
) -> WarmupResult:
    """Explore a repo and produce a RepoSetup.

    1. Scan for entry points (scanner_files + scanner_search)
    2. Try running with --help (sandbox_exec when available)
    3. Discover test commands (scanner_search)
    4. Persist RepoSetup to workspace wiki
    """
    errors: list[str] = []
    root = Path(repo).resolve()
    policy = _warmup_policy(registry)
    repo_setup = RepoSetup(source="warmup_partial")

    # 1. Find entry points
    try:
        from sysight.tools.scanner.files import list_files
        files_result = list_files(str(root), ext="py", max_results=500)
        py_files = [f.path for f in files_result.files]
    except Exception as e:
        errors.append(f"scanner_files failed: {e}")
        py_files = []

    if not py_files:
        errors.append("no Python files found")
        return WarmupResult(repo=str(root), repo_setup=repo_setup, errors=errors)

    # 2. Search for common entry-point patterns
    entry_candidates = _find_entry_candidates(root, py_files)
    if entry_candidates:
        repo_setup.entry_point = entry_candidates[0]

    # 3. Discover test commands
    try:
        from sysight.tools.scanner.search import search
        test_search = search(str(root), r"pytest|unittest", ext="py", fixed=False)
        if test_search.total_matches > 0:
            # Look for pytest configuration
            cfg = root / "pyproject.toml"
            if cfg.exists():
                repo_setup.test_commands = [["python", "-m", "pytest", "tests/"]]
            else:
                repo_setup.test_commands = [["python", "-m", "pytest"]]
    except Exception as e:
        errors.append(f"test discovery failed: {e}")

    # 4. Discover build commands
    if (root / "pyproject.toml").exists():
        repo_setup.build_commands = [["pip", "install", "-e", "."]]
    elif (root / "requirements.txt").exists():
        repo_setup.build_commands = [["pip", "install", "-r", "requirements.txt"]]

    # 5. Try to find a metric from common patterns
    if "train" in repo_setup.entry_point.lower():
        repo_setup.metric_grep = r"iter/s|step/s|throughput|loss"
        repo_setup.metric_lower_is_better = False
    elif "eval" in repo_setup.entry_point.lower() or "bench" in repo_setup.entry_point.lower():
        repo_setup.metric_grep = r"latency|time|ms"
        repo_setup.metric_lower_is_better = True

    repo_setup.source = "warmup_partial"
    if not errors:
        repo_setup.source = "warmup_verified"

    # 6. If there's a sandbox, try a minimal run to extract the metric
    # TODO: implement when sandbox_exec is available

    # 7. Persist to workspace wiki
    if knowledge:
        ns = knowledge.workspace_namespace(repo_root=str(root))
        summary = (
            f"# Repo Setup\n\n"
            f"Entry: `{repo_setup.entry_point}`\n\n"
            f"Tests: `{' '.join(repo_setup.test_commands[0]) if repo_setup.test_commands else 'none'}`\n\n"
            f"Metric: `{repo_setup.metric_grep}`\n\n"
            f"Source: {repo_setup.source}\n"
        )
        try:
            knowledge.write_page(
                f"workspaces/{ns}/overview.md",
                summary,
                title="Repo Overview",
                scope="workspace",
            )
        except Exception as e:
            errors.append(f"wiki write failed: {e}")

    return WarmupResult(repo=str(root), repo_setup=repo_setup, errors=errors)


def _warmup_policy(registry) -> object:
    from sysight.tools.registry import ToolPolicy
    return ToolPolicy(allowed_tools={"scanner_*", "sandbox_*"}, read_only=False)


def _find_entry_candidates(root: Path, py_files: list[str]) -> list[str]:
    """Find likely entry points: files with `if __name__ == '__main__'` or train/run patterns."""
    candidates = []
    for path in py_files:
        try:
            content = (root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if 'if __name__ == "__main__"' in content or "__main__" in content:
            candidates.append(f"python {path}")
        elif any(kw in path.lower() for kw in ("train", "run", "main")):
            candidates.append(f"python {path}")
    return candidates
