"""Shared data models, constants, and BaseScanner interface."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# ── ignore rules ─────────────────────────────────────────────────────────────

_IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build", ".pytest_cache", ".mypy_cache",
    ".cache", ".tox", ".eggs", "external", "third_party", "vendor",
    "site-packages", "generated",
}
_IGNORE_SUFFIXES = (".runfiles", ".dist-info", ".egg-info")
_IGNORE_EXTS = {
    ".pyc", ".pyd", ".pyo", ".so", ".dylib", ".dll", ".o", ".a", ".lib",
    ".class", ".jar", ".war", ".wasm",
    ".pb", ".onnx", ".pt", ".pth", ".bin",
    ".npy", ".npz", ".pkl", ".pickle", ".lock",
}

# Infrastructure file names — never surfaced as entry points
NON_ENTRY_NAMES = {"__init__.py", "__init__.pyi", "__main__.py", "conftest.py", "setup.py"}

# Scoring keyword sets (shared across all scanners)
ENTRY_FILE_HINTS: dict[str, int] = {
    "train": 3, "trainer": 2, "pretrain": 3, "finetune": 3, "fit": 2,
    "infer": 3, "inference": 3, "predict": 2, "generate": 2, "serve": 2,
    "eval": 1, "main": 1, "run": 1,
}
TRAINING_KW = (
    "optimizer.step", "loss.backward", "model.train",
    "dataloader", "trainer.fit", "train_epoch", "fit(",
)
INFERENCE_KW = (
    "model.eval", "torch.no_grad", "predict(", "generate(",
    "inference(", "infer(", "pipeline(",
)
CLI_HINTS = ("argparse", "click", "typer", "fire", "hydra")


def should_ignore(repo_root: Path, path: Path, include_runfiles: bool = False) -> bool:
    if path.suffix.lower() in _IGNORE_EXTS:
        return True
    for part in path.relative_to(repo_root).parts:
        low = part.lower()
        if low in _IGNORE_DIRS:
            return True
        if include_runfiles and low.endswith(".runfiles"):
            continue
        if any(low.endswith(s) for s in _IGNORE_SUFFIXES):
            return True
    return False


def norm(parts: list[str]) -> list[str]:
    return [p for p in parts if p and p != "."]


def score_by_content(src: str, has_main: bool, name: str,
                     extra_notes: list[str] | None = None) -> tuple[int, int, int, list[str]]:
    """Generic scoring used by all scanners."""
    tr = inf = gen = 0
    notes: list[str] = list(extra_notes or [])
    low = src.lower()

    if has_main:
        gen += 4
        notes.append("fn-main")
    for hint, w in ENTRY_FILE_HINTS.items():
        if hint in name:
            gen += w
            notes.append(f"filename:{hint}")
    for kw in TRAINING_KW:
        if kw in low:
            tr += 2
            notes.append(f"train:{kw}")
    for kw in INFERENCE_KW:
        if kw in low:
            inf += 2
            notes.append(f"infer:{kw}")
    return tr, inf, gen, notes


def match_brace(src: str, start: int) -> int:
    """Return index of the closing brace matching `src[start]`, or -1."""
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


# ── data models ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImportBinding:
    alias: str
    binding_type: str                # "module" | "symbol" | "include" | "import"
    target_module: str | None = None
    target_symbol: str | None = None
    target_file: str | None = None


@dataclass
class FunctionFacts:
    name: str
    qualified_name: str
    line: int
    calls: list[str] = field(default_factory=list)
    end_line: int | None = None          # None if scanner cannot determine
    is_gpu_kernel: bool = False          # True for __global__, @triton.jit, etc.
    extra: dict[str, str | int | float | list[str]] = field(default_factory=dict)


@dataclass
class CallSiteFacts:
    """A single Python call site captured at AST parse time.

    Syntax-level only — no type inference.  Receiver and keywords are
    string reprs of the AST nodes, not resolved values.

    id format: "<path>:<line>:<col>:<call_name>"
    """
    id: str                         # stable reference for LLM / optimizer
    path: str
    line: int
    col: int | None
    end_line: int | None
    end_col: int | None
    call_name: str                  # "to", "cuda", "copy_", "pin_memory"
    full_call_name: str | None      # "batch.to", "torch.cuda.synchronize"
    receiver: str | None            # "batch", "x", "self.data"
    args_repr: list[str]            # positional args as source strings
    keywords: dict[str, str]        # keyword args: {"device": "cuda", "non_blocking": "False"}
    enclosing_function: str | None  # qualified name of the containing function
    loop_depth: int                 # 0 = not inside any loop
    source_line: str                # raw source line for LLM reading


@dataclass
class FileFacts:
    path: str
    module_name: str
    language: str
    has_main_guard: bool
    cli_frameworks: list[str]
    top_level_calls: list[str]
    main_guard_calls: list[str]
    functions: dict[str, FunctionFacts]
    imports: dict[str, ImportBinding]
    training_score: int
    inference_score: int
    generic_score: int
    notes: list[str]
    callsites: list[CallSiteFacts] = field(default_factory=list)
    extra: dict[str, str | int | float | list[str]] = field(default_factory=dict)
    # e.g. {"gpu_tags": ["triton_jit", "nccl_comm"], "cuda_kernel_count": 3}


# ── BaseScanner ───────────────────────────────────────────────────────────────

class BaseScanner(ABC):
    """All language scanners implement this interface.

    To add a new language:
      1. Create ``sysight/analyzer/scanners/<lang>.py``  (subclass BaseScanner)
      2. Add the class to ``SCANNERS`` in ``sysight/analyzer/core.py``
    """

    #: Set of lowercase file extensions this scanner handles, e.g. {".py"}
    EXTENSIONS: frozenset[str] = frozenset()

    def __init__(self, repo_root: Path, include_runfiles: bool = False) -> None:
        self.root = repo_root
        self.include_runfiles = include_runfiles
        self.warnings: list[str] = []

    @abstractmethod
    def scan(self) -> dict[str, FileFacts]:
        """Return a mapping of repo-relative-path -> FileFacts for all files."""

    def scan_paths(self, paths: list[Path]) -> dict[str, FileFacts]:
        """Parse only the given absolute paths; skip rglob entirely.

        Default implementation calls ``_parse_one`` per path; subclasses may
        override for languages that require a global index before per-file parsing
        (e.g. Python module-name resolution, C++ include resolution).

        Returns repo-relative-path -> FileFacts.
        """
        out: dict[str, FileFacts] = {}
        for p in paths:
            if p.suffix.lower() not in self.EXTENSIONS:
                continue
            if should_ignore(self.root, p, include_runfiles=self.include_runfiles):
                continue
            rel = str(p.relative_to(self.root))
            try:
                facts = self._parse_one(p)
                if facts is not None:
                    out[rel] = facts
            except (OSError, UnicodeDecodeError, SyntaxError, RecursionError) as e:
                self.warnings.append(f"Skipped {rel}: {type(e).__name__}")
        return out

    def _parse_one(self, path: Path) -> FileFacts | None:
        """Parse a single file.  Override in subclasses that support targeted scan.

        Return None to signal that this scanner cannot handle the file.
        Default: return None (subclass must override).
        """
        return None
