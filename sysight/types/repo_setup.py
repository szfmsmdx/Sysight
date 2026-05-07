"""RepoSetup dataclass. Zero internal dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal


@dataclass
class RepoSetup:
    """Per-repo configuration discovered during WARMUP.

    Phase 1 (deterministic) populates: entry_point, minimal_run, constraints, etc.
    Phase 2 (LLM-guided) populates: smoke_test, metrics, test_scale, instrumentation, etc.
    """
    # --- Phase 1: Deterministic ---
    entry_point: str = ""
    minimal_run: list[str] = field(default_factory=list)
    test_commands: list[list[str]] = field(default_factory=list)
    build_commands: list[list[str]] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    source: Literal["warmup_verified", "warmup_partial", "manual", "benchmark_case"] = "manual"
    verified_at: str = ""

    # --- Phase 2: LLM-Guided Discovery ---

    # Environment
    python_version: str = ""
    python_bin: str = "python"
    gpu_available: bool = False
    nsys_available: bool = False
    key_packages: dict[str, str] = field(default_factory=dict)

    # Smoke test
    smoke_test_passed: bool = False
    smoke_test_command: str = ""
    smoke_test_exit_code: int = -1
    smoke_test_stdout_tail: str = ""
    smoke_test_stderr_tail: str = ""
    smoke_test_elapsed_ms: float = 0
    smoke_test_notes: str = ""

    # Metrics
    metric_name: str = ""
    metric_grep: str = ""
    metric_baseline: float | None = None
    metric_lower_is_better: bool = False
    all_metrics: list[dict] = field(default_factory=list)

    # Test-scale configuration
    test_scale_applicable: bool = False
    test_scale_config_file: str = ""
    test_scale_params: list[dict] = field(default_factory=list)
    test_scale_command: str = ""
    test_scale_notes: str = ""

    # Instrumentation
    needs_instrumentation: bool = False
    has_nvtx: bool = False
    has_custom_timer: bool = False
    existing_nvtx_tags: list[str] = field(default_factory=list)
    suggested_instrumentation: list[dict] = field(default_factory=list)

    # Profile validation
    profile_sqlite: str = ""
    profile_sqlite_valid: bool = False
    profile_nsys_rep: str = ""
    profile_nsys_rep_valid: bool = False
    profile_notes: str = ""

    # Summary
    warmup_summary: str = ""
    warmup_llm_errors: list[str] = field(default_factory=list)
    human_action_items: list[str] = field(default_factory=list)
    warmup_raw_output: dict = field(default_factory=dict)  # raw LLM JSON, for inspection

    # ── Serialization / caching ──

    def to_dict(self) -> dict:
        """Serialize all fields to a plain dict (JSON-safe)."""
        d = {}
        for f in fields(self):
            d[f.name] = getattr(self, f.name)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RepoSetup:
        """Deserialize from a plain dict."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def save_cache(self, cache_path: str | Path) -> None:
        """Persist to a JSON cache file."""
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load_cache(cls, cache_path: str | Path) -> RepoSetup | None:
        """Load from a JSON cache file. Returns None if missing or corrupt."""
        p = Path(cache_path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def is_fresh(self) -> bool:
        """Whether this warmup result is verified and usable for downstream stages."""
        return self.source == "warmup_verified" and bool(self.entry_point)

    # ── Export ──

    def to_execution_config(self) -> dict:
        """Export the subset needed by the optimizer loop."""
        return {
            "entry_point": self.entry_point,
            "minimal_run": self.minimal_run,
            "metric_name": self.metric_name,
            "metric_grep": self.metric_grep,
            "metric_baseline": self.metric_baseline,
            "metric_lower_is_better": self.metric_lower_is_better,
            "test_scale_command": self.test_scale_command,
            "test_commands": self.test_commands,
            "build_commands": self.build_commands,
            "env_vars": self.env_vars,
            "constraints": self.constraints,
            "source": self.source,
        }
