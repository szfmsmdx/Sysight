"""PipelineRunner — five-stage orchestration.

Template Method pattern — skeleton: pre → run → post → ledger.
Stages: WARMUP → ANALYZE → INSTRUMENT → LEARN → OPTIMIZE → EXECUTE → LEARN.

Usage:
  runner = PipelineRunner(registry, provider_factory, knowledge)
  result = runner.run_full("trace.sqlite", "./repo")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineResult:
    run_id: str = ""
    repo_setup: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    patches: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)


class PipelineRunner:
    """Orchestrates WARMUP → ANALYZE → INSTRUMENT → LEARN → OPTIMIZE → EXECUTE → LEARN."""

    def __init__(self, registry, provider_factory, knowledge=None):
        self._registry = registry
        self._provider_factory = provider_factory
        self._knowledge = knowledge

    def run_full(self, profile: str, repo: str) -> PipelineResult:
        """Run all stages end-to-end."""
        errors: list[str] = []
        stages: list[str] = []

        # 0. WARMUP
        try:
            warmup_result = self.run_warmup(repo)
            stages.append("warmup")
        except Exception as e:
            errors.append(f"warmup: {e}")
            # WARMUP failure is non-fatal — continue with defaults

        # 1. ANALYZE
        try:
            analyze_provider = self._provider_factory("analyze")
            analyze_result = self.run_analyze(profile, repo, analyze_provider)
            stages.append("analyze")
        except Exception as e:
            errors.append(f"analyze: {e}")
            return PipelineResult(errors=errors, stages_completed=stages)

        # 1.1 INSTRUMENT (targeted NVTX tagging based on findings)
        try:
            instrument_provider = self._provider_factory("instrument")
            instrument_result = self.run_instrument(
                analyze_result.finding_set, repo,
                instrument_provider,
                run_dir=analyze_result.run_dir,
            )
            stages.append("instrument")
        except Exception as e:
            errors.append(f"instrument: {e}")
            # non-fatal — continue without instrumentation

        # 1.2 LEARN (from analyze findings)
        try:
            learn_provider = self._provider_factory("learn")
            findings_json = _serialize_findings(analyze_result.finding_set)
            self.run_learn(
                analyze_result.run_id,
                learn_provider,
                findings_json=findings_json,
            )
            stages.append("learn")
        except Exception as e:
            errors.append(f"learn(after analyze): {e}")

        # 2. OPTIMIZE (plan only — no file changes)
        try:
            optimize_provider = self._provider_factory("optimize")
            patches = self.run_optimize(
                analyze_result.finding_set, repo, optimize_provider,
            )
            stages.append("optimize")
        except Exception as e:
            errors.append(f"optimize: {e}")
            return PipelineResult(
                run_id=analyze_result.run_id,
                findings=[f.__dict__ for f in analyze_result.finding_set.findings],
                errors=errors, stages_completed=stages,
            )

        # 3. EXECUTE (apply patches, smoke test, timer comparison)
        execute_result = None
        if patches:
            try:
                execute_result = self.run_execute(
                    patches, repo,
                    run_id=analyze_result.run_id,
                    analyze_run_dir=analyze_result.run_dir,
                )
                stages.append("execute")
            except Exception as e:
                errors.append(f"execute: {e}")

        # 3.1 LEARN (from analyze findings + execute patches)
        try:
            learn_provider = self._provider_factory("learn")
            findings_json = _serialize_findings(analyze_result.finding_set)
            patches_json = _serialize_patches(
                execute_result.patches if execute_result else []
            )
            self.run_learn(
                analyze_result.run_id,
                learn_provider,
                findings_json=findings_json,
                patches_json=patches_json,
            )
            if "learn" not in stages:
                stages.append("learn")
        except Exception as e:
            errors.append(f"learn(after execute): {e}")

        return PipelineResult(
            run_id=analyze_result.run_id,
            findings=[f.__dict__ for f in analyze_result.finding_set.findings],
            patches=[p.__dict__ for p in (execute_result.patches if execute_result else [])],
            errors=errors, stages_completed=stages,
        )

    def run_warmup(self, repo: str):
        from sysight.pipeline.warmup import run_warmup
        return run_warmup(repo, self._knowledge)

    def run_analyze(self, profile: str, repo: str, provider):
        from sysight.pipeline.analyze import run_analyze
        return run_analyze(profile, repo, self._registry, provider, self._knowledge)

    def run_instrument(self, findings, repo: str, provider, run_dir=None):
        from sysight.pipeline.instrument import run_instrument
        return run_instrument(
            findings, repo,
            run_dir=run_dir,
        )

    def run_optimize(self, findings, repo: str, provider):
        from sysight.pipeline.optimize import run_optimize
        return run_optimize(
            findings, repo, self._registry, provider,
        )

    def run_execute(self, patches, repo: str, *,
                    run_id: str = "",
                    analyze_run_dir=None):
        from sysight.pipeline.execute import run_execute
        return run_execute(
            patches, repo,
            run_id=run_id,
            analyze_run_dir=analyze_run_dir,
        )

    def run_learn(self, run_id: str, provider=None,
                  findings_json: str = "", patches_json: str = ""):
        from sysight.pipeline.learn import run_learn
        return run_learn(
            run_id, self._knowledge, provider,
            findings_json=findings_json,
            patches_json=patches_json,
        )


def _serialize_findings(finding_set) -> str:
    return json.dumps({
        "summary": finding_set.summary,
        "findings": [
            {k: getattr(f, k) for k in ("finding_id", "category", "title", "priority",
                                         "file_path", "function", "line", "metric",
                                         "description", "suggestion")}
            for f in finding_set.findings
        ],
    }, indent=2, ensure_ascii=False)


def _serialize_patches(patches) -> str:
    return json.dumps([
        {"patch_id": p.patch_id, "finding_ids": p.finding_ids,
         "status": p.status, "reason": getattr(p, "reason", ""),
         "diff": getattr(p, "diff", "")}
        for p in patches
    ], indent=2, ensure_ascii=False)
