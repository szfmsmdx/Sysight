"""PipelineRunner — four-stage orchestration.

Template Method pattern — skeleton: pre → run → post → ledger.
Stages: WARMUP → ANALYZE → LEARN → OPTIMIZE → LEARN.

Usage:
  runner = PipelineRunner(registry, provider_factory, knowledge)
  result = runner.run_full("trace.sqlite", "./repo")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class PipelineResult:
    run_id: str = ""
    repo_setup: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    patches: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)


class PipelineRunner:
    """Orchestrates WARMUP → ANALYZE → LEARN → OPTIMIZE → LEARN."""

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
            warmup_provider = self._provider_factory("warmup")
            warmup_result = self.run_warmup(repo, warmup_provider)
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

        # 1.1 LEARN (from analyze findings)
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

        # 2. OPTIMIZE
        try:
            optimize_provider = self._provider_factory("optimize")
            optimize_result = self.run_optimize(
                analyze_result.finding_set, repo, optimize_provider
            )
            stages.append("optimize")
        except Exception as e:
            errors.append(f"optimize: {e}")
            return PipelineResult(
                run_id=analyze_result.run_id,
                findings=[f.__dict__ for f in analyze_result.finding_set.findings],
                errors=errors, stages_completed=stages,
            )

        # 2.1 LEARN (from analyze findings + optimize patches)
        try:
            learn_provider = self._provider_factory("learn")
            findings_json = _serialize_findings(analyze_result.finding_set)
            patches_json = _serialize_patches(optimize_result.patches)
            self.run_learn(
                analyze_result.run_id,
                learn_provider,
                findings_json=findings_json,
                patches_json=patches_json,
            )
            if "learn" not in stages:
                stages.append("learn")
        except Exception as e:
            errors.append(f"learn(after optimize): {e}")

        return PipelineResult(
            run_id=analyze_result.run_id,
            findings=[f.__dict__ for f in analyze_result.finding_set.findings],
            patches=[p.__dict__ for p in optimize_result.patches],
            errors=errors, stages_completed=stages,
        )

    def run_warmup(self, repo: str, provider=None):
        from sysight.pipeline.warmup import run_warmup
        return run_warmup(repo, self._registry, provider, self._knowledge)

    def run_analyze(self, profile: str, repo: str, provider):
        from sysight.pipeline.analyze import run_analyze
        return run_analyze(profile, repo, self._registry, provider, self._knowledge)

    def run_optimize(self, findings, repo: str, provider):
        from sysight.pipeline.optimize import run_optimize
        return run_optimize(findings, repo, self._registry, provider, self._knowledge)

    def run_learn(self, run_id: str, provider=None,
                  findings_json: str = "", patches_json: str = ""):
        from sysight.pipeline.learn import run_learn
        return run_learn(
            run_id, self._knowledge, provider,
            findings_json=findings_json,
            patches_json=patches_json,
        )


def _serialize_findings(finding_set) -> str:
    """Serialize findings to JSON for learn input."""
    findings_data = []
    for f in finding_set.findings:
        findings_data.append({
            "finding_id": f.finding_id,
            "category": f.category,
            "title": f.title,
            "priority": f.priority,
            "file_path": f.file_path,
            "function": f.function,
            "line": f.line,
            "description": f.description,
            "suggestion": f.suggestion,
        })
    return json.dumps({
        "summary": finding_set.summary,
        "findings": findings_data,
    }, indent=2, ensure_ascii=False)


def _serialize_patches(patches) -> str:
    """Serialize patches to JSON for learn input."""
    patches_data = []
    for p in patches:
        patches_data.append({
            "patch_id": p.patch_id,
            "finding_id": p.finding_id,
            "status": p.status,
            "reason": getattr(p, "reason", ""),
            "diff": getattr(p, "diff", ""),
        })
    return json.dumps(patches_data, indent=2, ensure_ascii=False)
