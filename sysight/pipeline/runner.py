"""PipelineRunner — four-stage orchestration.

Template Method pattern — skeleton: pre → run → post → ledger.
Stages: WARMUP → ANALYZE → OPTIMIZE → LEARN.

Usage:
  runner = PipelineRunner(registry, provider_factory, knowledge)
  result = runner.run_full("trace.sqlite", "./repo")
"""

from __future__ import annotations

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
    """Orchestrates WARMUP → ANALYZE → OPTIMIZE → LEARN."""

    def __init__(self, registry, provider_factory, knowledge=None):
        self._registry = registry
        self._provider_factory = provider_factory
        self._knowledge = knowledge

    def run_full(self, profile: str, repo: str) -> PipelineResult:
        """Run all four stages end-to-end."""
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

        # 3. LEARN
        try:
            learn_provider = self._provider_factory("learn")
            self.run_learn(analyze_result.run_id, learn_provider)
            stages.append("learn")
        except Exception as e:
            errors.append(f"learn: {e}")

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

    def run_learn(self, run_id: str, provider=None):
        from sysight.pipeline.learn import run_learn
        return run_learn(run_id, self._knowledge, provider)
