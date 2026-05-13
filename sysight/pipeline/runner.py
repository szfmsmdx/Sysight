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

    def run_full(
        self,
        profile: str,
        repo: str,
        *,
        verbose: bool = False,
        max_trials: int = 5,
    ) -> PipelineResult:
        """Run all stages end-to-end."""
        errors: list[str] = []
        stages: list[str] = []

        warmup_result = None

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
            analyze_result = self.run_analyze(
                profile, repo, analyze_provider,
                repo_setup=warmup_result.repo_setup if warmup_result else None,
                verbose=verbose,
            )
            stages.append("analyze")
        except Exception as e:
            errors.append(f"analyze: {e}")
            return PipelineResult(errors=errors, stages_completed=stages)

        # 1.1 INSTRUMENT (targeted NVTX tagging based on findings)
        try:
            instrument_provider = self._provider_factory("instrument")
            self.run_instrument(
                analyze_result.finding_set, repo,
                instrument_provider,
                run_dir=analyze_result.run_dir,
            )
            stages.append("instrument")
        except Exception as e:
            errors.append(f"instrument: {e}")
            # non-fatal — continue without instrumentation

        # 1.2 LEARN(1) — post-analyze: workspace structure facts only
        try:
            learn_provider = self._provider_fallback("learn", "optimize", "analyze")
            findings_json = _serialize_findings(analyze_result.finding_set)
            self.run_learn(
                analyze_result.run_id,
                learn_provider,
                findings_json=findings_json,
                repo=repo,
                learn_stage="post_analyze",
            )
            stages.append("learn")
        except Exception as e:
            errors.append(f"learn(after analyze): {e}")

        # 2. OPTIMIZE — iterative patch trials with measurement loop
        optimize_loop_result = None
        try:
            optimize_provider = self._provider_factory("optimize")
            repo_setup_obj = warmup_result.repo_setup if warmup_result else None
            measurement_plan = analyze_result.measurement_plan
            _normalize_measurement_plan(measurement_plan, repo, repo_setup_obj)
            optimize_loop_result = self.run_optimize(
                analyze_result.finding_set, repo, optimize_provider,
                measurement_plan=measurement_plan,
                repo_setup=repo_setup_obj,
                verbose=verbose,
                max_trials=max_trials,
            )
            stages.append("optimize")
        except Exception as e:
            errors.append(f"optimize: {e}")
            return PipelineResult(
                run_id=analyze_result.run_id,
                findings=[f.__dict__ for f in analyze_result.finding_set.findings],
                errors=errors, stages_completed=stages,
            )

        # 3. EXECUTE is integrated into run_optimize_trials (trial loop).
        # 3.1 LEARN(2) — post-optimize: worklog + experience update, overview corrections only
        try:
            learn_provider = self._provider_fallback("learn", "optimize", "analyze")
            findings_json = _serialize_findings(analyze_result.finding_set)
            patches_json = _serialize_optimize_result(optimize_loop_result)
            self.run_learn(
                analyze_result.run_id,
                learn_provider,
                findings_json=findings_json,
                patches_json=patches_json,
                repo=repo,
                learn_stage="post_optimize",
            )
            if "learn" not in stages:
                stages.append("learn")
        except Exception as e:
            errors.append(f"learn(after optimize): {e}")

        return PipelineResult(
            run_id=analyze_result.run_id,
            findings=[f.__dict__ for f in analyze_result.finding_set.findings],
            patches=(
                [t.to_dict() for t in optimize_loop_result.trials]
                if optimize_loop_result else []
            ),
            errors=errors, stages_completed=stages,
        )

    def run_warmup(self, repo: str):
        from sysight.pipeline.warmup import run_warmup
        return run_warmup(repo, self._knowledge)

    def run_analyze(self, profile: str, repo: str, provider, repo_setup=None, verbose: bool = False):
        from sysight.pipeline.analyze import run_analyze
        return run_analyze(
            profile, repo, self._registry, provider, self._knowledge,
            repo_setup=repo_setup,
            verbose=verbose,
        )

    def run_instrument(self, findings, repo: str, provider, run_dir=None):
        from sysight.pipeline.instrument import run_instrument
        return run_instrument(
            findings, repo,
            provider=provider,
            registry=self._registry,
            run_dir=run_dir,
        )

    def run_optimize(self, findings, repo: str, provider,
                     measurement_plan=None, repo_setup=None,
                     verbose: bool = False, max_trials: int = 5):
        from sysight.pipeline.optimize import run_optimize_trials
        if measurement_plan is None:
            from sysight.types.optimization import MeasurementPlan
            measurement_plan = MeasurementPlan()
        return run_optimize_trials(
            findings, measurement_plan, repo, self._registry, provider,
            repo_setup=repo_setup,
            knowledge=self._knowledge,
            verbose=verbose,
            max_trials=max_trials,
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
                  findings_json: str = "", patches_json: str = "",
                  repo: str = "", learn_stage: str = ""):
        from sysight.pipeline.learn import run_learn
        return run_learn(
            run_id, self._knowledge, provider,
            findings_json=findings_json,
            patches_json=patches_json,
            repo=repo,
            learn_stage=learn_stage,
        )

    def _provider_fallback(self, *stages: str):
        for stage in stages:
            provider = self._provider_factory(stage)
            if provider:
                return provider
        return None


def _normalize_measurement_plan(measurement_plan, repo: str, repo_setup) -> None:
    """Apply the same cwd + python-interpreter fixes as _cmd_optimize.

    Mutates measurement_plan in-place.
    """
    import subprocess as _sp
    from pathlib import Path as _Path
    if measurement_plan is None:
        return
    if not measurement_plan.run_command and repo_setup and repo_setup.minimal_run:
        measurement_plan.run_command = repo_setup.minimal_run[:]
    # Replace generic 'python'/'python3' with the venv interpreter from warmup cache
    if (
        measurement_plan.run_command
        and measurement_plan.run_command[0] in ("python", "python3")
        and repo_setup
        and repo_setup.minimal_run
        and repo_setup.minimal_run[0] not in ("python", "python3")
    ):
        measurement_plan.run_command = [
            repo_setup.minimal_run[0]
        ] + measurement_plan.run_command[1:]
    # Compute cwd relative to git root (for worktree compatibility)
    if not measurement_plan.cwd:
        repo_abs = _Path(repo).resolve()
        git_out = _sp.run(
            ["git", "-C", str(repo_abs), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True,
        )
        if git_out.returncode == 0:
            git_root = _Path(git_out.stdout.strip())
            try:
                measurement_plan.cwd = str(repo_abs.relative_to(git_root))
            except ValueError:
                pass  # repo is the git root itself — cwd stays empty


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


def _serialize_optimize_result(optimize_loop_result) -> str:
    """Serialize OptimizeLoopResult to JSON for LEARN stage consumption."""
    if optimize_loop_result is None:
        return json.dumps([], ensure_ascii=False)
    return json.dumps({
        "run_id": optimize_loop_result.run_id,
        "best_commit": optimize_loop_result.best_commit,
        "accepted_count": optimize_loop_result.accepted_count,
        "rejected_count": optimize_loop_result.rejected_count,
        "baseline": optimize_loop_result.baseline.to_dict() if hasattr(optimize_loop_result.baseline, "to_dict") else {},
        "best_measurement": optimize_loop_result.best_measurement.to_dict() if hasattr(optimize_loop_result.best_measurement, "to_dict") else {},
        "trials": [t.to_dict() for t in optimize_loop_result.trials],
    }, indent=2, ensure_ascii=False, default=str)
