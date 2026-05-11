"""Outer agent loop orchestration for continuous repo optimization."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sysight.pipeline.worklog import WorklogRow, append_many, default_worklog_path
from sysight.types.findings import LocalizedFinding
from sysight.types.optimization import MeasurementPlan, MetricSpec
from sysight.types.repo_setup import RepoSetup
from sysight.utils.cache import cache_dir, safe_name


@dataclass
class AgentLoopIteration:
    iteration: int
    repo: str = ""
    profile: str = ""
    analyze_run_id: str = ""
    analyze_dir: str = ""
    optimize_dir: str = ""
    worktree_path: str = ""
    best_commit: str = ""
    metric_name: str = ""
    metric_before: float | None = None
    metric_after: float | None = None
    metric_delta_pct: float | None = None
    accepted_count: int = 0
    rejected_count: int = 0
    learn_summary: str = ""
    status: str = "ok"
    errors: list[str] = field(default_factory=list)


@dataclass
class AgentLoopResult:
    loop_id: str = ""
    run_dir: str = ""
    initial_repo: str = ""
    final_repo: str = ""
    initial_profile: str = ""
    final_profile: str = ""
    iterations: list[AgentLoopIteration] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["iterations"] = [asdict(item) for item in self.iterations]
        return data


def run_agent_loop(
    profile: str,
    repo: str,
    registry,
    provider_factory,
    knowledge=None,
    *,
    max_loops: int = 2,
    max_trials: int = 5,
    verbose: bool = False,
    refresh_profiles: bool = True,
    profile_command: str = "",
    stage_wall_seconds: int = 1800,
) -> AgentLoopResult:
    """Run analyze -> optimize trials -> learn repeatedly.

    Accepted optimizer commits advance the next outer-loop repository to the
    optimizer worktree. A refreshed Nsight SQLite profile is generated for the
    next iteration when possible.
    """
    root = Path(repo).resolve()
    profile_path = Path(profile).resolve()
    digest = hashlib.sha1(f"{root}|{profile_path}|agent-loop".encode()).hexdigest()[:8]
    loop_id = f"loop-{digest}-{int(time.time())}"
    run_dir = cache_dir("agent-loops", loop_id)

    errors: list[str] = []
    result = AgentLoopResult(
        loop_id=loop_id,
        run_dir=str(run_dir),
        initial_repo=str(root),
        final_repo=str(root),
        initial_profile=str(profile_path),
        final_profile=str(profile_path),
    )

    repo_setup = _load_repo_setup(root, knowledge, errors)
    current_repo = root
    current_profile = profile_path

    for idx in range(1, max(1, int(max_loops or 1)) + 1):
        iteration_dir = run_dir / f"iter-{idx:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        item = AgentLoopIteration(
            iteration=idx,
            repo=str(current_repo),
            profile=str(current_profile),
        )

        try:
            analyze_provider = provider_factory("analyze")
            if not analyze_provider:
                raise RuntimeError("no analyze provider configured")
            analyze_result = _run_analyze_iteration(
                current_profile,
                current_repo,
                registry,
                analyze_provider,
                knowledge,
                repo_setup,
                run_id=f"{loop_id}-iter-{idx:03d}",
                verbose=verbose,
                max_wall_seconds=stage_wall_seconds,
            )
            item.analyze_run_id = analyze_result.run_id
            item.analyze_dir = str(analyze_result.run_dir or "")
            item.errors.extend(analyze_result.errors)
        except Exception as e:
            item.status = "error"
            item.errors.append(f"analyze: {e}")
            result.errors.extend(item.errors)
            result.iterations.append(item)
            _record_iteration(run_dir, result, item)
            break

        try:
            optimize_provider = provider_factory("optimize")
            if not optimize_provider:
                raise RuntimeError("no optimize provider configured")
            optimize_result = _run_optimize_iteration(
                analyze_result,
                current_repo,
                registry,
                optimize_provider,
                knowledge,
                repo_setup,
                iteration_dir / "optimize",
                max_trials=max_trials,
                verbose=verbose,
                stage_wall_seconds=stage_wall_seconds,
                memory_repo=root,
            )
            item.optimize_dir = str(optimize_result.run_dir or "")
            item.worktree_path = optimize_result.worktree_path
            item.best_commit = optimize_result.best_commit
            item.accepted_count = optimize_result.accepted_count
            item.rejected_count = optimize_result.rejected_count
            item.metric_name = optimize_result.best_measurement.primary_metric
            item.metric_before = optimize_result.baseline.primary_value
            item.metric_after = optimize_result.best_measurement.primary_value
            item.metric_delta_pct = _metric_delta(
                optimize_result.baseline.primary_value,
                optimize_result.best_measurement.primary_value,
                analyze_result.measurement_plan,
            )
            item.errors.extend(optimize_result.errors)
        except Exception as e:
            item.status = "error"
            item.errors.append(f"optimize: {e}")
            result.errors.extend(item.errors)
            result.iterations.append(item)
            _record_iteration(run_dir, result, item)
            break

        if item.accepted_count > 0 and item.worktree_path:
            current_repo = Path(item.worktree_path).resolve()
            result.final_repo = str(current_repo)

        learn_provider = _provider_fallback(provider_factory, "learn", "optimize", "analyze")
        learn_summary = _learn_iteration(
            loop_id,
            idx,
            analyze_result,
            optimize_result,
            knowledge,
            learn_provider,
            repo=item.repo,
            memory_repo=root,
            stage_wall_seconds=stage_wall_seconds,
        )
        item.learn_summary = learn_summary

        if idx < max_loops and refresh_profiles:
            profile_status = run_dir / "profiles" / f"iter-{idx + 1:03d}.profile_refresh.json"
            refreshed = _refresh_profile(
                current_repo,
                current_profile,
                repo_setup,
                run_dir / "profiles",
                idx + 1,
                profile_command=profile_command,
            )
            if refreshed:
                current_profile = refreshed
                result.final_profile = str(current_profile)
            else:
                item.errors.append(f"profile refresh failed; reusing previous profile; see {profile_status}")

        result.iterations.append(item)
        _record_iteration(run_dir, result, item)

    _write_result(run_dir, result)
    return result


def _load_repo_setup(root: Path, knowledge, errors: list[str]) -> RepoSetup:
    try:
        from sysight.pipeline.warmup import load_or_run_repo_setup
        return load_or_run_repo_setup(str(root), knowledge)
    except Exception as e:
        errors.append(f"warmup: {e}")
        return RepoSetup()


def _run_analyze_iteration(
    profile: Path,
    repo: Path,
    registry,
    provider,
    knowledge,
    repo_setup: RepoSetup,
    *,
    run_id: str,
    verbose: bool,
    max_wall_seconds: int,
):
    from sysight.pipeline.analyze import run_analyze
    return run_analyze(
        str(profile),
        str(repo),
        registry,
        provider,
        knowledge,
        repo_setup=repo_setup,
        run_id=run_id,
        verbose=verbose,
        max_wall_seconds=max_wall_seconds,
    )


def _run_optimize_iteration(
    analyze_result,
    repo: Path,
    registry,
    provider,
    knowledge,
    repo_setup: RepoSetup,
    run_dir: Path,
    *,
    max_trials: int,
    verbose: bool,
    stage_wall_seconds: int,
    memory_repo: Path,
):
    from sysight.pipeline.optimize import run_optimize_trials
    finding_set, measurement_plan = _optimizer_inputs_from_analyze(
        analyze_result, repo_setup, repo=str(repo)
    )
    return run_optimize_trials(
        finding_set,
        measurement_plan,
        str(repo),
        registry,
        provider,
        repo_setup=repo_setup,
        knowledge=knowledge,
        run_dir=run_dir,
        max_trials=max_trials,
        verbose=verbose,
        max_agent_wall_seconds=stage_wall_seconds,
        memory_repo=str(memory_repo),
    )


def _optimizer_inputs_from_analyze(analyze_result, repo_setup: RepoSetup, repo: str = ""):
    import subprocess as _sp
    from pathlib import Path as _Path
    finding_set = analyze_result.finding_set
    measurement_plan = analyze_result.measurement_plan or MeasurementPlan()
    if not measurement_plan.run_command:
        measurement_plan.run_command = repo_setup.minimal_run[:]
    # Replace generic 'python'/'python3' with the venv interpreter from warmup cache
    if (
        measurement_plan.run_command
        and measurement_plan.run_command[0] in ("python", "python3")
        and repo_setup.minimal_run
        and repo_setup.minimal_run[0] not in ("python", "python3")
    ):
        measurement_plan.run_command = [
            repo_setup.minimal_run[0]
        ] + measurement_plan.run_command[1:]
    # Compute cwd relative to git root (for worktree compatibility)
    if not measurement_plan.cwd and repo:
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
    if not measurement_plan.metrics:
        metric = _fallback_metric_from_repo_setup(repo_setup)
        if metric:
            measurement_plan.metrics = [metric]
            measurement_plan.ensure_primary()

    if not finding_set.findings:
        entry = _entry_file_from_repo_setup(repo_setup)
        finding_set.summary = finding_set.summary or (
            "Analyzer produced no accepted findings; optimizer may inspect the repo "
            "and propose end-to-end performance experiments."
        )
        finding_set.findings = [
            LocalizedFinding(
                finding_id="C0:end_to_end",
                category="C0",
                title="End-to-end optimization target",
                priority="medium",
                confidence="unresolved",
                file_path=entry,
                line=1 if entry else None,
                metric=measurement_plan.primary_metric.name if measurement_plan.primary_metric else "",
                description=(
                    "Fallback finding created because analyze did not return accepted "
                    "code findings. Use the measurement plan and repo inspection to "
                    "choose a small, measurable optimization."
                ),
                suggestion="Read the entrypoint and hot-path files, then try one low-risk patch.",
                status="accepted",
            )
        ]
    return finding_set, measurement_plan


def _fallback_metric_from_repo_setup(repo_setup: RepoSetup) -> MetricSpec | None:
    if not repo_setup.minimal_run:
        return None
    return MetricSpec(
        name=repo_setup.metric_name or "iteration_ms",
        regex=r"(?:time(?:\s+per\s+iteration)?|iteration(?:_ms)?|latency)[^0-9\r\n]*([0-9]+(?:\.[0-9]+)?)\s*ms",
        group=1,
        aggregation="mean",
        lower_is_better=True,
        primary=True,
        unit="ms",
        source="warmup_fallback",
        rationale="Fallback metric inferred from warmup entry command when analyze did not return a measurement plan.",
    )


def _entry_file_from_repo_setup(repo_setup: RepoSetup) -> str:
    cmd = repo_setup.minimal_run or _split_entry(repo_setup.entry_point)
    for item in cmd:
        text = str(item)
        if text.endswith((".py", ".sh", ".ps1")):
            return text.replace("\\", "/")
    return ""


def _learn_iteration(
    loop_id: str,
    iteration: int,
    analyze_result,
    optimize_result,
    knowledge,
    provider,
    *,
    repo: str,
    memory_repo: Path,
    stage_wall_seconds: int,
) -> str:
    from sysight.pipeline.learn import run_learn
    findings_json = json.dumps(_findings_payload(analyze_result.finding_set), indent=2, ensure_ascii=False)
    optimize_json = json.dumps({
        "loop_id": loop_id,
        "iteration": iteration,
        "baseline": optimize_result.baseline.to_dict(),
        "best_measurement": optimize_result.best_measurement.to_dict(),
        "accepted_count": optimize_result.accepted_count,
        "rejected_count": optimize_result.rejected_count,
        "best_commit": optimize_result.best_commit,
        "trials": [trial.to_dict() for trial in optimize_result.trials],
    }, indent=2, ensure_ascii=False, default=str)
    learn = run_learn(
        f"{loop_id}-iter-{iteration:03d}",
        knowledge,
        provider,
        findings_json=findings_json,
        patches_json=optimize_json,
        repo=str(memory_repo),
        max_wall_seconds=min(stage_wall_seconds, 900) if stage_wall_seconds else 0,
    )
    return learn.summary


def _findings_payload(finding_set) -> dict:
    return {
        "summary": finding_set.summary,
        "findings": [
            {
                "finding_id": f.finding_id,
                "category": f.category,
                "title": f.title,
                "file_path": f.file_path,
                "line": f.line,
                "metric": f.metric,
                "description": f.description,
                "suggestion": f.suggestion,
                "status": f.status,
            }
            for f in finding_set.findings
        ],
    }


def _provider_fallback(provider_factory, *stages: str):
    for stage in stages:
        provider = provider_factory(stage)
        if provider:
            return provider
    return None


def _metric_delta(before: float | None, after: float | None, plan: MeasurementPlan) -> float | None:
    if before is None or after is None or before == 0:
        return None
    primary = plan.primary_metric
    lower = primary.lower_is_better if primary else True
    if lower:
        return (before - after) / before * 100.0
    return (after - before) / before * 100.0


def _refresh_profile(
    repo: Path,
    input_profile: Path,
    repo_setup: RepoSetup,
    profiles_dir: Path,
    iteration: int,
    *,
    profile_command: str = "",
) -> Path | None:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    prefix = profiles_dir / f"iter-{iteration:03d}"
    status_path = profiles_dir / f"iter-{iteration:03d}.profile_refresh.json"
    if profile_command:
        cmd = _profile_command_from_template(profile_command, repo, input_profile, prefix, iteration)
    else:
        if not shutil.which("nsys"):
            _write_profile_refresh_status(status_path, [], error="nsys not found on PATH")
            return None
        run_cmd = repo_setup.minimal_run or _split_entry(repo_setup.entry_point)
        if not run_cmd:
            _write_profile_refresh_status(status_path, [], error="no profile command or warmup minimal_run available")
            return None
        cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,cublas",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "-o",
            str(prefix),
            *run_cmd,
        ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=repo_setup_timeout(repo_setup),
        )
    except FileNotFoundError as e:
        _write_profile_refresh_status(status_path, cmd, error=str(e))
        return None
    except subprocess.TimeoutExpired as e:
        _write_profile_refresh_status(status_path, cmd, error=f"timed out after {e.timeout}s")
        return None
    _write_profile_process_logs(profiles_dir, iteration, "profile", proc)
    if proc.returncode != 0:
        _write_profile_refresh_status(status_path, cmd, returncode=proc.returncode)
        return None

    sqlite_path = prefix.with_suffix(".sqlite")
    if sqlite_path.exists():
        _write_profile_refresh_status(status_path, cmd, sqlite_path=str(sqlite_path), returncode=0)
        return sqlite_path

    rep_path = prefix.with_suffix(".nsys-rep")
    if not rep_path.exists() or not shutil.which("nsys"):
        _write_profile_refresh_status(status_path, cmd, error="nsys-rep not found after profile")
        return None
    export_cmd = [
        "nsys",
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(rep_path),
    ]
    try:
        proc = subprocess.run(export_cmd, cwd=str(repo), capture_output=True, text=True, timeout=600)
    except FileNotFoundError as e:
        _write_profile_refresh_status(status_path, cmd, export_cmd=export_cmd, error=str(e))
        return None
    except subprocess.TimeoutExpired as e:
        _write_profile_refresh_status(status_path, cmd, export_cmd=export_cmd, error=f"export timed out after {e.timeout}s")
        return None
    _write_profile_process_logs(profiles_dir, iteration, "export", proc)
    if proc.returncode == 0 and sqlite_path.exists():
        _write_profile_refresh_status(
            status_path,
            cmd,
            export_cmd=export_cmd,
            sqlite_path=str(sqlite_path),
            nsys_rep_path=str(rep_path),
            returncode=0,
        )
        return sqlite_path
    _write_profile_refresh_status(
        status_path,
        cmd,
        export_cmd=export_cmd,
        returncode=proc.returncode,
        error="sqlite export failed",
    )
    return None


def repo_setup_timeout(repo_setup: RepoSetup) -> int:
    return 1800 if repo_setup.minimal_run else 600


def _profile_command_from_template(
    command: str,
    repo: Path,
    input_profile: Path,
    prefix: Path,
    iteration: int,
) -> list[str]:
    rendered = command.format(
        repo=str(repo),
        input_profile=str(input_profile),
        output=str(prefix),
        sqlite=str(prefix.with_suffix(".sqlite")),
        nsys_rep=str(prefix.with_suffix(".nsys-rep")),
        iteration=iteration,
    )
    return shlex.split(rendered)


def _write_profile_process_logs(profiles_dir: Path, iteration: int, name: str, proc) -> None:
    stem = profiles_dir / f"iter-{iteration:03d}.{name}"
    stem.with_suffix(".stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    stem.with_suffix(".stderr.log").write_text(proc.stderr or "", encoding="utf-8")


def _write_profile_refresh_status(path: Path, cmd: list[str], **data) -> None:
    payload = {"cmd": cmd, **data}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _split_entry(entry_point: str) -> list[str]:
    if not entry_point:
        return []
    try:
        return shlex.split(entry_point)
    except ValueError:
        return entry_point.split()


def _record_iteration(run_dir: Path, result: AgentLoopResult, item: AgentLoopIteration) -> None:
    row = WorklogRow(
        loop_id=result.loop_id,
        iteration=item.iteration,
        stage="agent_loop",
        status=item.status,
        repo=item.repo,
        profile=item.profile,
        run_id=item.analyze_run_id,
        artifact_dir=str(run_dir / f"iter-{item.iteration:03d}"),
        commit_after=item.best_commit,
        metric_name=item.metric_name,
        metric_before=item.metric_before,
        metric_after=item.metric_after,
        metric_delta_pct=item.metric_delta_pct,
        accepted_count=item.accepted_count,
        rejected_count=item.rejected_count,
        coding_summary=item.learn_summary,
        reason="accepted" if item.accepted_count else "no accepted trial",
        errors=item.errors,
    )
    append_many([run_dir / "worklog.csv", default_worklog_path()], row)
    _write_result(run_dir, result)


def _write_result(run_dir: Path, result: AgentLoopResult) -> None:
    (run_dir / "agent_loop_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
