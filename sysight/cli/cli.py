"""Unified CLI for Sysight.

Facade pattern — single entry point that hides internal module structure.

Commands:
  sysight warmup   <repo>
  sysight analyze  <profile> --repo <repo>
  sysight optimize <analysis.json> --repo <repo>
  sysight learn    <run-id>
  sysight agent-loop <profile> --repo <repo>
  sysight full     <profile> --repo <repo>
  sysight tool     <category> <name> [args...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="sysight",
        description="Sysight: AI-powered GPU performance analysis and optimization",
    )
    sub = parser.add_subparsers(dest="command")

    # warmup
    p = sub.add_parser("warmup", help="Explore a repo (Phase 1 only — deterministic scanning)")
    p.add_argument("repo", help="Path to repo root")
    p.add_argument("--force", action="store_true", help="Force re-run even if warmup cache exists")

    # analyze
    p = sub.add_parser("analyze", help="Analyze an Nsight Systems profile")
    p.add_argument("profile", help="Path to .sqlite profile")
    p.add_argument("--repo", required=True, help="Path to repo root")
    p.add_argument("--debug", action="store_true", help="Print LLM I/O to terminal (always logged to debug.log)")
    p.add_argument("--dump-prompt", help="Write the first-turn analyze prompt bundle to this file or directory")
    p.add_argument("--no-run", action="store_true", help="Only build/dump the first-turn prompt; do not call the LLM")

    # instrument
    p = sub.add_parser("instrument", help="Targeted NVTX tagging based on analyzer findings")
    p.add_argument("run_id", help="Analyze run_id (or path to analyze_raw.json)")
    p.add_argument("--repo", required=True, help="Path to repo root")
    p.add_argument("--debug", action="store_true", help="Print LLM I/O to terminal (always logged to instrument_debug.log)")

    # optimize
    p = sub.add_parser("optimize", help="Optimize findings from an analysis")
    p.add_argument("run_id", help="Analyze run_id (or path to analyze_raw.json)")
    p.add_argument("--repo", required=True, help="Path to repo root")
    p.add_argument("--debug", action="store_true", help="Print LLM I/O to terminal (always logged to optimize_debug.log)")
    p.add_argument("--max-trials", type=int, default=5, help="Maximum optimizer trial iterations")
    p.add_argument("--legacy-patches", action="store_true", help="Use legacy patch batch + execute flow")

    # bench-optimize
    p = sub.add_parser("bench-optimize", help="Run optimizer benchmark against optimizer-bench cases")
    p.add_argument("cases", nargs="*", default=["case_1"], help="Case IDs to run (default: case_1)")
    p.add_argument("--all", action="store_true", help="Run all cases")
    p.add_argument("--debug", action="store_true", help="Print LLM I/O to terminal (always logged to optimize_debug.log)")
    p.add_argument("--bench-dir", default="optimizer-bench", help="Path to optimizer-bench directory")

    # learn
    p = sub.add_parser("learn", help="Post-session learning")
    p.add_argument("run_id", help="Run ID to process")
    p.add_argument("--repo", default="", help="Optional repo root for workspace namespace")

    # outer agent loop
    p = sub.add_parser("agent-loop", help="Run repeated analyze -> optimize -> learn loops")
    p.add_argument("profile", help="Path to initial .sqlite profile")
    p.add_argument("--repo", required=True, help="Path to repo root")
    p.add_argument("--max-loops", type=int, default=2, help="Maximum outer loop iterations")
    p.add_argument("--max-trials", type=int, default=5, help="Optimizer trials per outer loop")
    p.add_argument("--stage-wall-seconds", type=int, default=1800, help="Wall-clock limit for one LLM sub-stage; 0 disables")
    p.add_argument("--debug", action="store_true", help="Print LLM I/O to terminal")
    p.add_argument("--no-profile-refresh", action="store_true", help="Reuse the input profile between outer loops")
    p.add_argument(
        "--profile-command",
        default="",
        help=(
            "Optional profile command template. Supports {repo}, {output}, "
            "{input_profile}, {sqlite}, {nsys_rep}, {iteration}."
        ),
    )

    # full pipeline
    p = sub.add_parser("full", help="Run the full pipeline")
    p.add_argument("profile", help="Path to .sqlite profile")
    p.add_argument("--repo", required=True, help="Path to repo root")

    # tool (for external agent use)
    p = sub.add_parser("tool", help="Execute a single tool directly")
    p.add_argument("category", help="Tool category: scanner, nsys_sql, memory")
    p.add_argument("name", help="Tool name")
    p.add_argument("--write", action="store_true", help="Allow non-read-only tools such as memory_write")
    p.add_argument("args", nargs="*", help="Tool arguments as key=value pairs")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    dispatch = {
        "warmup": _cmd_warmup,
        "analyze": _cmd_analyze,
        "instrument": _cmd_instrument,
        "optimize": _cmd_optimize,
        "bench-optimize": _cmd_bench_optimize,
        "learn": _cmd_learn,
        "agent-loop": _cmd_agent_loop,
        "full": _cmd_full,
        "tool": _cmd_tool,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)


def _setup():
    """Create registry, load configs."""
    from sysight.tools.registry import ToolRegistry
    from sysight.tools import register_all_tools
    from sysight.agent.config_loader import load_config
    from sysight.agent.provider import create_provider
    from sysight.wiki.store import WikiRepository

    registry = ToolRegistry()
    register_all_tools(registry)

    try:
        configs = load_config()
    except FileNotFoundError:
        configs = {}

    def provider_factory(stage: str):
        cfg = configs.get(stage)
        if not cfg or not cfg.api_key:
            return None
        return create_provider(cfg)

    knowledge = WikiRepository()
    return registry, provider_factory, knowledge


def _cmd_warmup(args):
    registry, _, knowledge = _setup()

    from sysight.pipeline.warmup import run_warmup
    result = run_warmup(args.repo, knowledge, force=getattr(args, 'force', False))

    s = result.summary
    print(f"\n{'='*60}")
    print(f"  Warmup 完成 — {s.get('source', '?')}")
    print(f"{'='*60}")
    print(f"  入口:     {s.get('entry_point', '?')}")
    print(f"  profile:  {s.get('profile_sqlite', '?')}")
    print(f"  热路径:   {s.get('hot_path_count', 0)} 文件")

    warns = s.get('warnings', [])
    errs = s.get('errors', [])
    if warns or errs:
        print()
        for w in warns:
            print(f"  ⚠ {w}")
        for e in errs:
            print(f"  ✗ {e}")

    print(f"\n  Cache: {result.summary.get('cache_path', '')}")
    print(f"  Overview: {result.summary.get('overview_path', '')}")


def _cmd_analyze(args):
    registry, provider_factory, knowledge = _setup()
    from sysight.pipeline.warmup import load_or_run_repo_setup
    repo_setup = load_or_run_repo_setup(args.repo, knowledge)
    if args.dump_prompt or args.no_run:
        from sysight.pipeline.analyze import build_analyze_first_turn
        bundle = build_analyze_first_turn(
            args.profile, args.repo, registry, knowledge, repo_setup=repo_setup,
        )
        if args.dump_prompt:
            _write_prompt_dump(args.dump_prompt, bundle)
        if args.no_run:
            print(json.dumps({
                "status": "prompt_built",
                "dump_prompt": args.dump_prompt,
                "system_chars": len(bundle["system_prompt"]),
                "user_chars": len(bundle["messages"][0]["content"]),
                "tools": len(bundle.get("tools") or []),
            }, indent=2, ensure_ascii=False))
            return

    provider = provider_factory("analyze")
    if not provider:
        print("Error: no analyze provider configured", file=sys.stderr)
        sys.exit(1)

    from sysight.pipeline.analyze import run_analyze
    result = run_analyze(
        args.profile, args.repo, registry, provider, knowledge,
        repo_setup=repo_setup,
        verbose=getattr(args, 'debug', False),
    )
    # Output location already printed by run_analyze; print run_id for scripting
    print(result.run_id)


def _resolve_analyze_raw(run_id_or_path: str) -> Path:
    """Resolve a run_id or file path to an analyze_raw.json Path."""
    p = Path(run_id_or_path)
    # Direct file path
    if p.suffix == ".json" and p.exists():
        return p
    # Directory containing analyze_raw.json
    if p.is_dir():
        candidate = p / "analyze_raw.json"
        if candidate.exists():
            return candidate
    # run_id lookup in current cache location, then legacy location.
    from sysight.utils.cache import cache_dir
    for base in (
        cache_dir("analysis-runs"),
        Path.cwd() / ".sysight" / "analysis-runs",
    ):
        candidate = base / run_id_or_path / "analyze_raw.json"
        if candidate.exists():
            return candidate
    print(f"Error: cannot resolve analyze_raw.json from '{run_id_or_path}'", file=sys.stderr)
    sys.exit(1)


def _load_findings(run_id_or_path: str):
    """Parse analyze_raw.json (by run_id or path) into LocalizedFindingSet."""
    path = _resolve_analyze_raw(run_id_or_path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    from sysight.types.findings import LocalizedFindingSet, LocalizedFinding
    return LocalizedFindingSet(
        run_id=data.get("run_id", ""),
        summary=data.get("summary", ""),
        findings=[
            LocalizedFinding(
                finding_id=f.get("finding_id", ""),
                category=f.get("category", ""),
                title=f.get("title", ""),
                priority=f.get("priority", "medium"),
                confidence=f.get("confidence", "unresolved"),
                metric=f.get("metric", ""),
                file_path=f.get("file_path"),
                function=f.get("function"),
                line=f.get("line"),
                description=f.get("description", ""),
                suggestion=f.get("suggestion", ""),
                status=f.get("status", "accepted"),
            )
            for f in data.get("findings", [])
            if f.get("status") == "accepted"
        ],
    )


def _load_measurement_plan(run_id_or_path: str):
    path = _resolve_analyze_raw(run_id_or_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    from sysight.types.optimization import MeasurementPlan
    return MeasurementPlan.from_dict(data.get("measurement_plan") or {})


def _cmd_instrument(args):
    registry, provider_factory, knowledge = _setup()
    provider = provider_factory("instrument")
    if not provider:
        print("Error: no instrument provider configured", file=sys.stderr)
        sys.exit(1)

    findings = _load_findings(args.run_id)
    raw_path = _resolve_analyze_raw(args.run_id)
    run_dir = raw_path.parent  # .sysight/analysis-runs/<run_id>/

    from sysight.pipeline.instrument import run_instrument
    result = run_instrument(
        findings, args.repo,
        verbose=getattr(args, 'debug', False),
        run_dir=run_dir,
    )

    s = result.summary
    warns = result.warnings
    errs  = result.errors

    merge_warns = [w for w in warns if w.startswith("__merge__:")]
    skip_warns  = [w for w in warns if w.startswith("__skip__:")]
    other_warns = [w for w in warns if not w.startswith(("__merge__:", "__skip__:"))]

    # 从合并链中提取最终合并后的 timer 数量（去重 merged label）
    final_timers = s.get("timer_count", 0) - len(merge_warns)

    print(f"\n  ✅  埋点完成  |  {final_timers} 个计时器  |  {len(result.modified_files)} 个文件")

    if merge_warns:
        print(f"  📦  {len(merge_warns)} 处重叠范围已自动合并")
    if skip_warns:
        print(f"  ⏭   {len(skip_warns)} 处已跳过（重复埋点）")
    if other_warns:
        for w in other_warns:
            print(f"  ⚠   {w}")
    if errs:
        for e in errs:
            print(f"  ✗   {e}")


def _write_prompt_dump(target: str, bundle: dict) -> None:
    path = Path(target)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return

    path.mkdir(parents=True, exist_ok=True)
    (path / "system_prompt.txt").write_text(bundle["system_prompt"], encoding="utf-8")
    (path / "user_prompt.txt").write_text(bundle["messages"][0]["content"], encoding="utf-8")
    (path / "full_prompt.txt").write_text(bundle["full_prompt"], encoding="utf-8")
    (path / "first_turn.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _cmd_optimize(args):
    registry, provider_factory, knowledge = _setup()
    provider = provider_factory("optimize")
    if not provider:
        print("Error: no optimize provider configured", file=sys.stderr)
        sys.exit(1)

    findings = _load_findings(args.run_id)
    raw_path = _resolve_analyze_raw(args.run_id)
    analyze_run_dir = raw_path.parent  # .sysight/analysis-runs/<run_id>/
    from sysight.pipeline.warmup import load_or_run_repo_setup
    repo_setup = load_or_run_repo_setup(args.repo, knowledge)

    if not getattr(args, "legacy_patches", False):
        measurement_plan = _load_measurement_plan(args.run_id)
        if not measurement_plan.run_command:
            measurement_plan.run_command = repo_setup.minimal_run
        if not measurement_plan.is_valid():
            print("Error: analyze_raw.json has no valid measurement_plan", file=sys.stderr)
            sys.exit(1)
        from sysight.pipeline.optimize import run_optimize_trials
        result = run_optimize_trials(
            findings,
            measurement_plan,
            args.repo,
            registry,
            provider,
            repo_setup=repo_setup,
            knowledge=knowledge,
            verbose=getattr(args, 'debug', False),
            max_trials=getattr(args, "max_trials", 5),
        )
        print(json.dumps({
            "run_id": result.run_id,
            "worktree_path": result.worktree_path,
            "best_commit": result.best_commit,
            "accepted_count": result.accepted_count,
            "rejected_count": result.rejected_count,
            "errors": result.errors,
        }, indent=2, ensure_ascii=False, default=str))
        return

    from sysight.pipeline.optimize import run_optimize
    patches = run_optimize(
        findings, args.repo, registry, provider,
        verbose=getattr(args, 'debug', False),
    )

    if not patches:
        print("No patches generated.", file=sys.stderr)
        return

    from sysight.pipeline.execute import run_execute
    result = run_execute(
        patches, args.repo,
        run_id=findings.run_id,
        analyze_run_dir=analyze_run_dir,
    )
    # run_id already printed by run_execute; print result for scripting
    print(result.run_id)


def _cmd_bench_optimize(args):
    """Run optimizer benchmark."""
    from sysight.benchmark.optimizer_runner import OptimizerBenchmarkRunner

    case_ids = args.cases
    if args.all:
        bench_dir = Path(args.bench_dir)
        cases_dir = bench_dir / "cases"
        if cases_dir.is_dir():
            case_ids = sorted(
                d.name for d in cases_dir.iterdir()
                if d.is_dir() and d.name.startswith("case_")
            )
        if not case_ids:
            print("Error: no cases found", file=sys.stderr)
            sys.exit(1)

    runner = OptimizerBenchmarkRunner(
        bench_dir=args.bench_dir,
        debug=getattr(args, 'debug', False),
    )
    result = runner.run(case_ids)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _cmd_learn(args):
    _, provider_factory, knowledge = _setup()
    provider = _provider_fallback(provider_factory, "learn", "optimize", "analyze")
    from sysight.pipeline.learn import run_learn
    result = run_learn(args.run_id, knowledge, provider, repo=getattr(args, "repo", ""))
    print(json.dumps({
        "run_id": result.run_id,
        "summary": result.summary,
        "worklog_path": result.worklog_path,
        "updates": len(result.memory_updates),
        "errors": result.errors,
    }, indent=2, ensure_ascii=False))


def _cmd_agent_loop(args):
    registry, provider_factory, knowledge = _setup()
    from sysight.pipeline.agent_loop import run_agent_loop
    result = run_agent_loop(
        args.profile,
        args.repo,
        registry,
        provider_factory,
        knowledge,
        max_loops=getattr(args, "max_loops", 2),
        max_trials=getattr(args, "max_trials", 5),
        verbose=getattr(args, "debug", False),
        refresh_profiles=not getattr(args, "no_profile_refresh", False),
        profile_command=getattr(args, "profile_command", ""),
        stage_wall_seconds=getattr(args, "stage_wall_seconds", 1800),
    )
    print(json.dumps({
        "loop_id": result.loop_id,
        "run_dir": result.run_dir,
        "iterations": len(result.iterations),
        "final_repo": result.final_repo,
        "final_profile": result.final_profile,
        "errors": result.errors,
    }, indent=2, ensure_ascii=False, default=str))


def _provider_fallback(provider_factory, *stages: str):
    for stage in stages:
        provider = provider_factory(stage)
        if provider:
            return provider
    return None


def _cmd_full(args):
    registry, provider_factory, knowledge = _setup()

    from sysight.pipeline.runner import PipelineRunner
    runner = PipelineRunner(registry, provider_factory, knowledge)
    result = runner.run_full(args.profile, args.repo)

    print(json.dumps({
        "run_id": result.run_id,
        "stages_completed": result.stages_completed,
        "findings": len(result.findings),
        "patches": len(result.patches),
        "errors": result.errors,
    }, indent=2, ensure_ascii=False))


def _cmd_tool(args):
    registry, _, _ = _setup()
    tool_name = f"{args.category}_{args.name}"

    tool_args = {}
    for a in args.args:
        if "=" in a:
            k, v = a.split("=", 1)
            if v == "-":
                v = sys.stdin.read()
            elif v.startswith("@"):
                v = Path(v[1:]).read_text(encoding="utf-8")
            tool_args[k] = v
        else:
            tool_args[a] = True

    from sysight.tools.registry import ToolPolicy
    policy = ToolPolicy(allowed_tools={f"{args.category}_*"}, read_only=not args.write)
    result = registry.execute(tool_name, tool_args, policy)

    if result.status == "ok" and result.data is not None:
        from dataclasses import asdict
        try:
            print(json.dumps(asdict(result.data), indent=2, ensure_ascii=False, default=str))
        except (TypeError, AttributeError):
            print(json.dumps({"data": str(result.data)}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"status": result.status, "error": result.error}, indent=2))


if __name__ == "__main__":
    main()
