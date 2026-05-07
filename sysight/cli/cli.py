"""Unified CLI for Sysight.

Facade pattern — single entry point that hides internal module structure.

Commands:
  sysight warmup   <repo>
  sysight analyze  <profile> --repo <repo>
  sysight optimize <analysis.json> --repo <repo>
  sysight learn    <run-id>
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

    # learn
    p = sub.add_parser("learn", help="Post-session learning")
    p.add_argument("run_id", help="Run ID to process")

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
        "learn": _cmd_learn,
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
    result = run_warmup(args.repo, registry, knowledge, force=getattr(args, 'force', False))

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

    print(f"\n  Cache: .sysight/warmup-caches/<hash>.json")
    print(f"  Overview: {result.summary.get('overview_path', '')}")


def _cmd_analyze(args):
    registry, provider_factory, knowledge = _setup()
    if args.dump_prompt or args.no_run:
        from sysight.pipeline.analyze import build_analyze_first_turn
        bundle = build_analyze_first_turn(args.profile, args.repo, registry, knowledge)
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
    # run_id lookup in .sysight/analysis-runs/
    candidate = Path.cwd() / ".sysight" / "analysis-runs" / run_id_or_path / "analyze_raw.json"
    if candidate.exists():
        return candidate
    print(f"Error: cannot resolve analyze_raw.json from '{run_id_or_path}'", file=sys.stderr)
    sys.exit(1)


def _load_findings(run_id_or_path: str):
    """Parse analyze_raw.json (by run_id or path) into LocalizedFindingSet."""
    path = _resolve_analyze_raw(run_id_or_path)
    data = json.loads(path.read_text())
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
    from sysight.pipeline.warmup import _warmup_cache_path
    from sysight.types.repo_setup import RepoSetup
    cache_path = _warmup_cache_path(Path(args.repo))
    repo_setup = RepoSetup.load_cache(cache_path)
    result = run_instrument(
        findings, args.repo, registry, provider, knowledge,
        repo_setup=repo_setup,
        verbose=getattr(args, 'debug', False),
        run_dir=run_dir,
    )

    s = result.summary
    print(f"\n{'='*60}")
    print(f"  Instrument 完成")
    print(f"{'='*60}")
    print(f"  入口:         {s.get('entry_point', '?')}")
    print(f"  smoke:        {'✓ 通过' if s.get('smoke_test_passed') else '? 未验证'}")
    print(f"  NVTX 已有:    {'是' if s.get('has_nvtx') else '否'}")
    print(f"  建议打标:     {s.get('suggested_tags', 0)} 处")

    hai = s.get('human_action_items', [])
    if hai:
        print(f"\n  ⚠ 需要人工操作 ({len(hai)} 项):")
        for item in hai:
            print(f"    • {item}")

    warns = s.get('warnings', [])
    errs = s.get('errors', [])
    if warns or errs:
        print()
        for w in warns:
            print(f"  ⚠ {w}")
        for e in errs:
            print(f"  ✗ {e}")


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

    from sysight.pipeline.optimize import run_optimize
    result = run_optimize(findings, args.repo, registry, provider, knowledge)
    print(json.dumps({
        "run_id": result.run_id,
        "patches": [
            {"patch_id": p.patch_id, "finding_id": p.finding_id,
             "status": p.status, "reason": p.reason}
            for p in result.patches
        ],
        "errors": result.errors,
    }, indent=2, ensure_ascii=False))


def _cmd_learn(args):
    _, _, knowledge = _setup()
    from sysight.pipeline.learn import run_learn
    result = run_learn(args.run_id, knowledge)
    print(json.dumps({
        "run_id": result.run_id,
        "worklog": result.worklog[:500],
        "experiences": len(result.experiences),
        "errors": result.errors,
    }, indent=2, ensure_ascii=False))


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
