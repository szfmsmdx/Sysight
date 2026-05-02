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
    p = sub.add_parser("warmup", help="Explore a repo and create RepoSetup")
    p.add_argument("repo", help="Path to repo root")

    # analyze
    p = sub.add_parser("analyze", help="Analyze an Nsight Systems profile")
    p.add_argument("profile", help="Path to .sqlite profile")
    p.add_argument("--repo", required=True, help="Path to repo root")

    # optimize
    p = sub.add_parser("optimize", help="Optimize findings from an analysis")
    p.add_argument("analysis_json", help="Path to analysis.json")
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
    p.add_argument("args", nargs="*", help="Tool arguments as key=value pairs")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    dispatch = {
        "warmup": _cmd_warmup,
        "analyze": _cmd_analyze,
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

    configs = load_config()

    def provider_factory(stage: str):
        cfg = configs.get(stage)
        if not cfg or not cfg.api_key:
            return None
        return create_provider(cfg)

    knowledge = WikiRepository()
    return registry, provider_factory, knowledge


def _cmd_warmup(args):
    registry, provider_factory, knowledge = _setup()
    provider = provider_factory("warmup")
    if not provider:
        print("Error: no warmup provider configured", file=sys.stderr)
        sys.exit(1)

    from sysight.pipeline.warmup import run_warmup
    result = run_warmup(args.repo, registry, provider, knowledge)
    print(json.dumps({
        "repo": result.repo,
        "entry_point": result.repo_setup.entry_point,
        "metric_grep": result.repo_setup.metric_grep,
        "source": result.repo_setup.source,
        "errors": result.errors,
    }, indent=2, ensure_ascii=False))


def _cmd_analyze(args):
    registry, provider_factory, knowledge = _setup()
    provider = provider_factory("analyze")
    if not provider:
        print("Error: no analyze provider configured", file=sys.stderr)
        sys.exit(1)

    from sysight.pipeline.analyze import run_analyze
    result = run_analyze(args.profile, args.repo, registry, provider, knowledge)
    output = {
        "run_id": result.run_id,
        "summary": result.finding_set.summary,
        "findings": [
            {
                "finding_id": f.finding_id,
                "category": f.category,
                "title": f.title,
                "priority": f.priority,
                "file_path": f.file_path,
                "function": f.function,
                "line": f.line,
                "confidence": f.confidence,
                "description": f.description,
                "suggestion": f.suggestion,
                "status": f.status,
            }
            for f in result.finding_set.findings
        ],
        "rejected": len(result.finding_set.rejected),
        "errors": result.errors,
        "elapsed_ms": result.elapsed_ms,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _cmd_optimize(args):
    registry, provider_factory, knowledge = _setup()
    provider = provider_factory("optimize")
    if not provider:
        print("Error: no optimize provider configured", file=sys.stderr)
        sys.exit(1)

    data = json.loads(Path(args.analysis_json).read_text())
    from sysight.types.findings import LocalizedFindingSet, LocalizedFinding
    findings = LocalizedFindingSet(
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
        ],
    )

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
            tool_args[k] = v
        else:
            tool_args[a] = True

    from sysight.tools.registry import ToolPolicy
    policy = ToolPolicy(allowed_tools={f"{args.category}_*"}, read_only=True)
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
