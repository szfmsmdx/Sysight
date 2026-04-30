"""Command-line interface for sysight.optimizer."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .plan import resolve_findings, run_optimizer_agent, _emit

logger = logging.getLogger(__name__)


def add_optimizer_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the optimizer command to the main parser."""
    parser = subparsers.add_parser("optimize", help="Generate code patches from analyzer findings")
    parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Path to the repository to patch (default: cwd)",
    )
    # All input sources are optional — auto-discover from .sysight/analysis-runs/ if omitted
    parser.add_argument(
        "--findings",
        type=str,
        default=None,
        help="Path to findings.json or analysis last_message.txt (optional, auto-discovered if omitted)",
    )
    parser.add_argument(
        "--nsys-artifact",
        type=str,
        default=None,
        help="Path to .sysight/nsys/*.json artifact from analyzer (optional)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to output patch_plan.json (default: .sysight/optim-runs/run-<id>/patch_plan.json)",
    )
    parser.add_argument(
        "--dump-findings",
        type=str,
        default=None,
        help="Write resolved findings.json to this path and exit (no agent call)",
    )
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="",
        help="Path to Sysight memory directory",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Memory namespace",
    )


def dispatch_optimizer(args: argparse.Namespace) -> int:
    """Run the optimizer workflow from CLI arguments."""
    from sysight.shared.findings import write_findings_json

    repo_path = Path(args.repo).resolve()

    _emit(f"optimizer 启动，repo: {repo_path}")

    findings = resolve_findings(
        findings_path=args.findings,
        nsys_artifact=getattr(args, "nsys_artifact", None),
    )

    if not findings:
        return 1

    # --dump-findings: write findings and exit
    if args.dump_findings:
        dump_path = write_findings_json(findings, args.dump_findings)
        _emit(f"findings ({len(findings)} 条) 已写出: {dump_path}")
        return 0

    plan, artifact_dir = run_optimizer_agent(
        repo_root=str(repo_path),
        findings=findings,
        memory_dir=args.memory_dir,
        namespace=args.namespace,
    )

    if not plan:
        return 1

    # Determine output path
    if args.out:
        out_path = Path(args.out).resolve()
    elif artifact_dir:
        out_path = artifact_dir / "patch_plan.json"
    else:
        out_path = Path("patch_plan.json").resolve()

    out_path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    _emit(f"patch_plan 已写出: {out_path}  ({len(plan.patches)} 个 patch)")
    return 0


def main() -> int:
    """Standalone entrypoint for testing."""
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_optimizer_subparser(subparsers)
    args = parser.parse_args()

    if args.command == "optimize":
        return dispatch_optimizer(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
