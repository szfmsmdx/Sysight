"""Command-line interface for sysight.optimizer."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .plan import run_optimizer_agent

logger = logging.getLogger(__name__)


def add_optimizer_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the optimizer command to the main parser."""
    parser = subparsers.add_parser("optimize", help="Generate code patches from findings.json")
    parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Path to the repository to analyze and patch",
    )
    parser.add_argument(
        "--findings",
        type=str,
        required=True,
        help="Path to the findings.json input file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="patch_plan.json",
        help="Path to output patch_plan.json",
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
    repo_path = Path(args.repo).resolve()
    findings_path = Path(args.findings).resolve()
    out_path = Path(args.out).resolve()

    if not findings_path.exists():
        logger.error("Findings file not found: %s", findings_path)
        return 1

    logger.info("Starting optimizer agent on %s", repo_path)
    logger.info("Reading findings from %s", findings_path)

    plan = run_optimizer_agent(
        repo_root=str(repo_path),
        findings_path=str(findings_path),
        memory_dir=args.memory_dir,
        namespace=args.namespace,
    )

    if not plan:
        logger.error("Optimizer agent failed or returned no patches.")
        return 1

    out_path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Patch plan written to %s", out_path)
    return 0


def main() -> int:
    """Standalone entrypoint for testing."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_optimizer_subparser(subparsers)
    args = parser.parse_args()

    if args.command == "optimize":
        return dispatch_optimizer(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
