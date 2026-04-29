"""Command-line interface for sysight.executor."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sysight.optimizer.models import PatchPlan
from .execute import execute_patch_plan

logger = logging.getLogger(__name__)


def add_executor_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the executor command to the main parser."""
    parser = subparsers.add_parser("execute", help="Execute a patch plan and evaluate metrics")
    parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Path to the repository to patch and measure",
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to the patch_plan.json input file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="execution_report.json",
        help="Path to output execution_report.json",
    )


def dispatch_executor(args: argparse.Namespace) -> int:
    """Run the executor workflow from CLI arguments."""
    repo_path = Path(args.repo).resolve()
    plan_path = Path(args.plan).resolve()
    out_path = Path(args.out).resolve()

    if not plan_path.exists():
        logger.error("Patch plan file not found: %s", plan_path)
        return 1

    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        plan = PatchPlan.from_dict(data)
    except Exception as e:
        logger.error("Failed to parse patch_plan.json: %s", e)
        return 1

    logger.info("Starting executor on %s", repo_path)
    
    report = execute_patch_plan(
        repo_root=str(repo_path),
        plan=plan,
    )

    out_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Execution report written to %s", out_path)
    return 0


def main() -> int:
    """Standalone entrypoint for testing."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_executor_subparser(subparsers)
    args = parser.parse_args()

    if args.command == "execute":
        return dispatch_executor(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
