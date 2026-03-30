"""Minimal CLI for the analysis MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

from nsys_agent.analysis import (
    build_evidence_report,
    format_analysis_markdown,
    format_analysis_report,
    format_info,
    format_summary,
    run_analysis,
)
from nsys_agent.analysis.skills import all_skills, get_skill
from nsys_agent.annotation import save_findings
from nsys_agent.profile import open as open_profile


def _add_trim(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trim",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        default=None,
        help="Time window in seconds",
    )


def _parse_trim(args) -> tuple[int, int] | None:
    if args.trim:
        return (int(args.trim[0] * 1e9), int(args.trim[1] * 1e9))
    return None


def _resolve_markdown_output(profile_path: str, gpu: int | None, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    stem = Path(profile_path).stem
    suffix = f".gpu{gpu}" if gpu is not None else ""
    return Path("outputs") / f"{stem}{suffix}.analysis.md"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsys-agent",
        description="Lightweight Torch-first analysis for Nsight Systems SQLite profiles",
    )
    sub = parser.add_subparsers(dest="command")

    info = sub.add_parser("info", help="Show profile metadata")
    info.add_argument("profile", help="Path to .sqlite or .nsys-rep")

    summary = sub.add_parser("summary", help="Show per-GPU summary")
    summary.add_argument("profile", help="Path to .sqlite or .nsys-rep")
    summary.add_argument("--gpu", type=int, default=None, help="Optional GPU device ID")
    _add_trim(summary)

    analyze = sub.add_parser("analyze", help="Run lightweight analysis")
    analyze.add_argument("profile", help="Path to .sqlite or .nsys-rep")
    analyze.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    analyze.add_argument(
        "--findings",
        default=None,
        help="Optional path to findings JSON output",
    )
    analyze.add_argument(
        "--markdown",
        default=None,
        help="Optional path to markdown report output. Defaults to outputs/<profile>.analysis.md",
    )
    _add_trim(analyze)

    skill = sub.add_parser("skill", help="Run adapted built-in analysis skills")
    skill_sub = skill.add_subparsers(dest="skill_action")

    skill_list = skill_sub.add_parser("list", help="List built-in skills")
    skill_list.set_defaults(skill_action="list")

    skill_run = skill_sub.add_parser("run", help="Run one built-in skill")
    skill_run.add_argument("skill_name", help="Skill name")
    skill_run.add_argument("profile", help="Path to .sqlite or .nsys-rep")
    skill_run.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    _add_trim(skill_run)
    skill_run.set_defaults(skill_action="run")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "skill":
        if args.skill_action == "list":
            for skill in all_skills():
                print(f"{skill.name:<24s} {skill.description}")
            return
        if args.skill_action == "run":
            trim = _parse_trim(args)
            with open_profile(args.profile) as prof:
                skill = get_skill(args.skill_name)
                if not skill:
                    raise SystemExit(f"Unknown skill: {args.skill_name}")
                result = skill.run(prof, args.gpu, trim)
                print(skill.format(result))
            return
        raise SystemExit("Usage: nsys-agent skill {list,run} ...")

    with open_profile(args.profile) as prof:
        if args.command == "info":
            print(format_info(prof, args.profile))
            return

        trim = _parse_trim(args)

        if args.command == "summary":
            print(format_summary(prof, args.gpu, trim))
            return

        if args.command == "analyze":
            data = run_analysis(prof, args.gpu, trim)
            print(format_analysis_report(data))
            markdown_path = _resolve_markdown_output(args.profile, args.gpu, args.markdown)
            markdown = format_analysis_markdown(data, args.profile)
            _write_text(markdown_path, markdown)
            print(f"\nMarkdown report written to {markdown_path}")
            if args.findings:
                report = build_evidence_report(data)
                save_findings(report, args.findings)
                print(f"\nFindings written to {args.findings}")
            return
