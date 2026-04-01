"""Minimal CLI for the analysis MVP."""

from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path

from sysight.analysis import (
    build_evidence_report,
    format_analysis_markdown,
    format_analysis_report,
    format_info,
    format_summary,
    format_workflow_route,
    resolve_workflow_route,
    run_analysis,
)
from sysight.analysis.mfu import compute_theoretical_flops, format_region_mfu, format_theoretical_flops
from sysight.analysis.skills import all_skills, get_skill
from sysight.annotation import save_findings
from sysight.profile import open as open_profile


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
    trim = getattr(args, "trim", None)
    if trim:
        return (int(trim[0] * 1e9), int(trim[1] * 1e9))
    return None


def _coerce_cli_value(raw: str):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _parse_key_value_args(items: list[str] | None) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Expected KEY=VALUE for --arg, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Empty key in --arg {item!r}")
        parsed[key] = _coerce_cli_value(value.strip())
    return parsed


def _resolve_markdown_output(profile_path: str, gpu: int | None, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    stem = Path(profile_path).stem
    suffix = f".gpu{gpu}" if gpu is not None else ""
    return Path("outputs") / f"{stem}{suffix}.report.md"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sysight",
        description="Lightweight Torch-first analysis for Nsight Systems SQLite profiles",
    )
    sub = parser.add_subparsers(dest="command")

    info = sub.add_parser("info", help="Show profile metadata")
    info.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")

    summary = sub.add_parser("summary", help="Show per-GPU summary")
    summary.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")
    summary.add_argument("--gpu", type=int, default=None, help="Optional GPU device ID")
    _add_trim(summary)

    report = sub.add_parser("report", help="Generate markdown-oriented performance report")
    report.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")
    report.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    report.add_argument(
        "--findings",
        default=None,
        help="Optional path to findings JSON output",
    )
    report.add_argument(
        "--markdown",
        default=None,
        help="Optional path to markdown report output. Defaults to outputs/<profile>.report.md",
    )
    report.add_argument("--workspace", default=None, help="Optional workspace root")
    report.add_argument(
        "--program",
        default="program.md",
        help="Workspace contract file name or path. Defaults to program.md",
    )
    _add_trim(report)

    analyze = sub.add_parser("analyze", help="Run lightweight analysis")
    analyze.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")
    analyze.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    analyze.add_argument(
        "--findings",
        default=None,
        help="Optional path to findings JSON output",
    )
    analyze.add_argument(
        "--markdown",
        default=None,
        help="Optional path to markdown report output. Defaults to outputs/<profile>.report.md",
    )
    analyze.add_argument("--workspace", default=None, help="Optional workspace root")
    analyze.add_argument(
        "--program",
        default="program.md",
        help="Workspace contract file name or path. Defaults to program.md",
    )
    _add_trim(analyze)

    skill = sub.add_parser("skill", help="Run adapted built-in analysis skills")
    skill_sub = skill.add_subparsers(dest="skill_action")

    skill_list = skill_sub.add_parser("list", help="List built-in skills")
    skill_list.set_defaults(skill_action="list")

    skill_run = skill_sub.add_parser("run", help="Run one built-in skill")
    skill_run.add_argument("skill_name", help="Skill name")
    skill_run.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")
    skill_run.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    skill_run.add_argument(
        "--workspace",
        default=None,
        help="Optional workspace root used by agent workflow-aware skills",
    )
    skill_run.add_argument(
        "--program",
        default="program.md",
        help="Workspace contract file name or path. Defaults to program.md",
    )
    skill_run.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional skill argument. Can be repeated.",
    )
    _add_trim(skill_run)
    skill_run.set_defaults(skill_action="run")

    overlap = sub.add_parser("overlap", help="Show compute/NCCL overlap breakdown")
    overlap.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")
    overlap.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    _add_trim(overlap)

    nccl = sub.add_parser("nccl", help="Show NCCL collective breakdown")
    nccl.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")
    nccl.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    _add_trim(nccl)

    iters = sub.add_parser("iters", help="Show per-iteration timing")
    iters.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")
    iters.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    _add_trim(iters)

    schema = sub.add_parser("schema", help="Inspect SQLite schema")
    schema.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")

    theoretical_flops = sub.add_parser(
        "theoretical-flops", help="Compute theoretical FLOPs for transformer operations"
    )
    theoretical_flops.add_argument("operation", help="Operation name")
    theoretical_flops.add_argument("--hidden-dim", type=int, default=0, help="Hidden dimension")
    theoretical_flops.add_argument("--seq-len", type=int, default=0, help="Sequence length")
    theoretical_flops.add_argument("--num-layers", type=int, default=1, help="Number of layers")
    theoretical_flops.add_argument("--ffn-dim", type=int, default=None, help="FFN dimension")
    theoretical_flops.add_argument("--batch-size", type=int, default=1, help="Batch size")
    theoretical_flops.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="1=fwd only, 3=fwd+bwd, 4=fwd+bwd+checkpointing",
    )
    theoretical_flops.add_argument("--M", dest="m_dim", type=int, default=0, help="Linear M dim")
    theoretical_flops.add_argument("--N", dest="n_dim", type=int, default=0, help="Linear N dim")
    theoretical_flops.add_argument("--K", dest="k_dim", type=int, default=0, help="Linear K dim")

    route = sub.add_parser("route", help="Resolve the agent workflow mode from profile/workspace inputs")
    route.add_argument("--profile", default=None, help="Optional path to .sqlite, .sqlite3, or .nsys-rep")
    route.add_argument("--workspace", default=None, help="Optional workspace root")
    route.add_argument(
        "--program",
        default="program.md",
        help="Workspace contract file name or path. Defaults to program.md",
    )

    region_mfu = sub.add_parser("region-mfu", help="Compute MFU for a named NVTX region or kernel")
    region_mfu.add_argument("profile", help="Path to .sqlite, .sqlite3, or .nsys-rep")
    region_mfu.add_argument("--name", required=True, help="NVTX region text or kernel name")
    region_mfu.add_argument(
        "--theoretical-flops",
        required=True,
        type=float,
        help="Theoretical FLOPs for the region or step",
    )
    region_mfu.add_argument(
        "--source",
        choices=["nvtx", "kernel"],
        default="nvtx",
        help="Whether to match an NVTX range or kernels directly",
    )
    region_mfu.add_argument("--peak-tflops", type=float, default=None, help="Override GPU peak TFLOPS")
    region_mfu.add_argument("--num-gpus", type=int, default=1, help="Scale peak by number of GPUs")
    region_mfu.add_argument(
        "--occurrence-index",
        type=int,
        default=1,
        help="Which NVTX occurrence to analyze (1-based)",
    )
    region_mfu.add_argument("--gpu", type=int, default=None, help="Target GPU device ID")
    region_mfu.add_argument(
        "--match-mode",
        choices=["contains", "exact", "startswith"],
        default="contains",
        help="Name matching strategy",
    )

    return parser


def _run_named_skill(args, skill_name: str) -> None:
    trim = _parse_trim(args)
    extra_args = _parse_key_value_args(getattr(args, "arg", None))
    with open_profile(args.profile) as prof:
        skill = get_skill(skill_name)
        if not skill:
            raise SystemExit(f"Unknown skill: {skill_name}")
        result = skill.run(prof, args.gpu if hasattr(args, "gpu") else None, trim, **extra_args)
        print(skill.format(result))


def _run_report_like(args, prof) -> None:
    trim = _parse_trim(args)
    data = run_analysis(prof, args.gpu, trim)
    route = resolve_workflow_route(
        profile_path=args.profile,
        workspace_root=getattr(args, "workspace", None),
        program_path=getattr(args, "program", None),
    )
    data["workflow_route"] = route.to_dict()

    markdown_path = _resolve_markdown_output(args.profile, args.gpu, args.markdown).resolve()
    requested_resolved = Path(args.profile).resolve()
    resolved_sqlite = Path(prof.path).resolve()
    data["requested_profile_path"] = args.profile
    data["resolved_profile_path"] = str(resolved_sqlite)
    data["input_was_converted"] = requested_resolved != resolved_sqlite
    data["markdown_path"] = str(markdown_path)

    findings_path = None
    if args.findings:
        findings_path = Path(args.findings).resolve()
        data["findings_path"] = str(findings_path)

    markdown = format_analysis_markdown(data, args.profile)
    _write_text(markdown_path, markdown)
    print(format_analysis_report(data))

    if findings_path:
        report = build_evidence_report(data)
        save_findings(report, str(findings_path))


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
                result = skill.run(
                    prof,
                    args.gpu,
                    trim,
                    profile_path=args.profile,
                    workspace=args.workspace,
                    program=args.program,
                    **_parse_key_value_args(args.arg),
                )
                print(skill.format(result))
            return
        raise SystemExit("Usage: sysight skill {list,run} ...")

    if args.command == "route":
        route = resolve_workflow_route(
            profile_path=args.profile,
            workspace_root=args.workspace,
            program_path=args.program,
        )
        print(format_workflow_route(route))
        return

    if args.command == "theoretical-flops":
        result = compute_theoretical_flops(
            args.operation,
            hidden_dim=args.hidden_dim,
            seq_len=args.seq_len,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            batch_size=args.batch_size,
            multiplier=args.multiplier,
            m_dim=args.m_dim,
            n_dim=args.n_dim,
            k_dim=args.k_dim,
        )
        print(format_theoretical_flops(result))
        return

    if args.command == "region-mfu":
        with open_profile(args.profile) as prof:
            skill = get_skill("region_mfu")
            result = skill.run(
                prof,
                args.gpu,
                None,
                name=args.name,
                theoretical_flops=args.theoretical_flops,
                source=args.source,
                peak_tflops=args.peak_tflops,
                num_gpus=args.num_gpus,
                occurrence_index=args.occurrence_index,
                match_mode=args.match_mode,
            )
            print(format_region_mfu(result))
        return

    if args.command == "overlap":
        _run_named_skill(args, "overlap_breakdown")
        return

    if args.command == "nccl":
        _run_named_skill(args, "nccl_breakdown")
        return

    if args.command == "iters":
        _run_named_skill(args, "iteration_timing")
        return

    if args.command == "schema":
        _run_named_skill(args, "schema_inspect")
        return

    if args.command in {"analyze", "report"}:
        with open_profile(args.profile) as prof:
            _run_report_like(args, prof)
        return

    with open_profile(args.profile) as prof:
        if args.command == "info":
            print(format_info(prof, args.profile))
            return

        trim = _parse_trim(args)

        if args.command == "summary":
            print(format_summary(prof, args.gpu, trim))
            return
