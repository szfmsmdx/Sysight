"""Agent workflow routing and workspace program contract helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re

_REQUIRED_SECTION_HINTS = {
    "task": ("task", "任务", "goal", "目标"),
    "project": ("project", "项目", "背景"),
    "framework": ("framework", "框架", "stack", "技术栈"),
    "entry": ("entry", "启动", "run", "运行", "command"),
    "performance_goal": ("performance", "优化目标", "latency", "throughput", "mfu"),
    "important_paths": ("important path", "关键路径", "key path", "模块", "paths"),
    "constraints": ("constraint", "约束", "限制"),
    "success_criteria": ("success", "验收", "成功标准", "criteria"),
    "output_contract": ("output", "输出", "deliverable", "report"),
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class ProgramContract:
    """Parsed workspace contract from ``program.md``."""

    workspace_root: str
    program_path: str
    sections: dict[str, str]
    missing_sections: list[str] = field(default_factory=list)
    task: str = ""
    project: str = ""
    framework: str = ""
    entry: str = ""
    performance_goal: str = ""
    important_paths: list[str] = field(default_factory=list)
    constraints: str = ""
    success_criteria: str = ""
    output_contract: str = ""

    @property
    def is_complete(self) -> bool:
        return not self.missing_sections

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowRoute:
    """Resolved agent workflow path for one user request."""

    mode: str
    ready: bool
    summary: str
    profile_path: str = ""
    workspace_root: str = ""
    program_path: str = ""
    missing_inputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    program_contract: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.strip().lower()).strip()


def _parse_markdown_sections(text: str) -> dict[str, str]:
    current_heading = "preamble"
    buckets: dict[str, list[str]] = {current_heading: []}
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            current_heading = _normalize_heading(match.group(2))
            buckets.setdefault(current_heading, [])
            continue
        buckets.setdefault(current_heading, []).append(line.rstrip())
    return {
        heading: "\n".join(lines).strip()
        for heading, lines in buckets.items()
        if "\n".join(lines).strip()
    }


def _match_section_key(headings: dict[str, str], canonical_key: str) -> str:
    hints = _REQUIRED_SECTION_HINTS[canonical_key]
    for heading in headings:
        if any(hint in heading for hint in hints):
            return heading
    return ""


def _clean_summary_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        stripped = stripped.lstrip("-*0123456789. ").strip()
        if stripped:
            return stripped
    return ""


def _extract_paths(text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*0123456789. ").strip()
        if not stripped:
            continue
        for match in re.findall(r"`([^`]+)`", stripped):
            if "/" in match or match.endswith(".py") or match.endswith(".md"):
                if match not in seen:
                    seen.add(match)
                    paths.append(match)
        if "/" in stripped and stripped not in seen:
            seen.add(stripped)
            paths.append(stripped)
    return paths[:8]


def load_program_contract(
    workspace_root: str | Path,
    program_path: str | Path = "program.md",
) -> ProgramContract:
    """Load and parse the workspace-level ``program.md`` contract."""
    workspace = Path(workspace_root).resolve()
    program = Path(program_path)
    if not program.is_absolute():
        program = workspace / program
    program = program.resolve()
    text = program.read_text(encoding="utf-8")
    raw_sections = _parse_markdown_sections(text)

    mapped_sections: dict[str, str] = {}
    missing_sections: list[str] = []
    for canonical_key in _REQUIRED_SECTION_HINTS:
        heading = _match_section_key(raw_sections, canonical_key)
        if heading:
            mapped_sections[canonical_key] = raw_sections[heading]
        else:
            missing_sections.append(canonical_key)

    return ProgramContract(
        workspace_root=str(workspace),
        program_path=str(program),
        sections=mapped_sections,
        missing_sections=missing_sections,
        task=_clean_summary_text(mapped_sections.get("task", "")),
        project=_clean_summary_text(mapped_sections.get("project", "")),
        framework=_clean_summary_text(mapped_sections.get("framework", "")),
        entry=_clean_summary_text(mapped_sections.get("entry", "")),
        performance_goal=_clean_summary_text(mapped_sections.get("performance_goal", "")),
        important_paths=_extract_paths(mapped_sections.get("important_paths", "")),
        constraints=_clean_summary_text(mapped_sections.get("constraints", "")),
        success_criteria=_clean_summary_text(mapped_sections.get("success_criteria", "")),
        output_contract=_clean_summary_text(mapped_sections.get("output_contract", "")),
    )


def resolve_workflow_route(
    *,
    profile_path: str | None = None,
    workspace_root: str | None = None,
    program_path: str | None = None,
) -> WorkflowRoute:
    """Route a user request into the current supported agent workflow mode."""
    profile = (profile_path or "").strip()
    workspace = (workspace_root or "").strip()
    program_name = (program_path or "program.md").strip() or "program.md"

    missing_inputs: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    contract: ProgramContract | None = None

    profile_exists = False
    if profile:
        profile_exists = Path(profile).expanduser().exists()
        if not profile_exists:
            missing_inputs.append("valid profile path")
            warnings.append(f"Profile path does not exist yet: {profile}")

    workspace_exists = False
    if workspace:
        workspace_path = Path(workspace).expanduser()
        workspace_exists = workspace_path.exists() and workspace_path.is_dir()
        if not workspace_exists:
            warnings.append(f"Workspace path is not a readable directory: {workspace}")
        else:
            try:
                contract = load_program_contract(workspace_path, program_name)
                if contract.missing_sections:
                    warnings.append(
                        "program.md is missing recommended sections: "
                        + ", ".join(contract.missing_sections)
                    )
            except FileNotFoundError:
                warnings.append(
                    f"Workspace mode expects `{program_name}` under `{workspace_path.resolve()}`"
                )
                recommendations.append(
                    "补充 workspace/program.md，至少写明任务、框架、启动方式、性能目标、约束和输出要求。"
                )
            except OSError as exc:
                warnings.append(f"Failed to read program contract: {exc}")

    if profile_exists and contract:
        summary = "走 workspace-aware analysis：已有 profile，同时加载 workspace/program.md 增强 agent 上下文。"
        recommendations.extend(
            [
                "优先让 agent 使用 program.md 中的启动方式、性能目标和关键路径解释 findings。",
                "当前阶段先做分析与报告，不自动触发 profile 或代码优化闭环。",
            ]
        )
        return WorkflowRoute(
            mode="workspace-aware",
            ready=True,
            summary=summary,
            profile_path=profile,
            workspace_root=contract.workspace_root,
            program_path=contract.program_path,
            missing_inputs=missing_inputs,
            warnings=warnings,
            recommendations=recommendations,
            program_contract=contract.to_dict(),
        )

    if profile_exists:
        summary = "走 profile-only analysis：当前只基于 profile 产出分析摘要和 report.md。"
        if workspace and not contract:
            recommendations.append(
                "如需更强的代码/项目语义归因，请补充 workspace 下的 program.md 后再重跑分析。"
            )
        else:
            recommendations.append(
                "如需结合项目入口、优化目标和关键模块解释 findings，可补充 workspace/program.md。"
            )
        return WorkflowRoute(
            mode="profile-only",
            ready=True,
            summary=summary,
            profile_path=profile,
            workspace_root=contract.workspace_root if contract else workspace,
            program_path=contract.program_path if contract else "",
            missing_inputs=missing_inputs,
            warnings=warnings,
            recommendations=recommendations,
            program_contract=contract.to_dict() if contract else None,
        )

    if contract:
        summary = "检测到 workspace/program.md，但当前还没有 profile；现阶段可先完成项目上下文准备。"
        recommendations.extend(
            [
                "补一个 `.sqlite` / `.sqlite3` / `.nsys-rep` 路径后，即可进入标准 analysis/report 流程。",
                "后续补 profile automation 时，可直接复用这份 program.md 作为 agent 合同。",
            ]
        )
        return WorkflowRoute(
            mode="workspace-context-only",
            ready=False,
            summary=summary,
            workspace_root=contract.workspace_root,
            program_path=contract.program_path,
            warnings=warnings,
            recommendations=recommendations,
            program_contract=contract.to_dict(),
        )

    recommendations.append(
        "至少提供一个 profile 路径；若还希望 agent 结合项目语义工作，再额外提供 workspace/program.md。"
    )
    return WorkflowRoute(
        mode="insufficient-input",
        ready=False,
        summary="当前既没有可读 profile，也没有满足协议的 workspace/program.md。",
        missing_inputs=missing_inputs,
        warnings=warnings,
        recommendations=recommendations,
    )


def format_workflow_route(route: WorkflowRoute) -> str:
    """Format a route decision for terminal-oriented agent intake checks."""
    lines = ["Agent Workflow Route"]
    lines.append(f"- Mode: {route.mode}")
    lines.append(f"- Ready: {'yes' if route.ready else 'no'}")
    lines.append(f"- Summary: {route.summary}")
    if route.profile_path:
        lines.append(f"- Profile: {route.profile_path}")
    if route.workspace_root:
        lines.append(f"- Workspace: {route.workspace_root}")
    if route.program_path:
        lines.append(f"- Program: {route.program_path}")
    contract = route.program_contract or {}
    if contract:
        if contract.get("task"):
            lines.append(f"- Task: {contract['task']}")
        if contract.get("framework"):
            lines.append(f"- Framework: {contract['framework']}")
        if contract.get("entry"):
            lines.append(f"- Entry: {contract['entry']}")
        if contract.get("performance_goal"):
            lines.append(f"- Performance Goal: {contract['performance_goal']}")
        if contract.get("success_criteria"):
            lines.append(f"- Success Criteria: {contract['success_criteria']}")
        if contract.get("important_paths"):
            lines.append(
                "- Important Paths: " + ", ".join(str(item) for item in contract["important_paths"][:4])
            )
    for warning in route.warnings:
        lines.append(f"- Warning: {warning}")
    for recommendation in route.recommendations:
        lines.append(f"- Next: {recommendation}")
    return "\n".join(lines)
