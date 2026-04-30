"""Orchestration logic for sysight.optimizer.

Reads findings (from analyzer output or pre-made JSON), builds prompt,
dispatches to LLM, and parses the resulting patch_plan.json.

Artifacts are written to .sysight/optim-runs/run-<id>/.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sysight.shared.memory.store import build_memory_brief
from sysight.shared.findings import Finding, extract_findings, write_findings_json
from .models import PatchPlan


logger = logging.getLogger(__name__)


def _emit(msg: str) -> None:
    """Print a timestamped info line to stderr (always visible regardless of log level)."""
    print(f"[info] {msg}", file=sys.stderr, flush=True)


def _skill_path() -> Path:
    return Path(__file__).parent / "SKILL.txt"


def _create_optim_run_dir() -> Path:
    base_dir = Path.cwd() / ".sysight" / "optim-runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=base_dir))


def resolve_findings(
    findings_path: str | None = None,
    nsys_artifact: str | None = None,
) -> list[Finding]:
    """Resolve findings from one of several input sources.

    Priority:
      1. --findings: explicit findings.json or analysis last_message.txt path
      2. --nsys-artifact: explicit .sysight/nsys/*.json artifact path
      3. Auto-discover: latest analysis-run under cwd/.sysight/analysis-runs/
      4. Auto-discover: latest nsys artifact under cwd/.sysight/nsys/

    Auto-discover always uses cwd (where the sysight command is run from),
    which is where analyzer also writes its artifacts.
    """
    from sysight.shared.findings import find_latest_analysis_output, find_latest_nsys_artifact

    # Source 1: explicit findings file
    if findings_path:
        _emit(f"从指定文件读取 findings: {findings_path}")
        findings = extract_findings(findings_path)
        if findings:
            _emit(f"读取到 {len(findings)} 个 finding")
            return findings
        _emit(f"警告：{findings_path} 中未提取到有效 finding")

    # Source 2: explicit nsys artifact
    if nsys_artifact:
        _emit(f"从 nsys artifact 读取 findings: {nsys_artifact}")
        findings = extract_findings(nsys_artifact)
        if findings:
            _emit(f"读取到 {len(findings)} 个 finding")
            return findings
        _emit(f"警告：{nsys_artifact} 中未提取到有效 finding")

    # Source 3: auto-discover — always from cwd, same place analyzer writes to
    cwd = Path.cwd()
    _emit(f"自动发现最新 analysis-run（{cwd}/.sysight/analysis-runs/）")
    analysis_out = find_latest_analysis_output(cwd)
    if analysis_out:
        _emit(f"发现: {analysis_out}")
        findings = extract_findings(analysis_out)
        if findings:
            _emit(f"读取到 {len(findings)} 个 finding")
            return findings
        _emit("该 analysis-run 不含 code-level findings，尝试 nsys artifact")

    # Source 4: auto-discover nsys artifact
    nsys_out = find_latest_nsys_artifact(cwd)
    if nsys_out:
        _emit(f"发现 nsys artifact: {nsys_out}")
        findings = extract_findings(nsys_out)
        if findings:
            _emit(f"读取到 {len(findings)} 个 finding")
            return findings

    _emit("错误：未找到有效 findings，请先运行 analyzer 或手动指定 --findings")
    return []


def _parse_tokens_from_stderr(stderr_path: Path) -> int | None:
    """Parse token count from codex stderr output.

    Codex prints a footer like:
        tokens used
        57,817
    at the end of stderr. Returns the integer token count, or None if not found.
    """
    try:
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    for i in range(len(lines) - 1, 0, -1):
        if lines[i - 1].strip().lower() == "tokens used":
            raw = lines[i].strip().replace(",", "")
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def _estimate_prompt_tokens(prompt: str) -> int:
    """Rough estimate of prompt token count: ~4 chars per token for mixed CJK/English."""
    return max(1, round(len(prompt) / 4))


def build_optimizer_prompt(
    repo_root: str,
    findings: list[Finding],
    memory_dir: str = "",
    namespace: str | None = None,
) -> str:
    """Build the prompt for the optimizer agent."""
    import os as _os
    import sys as _sys

    skill_text = _skill_path().read_text(encoding="utf-8")

    findings_json = json.dumps([f.to_dict() for f in findings], indent=2, ensure_ascii=False)

    memory_brief = ""
    if memory_dir:
        memory_brief = build_memory_brief(
            memory_dir,
            repo_root=repo_root,
            namespace=namespace,
        )

    # Tell the agent exactly which python binary and sysight root to use,
    # so `python3 -m sysight.analyzer.cli scanner ...` works reliably.
    _sysight_root = str(Path(__file__).resolve().parents[2])
    _python_bin = _sys.executable
    _pp = _os.environ.get("PYTHONPATH", "")
    _scanner_env = f"PYTHONPATH={_sysight_root}:{_pp}" if _pp else f"PYTHONPATH={_sysight_root}"

    # Find bench python: prefer repo-level .venv, then bench-dir .venv, else fall back.
    _repo_path = Path(repo_root)
    _bench_python = _python_bin  # fallback
    for _candidate in (
        _repo_path / ".venv" / "bin" / "python",
        _repo_path.parent.parent / ".venv" / "bin" / "python",  # nsys-bench/.venv
        _repo_path.parent / ".venv" / "bin" / "python",
    ):
        if _candidate.exists():
            _bench_python = str(_candidate)
            break

    from sysight.shared.memory.store import default_memory_root as _default_memory_root
    _memory_root = memory_dir if memory_dir else str(_default_memory_root())

    env_block = (
        f"Python: {_python_bin}\n"
        f"Bench Python: {_bench_python}  (use this in run_cmd)\n"
        f"Sysight Root: {_sysight_root}\n"
        f"Scanner command: {_scanner_env} {_python_bin} -m sysight.analyzer.cli scanner\n"
        f"Memory root: {_memory_root}"
    )

    prompt = (
        f"{skill_text}\n\n"
        f"==== CONTEXT ====\n"
        f"Repo Root: {repo_root}\n"
        f"{env_block}\n"
        f"Memory:\n{memory_brief}\n"
        f"==== FINDINGS ====\n"
        f"Please analyze these findings and generate a PatchPlan:\n"
        f"```json\n{findings_json}\n```\n"
    )

    return prompt


def parse_patch_plan_output(stdout: str) -> PatchPlan | None:
    """Extract the PatchPlan JSON from agent output."""
    matches = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", stdout))
    if not matches:
        return None

    last_json_str = matches[-1].group(1)
    try:
        data = json.loads(last_json_str)
        return PatchPlan.from_dict(data)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON block from agent output: %s", e)
        return None


def run_optimizer_agent(
    repo_root: str,
    findings: list[Finding],
    memory_dir: str = "",
    namespace: str | None = None,
    mock_stdout: str | None = None,
) -> tuple[PatchPlan | None, Path | None]:
    """Run the optimizer agent and return (PatchPlan, artifact_dir).

    Artifacts are written to .sysight/optim-runs/run-<id>/.
    """
    if mock_stdout is not None:
        return parse_patch_plan_output(mock_stdout), None

    prompt = build_optimizer_prompt(
        repo_root=repo_root,
        findings=findings,
        memory_dir=memory_dir,
        namespace=namespace,
    )

    artifact_dir = _create_optim_run_dir()
    prompt_path = artifact_dir / "prompt.txt"
    output_path = artifact_dir / "last_message.txt"
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    cmd = [
        "codex",
        "exec",
        "--cd", repo_root,
        "--sandbox", "workspace-write",
        "--color", "never",
        "--output-last-message", str(output_path),
    ]
    if not Path(repo_root).joinpath(".git").exists():
        cmd.append("--skip-git-repo-check")
    cmd.append("-")

    # Inject PYTHONPATH so that `python3 -m sysight.analyzer.cli` works inside
    # the Codex sandbox regardless of whether the package is pip-installed.
    import os as _os
    _project_root = str(Path(__file__).resolve().parents[2])
    _env = dict(_os.environ)
    existing_pp = _env.get("PYTHONPATH", "")
    _env["PYTHONPATH"] = f"{_project_root}:{existing_pp}" if existing_pp else _project_root

    _emit(f"optimizer agent 启动: {' '.join(cmd)}")
    _emit(f"optim 工件目录: {artifact_dir}")
    _emit(f"prompt: {prompt_path}")
    _emit(f"输出: {output_path}")

    started = time.monotonic()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, \
             stderr_path.open("w", encoding="utf-8") as stderr_f:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
                env=_env,
                start_new_session=True,
            )
            _emit(f"optimizer agent 运行中，思考日志写入: {stderr_path}")
            process.communicate(prompt)
    except Exception as exc:
        _emit(f"错误：optimizer agent 启动失败: {exc}")
        return None, artifact_dir

    elapsed = time.monotonic() - started
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    if process.returncode != 0:
        _emit(f"错误：optimizer agent 退出码 {process.returncode}，用时 {elapsed_str}")
        return None, artifact_dir

    if not output_path.exists():
        _emit("错误：codex 未生成输出文件")
        return None, artifact_dir

    output_text = output_path.read_text(encoding="utf-8")
    plan = parse_patch_plan_output(output_text)

    # Token statistics (same as analyzer)
    tokens_used = _parse_tokens_from_stderr(stderr_path)
    prompt_tokens_est = _estimate_prompt_tokens(prompt)
    if tokens_used is not None:
        prompt_pct = prompt_tokens_est / tokens_used * 100 if tokens_used > 0 else 0.0
        token_str = f"  prompt_tokens: ~{prompt_tokens_est:,} ({prompt_pct:.1f}%),  all tokens used: {tokens_used:,}"
    else:
        token_str = ""

    if plan:
        _emit(f"optimizer agent 完成，用时 {elapsed_str}，生成 {len(plan.patches)} 个 patch{token_str}")
    else:
        _emit(f"optimizer agent 完成，用时 {elapsed_str}，但未解析到有效 PatchPlan{token_str}")

    # Flush memory updates from agent output (same as analyzer)
    _flush_memory_from_output(output_text, memory_dir=memory_dir, repo_root=repo_root, namespace=namespace)

    return plan, artifact_dir


def _flush_memory_from_output(
    output_text: str,
    *,
    memory_dir: str = "",
    repo_root: str = "",
    namespace: str | None = None,
) -> None:
    """Parse memory_updates from agent JSON output and persist them."""
    import json as _json
    from sysight.shared.memory.store import apply_memory_updates, default_memory_root

    # Find the last JSON block in output
    matches = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", output_text))
    if not matches:
        return
    try:
        data = _json.loads(matches[-1].group(1))
    except (_json.JSONDecodeError, IndexError):
        return

    updates = data.get("memory_updates")
    workspace_mem = data.get("workspace_memory") or None
    experience_mem = data.get("experience_memory") or None

    has_updates = updates or workspace_mem or experience_mem
    if not has_updates:
        return

    root = memory_dir or str(default_memory_root())
    try:
        apply_memory_updates(
            root,
            list(updates) if updates else [],
            repo_root=repo_root or None,
            namespace=namespace,
        )
        # Legacy fields
        from sysight.shared.memory.store import write_memory_file as _wm, _resolve_root as _rr
        from pathlib import Path as _Path
        _base = _rr(root)
        if workspace_mem:
            _wm(str(_base), "workspace.md", "\n\n---\n\n" + workspace_mem.strip(), append=True)
        if experience_mem:
            _wm(str(_base), "experience.md", "\n\n---\n\n" + experience_mem.strip(), append=True)
        _emit(f"memory 已写入: {_base}")
    except Exception as exc:
        _emit(f"警告：memory 写入失败: {exc}")
