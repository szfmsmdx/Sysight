"""Agent-driven localization helpers for nsys analysis.

This module isolates prompt building, backend registration, Codex CLI
execution, and result parsing from the main nsys orchestration entrypoint.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ...shared.memory.store import build_memory_brief, resolve_memory_path
from .sql_cli import run_sql_nvtx, run_sql_sync, run_sql_memcpy, run_sql_kernels, run_sql_gaps, run_sql_kernel_launch

from .models import (
    EvidenceWindow,
    LocalizationAnchor,
    LocalizationQuestion,
    LocalizationResult,
    NsysAnalysisRequest,
    NsysFinding,
)

_CLI_INVESTIGATORS: dict[str, callable] = {}


def register_cli_investigator(name: str, runner) -> None:
    """Register a thin CLI backend runner for Code localization."""
    _CLI_INVESTIGATORS[name] = runner


def has_cli_investigator(name: str) -> bool:
    return name in _CLI_INVESTIGATORS


def run_code_localization(
    request: NsysAnalysisRequest,
    *,
    summary: str,
    findings: list[NsysFinding],
    windows: list[EvidenceWindow],
    sqlite_path: str,
    bottleneck_summary=None,
    hotspots=None,
    profile_report_text: str = "",
) -> LocalizationResult:
    backend = request.localization_backend or "codex"
    runner = _CLI_INVESTIGATORS.get(backend)
    if runner is None:
        return LocalizationResult(
            backend=backend,
            status="error",
            prompt="",
            error=f"未注册的 localization backend：{backend}",
        )

    prompt = _build_localization_prompt(
        request,
        summary=summary,
        findings=findings,
        windows=windows,
        sqlite_path=sqlite_path,
        bottleneck_summary=bottleneck_summary,
        hotspots=hotspots,
        profile_report_text=profile_report_text,
    )
    return runner(prompt, request)


# Backward-compatible alias for tests and existing imports.
_run_code_localization = run_code_localization


def _analyzer_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _memory_dir() -> Path:
    # Runtime memory lives under workspace-local .sysight/, not under analyzer assets.
    return Path(__file__).resolve().parents[3] / ".sysight" / "memory"


def _task_path() -> Path:
    return _analyzer_dir() / "SKILL.txt"


def _read_harness_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _build_localization_prompt(
    request: NsysAnalysisRequest,
    *,
    summary: str,
    findings: list[NsysFinding],
    windows: list[EvidenceWindow],
    sqlite_path: str,
    bottleneck_summary=None,
    hotspots=None,
    profile_report_text: str = "",
) -> str:
    """Build the localization prompt from TASK.txt template + profile report.

    The prompt is intentionally minimal:
      - Profile report (raw numbers, no heuristic findings)
      - Pre-injected SQL results (nvtx/sync/memcpy) to reduce codex CLI calls
      - SQLite/repo paths
      - Task instructions loaded from TASK.txt (editable by humans)

    Codex is expected to reason from the numbers and use CLI tools autonomously.
    We do NOT inject pre-generated findings, questions, or evidence windows.
    """
    task_text = _read_harness_file(_task_path())

    # Build memory brief for prompt injection
    memory_brief = build_memory_brief(
        str(_memory_dir()),
        repo_root=request.repo_root,
        namespace=getattr(request, "memory_namespace", None),
    )

    # Fill in template placeholders
    prompt = task_text.replace("{{PROFILE_REPORT}}", profile_report_text)
    prompt = prompt.replace("{{SQLITE_PATH}}", sqlite_path)
    prompt = prompt.replace("{{REPO_ROOT}}", request.repo_root or ".")
    prompt = prompt.replace("{{PRE_INJECTED_SQL}}", _build_pre_injected_sql(sqlite_path))
    prompt = prompt.replace("{{WORKSPACE_MEMORY}}", "")
    prompt = prompt.replace("{{EXPERIENCE_MEMORY_PATH}}", str(_memory_dir()))
    prompt = prompt.replace("{{MEMORY_BRIEF}}", memory_brief)

    return prompt


def _build_pre_injected_sql(sqlite_path: str) -> str:
    """Pre-run SQL queries and inject results into prompt.

    Covers all high-signal profile-side data so codex rarely needs to
    call nsys-sql on its own. Sections: nvtx, sync, memcpy, kernels,
    gaps, kernel-launch.
    """
    lines: list[str] = []
    lines.append("────────────────────────────────────────────────────────────────")
    lines.append("  预注入 Profile 数据（已覆盖主要维度，无新疑问时无需重调 CLI）")
    lines.append("────────────────────────────────────────────────────────────────")

    try:
        nvtx = run_sql_nvtx(sqlite_path, limit=20)
        if nvtx.nvtx_ranges:
            lines.append("")
            lines.append("NVTX range 耗时（top 20，按总耗时排序）：")
            for r in nvtx.nvtx_ranges:
                total_ms = r.total_ns / 1e6
                avg_ms = r.avg_ns / 1e6
                lines.append(f"  {r.text:<40}  count={r.count:<5}  total={total_ms:.3f}ms  avg={avg_ms:.3f}ms")
    except Exception:
        pass

    try:
        sync = run_sql_sync(sqlite_path)
        if sync.sync_events:
            lines.append("")
            lines.append(f"CUDA 同步事件（wall pct={sync.sync_wall_pct}%）：")
            for s in sync.sync_events:
                total_ms = s.total_ns / 1e6
                lines.append(f"  {s.sync_type:<60}  count={s.count:<5}  total={total_ms:.3f}ms  avg={s.avg_ns/1e3:.1f}us")
    except Exception:
        pass

    try:
        memcpy = run_sql_memcpy(sqlite_path)
        if memcpy.memcpy_ops:
            lines.append("")
            lines.append("内存拷贝统计：")
            for m in memcpy.memcpy_ops:
                total_mb = m.total_bytes / 1e6
                total_ms = m.total_ns / 1e6
                lines.append(f"  {m.direction:<6}  count={m.count:<5}  {total_mb:.2f}MB  {total_ms:.3f}ms  {m.avg_bw_gbps:.2f}GB/s")
    except Exception:
        pass

    try:
        kernels = run_sql_kernels(sqlite_path, limit=15)
        if kernels.kernels:
            lines.append("")
            lines.append("Top GPU Kernel（按总耗时，top 15）：")
            for k in kernels.kernels:
                total_ms = k.total_ns / 1e6
                avg_us = k.avg_ns / 1e3
                lines.append(f"  {k.name[:60]:<60}  count={k.count:<5}  total={total_ms:.3f}ms  avg={avg_us:.1f}us")
    except Exception:
        pass

    try:
        gaps = run_sql_gaps(sqlite_path, limit=10)
        if gaps.gaps:
            lines.append("")
            lines.append(f"GPU 空闲气泡（总空闲={gaps.total_gap_ms:.1f}ms，top 10）：")
            for g in gaps.gaps:
                lines.append(f"  gap={g.gap_ms:.3f}ms  before={g.before_kernel[:50]}  after={g.after_kernel[:50]}")
    except Exception:
        pass

    try:
        kl = run_sql_kernel_launch(sqlite_path, limit=10)
        if kl.launches:
            lines.append("")
            lines.append(f"Kernel Launch 延迟（API→GPU，avg={kl.avg_launch_us:.1f}us，top 10 高延迟）：")
            for l in kl.launches:
                lines.append(f"  {l.kernel_name[:55]:<55}  launch_us={l.launch_us:.1f}")
    except Exception:
        pass

    lines.append("────────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def _emit_status(level: str, message: str) -> None:
    print(f"[{level}] {message}", file=sys.stderr, flush=True)


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
    # Walk lines from the end looking for the pattern
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


def _run_codex_cli(prompt: str, request: NsysAnalysisRequest) -> LocalizationResult:
    repo_root = request.repo_root or "."
    artifact_dir = _create_localization_artifact_dir()
    prompt_path = artifact_dir / "prompt.txt"
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    output_path = artifact_dir / "last_message.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    command = [
        "codex",
        "exec",
        "--cd",
        repo_root,
        "--sandbox",
        "workspace-write",
        "--color",
        "never",
        "--output-last-message",
        str(output_path),
    ]
    if request.localization_model:
        command.extend(["--model", request.localization_model])
    if not Path(repo_root).joinpath(".git").exists():
        command.append("--skip-git-repo-check")
    command.append("-")  # read prompt from stdin

    _write_localization_manifest(artifact_dir, {
        "status": "starting",
        "backend": "codex",
        "repo_root": repo_root,
        "command": command,
        "prompt_path": str(prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "output_path": str(output_path),
    })

    if request.emit_progress_info:
        _emit_status("info", f"codex 子进程启动: {' '.join(command)}")
        _emit_status("info", f"codex 工件目录: {artifact_dir}")
        _emit_status("info", f"codex prompt: {prompt_path}")
        _emit_status("info", f"codex stdout: {stdout_path}")
        _emit_status("info", f"codex stderr: {stderr_path}")
        _emit_status("info", f"codex 最终结果: {output_path}")

    return _run_codex_cli_sync(command, prompt, str(output_path), request, artifact_dir, prompt_path, stdout_path, stderr_path)


def _read_localization_output(output_path: str) -> str:
    try:
        return Path(output_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _create_localization_artifact_dir() -> Path:
    base_dir = Path.cwd() / ".sysight" / "codex_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=base_dir))


def _write_localization_manifest(artifact_dir: Path, payload: dict[str, object]) -> None:
    try:
        (artifact_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _extract_json_payload(text: str) -> str:
    fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return ""


def _parse_questions(data: dict[str, object]) -> list[LocalizationQuestion]:
    questions: list[LocalizationQuestion] = []
    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list):
        return questions
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or "").strip()
        if not question_id:
            continue
        line = item.get("line")
        try:
            line_no = int(line) if line is not None else None
        except (TypeError, ValueError):
            line_no = None
        raw_window_ids = item.get("window_ids")
        window_ids = [str(v).strip() for v in raw_window_ids if str(v).strip()] if isinstance(raw_window_ids, list) else []
        questions.append(LocalizationQuestion(
            question_id=question_id,
            problem_id=str(item.get("problem_id") or "").strip(),
            category=str(item.get("category") or "").strip(),
            title=str(item.get("title") or "").strip(),
            file_path=str(item.get("file_path") or "").strip(),
            line=line_no,
            function=str(item.get("function") or "").strip(),
            rationale=str(item.get("rationale") or "").strip(),
            suggestion=str(item.get("suggestion") or "").strip(),
            status=str(item.get("status") or "unknown").strip() or "unknown",
            window_ids=window_ids,
        ))
    return questions


def _parse_anchors(data: dict[str, object]) -> list[LocalizationAnchor]:
    anchors: list[LocalizationAnchor] = []
    raw_anchors = data.get("anchors")
    if not isinstance(raw_anchors, list):
        return anchors
    for item in raw_anchors:
        if not isinstance(item, dict):
            continue
        window_id = str(item.get("window_id") or "").strip()
        if not window_id:
            continue
        line = item.get("line")
        try:
            line_no = int(line) if line is not None else None
        except (TypeError, ValueError):
            line_no = None
        anchors.append(LocalizationAnchor(
            window_id=window_id,
            problem_id=str(item.get("problem_id") or "").strip(),
            category=str(item.get("category") or "").strip(),
            event_name=str(item.get("event_name") or "").strip(),
            file_path=str(item.get("file_path") or "").strip(),
            line=line_no,
            function=str(item.get("function") or "").strip(),
            rationale=str(item.get("rationale") or "").strip(),
            suggestion=str(item.get("suggestion") or "").strip(),
            status=str(item.get("status") or "unknown").strip() or "unknown",
        ))
    return anchors


def _parse_memory_updates(data: dict[str, object]) -> list[dict[str, object]]:
    updates: list[dict[str, object]] = []
    raw_updates = data.get("memory_updates")
    if not isinstance(raw_updates, list):
        return updates
    for item in raw_updates:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        content = item.get("content")
        if not path or not isinstance(content, str):
            continue
        normalized = content.strip()
        if not normalized:
            continue
        updates.append({
            "path": path,
            "content": normalized,
            "append": bool(item.get("append", True)),
        })
    return updates


def _parse_localization_output(text: str) -> tuple[str, list[LocalizationQuestion], list[LocalizationAnchor], str | None, str | None, list[dict[str, object]]]:
    """Returns (summary, questions, anchors, workspace_mem, experience_mem, memory_updates)."""
    payload = _extract_json_payload(text)
    if not payload:
        return "", [], [], None, None, []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "", [], [], None, None, []
    if not isinstance(data, dict):
        return "", [], [], None, None, []

    summary = str(data.get("summary") or "").strip()
    questions = _parse_questions(data)
    anchors = _parse_anchors(data)
    workspace_memory = data.get("workspace_memory") or None
    experience_memory = data.get("experience_memory") or None
    if isinstance(workspace_memory, str):
        workspace_memory = workspace_memory.strip() or None
    if isinstance(experience_memory, str):
        experience_memory = experience_memory.strip() or None
    memory_updates = _parse_memory_updates(data)
    return summary, questions, anchors, workspace_memory, experience_memory, memory_updates


def _legacy_memory_dir(memory_dir: Path | None = None) -> Path:
    runtime_dir = memory_dir or _memory_dir()
    return runtime_dir.parent.parent / "sysight" / "memory"


def _migrate_legacy_memory(memory_dir: Path) -> None:
    legacy_dir = _legacy_memory_dir(memory_dir)
    if not legacy_dir.exists():
        return
    memory_dir.mkdir(parents=True, exist_ok=True)
    for name in ("workspace.md", "experience.md"):
        legacy_path = legacy_dir / name
        if not legacy_path.is_file():
            continue
        try:
            legacy_text = legacy_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not legacy_text:
            continue
        target_path = memory_dir / name
        dedup = name == "experience.md"
        _append_memory(target_path, legacy_text, dedup=dedup)


def _flush_memory(
    workspace_mem: str | None,
    experience_mem: str | None,
    memory_updates: list[dict[str, object]] | None = None,
) -> None:
    """Persist codex-provided memory updates under runtime `.sysight/memory`."""
    memory_dir = _memory_dir()
    _migrate_legacy_memory(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    updates: list[dict[str, object]] = []
    if workspace_mem:
        updates.append({"path": "workspace", "content": workspace_mem, "append": True})
    if experience_mem:
        updates.append({"path": "experience", "content": experience_mem, "append": True})
    if memory_updates:
        updates.extend(memory_updates)

    for update in updates:
        _write_memory_update(memory_dir, update)


def _append_memory(path: Path, content: str, *, dedup: bool = False) -> None:
    """Append a logical memory block with separators and optional exact dedup."""
    block = content.strip()
    if not block:
        return
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if dedup and block in existing:
            return
        separator = "\n\n---\n\n" if existing.strip() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(existing + separator + block, encoding="utf-8")
    except OSError:
        pass


def _write_memory_update(memory_dir: Path, update: dict[str, object]) -> None:
    path = str(update.get("path") or "").strip()
    content = str(update.get("content") or "").strip()
    append = bool(update.get("append", True))
    if not path or not content:
        return

    _, rel_path, target_path = resolve_memory_path(memory_dir, path)
    target_name = Path(rel_path).name
    if append and target_name in {"workspace.md", "experience.md"}:
        _append_memory(target_path, content, dedup=target_name == "experience.md")
        return

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        existing = target_path.read_text(encoding="utf-8") if append and target_path.exists() else ""
        target_path.write_text(existing + content if append else content, encoding="utf-8")
    except OSError:
        pass


def _run_codex_cli_sync(
    command: list[str],
    prompt: str,
    output_path: str,
    request: NsysAnalysisRequest,
    artifact_dir: Path,
    prompt_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> LocalizationResult:
    # Inject PYTHONPATH so that `python3 -m sysight.analyzer.cli` works inside
    # the Codex sandbox regardless of whether the package is pip-installed.
    # _analyzer_dir() points to sysight/analyzer/; the project root is 2 levels up.
    import os as _os
    _project_root = str(Path(__file__).resolve().parents[3])
    _env = dict(_os.environ)
    existing_pp = _env.get("PYTHONPATH", "")
    _env["PYTHONPATH"] = f"{_project_root}:{existing_pp}" if existing_pp else _project_root

    started = time.monotonic()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, \
             stderr_path.open("w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=_env,
                start_new_session=True,
            )
            _write_localization_manifest(artifact_dir, {
                "status": "running",
                "backend": "codex",
                "pid": process.pid,
                "command": command,
                "prompt_path": str(prompt_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "output_path": output_path,
            })
            if request.emit_progress_info:
                _emit_status("info", "codex 调查进行中，等待细定位结果")
            process.communicate(prompt)
    except Exception as exc:
        return LocalizationResult(
            backend="codex",
            status="error",
            prompt=prompt,
            error=str(exc),
            command=command,
            output_path=output_path,
            artifact_dir=str(artifact_dir),
            prompt_path=str(prompt_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    output = _read_localization_output(output_path)
    summary, questions, anchors, workspace_mem, experience_mem, memory_updates = _parse_localization_output(output)
    _flush_memory(workspace_mem, experience_mem, memory_updates)
    elapsed = time.monotonic() - started
    tokens_used = _parse_tokens_from_stderr(stderr_path)
    prompt_tokens_est = _estimate_prompt_tokens(prompt)
    final_status = "ok" if process.returncode == 0 else "error"
    _write_localization_manifest(artifact_dir, {
        "status": final_status,
        "backend": "codex",
        "pid": process.pid,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_used": tokens_used,
        "prompt_tokens_est": prompt_tokens_est,
        "command": command,
        "prompt_path": str(prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "output_path": output_path,
    })

    if process.returncode != 0:
        return LocalizationResult(
            backend="codex",
            status="error",
            prompt=prompt,
            output=output,
            error=f"codex 退出码 {process.returncode}",
            command=command,
            output_path=output_path,
            pid=process.pid,
            summary=summary,
            anchors=anchors,
            questions=questions,
            artifact_dir=str(artifact_dir),
            prompt_path=str(prompt_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    if request.emit_progress_info:
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        if tokens_used is not None:
            prompt_pct = prompt_tokens_est / tokens_used * 100 if tokens_used > 0 else 0.0
            token_str = f"  prompt_tokens: ~{prompt_tokens_est:,} ({prompt_pct:.1f}%),  all tokens used: {tokens_used:,}"
        else:
            token_str = ""
        _emit_status("info", f"codex 调查完成，用时 {elapsed_str}{token_str}，结构化输出: {output_path}")

    return LocalizationResult(
        backend="codex",
        status="ok",
        prompt=prompt,
        output=output or "Codex 已完成，但未返回有效文本结果。",
        command=command,
        output_path=output_path,
        pid=process.pid,
        summary=summary,
        anchors=anchors,
        questions=questions,
        artifact_dir=str(artifact_dir),
        prompt_path=str(prompt_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


register_cli_investigator("codex", _run_codex_cli)


__all__ = [
    "register_cli_investigator",
    "has_cli_investigator",
    "run_code_localization",
    "_run_code_localization",
    "_build_localization_prompt",
]