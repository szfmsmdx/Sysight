"""ANALYZE stage — per-profile, 1 AgentLoop → LocalizedFindingSet.

Outputs to .sysight/analysis-runs/<run_id>/:
  debug.log          — turn-by-turn LLM I/O
  analyze_raw.json   — findings + context_stats
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sysight.agent.loop import AgentLoop, AgentTask
from sysight.benchmark.debug import DebugProvider
from sysight.types.findings import LocalizedFinding, LocalizedFindingSet


@dataclass
class AnalyzeResult:
    run_id: str = ""
    run_dir: Path | None = None          # .sysight/analysis-runs/<run_id>/
    finding_set: LocalizedFindingSet = field(default_factory=lambda: LocalizedFindingSet(run_id=""))
    errors: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    context_stats: dict = field(default_factory=dict)
    provider_error: dict = field(default_factory=dict)
    evidence_pack: dict = field(default_factory=dict)
    elapsed_ms: float = 0
    backoff_ms: float = 0


def run_analyze(
    profile: str,
    repo: str,
    registry,
    provider,
    knowledge=None,
    run_id: str = "",
    verbose: bool = False,
) -> AnalyzeResult:
    """Analyze a profile and produce a LocalizedFindingSet.

    Always writes debug.log and analyze_raw.json to
    .sysight/analysis-runs/<run_id>/.  Pass verbose=True to also echo
    the LLM interaction to the terminal (stderr).
    """
    errors: list[str] = []
    root = Path(repo).resolve()
    sqlite_path = Path(profile).resolve()
    ns = knowledge.workspace_namespace(repo_root=str(root)) if knowledge else "default"

    if not run_id:
        # Stable short hash from profile path + repo path
        digest = hashlib.sha1(
            f"{sqlite_path}|{root}".encode()
        ).hexdigest()[:8]
        run_id = f"run-{digest}"

    # Create output directory
    run_dir = Path.cwd() / ".sysight" / "analysis-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build prompt bundle (shared with build_analyze_first_turn)
    bundle = _build_analyze_bundle(
        sqlite_path=sqlite_path,
        root=root,
        ns=ns,
        registry=registry,
        knowledge=knowledge,
    )
    evidence_pack = bundle["evidence_pack"]
    system_prompt = bundle["system_prompt"]
    user_prompt = bundle["user_prompt"]

    # 2. Wrap provider with DebugProvider (always logs to file; verbose → stderr)
    debug_log: list[dict] = []
    log_file = str(run_dir / "debug.log")
    wrapped = DebugProvider(provider, debug_log, verbose=verbose, log_file=log_file)

    # 3. Run AgentLoop
    policy = _make_analyze_policy()
    loop = AgentLoop(wrapped, registry, policy)
    task = AgentTask(
        run_id=run_id, task_id=f"{run_id}-analyze",
        task_type="analyze",
        system_prompt=system_prompt, user_prompt=user_prompt,
        max_turns=50, max_wall_seconds=0,
    )

    result = loop.run(task)

    # 4. Parse output
    finding_set = LocalizedFindingSet(run_id=run_id)
    if result.output:
        finding_set = _parse_finding_set(result.output, run_id)

    # 5. Validate findings
    accepted, rejected = _validate_findings(finding_set.findings, root)
    finding_set.findings = accepted
    finding_set.rejected = rejected

    # 6. Write analyze_raw.json
    analyze_raw = {
        "run_id": run_id,
        "summary": finding_set.summary,
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
                "evidence_refs": f.evidence_refs,
                "metric": f.metric,
                "description": f.description,
                "suggestion": f.suggestion,
                "status": f.status,
                "reject_reason": f.reject_reason,
            }
            for f in (accepted + rejected)
        ],
        "rejected_count": len(rejected),
        "errors": errors,  # populated below
        "tool_calls": result.tool_calls,
        "context_stats": result.context_stats,
        "provider_error": result.provider_error,
        "evidence_pack": evidence_pack,
        "elapsed_ms": result.elapsed_ms,
        "backoff_ms": result.backoff_ms,
    }

    # 7. Record to ledger
    if knowledge:
        try:
            from sysight.wiki.ledger import RunLedger, RunRecord
            ledger = RunLedger()
            ledger.init()
            ledger.record_session(RunRecord(
                run_id=run_id, status="ok", repo_root=str(root),
                profile_hash=str(sqlite_path), memory_namespace=ns,
            ))
            ledger.record_findings(run_id, [
                {"finding_id": f.finding_id, "category": f.category,
                 "file_path": f.file_path, "line": f.line, "function": f.function,
                 "confidence": f.confidence, "status": f.status,
                 "reject_reason": f.reject_reason}
                for f in (accepted + rejected)
            ])
        except Exception as e:
            errors.append(f"ledger write failed: {e}")

    errors.extend(result.errors)

    (run_dir / "analyze_raw.json").write_text(
        json.dumps(analyze_raw, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ── Always print the output location so users can find logs ──
    sep = "=" * 60
    print(f"\n{sep}", file=sys.stderr)
    print(f"  ANALYZE COMPLETE  run_id={run_id}", file=sys.stderr)
    print(f"  Output dir : {run_dir}", file=sys.stderr)
    print(f"  debug.log  : {run_dir / 'debug.log'}", file=sys.stderr)
    print(f"  results    : {run_dir / 'analyze_raw.json'}", file=sys.stderr)
    print(f"{sep}\n", file=sys.stderr)

    return AnalyzeResult(
        run_id=run_id,
        run_dir=run_dir,
        finding_set=finding_set,
        errors=errors,
        tool_calls=result.tool_calls,
        context_stats=result.context_stats,
        provider_error=result.provider_error,
        evidence_pack=evidence_pack,
        elapsed_ms=result.elapsed_ms,
        backoff_ms=result.backoff_ms,
    )


def build_analyze_first_turn(
    profile: str,
    repo: str,
    registry=None,
    knowledge=None,
) -> dict:
    """Build the exact first-turn analyze request without calling an LLM."""
    root = Path(repo).resolve()
    sqlite_path = Path(profile).resolve()
    ns = knowledge.workspace_namespace(repo_root=str(root)) if knowledge else "default"

    bundle = _build_analyze_bundle(
        sqlite_path=sqlite_path,
        root=root,
        ns=ns,
        registry=registry,
        knowledge=knowledge,
    )
    tools = registry.as_openai_tools(_make_analyze_policy()) if registry else []
    return {
        "system_prompt": bundle["system_prompt"],
        "messages": [{"role": "user", "content": bundle["user_prompt"]}],
        "tools": tools,
        "evidence_pack": bundle["evidence_pack"],
        "full_prompt": (
            "[SYSTEM]\n"
            f"{bundle['system_prompt']}\n\n"
            "[USER]\n"
            f"{bundle['user_prompt']}"
        ),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_analyze_bundle(
    *,
    sqlite_path: Path,
    root: Path,
    ns: str,
    registry=None,
    knowledge=None,
) -> dict:
    """Build the prompt bundle (system + user + evidence_pack) shared by
    run_analyze and build_analyze_first_turn."""
    global_brief, evidence_pack = _build_global_brief(sqlite_path, root, ns)
    memory_refs = _build_memory_refs(root, knowledge, ns)
    evidence_pack["memory_refs"] = memory_refs

    from sysight.agent.prompts.loader import PromptLoader
    loader = PromptLoader()
    system_prompt = loader.build_system_prompt("analyze")
    user_prompt = _build_analyze_user_prompt(
        global_brief=global_brief,
        memory_refs=memory_refs,
        sqlite_path=sqlite_path,
        repo_root=root,
        namespace=ns,
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "evidence_pack": evidence_pack,
    }


def _make_analyze_policy():
    from sysight.tools.registry import ToolPolicy
    return ToolPolicy(
        allowed_tools={"scanner_*", "nsys_sql_*", "memory_search", "memory_read"},
        read_only=True,
        max_reads_per_file=4,
    )


def _build_global_brief(
    sqlite_path: Path,
    repo_root: Path,
    namespace: str,
) -> tuple[str, dict]:
    """Build compact global profile statistics for the analyzer prompt.

    The brief is deliberately descriptive rather than prescriptive: it reports
    broad profile facts but does not inject source candidates or suspicious
    points. The LLM should form and verify hypotheses through tools.
    """
    from sysight.agent.context import to_jsonable

    pack: dict = {
        "repo_root": str(repo_root),
        "sqlite_path": str(sqlite_path),
        "namespace": namespace,
        "sections": {},
        "tool_errors": [],
    }

    def capture(section: str, fn, *args, **kwargs):
        try:
            data = fn(*args, **kwargs)
            pack["sections"][section] = to_jsonable(data)
            return data
        except Exception as e:
            item = {"section": section, "error": str(e)}
            pack["tool_errors"].append(item)
            return None

    from sysight.tools.nsys_sql.gaps import gaps
    from sysight.tools.nsys_sql.kernels import kernels
    from sysight.tools.nsys_sql.launch import launch
    from sysight.tools.nsys_sql.memcpy import memcpy
    from sysight.tools.nsys_sql.nccl import nccl
    from sysight.tools.nsys_sql.nvtx import nvtx
    from sysight.tools.nsys_sql.overlap import overlap
    from sysight.tools.nsys_sql.sync import sync

    machine_info = _read_machine_info(sqlite_path)
    event_inventory = _read_event_inventory(sqlite_path)
    kernel_data = capture("top_kernels", kernels, str(sqlite_path), limit=5)
    nvtx_data = capture("nvtx_ranges", nvtx, str(sqlite_path), limit=5, include_counts=False)
    gap_data = capture("gpu_idle_gaps", gaps, str(sqlite_path), min_gap_ns=1_000_000, limit=5)
    sync_data = capture("sync", sync, str(sqlite_path))
    memcpy_data = capture("memcpy", memcpy, str(sqlite_path))
    launch_data = capture("kernel_launch", launch, str(sqlite_path), limit=5)
    nccl_data = capture("nccl", nccl, str(sqlite_path), limit=8)
    overlap_data = capture("compute_comm_overlap", overlap, str(sqlite_path))
    iteration_summary = _read_iteration_summary(sqlite_path)

    pack["sections"]["machine_info"] = machine_info
    pack["sections"]["event_inventory"] = event_inventory
    pack["sections"]["iteration_summary"] = iteration_summary

    # Derive trace duration
    trace_ns = 0
    if kernel_data and getattr(kernel_data, "trace_duration_ns", 0):
        trace_ns = int(kernel_data.trace_duration_ns)
    elif nvtx_data and getattr(nvtx_data, "trace_duration_ns", 0):
        trace_ns = int(nvtx_data.trace_duration_ns)

    total_kernel_ns = int(getattr(kernel_data, "total_kernel_ns", 0) or 0) if kernel_data else 0
    gpu_active_pct = round(total_kernel_ns / trace_ns * 100, 1) if trace_ns else 0
    gpu_idle_pct = round(100 - gpu_active_pct, 1)
    sync_ns = int(getattr(sync_data, "total_sync_ns", 0) or 0) if sync_data else 0
    nccl_ns = int(getattr(nccl_data, "total_nccl_ns", 0) or 0) if nccl_data else 0
    memcpy_ns = int(getattr(memcpy_data, "total_ns", 0) or 0) if memcpy_data else 0
    gpu_idle_ns = max(0, trace_ns - total_kernel_ns)

    # ── Render markdown ──
    lines: list[str] = [
        "# Profile 统计报告",
        "",
        f"- SQLite: `{sqlite_path}`",
        f"- Repo root: `{repo_root}`",
        f"- Memory namespace: `{namespace}`",
        f"- Trace 时长: {_fmt_ms(trace_ns)}" if trace_ns else "- Trace 时长: unknown",
    ]
    _append_machine_summary(lines, machine_info, total_kernel_ns, gpu_active_pct, gpu_idle_pct)
    _append_event_summary(lines, event_inventory)
    _append_time_distribution(lines, gpu_idle_ns, total_kernel_ns, sync_ns, nccl_ns, memcpy_ns, trace_ns)
    _append_device_workload(lines, machine_info)
    _append_iteration_summary(lines, iteration_summary)
    _append_kernel_table(lines, kernel_data, trace_ns)
    _append_nvtx_table(lines, nvtx_data)
    _append_gap_table(lines, gap_data)
    _append_sync_table(lines, sync_data)
    _append_memcpy_table(lines, memcpy_data)
    _append_launch_table(lines, launch_data)
    _append_nccl_table(lines, nccl_data)
    _append_overlap_section(lines, overlap_data, pack)
    _append_tool_errors(lines, pack)

    return "\n".join(lines), pack


def _append_machine_summary(lines, machine_info, total_kernel_ns, gpu_active_pct, gpu_idle_pct):
    if not machine_info:
        return
    gpu_models = machine_info.get('gpu_models', [])
    gpu_str = ', '.join(gpu_models[:2]) if gpu_models else 'unknown'
    lines.append(f"- GPU: {gpu_str}")
    topo = machine_info.get('topology', 'unknown')
    active = machine_info.get('active_gpu_count', 0)
    if active > 0:
        lines.append(f"- 拓扑: {topo}  |  GPU 活跃: {_fmt_ms(total_kernel_ns)} ({gpu_active_pct}%)  |  GPU 空闲: {gpu_idle_pct}%")
    if machine_info.get("gpu_specs"):
        spec = machine_info["gpu_specs"]
        lines.append(
            f"- 单卡: {_fmt_bytes(spec.get('total_memory', 0))}, "
            f"SM={spec.get('sm_count', 'unknown')}, "
            f"BW≈{_fmt_bandwidth(spec.get('memory_bandwidth', 0))}"
        )


def _append_event_summary(lines, event_inventory):
    if not event_inventory:
        return
    lines.append(
        f"- 事件: {event_inventory.get('kernel_count', 0)} kernels, "
        f"{event_inventory.get('runtime_count', 0)} runtime, "
        f"{event_inventory.get('memcpy_count', 0)} memcpy, "
        f"{event_inventory.get('sync_count', 0)} sync, "
        f"{event_inventory.get('stream_count', 0)} streams"
    )


def _append_time_distribution(lines, gpu_idle_ns, kernel_ns, sync_ns, nccl_ns, memcpy_ns, trace_ns):
    lines.extend([
        "",
        "## 时间分布（wall/union 时间维度，近似）",
        "| 类别 | 时长 | Wall % |",
        "|------|------:|-------:|",
        f"| gpu_idle | {_fmt_ms(gpu_idle_ns)} | {_fmt_pct(gpu_idle_ns, trace_ns)} |",
        f"| gpu_compute | {_fmt_ms(kernel_ns)} | {_fmt_pct(kernel_ns, trace_ns)} |",
        f"| sync_wait | {_fmt_ms(sync_ns)} | {_fmt_pct(sync_ns, trace_ns)} |",
        f"| gpu_comm | {_fmt_ms(nccl_ns)} | {_fmt_pct(nccl_ns, trace_ns)} |",
        f"| gpu_memcpy | {_fmt_ms(memcpy_ns)} | {_fmt_pct(memcpy_ns, trace_ns)} |",
    ])


def _append_device_workload(lines, machine_info):
    if not machine_info.get("device_workload") or int(machine_info.get("active_gpu_count", 0)) <= 1:
        return
    lines.extend([
        "",
        "## GPU 负载分布",
        "| Device | Kernels | Streams | GPU Time | Share |",
        "|-------:|--------:|--------:|---------:|------:|",
    ])
    total_device_ns = sum(int(d.get("total_ns", 0)) for d in machine_info["device_workload"])
    for item in machine_info["device_workload"][:16]:
        lines.append(
            f"| {item.get('device_id')} | {item.get('kernels')} | {item.get('streams')} | "
            f"{_fmt_ms(item.get('total_ns', 0))} | {_fmt_pct(item.get('total_ns', 0), total_device_ns)} |"
        )


def _append_iteration_summary(lines, iteration_summary):
    if not iteration_summary:
        return
    lines.extend([
        "",
        "## Iteration 时间分布",
        f"- iteration ranges: {iteration_summary.get('count', 0)}; "
        f"首轮: {_fmt_ms(iteration_summary.get('first_ns', 0))}; "
        f"稳态均值(排除首轮): {_fmt_ms(iteration_summary.get('steady_avg_ns', 0))}; "
        f"稳态中位数: {_fmt_ms(iteration_summary.get('steady_median_ns', 0))}",
        f"- 稳态 min/max: {_fmt_ms(iteration_summary.get('steady_min_ns', 0))} / "
        f"{_fmt_ms(iteration_summary.get('steady_max_ns', 0))}",
    ])


def _append_kernel_table(lines, kernel_data, trace_ns):
    if not kernel_data:
        return
    total_kernel_ns = int(getattr(kernel_data, "total_kernel_ns", 0) or 0)
    lines.extend([
        "",
        "## GPU Kernel 总时长 Top 5",
        "| # | Kernel | Count | Total | Avg | Max | Trace % |",
        "|---|--------|------:|------:|----:|----:|--------:|",
    ])
    for idx, k in enumerate(getattr(kernel_data, "kernels", [])[:5], 1):
        lines.append(
            f"| {idx} | `{_short(k.name, 72)}` | {k.count} | {_fmt_ms(k.total_ns)} | "
            f"{_fmt_ms(k.avg_ns)} | {_fmt_ms(k.max_ns)} | {_fmt_pct(k.total_ns, trace_ns)} |"
        )
    if total_kernel_ns:
        lines.append(f"- Sum GPU kernel time: {_fmt_ms(total_kernel_ns)} inclusive across devices/streams.")


def _append_nvtx_table(lines, nvtx_data):
    if not nvtx_data or not getattr(nvtx_data, "ranges", None):
        return
    lines.extend([
        "",
        "## NVTX Range 分布 Top 5",
        "| # | Range | Count | Total | Avg | Max | Kernels | Runtime | Memcpy | Sync |",
        "|---|-------|------:|------:|----:|----:|--------:|--------:|-------:|-----:|",
    ])
    for idx, r in enumerate(nvtx_data.ranges[:5], 1):
        lines.append(
            f"| {idx} | `{_short(r.name, 64)}` | {r.count} | {_fmt_ms(r.total_ns)} | "
            f"{_fmt_ms(r.avg_ns)} | {_fmt_ms(r.max_ns)} | {r.kernel_count} | "
            f"{r.runtime_count} | {r.memcpy_count} | {r.sync_count} |"
        )


def _append_gap_table(lines, gap_data):
    if not gap_data:
        return
    lines.extend([
        "",
        "## GPU 空闲间隙",
        f"- 已报告 gap >= 1ms: {gap_data.gap_count}; 展示条目的总 gap: {_fmt_ms(gap_data.total_gap_ns)}.",
        "| # | Stream | Gap | Before | After |",
        "|---|-------:|----:|--------|-------|",
    ])
    for idx, g in enumerate(getattr(gap_data, "gaps", [])[:5], 1):
        lines.append(
            f"| {idx} | {g.stream_id} | {_fmt_ms(g.gap_ns)} | "
            f"`{_short(g.before_kernel or '', 44)}` | `{_short(g.after_kernel or '', 44)}` |"
        )


def _append_sync_table(lines, sync_data):
    if not sync_data:
        return
    lines.extend([
        "",
        "## 同步摘要",
        f"- 同步总时长: {_fmt_ms(sync_data.total_sync_ns)}; union wall pct: {sync_data.sync_wall_pct:.1f}%.",
        "| Type | Count | Total | Avg | Max |",
        "|------|------:|------:|----:|----:|",
    ])
    for item in getattr(sync_data, "sync_events", [])[:8]:
        lines.append(
            f"| `{_short(item.sync_type, 48)}` | {item.count} | {_fmt_ms(item.total_ns)} | "
            f"{_fmt_ms(item.avg_ns)} | {_fmt_ms(item.max_ns)} |"
        )


def _append_memcpy_table(lines, memcpy_data):
    if not memcpy_data:
        return
    lines.extend([
        "",
        "## 内存搬运",
        f"- 总字节数: {_fmt_bytes(memcpy_data.total_bytes)}; 总 copy 时长: {_fmt_ms(memcpy_data.total_ns)}.",
        "| Direction | Count | Bytes | Time | Avg BW |",
        "|-----------|------:|------:|-----:|-------:|",
    ])
    for item in getattr(memcpy_data, "memcpy_ops", [])[:8]:
        lines.append(
            f"| {item.direction} | {item.count} | {_fmt_bytes(item.total_bytes)} | "
            f"{_fmt_ms(item.total_ns)} | {item.avg_bw_gbps:.2f} GB/s |"
        )


def _append_launch_table(lines, launch_data):
    if not launch_data:
        return
    lines.extend([
        "",
        "## Kernel Launch 开销",
        f"- 展示条目的平均 overhead: {launch_data.avg_overhead_us:.1f}us; max: {launch_data.max_overhead_us:.1f}us.",
        "- 注意: launch overhead 依赖 CUPTI correlation 语义，异常大值需要结合 runtime/NVTX/source 二次验证。",
        "| # | Kernel | API | Kernel | Overhead |",
        "|---|--------|----:|-------:|---------:|",
    ])
    for idx, item in enumerate(getattr(launch_data, "entries", [])[:5], 1):
        lines.append(
            f"| {idx} | `{_short(item.kernel_name, 58)}` | {item.api_ms:.3f}ms | "
            f"{item.kernel_ms:.3f}ms | {item.overhead_us:.1f}us |"
        )


def _append_nccl_table(lines, nccl_data):
    if not nccl_data:
        return
    lines.extend([
        "",
        "## NCCL 摘要",
        f"- NCCL ops: {nccl_data.total_ops}; NCCL 总时长: {_fmt_ms(nccl_data.total_nccl_ns)}.",
        "| Stream | Ops | Total | Avg |",
        "|-------:|----:|------:|----:|",
    ])
    for item in getattr(nccl_data, "streams", [])[:8]:
        lines.append(
            f"| {item.stream_id} | {item.op_count} | {_fmt_ms(item.total_ns)} | {_fmt_ms(item.avg_ns)} |"
        )


def _append_overlap_section(lines, overlap_data, pack):
    if not overlap_data:
        return
    if overlap_data.compute_kernels > 0 and overlap_data.nccl_kernels > 0:
        lines.extend([
            "",
            "## 计算 / 通信 Overlap",
            f"- Span: {_fmt_ms(overlap_data.total_span_ns)}; overlap: {_fmt_ms(overlap_data.overlap_ns)} "
            f"({overlap_data.overlap_pct:.1f}% of NCCL span approximation).",
            f"- Compute-only: {_fmt_ms(overlap_data.compute_only_ns)}; NCCL-only: {_fmt_ms(overlap_data.nccl_only_ns)}; "
            f"kernels: compute={overlap_data.compute_kernels}, nccl={overlap_data.nccl_kernels}.",
            f"- Note: {overlap_data.note}",
        ])
    else:
        pack["tool_errors"].append({
            "section": "compute_comm_overlap",
            "error": (
                "Stream-level overlap approximation is not useful for this profile "
                f"(compute_kernels={overlap_data.compute_kernels}, "
                f"nccl_kernels={overlap_data.nccl_kernels}). "
                "该 stream-level overlap 近似不适用于本 profile；请改用 nsys_sql_nccl、kernels 和 NVTX 证据判断。"
            ),
        })


def _append_tool_errors(lines, pack):
    if not pack["tool_errors"]:
        return
    lines.extend(["", "## 工具 / 数据质量备注"])
    for item in pack["tool_errors"]:
        lines.append(f"- {item['section']}: {item['error']}")


def _read_machine_info(sqlite_path: Path) -> dict:
    """Read target GPU/host info from Nsight target metadata tables."""
    import sqlite3
    from collections import Counter

    info: dict = {
        "available_gpu_count": 0,
        "active_gpu_count": 0,
        "cuda_device_count": 0,
        "gpu_models": [],
        "gpu_specs": {},
        "cpu": {},
        "nic_count": 0,
        "nics": [],
        "device_workload": [],
        "topology": "unknown",
    }

    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        tables = _sqlite_tables(conn)
        has = tables.__contains__

        if has("TARGET_INFO_GPU"):
            rows = conn.execute("SELECT * FROM TARGET_INFO_GPU").fetchall()
            info["available_gpu_count"] = len(rows)
            counts = Counter(str(r["name"] or "unknown") for r in rows)
            info["gpu_models"] = [f"{name} x{count}" for name, count in counts.items()]
            if rows:
                r = rows[0]
                info["gpu_specs"] = {
                    "total_memory": int(r["totalMemory"] or 0) if "totalMemory" in r.keys() else 0,
                    "memory_bandwidth": int(r["memoryBandwidth"] or 0) if "memoryBandwidth" in r.keys() else 0,
                    "sm_count": int(r["smCount"] or 0) if "smCount" in r.keys() else 0,
                    "compute_capability": (
                        f"{int(r['computeMajor'])}.{int(r['computeMinor'])}"
                        if "computeMajor" in r.keys() and r["computeMajor"] is not None else "unknown"
                    ),
                }

        if has("TARGET_INFO_CUDA_DEVICE"):
            rows = conn.execute("SELECT DISTINCT cudaId FROM TARGET_INFO_CUDA_DEVICE").fetchall()
            info["cuda_device_count"] = len(rows)

        if has("TARGET_INFO_SYSTEM_ENV"):
            wanted = {"CpuCores", "CpuSpeedMhz", "CpuArchitecture", "Hostname", "OSRuntime"}
            for row in conn.execute("SELECT name, value FROM TARGET_INFO_SYSTEM_ENV"):
                if row["name"] in wanted:
                    info["cpu"][row["name"]] = row["value"]

        if has("TARGET_INFO_NIC_INFO"):
            rows = conn.execute("SELECT DISTINCT name FROM TARGET_INFO_NIC_INFO ORDER BY name").fetchall()
            info["nics"] = [str(r["name"]) for r in rows if r["name"]]
            info["nic_count"] = len(info["nics"])

        if has("CUPTI_ACTIVITY_KIND_KERNEL"):
            rows = conn.execute(
                """
                SELECT deviceId, COUNT(*) AS kernels, COUNT(DISTINCT streamId) AS streams,
                       SUM([end]-start) AS total_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                GROUP BY deviceId
                ORDER BY total_ns DESC
                """
            ).fetchall()
            info["device_workload"] = [
                {
                    "device_id": int(r["deviceId"]),
                    "kernels": int(r["kernels"] or 0),
                    "streams": int(r["streams"] or 0),
                    "total_ns": int(r["total_ns"] or 0),
                }
                for r in rows
            ]
            info["active_gpu_count"] = len(rows)

        host_states: set[str] = set()
        if has("TARGET_INFO_SYSTEM_ENV"):
            for row in conn.execute("SELECT DISTINCT devStateName FROM TARGET_INFO_SYSTEM_ENV"):
                if row["devStateName"]:
                    host_states.add(str(row["devStateName"]))
        if has("TARGET_INFO_NIC_INFO"):
            for row in conn.execute("SELECT DISTINCT stateName FROM TARGET_INFO_NIC_INFO"):
                if row["stateName"]:
                    host_states.add(str(row["stateName"]))

        host_count = len(host_states) or 1
        active = int(info["active_gpu_count"] or 0)
        if host_count > 1 and active > 1:
            info["topology"] = f"多机多卡迹象({host_count} host states, {active} active GPUs)"
        elif host_count > 1:
            info["topology"] = f"多机/多 target 迹象({host_count} host states)"
        elif active > 1:
            info["topology"] = f"单机多卡({active} active GPUs)"
        elif active == 1:
            visible = int(info["available_gpu_count"] or 0)
            suffix = f"，节点可见 {visible} 张 GPU" if visible > 1 else ""
            info["topology"] = f"单机单卡 workload{suffix}"
        else:
            info["topology"] = "未观察到 GPU kernel 活动"

        conn.close()
    except Exception as e:
        info["error"] = str(e)
    return info


def _read_event_inventory(sqlite_path: Path) -> dict:
    """Read event counts and table availability from a profile."""
    import sqlite3

    table_map = {
        "kernel_count": "CUPTI_ACTIVITY_KIND_KERNEL",
        "runtime_count": "CUPTI_ACTIVITY_KIND_RUNTIME",
        "memcpy_count": "CUPTI_ACTIVITY_KIND_MEMCPY",
        "sync_count": "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        "nvtx_count": "NVTX_EVENTS",
    }
    result: dict = {k: 0 for k in table_map}
    result.update({"available_tables": [], "stream_count": 0,
                   "kernel_process_count": 0, "runtime_thread_count": 0})
    try:
        conn = sqlite3.connect(str(sqlite_path))
        tables = _sqlite_tables(conn)
        result["available_tables"] = [tbl for tbl in table_map.values() if tbl in tables]
        for key, table in table_map.items():
            if table in tables:
                result[key] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            result["stream_count"] = int(conn.execute(
                "SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0] or 0)
            result["kernel_process_count"] = int(conn.execute(
                "SELECT COUNT(DISTINCT globalPid) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0] or 0)
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
            result["runtime_thread_count"] = int(conn.execute(
                "SELECT COUNT(DISTINCT globalTid) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            ).fetchone()[0] or 0)
        conn.close()
    except Exception as e:
        result["error"] = str(e)
    return result


def _read_iteration_summary(sqlite_path: Path) -> dict:
    """Summarize NVTX ranges that look like training/inference iterations."""
    import sqlite3
    from statistics import median

    result: dict = {}
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        tables = _sqlite_tables(conn)
        if "NVTX_EVENTS" not in tables:
            conn.close()
            return result
        rows = conn.execute(
            "SELECT text AS name, start, [end] FROM NVTX_EVENTS "
            "WHERE [end] IS NOT NULL AND [end] > start AND text IS NOT NULL "
            "ORDER BY start"
        ).fetchall()
        durations: list[int] = []
        for row in rows:
            name = str(row["name"] or "")
            if re.search(r"^(iter|iteration|step)([_:/-]?\\d+|[_:/-])", name, re.IGNORECASE):
                durations.append(int(row["end"]) - int(row["start"]))
        conn.close()
        if not durations:
            return result
        steady = durations[1:] if len(durations) > 1 else durations
        result = {
            "count": len(durations),
            "first_ns": durations[0],
            "steady_avg_ns": int(sum(steady) / len(steady)) if steady else 0,
            "steady_median_ns": int(median(steady)) if steady else 0,
            "steady_min_ns": min(steady) if steady else 0,
            "steady_max_ns": max(steady) if steady else 0,
        }
    except Exception as e:
        result["error"] = str(e)
    return result


def _sqlite_tables(conn) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _build_memory_refs(repo_root: Path, knowledge, namespace: str) -> list[dict]:
    """Return memory/wiki references for on-demand reads."""
    if not knowledge:
        return []

    memory_root = Path(getattr(knowledge, "root", Path.cwd() / ".sysight" / "memory"))
    wiki = memory_root / "wiki"
    refs = [
        {
            "label": "warmup_overview",
            "tool": "memory_read",
            "path": f"workspaces/{namespace}/overview.md",
            "when": "需要入口命令、active config、profile artifact、代码地图时读取。",
        },
        {
            "label": "experience_context",
            "tool": "memory_read",
            "path": "experience",
            "when": "需要历史性能经验、常见反模式或可复用判断规则时读取。",
        },
        {
            "label": "wiki_index",
            "tool": "memory_read",
            "path": "INDEX.md",
            "when": "不确定有哪些 wiki 页面时读取。",
        },
        {
            "label": "targeted_memory_search",
            "tool": "memory_search",
            "path": "",
            "when": "按 kernel/NVTX/配置键/性能机制关键词搜索相关经验；可带 namespace 限定。",
        },
    ]
    for ref in refs:
        path = ref.get("path")
        try:
            if path:
                ref["exists"] = bool(knowledge.read_page(path))
            else:
                ref["exists"] = wiki.exists()
        except Exception:
            ref["exists"] = False
    return refs


def _build_analyze_user_prompt(
    *,
    global_brief: str,
    memory_refs: list[dict],
    sqlite_path: Path,
    repo_root: Path,
    namespace: str,
) -> str:
    workspace_path = next(
        (ref["path"] for ref in memory_refs if ref.get("label") == "warmup_overview" and ref.get("path")),
        "",
    )

    parts = [global_brief]
    if workspace_path:
        parts.append(f"workspace_path: {workspace_path}")

    parts.extend([
        _memory_refs_block(memory_refs, namespace),
        "按 SOP 挖掘所有可优化点，只输出 JSON findings。",
    ])

    return "\n".join(part for part in parts if part)


def _memory_refs_block(memory_refs: list[dict], namespace: str) -> str:
    lines = [
        "Memory refs:",
        f"- namespace: `{namespace}`",
    ]
    if memory_refs:
        for ref in memory_refs:
            label = ref.get("label", "")
            path = ref.get("path", "")
            exists = ref.get("exists")
            when = ref.get("when", "")
            if path:
                lines.append(f"- {label}: `{path}` (exists={exists}) — {when}")
    return "\n".join(lines)


def _fmt_ms(ns: int | float) -> str:
    value = float(ns or 0)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}s"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}ms"
    if value >= 1_000:
        return f"{value / 1_000:.1f}us"
    return f"{value:.0f}ns"


def _fmt_pct(ns: int | float, total_ns: int | float) -> str:
    if not total_ns:
        return "n/a"
    return f"{float(ns or 0) / float(total_ns) * 100:.1f}%"


def _fmt_bytes(num: int | float) -> str:
    value = float(num or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f}{unit}" if unit != "B" else f"{value:.0f}B"
        value /= 1024
    return f"{value:.2f}TB"


def _fmt_bandwidth(num: int | float) -> str:
    """Format bytes/s into human-readable bandwidth string."""
    value = float(num or 0)
    if value <= 0:
        return "unknown"
    for unit in ("B/s", "KB/s", "MB/s", "GB/s", "TB/s"):
        if value < 1_000 or unit == "TB/s":
            return f"{value:.2f}{unit}"
        value /= 1_000
    return f"{value:.2f}TB/s"


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _parse_finding_set(data: dict, run_id: str) -> LocalizedFindingSet:
    """Parse LLM output into LocalizedFindingSet.

    Accepts both SOTA field names (evidence, file) and legacy field names
    (evidence_refs, file_path) for compatibility.
    """
    from sysight.types.findings import make_finding_id

    findings = []

    for f in data.get("findings", []):
        evidence_refs = f.get("evidence_refs") or f.get("evidence", [])
        file_path = f.get("file_path") or f.get("file")
        category = f.get("category", "")
        function = f.get("function")
        line = f.get("line")
        finding_id = f.get("finding_id") or make_finding_id(category, file_path, line, function)

        # Auto-derive confidence from evidence presence
        confidence = f.get("confidence")
        if not confidence:
            confidence = "confirmed" if evidence_refs else "unresolved"

        findings.append(LocalizedFinding(
            finding_id=finding_id,
            category=category,
            title=f.get("title", ""),
            priority=f.get("priority", "medium"),
            confidence=confidence,
            evidence_refs=evidence_refs,
            metric=f.get("metric", ""),
            file_path=file_path,
            function=function,
            line=line,
            description=f.get("description", ""),
            suggestion=f.get("suggestion", ""),
            status=f.get("status", "accepted"),
            reject_reason=f.get("reject_reason", ""),
        ))

    return LocalizedFindingSet(
        run_id=run_id,
        summary=data.get("summary", ""),
        findings=findings,
    )


def _validate_findings(
    findings: list[LocalizedFinding],
    repo_root: Path,
) -> tuple[list[LocalizedFinding], list[LocalizedFinding]]:
    """Validate findings: path containment, file existence, dedup."""
    accepted: list[LocalizedFinding] = []
    rejected: list[LocalizedFinding] = []
    seen: set[str] = set()

    def _reject(f: LocalizedFinding, reason: str) -> None:
        f.status = "rejected"
        f.reject_reason = reason
        rejected.append(f)

    for f in findings:
        # Dedup
        key = f"{f.category}:{f.file_path}:{f.line}:{f.function}"
        if key in seen:
            continue
        seen.add(key)

        # Path containment
        if f.file_path:
            try:
                full = (repo_root / f.file_path).resolve()
                if not str(full).startswith(str(repo_root.resolve())):
                    _reject(f, "path outside repo")
                    continue
                if not full.exists():
                    _reject(f, "file not found")
                    continue
            except (OSError, ValueError):
                _reject(f, "invalid path")
                continue

        accepted.append(f)

    return accepted, rejected
