"""nsys — profile analyzer for NVIDIA Nsight Systems.

Entry point: analyze_nsys()

Returns: NsysDiag  (structured bottleneck + findings + repo hotspot mapping)

Pipeline stages T1–T8 as documented in analyzer.md v3:
  T1: resolve_profile_input  (.nsys-rep → sqlite; action_required if missing)
  T2: inspect_schema         (runtime schema probing → capabilities, table_roles)
  T3: extract_trace          (sqlite → NsysTrace with TimelineEvents)
  T4: classify_bottlenecks   (interval-union → BottleneckSummary + NsysFinding list)
  T5: Event stats built inside classify_bottlenecks
  T6: RepoMapper             (sample → repo file/function)  [via derive_repo_scope]
  T7: ContextEnricher        (callers/callees for high-confidence matches)
  T8: ReportSynthesizer      (assemble NsysDiag + EvidenceLink)

Scope-aware repo scanning:
  By default, analyze_nsys() does NOT do full repo scans for large repos.
  It derives a RepoScope from the trace evidence and scans only relevant files.
  Full scan is opt-in via NsysAnalysisRequest.repo_scope_mode="full".

Sub-module layout:
  input.py    — T1: ProfileInputResolver
  schema.py   — T2: SchemaInspector
  extract.py  — T3: TraceExtractor
  intervals.py — Interval math (union, intersect, gaps)
  classify.py — T4: BottleneckClassifier + NsysFinding generation
  models.py   — All dataclass definitions
"""

from __future__ import annotations

import logging
from pathlib import Path

from .classify import classify_bottlenecks
from .extract import (
    extract_trace,
    resolve_profile_input,
    inspect_schema,
    union_intervals, union_ns, intersect_total, find_gaps,
)
from .models import (
    BottleneckSummary,
    EvidenceLink,
    MappedHotspot,
    NsysDiag,
    NsysAnalysisRequest,
    NsysFinding,
    NsysTrace,
    ProfileInput,
    SchemaInfo,
    TaskDraft,
    TimelineEvent,
)

from ..analyzer import (
    RepoScope,
    ContextBudget,
    FileFacts,
    scan_repo,
    build_repo_index,
    build_callsite_index,
    build_dag,
    derive_analysis_scope,
    lookup_by_file_line,
    lookup_by_symbol,
    callers_of,
    callees_of,
    fuzzy_file_match,
    search_calls,
)

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_nsys",
    "derive_repo_scope",
    "inspect_schema",
    "extract_trace",
    "classify_bottlenecks",
    # Models
    "NsysDiag", "NsysAnalysisRequest", "NsysFinding",
    "BottleneckSummary", "MappedHotspot", "EvidenceLink",
    "NsysTrace", "ProfileInput", "SchemaInfo", "TimelineEvent",
]


# ── Main entry point ─────────────────────────────────────────────────────────


def analyze_nsys(
    request: NsysAnalysisRequest,
) -> NsysDiag:
    """Main entry point for nsys profile analysis.

    Pipeline order:
      1. T1: resolve .nsys-rep → sqlite  (if missing, return action_required)
      2. T2: probe schema → capabilities & table_roles
      3. T3: extract timeline events → NsysTrace
      4.     derive_repo_scope(trace) → RepoScope (if include_repo_context)
      5. T4: classify bottlenecks → BottleneckSummary + list[NsysFinding]
      6.     map hotspots to repo (scope-aware scan)
      7.     enrich with callers/callees
      8.     synthesize NsysDiag + EvidenceLink
    """
    profile_path = Path(request.profile_path)

    # ── T1: Input resolution ─────────────────────────────────────────────────
    prof_input = resolve_profile_input(
        str(profile_path),
        sqlite_path=request.sqlite_path,
    )

    if prof_input.action_required or prof_input.sqlite_path is None:
        return NsysDiag(
            status="action_required",
            profile_path=str(profile_path),
            sqlite_path=prof_input.sqlite_path,
            required_action=prof_input.reason or (
                f"需要导出 SQLite 文件：请运行 `nsys export -t sqlite "
                f"{profile_path}` 生成 .sqlite 文件后重试。"
            ),
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            task_drafts=[],
            repo_warnings=[],
            summary="无法继续分析：缺少 SQLite 导出文件。",
        )

    sqlite_path = prof_input.sqlite_path

    # ── T2: Schema inspection ────────────────────────────────────────────────
    try:
        schema = inspect_schema(sqlite_path)
    except Exception as e:
        logger.error("Schema inspection failed: %s", e)
        return NsysDiag(
            status="error",
            profile_path=str(profile_path),
            sqlite_path=sqlite_path,
            required_action=None,
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            task_drafts=[],
            repo_warnings=[f"Schema 检查错误：{e}"],
            summary=f"读取 SQLite Schema 失败：{e}",
        )

    if "cuda_kernel" not in schema.capabilities:
        return NsysDiag(
            status="error",
            profile_path=str(profile_path),
            sqlite_path=sqlite_path,
            required_action=(
                "Profile 中未包含 GPU 内核数据。"
                "请重新采集并启用 CUDA 内核追踪："
                "nsys profile --trace=cuda,nvtx,osrt ..."
            ),
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            task_drafts=[],
            repo_warnings=schema.warnings,
            summary="未在 Profile 中找到 GPU 内核数据。",
        )

    # ── T3: Trace extraction ─────────────────────────────────────────────────
    try:
        trace = extract_trace(sqlite_path, schema)
    except Exception as e:
        logger.error("Trace extraction failed: %s", e)
        return NsysDiag(
            status="error",
            profile_path=str(profile_path),
            sqlite_path=sqlite_path,
            required_action=None,
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            task_drafts=[],
            repo_warnings=schema.warnings + [f"Trace 提取错误：{e}"],
            summary=f"提取 Trace 失败：{e}",
        )

    # ── T4: Bottleneck classification + Findings ─────────────────────────────
    try:
        bottleneck_summary, findings = classify_bottlenecks(trace)
    except Exception as e:
        logger.error("Bottleneck classification failed: %s", e)
        bottleneck_summary = None
        findings = []
        trace.warnings.append(f"分类分析错误：{e}")

    # ── T6–T7: Repository mapping (scope-aware) ──────────────────────────────
    hotspots: list[MappedHotspot] = []
    evidence_links: list[EvidenceLink] = []
    repo_files: dict[str, FileFacts] = {}
    repo_warnings: list[str] = list(trace.warnings)

    if request.include_repo_context and request.repo_root:
        try:
            hotspots, evidence_links, mapping_warnings, repo_files = _map_hotspots_to_repo(
                trace, findings, request,
            )
            repo_warnings.extend(mapping_warnings)
        except Exception as e:
            logger.warning("Repo mapping failed (non-fatal): %s", e)
            repo_warnings.append(f"代码仓库映射已跳过：{e}")

    # ── T8: Synthesize NsysDiag ─────────────────────────────────────────────
    summary = _build_summary(bottleneck_summary, findings, trace)

    # Generate TaskDrafts for findings that have repo context
    task_drafts = _generate_task_drafts(findings, hotspots, request, trace, repo_files)

    return NsysDiag(
        status="ok",
        profile_path=str(profile_path),
        sqlite_path=sqlite_path,
        required_action=None,
        bottlenecks=bottleneck_summary,
        findings=findings,
        hotspots=hotspots,
        evidence_links=evidence_links,
        task_drafts=task_drafts,
        repo_warnings=repo_warnings,
        summary=summary,
    )


# ── Repo mapping (T6–T7) ─────────────────────────────────────────────────────

def _map_hotspots_to_repo(
    trace: NsysTrace,
    findings: list[NsysFinding],
    request: NsysAnalysisRequest,
) -> tuple[list[MappedHotspot], list[EvidenceLink], list[str], dict[str, FileFacts]]:
    """Map trace hotspots and findings to repository source locations."""
    from .models import SampleHotspot, SourceFrame
    from collections import Counter

    warnings: list[str] = []
    repo_root = Path(request.repo_root)

    if not repo_root.is_dir():
        warnings.append(f"代码仓库路径不存在：{repo_root}")
        return [], [], warnings, {}

    # Derive repo scope from trace evidence
    scope = derive_repo_scope(trace, str(repo_root))
    scope.mode = request.repo_scope_mode
    scope.max_files = request.max_repo_files
    scope.max_file_bytes = request.max_file_bytes
    scope.include_runfiles = _should_include_runfiles(repo_root)

    # Scan repo (targeted by default)
    files, scan_warnings = scan_repo(repo_root, scope=scope)
    warnings.extend(scan_warnings)

    if not files:
        warnings.append("未找到匹配 trace 范围的源文件。")
        return [], [], warnings, {}

    index = build_repo_index(files)

    # Build hotspots from CPU samples
    hotspots: list[MappedHotspot] = []
    cpu_samples = [ev for ev in trace.events if ev.is_sample]

    if cpu_samples:
        counter: Counter[str] = Counter(ev.name for ev in cpu_samples)
        total = len(cpu_samples)
        for symbol, count in counter.most_common(request.top_hotspots):
            pct = count / total
            frame = SourceFrame(symbol=symbol, source_file=None, source_line=None)
            sample = SampleHotspot(frame=frame, count=count, pct=pct)

            # Try to map to repo
            matches = lookup_by_symbol(symbol, index)
            if matches and matches[0].confidence >= request.map_confidence_threshold:
                m = matches[0]
                fn_name = m.function.qualified_name if m.function else None
                callers = callers_of(fn_name, files) if fn_name else []
                callees_list = callees_of(fn_name, files) if fn_name else []
                hotspots.append(MappedHotspot(
                    sample=sample,
                    repo_file=m.repo_file,
                    function=fn_name,
                    match_confidence=m.confidence,
                    match_reason=m.reason,
                    callers=callers,
                    callees=callees_list,
                    alternatives=[str(a) for a in m.alternatives],
                ))
            else:
                hotspots.append(MappedHotspot(
                    sample=sample,
                    repo_file=None,
                    function=None,
                    match_confidence=0.0,
                    match_reason="none",
                    callers=[],
                    callees=[],
                    alternatives=[],
                ))

    # Build evidence links from kernel findings
    evidence_links: list[EvidenceLink] = []
    _link_counters: dict[str, int] = {}
    for finding in findings:
        for hotspot_name in finding.related_hotspots[:3]:
            matches = lookup_by_symbol(hotspot_name, index)
            if matches and matches[0].confidence >= request.map_confidence_threshold:
                m = matches[0]
                fn_name = m.function.qualified_name if m.function else None
                _link_key = f"{finding.category}:{hotspot_name}:symbol"
                _link_idx = _link_counters.get(_link_key, 0)
                _link_counters[_link_key] = _link_idx + 1
                _link_id = f"{_link_key}:{_link_idx}"
                evidence_links.append(EvidenceLink(
                    bottleneck_category=finding.category,
                    event_name=hotspot_name,
                    event_category=finding.category,
                    hotspot_function=fn_name,
                    hotspot_file=m.repo_file,
                    link_type="symbol",
                    reason=f"Kernel name matched repo symbol with confidence {m.confidence:.2f}",
                    confidence=m.confidence * finding.confidence,
                    id=_link_id,
                ))

    # Build NVTX region → repo callsite mapping (补充：不依赖 CPU backtrace)
    nvtx_hotspots = _map_nvtx_regions_to_repo(trace, files)
    hotspots.extend(nvtx_hotspots)

    return hotspots, evidence_links, warnings, files


# NVTX region 名黑名单：系统/框架级噪音，无法映射到用户代码
_NVTX_NOISE = frozenset({
    "Holding GIL", "Waiting for GIL", "GIL", "PyEval",
    "cudaLaunchKernel", "cudaMemcpy", "cudaStreamSynchronize",
    "ncclAllReduce", "ncclBroadcast", "ncclReduceScatter",
    "cublasGemmEx", "cublasGemmStridedBatchedEx",
})

# NVTX range 调用的函数名（Python 侧常见写法）
_NVTX_CALL_NAMES = frozenset({
    "range_push", "range", "mark",
    "torch.cuda.nvtx.range_push", "nvtx.range_push",
    "range_start",
})


def _is_nvtx_noise(name: str) -> bool:
    """过滤系统级 NVTX region 名（iter_NNN、地址、空名等）。"""
    import re
    if not name or len(name) < 2:
        return True
    if name in _NVTX_NOISE:
        return True
    # iter_NNN / step_NNN 等纯数字迭代标记
    if re.match(r'^(iter|step|epoch|batch)_?\d+$', name, re.IGNORECASE):
        return True
    # 纯十六进制地址
    if re.match(r'^0x[0-9a-fA-F]+$', name):
        return True
    return False


def _map_nvtx_regions_to_repo(
    trace: "NsysTrace",
    files: "dict[str, FileFacts]",
) -> "list[MappedHotspot]":
    """将 NVTX region 名映射到 repo 里对应的 nvtx.range_push 调用点。

    策略：
      1. 从 trace NVTX events 提取 distinct 的 region 名（过滤噪音）
      2. 在 callsite index 里找 call_name 为 range_push/range/mark 的条目
      3. 过滤 args_repr 或 source_line 中包含该 region 名的条目
      4. 置信度 = 0.60（精确字符串匹配）
    """
    from .models import SampleHotspot, SourceFrame
    from collections import Counter

    if not files:
        return []

    nvtx_events = [ev for ev in trace.events if ev.category == "nvtx" and ev.name]
    if not nvtx_events:
        return []

    # 按 region 名统计出现次数（用于排序展示）
    counter: Counter[str] = Counter(
        ev.name for ev in nvtx_events if not _is_nvtx_noise(ev.name)
    )
    if not counter:
        return []

    # 建立 callsite index（call_name → [CallSiteFacts]）
    callsite_idx = build_callsite_index(files)

    # 收集所有 NVTX 调用点
    nvtx_callsites = []
    for call_name, cslist in callsite_idx.items():
        if call_name in _NVTX_CALL_NAMES or call_name.endswith("range_push") or call_name.endswith("range"):
            nvtx_callsites.extend(cslist)

    if not nvtx_callsites:
        return []

    hotspots: list[MappedHotspot] = []
    seen: set[str] = set()  # 去重：(repo_file, line, region_name)

    for region_name, count in counter.most_common(20):
        matched_cs = None
        region_lower = region_name.lower()

        for cs in nvtx_callsites:
            # 检查 args_repr 或 source_line 是否包含 region 名
            text = " ".join(cs.args_repr) + " " + (cs.source_line or "")
            if region_lower in text.lower() or region_name in text:
                matched_cs = cs
                break

        if matched_cs is None:
            continue

        key = (matched_cs.path, matched_cs.line, region_name)
        if key in seen:
            continue
        seen.add(key)

        frame = SourceFrame(symbol=region_name, source_file=matched_cs.path, source_line=matched_cs.line)
        total_nvtx = len(nvtx_events)
        pct = count / total_nvtx if total_nvtx > 0 else 0.0
        sample = SampleHotspot(frame=frame, count=count, pct=pct)

        hotspots.append(MappedHotspot(
            sample=sample,
            repo_file=matched_cs.path,
            function=matched_cs.enclosing_function,
            match_confidence=0.60,
            match_reason="nvtx_region",
            callers=[],
            callees=[],
            alternatives=[],
        ))

    return hotspots


def _should_include_runfiles(repo_root: Path) -> bool:
    """Packaged training artifacts often keep real source under *.runfiles."""
    return any(p.is_dir() for p in repo_root.glob("*.runfiles"))


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary(
    bottlenecks: BottleneckSummary | None,
    findings: list[NsysFinding],
    trace: NsysTrace,
) -> str:
    """构建供 LLM 快速阅读的文字摘要。"""
    lines: list[str] = []

    if bottlenecks:
        idle_pct = bottlenecks.gpu_idle_ns / bottlenecks.total_ns * 100 if bottlenecks.total_ns else 0
        lines.append(
            f"Trace 时长：{bottlenecks.total_ns / 1e6:.1f}ms，"
            f"GPU 活跃：{bottlenecks.gpu_active_ns / 1e6:.1f}ms（{100 - idle_pct:.1f}%），"
            f"GPU 空闲：{bottlenecks.gpu_idle_ns / 1e6:.1f}ms（{idle_pct:.1f}%）。"
        )
        if bottlenecks.labels:
            top = bottlenecks.labels[0]
            _CAT_ZH = {
                "gpu_compute":  "GPU 计算",
                "gpu_comm":     "GPU 通信（NCCL）",
                "gpu_memcpy":   "GPU 内存拷贝",
                "gpu_idle":     "GPU 空闲",
                "sync_wait":    "同步等待",
            }
            cat_zh = _CAT_ZH.get(top.category, top.category)
            lines.append(
                f"主要瓶颈：{cat_zh}"
                f"（占 trace {top.pct_of_trace * 100:.1f}%）。"
            )

    crit = [f for f in findings if f.severity == "critical"]
    warn = [f for f in findings if f.severity == "warning"]
    if crit:
        lines.append(f"严重问题 {len(crit)} 条：" + "；".join(f.title for f in crit[:3]))
    if warn:
        lines.append(f"警告 {len(warn)} 条：" + "；".join(f.title for f in warn[:3]))

    if not lines:
        lines.append(f"Trace {trace.profile_path}：已提取 {len(trace.events)} 个事件。")

    return " ".join(lines)


# ── RepoScope derivation from trace evidence ─────────────────────────────────

def derive_repo_scope(
    trace: NsysTrace,
    repo_root: str,
) -> RepoScope:
    """Generate a targeted RepoScope from profile evidence.

    Extracts candidate source files, symbols, and kernel names from:
      - TimelineEvent.extra["source_file"]
      - GPU kernel names
      - NVTX range names

    Priority-based scope: seed_files from profile evidence are highest priority;
    GPU-hinted file expansion is lower priority.

    Returns a RepoScope with mode="targeted" and reasonable defaults.
    Called AFTER T3 (trace extraction) but BEFORE T6 (repo scanning).
    """
    seed_files: list[str] = []
    seed_symbols: list[str] = []
    seed_kernels: list[str] = []

    for event in trace.events:
        if event.category in {"gpu_compute", "gpu_comm"}:
            seed_kernels.append(event.name)
        if event.category == "cuda_api" and "source_file" in event.extra:
            sf = event.extra["source_file"]
            if isinstance(sf, str) and sf:
                seed_files.append(sf)
        if event.category == "nvtx":
            name = event.name.strip()
            if name and not name.upper().startswith("NCCL"):
                seed_symbols.append(name)

    return RepoScope(
        mode="targeted",
        seed_files=sorted(set(seed_files)),
        seed_symbols=sorted(set(seed_symbols)),
        seed_kernels=sorted(set(seed_kernels)),
        include_extensions={".py", ".cpp", ".cc", ".cxx", ".c",
                            ".h", ".hpp", ".cu", ".cuh"},
        follow_imports_depth=1,
        max_files=500,
        max_file_bytes=512_000,
        include_gpu_related=True,
    )


# ── T8 helper: TaskDraft generation ─────────────────────────────────────────

# Finding category → (hypothesis template, verification metric)
_TASK_DRAFT_TEMPLATES: dict[str, tuple[str, str]] = {
    # Matches classify.py categories
    "gpu_memcpy_hotspot": (
        "H2D/D2H memcpy 量异常（占 trace {pct:.0f}%），待确认 host↔device 数据传输的代码位置",
        "memcpy total_ms 下降；GPU idle gap 减少",
    ),
    "gpu_idle": (
        "GPU idle 时间过高（占 trace {pct:.0f}%），待确认 CPU 侧瓶颈位置",
        "GPU idle 时间下降；GPU active 时间上升",
    ),
    "gpu_comm_hotspot": (
        "通信开销较高（占 trace {pct:.0f}%），待确认 backward/all_reduce 调用位置",
        "通信与计算重叠率上升；通信占比下降",
    ),
    "sync_wait": (
        "CPU 同步等待时间过长（占 trace {pct:.0f}%），待确认 synchronize/item 调用位置",
        "同步等待时间下降",
    ),
    # Also handle derived category names if classify.py ever emits them
    "gpu_memcpy_h2d": (
        "H2D memcpy 量异常（占 trace {pct:.0f}%），待确认 host→device 数据传输的代码位置",
        "H2D memcpy total_ms 下降；GPU idle gap 减少",
    ),
    "gpu_memcpy_d2h": (
        "D2H memcpy 量异常（占 trace {pct:.0f}%），待确认 device→host 数据传输的代码位置",
        "D2H memcpy total_ms 下降",
    ),
}

# Finding category → list of rg search seed specs.
# These are ENTRY POINTS for investigation, not conclusions.
# pattern uses Python raw string; CLI usage: rg -n "<pattern>" <repo>
_SEARCH_SPECS: dict[str, list[dict]] = {
    "sync_wait": [
        {
            "pattern": r"synchronize\(|\.item\(|\.cpu\(|barrier\(",
            "kind": "rg",
            "rationale": "sync_wait finding: look for host-side sync calls in Python code",
        },
    ],
    "gpu_memcpy_hotspot": [
        {
            "pattern": r"\.to\(device\)|\.to\(.*cuda|\.cuda\(\)|pin_memory|non_blocking",
            "kind": "rg",
            "rationale": "memcpy finding: look for tensor-to-device transfers and pin_memory settings",
        },
    ],
    "gpu_memcpy_h2d": [
        {
            "pattern": r"\.to\(device\)|\.to\(.*cuda|\.cuda\(\)|pin_memory|non_blocking",
            "kind": "rg",
            "rationale": "H2D memcpy: look for tensor-to-device transfers",
        },
    ],
    "gpu_memcpy_d2h": [
        {
            "pattern": r"\.cpu\(\)|\.numpy\(\)|\.item\(\)",
            "kind": "rg",
            "rationale": "D2H memcpy: look for device-to-host transfers (logging, loss scalar, metric)",
        },
    ],
    "gpu_idle": [
        {
            "pattern": r"synchronize\(|DataLoader|prefetch|barrier\(|time\.sleep",
            "kind": "rg",
            "rationale": "gpu_idle: look for CPU-side stalls, pipeline gaps, DataLoader blocking",
        },
    ],
    "gpu_comm_hotspot": [
        {
            "pattern": r"all_reduce|backward\(|reduce_scatter|allgather|dist\.",
            "kind": "rg",
            "rationale": "comm hotspot: look for collective ops and backward pass calls",
        },
    ],
}


def _generate_task_drafts(
    findings: list[NsysFinding],
    hotspots: list[MappedHotspot],
    request: NsysAnalysisRequest,
    trace: NsysTrace,
    repo_files: dict[str, FileFacts] | None = None,
) -> list[TaskDraft]:
    """Generate deterministic TaskDrafts from findings.

    Fills evidence_windows from the trace (top slow events per finding category)
    and search_specs from the static mapping above.
    candidate_callsites is empty — populated by the LLM investigator later.
    """
    drafts: list[TaskDraft] = []
    seen_categories: set[str] = set()

    repo_files = repo_files or {}
    callsite_index = build_callsite_index(repo_files) if repo_files else {}

    # Pre-build NVTX event list once for evidence window enrichment
    nvtx_events = [ev for ev in trace.events if ev.category == "nvtx"]
    # gpu_active_events includes memcpy: GPU is "active" during memcpy,
    # so idle gaps should be gaps where none of these are running.
    gpu_compute_events = [ev for ev in trace.events
                          if ev.category in ("gpu_compute", "gpu_comm", "gpu_memcpy")
                          and ev.dur_ns > 0]

    for idx, finding in enumerate(findings):
        cat = finding.category
        if cat not in _TASK_DRAFT_TEMPLATES:
            continue
        # One draft per category (dedup by category)
        if cat in seen_categories:
            continue
        seen_categories.add(cat)

        hyp_template, metric = _TASK_DRAFT_TEMPLATES[cat]
        pct = _extract_pct_from_finding(finding)
        hypothesis = hyp_template.format(pct=pct)

        draft_id = f"{cat}_{idx:03d}_draft"

        evidence_windows = _build_evidence_windows(cat, trace, nvtx_events, gpu_compute_events)
        search_specs = list(_SEARCH_SPECS.get(cat, []))
        candidate_callsites = _find_task_candidate_callsites(cat, repo_files, callsite_index)

        drafts.append(TaskDraft(
            id=draft_id,
            finding_id=finding.stable_id or _finding_id(finding, idx),
            hypothesis=hypothesis,
            verification_metric=metric,
            candidate_callsites=candidate_callsites,
            target_locations=[],
            inferred_by="deterministic",
            evidence_windows=evidence_windows,
            search_specs=search_specs,
        ))

    return drafts


def _find_task_candidate_callsites(
    category: str,
    repo_files: dict[str, FileFacts],
    callsite_index: dict,
    limit: int = 5,
) -> list[str]:
    """Return deterministic callsite ids that seed LLM investigation.

    Returns ids in the form "<path>:<line>:<col>:<call_name>" so the
    render layer and LLM investigator can resolve them with get_callsite_context().
    The caller (render.py) shows file:line:fn:source_line from the id.
    """
    if not repo_files:
        return []
    scope = derive_analysis_scope(category, repo_files)
    candidates = search_calls(callsite_index, scope, limit=limit * 4)
    candidates = [c for c in candidates if not _is_task_candidate_noise(category, c.source_line)]
    # Prefer candidates whose enclosing function looks like a training iteration
    top = candidates[:limit]
    return [c.id for c in top]


def _is_task_candidate_noise(category: str, source_line: str) -> bool:
    """Filter syntax matches that are not useful investigation seeds."""
    if category in ("gpu_memcpy_hotspot", "gpu_memcpy_h2d", "gpu_memcpy_d2h"):
        # Triton tensor dtype casts look like `.to(tl.float32)` but are not
        # host/device transfers. Keep them out of deterministic seed slots.
        if ".to(tl." in source_line:
            return True
    return False


def _build_evidence_windows(
    category: str,
    trace: NsysTrace,
    nvtx_events: list[TimelineEvent],
    gpu_compute_events: list[TimelineEvent],
    top_n: int = 5,
) -> list[dict]:
    """Extract top-N slowest time windows for a finding category.

    For each window we record: timestamp, device, stream, event name,
    neighboring GPU kernels (before/after), and overlapping NVTX labels.

    Returns up to top_n dicts, sorted by duration descending.
    """
    # Choose source events based on finding category
    if category in ("sync_wait",):
        source_events = [ev for ev in trace.events
                         if ev.category == "sync_wait" and ev.dur_ns > 0]
    elif category in ("gpu_memcpy_hotspot", "gpu_memcpy_h2d", "gpu_memcpy_d2h"):
        source_events = [ev for ev in trace.events
                         if ev.category == "gpu_memcpy" and ev.dur_ns > 0]
    elif category in ("gpu_idle",):
        # Use large gaps between GPU events as windows
        return _build_idle_windows(trace, nvtx_events, gpu_compute_events, top_n)
    elif category in ("gpu_comm_hotspot",):
        source_events = [ev for ev in trace.events
                         if ev.category == "gpu_comm" and ev.dur_ns > 0]
    else:
        return []

    if not source_events:
        return []

    top = sorted(source_events, key=lambda e: e.dur_ns, reverse=True)[:top_n]
    windows = []
    for ev in top:
        windows.append(_make_window(ev, nvtx_events, gpu_compute_events))
    return windows


def _build_idle_windows(
    trace: NsysTrace,
    nvtx_events: list[TimelineEvent],
    gpu_compute_events: list[TimelineEvent],
    top_n: int,
) -> list[dict]:
    """Build evidence windows from GPU idle gaps (between compute kernels)."""
    from .extract import find_gaps
    if not gpu_compute_events:
        return []
    intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in gpu_compute_events]
    trace_start = trace.trace_start_ns or min(s for s, _ in intervals)
    trace_end = trace.trace_end_ns or max(e for _, e in intervals)
    gaps = find_gaps(intervals, trace_start, trace_end, min_gap_ns=1_000_000)
    top_gaps = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:top_n]

    windows = []
    for gs, ge in top_gaps:
        # Find the kernel just before and after the gap
        before = _find_kernel_before(gs, gpu_compute_events)
        after = _find_kernel_after(ge, gpu_compute_events)
        overlap_nvtx = _overlapping_nvtx_labels(gs, ge, nvtx_events)
        windows.append({
            "start_ms": round(gs / 1e6, 3),
            "end_ms": round(ge / 1e6, 3),
            "duration_ms": round((ge - gs) / 1e6, 3),
            "device": None,
            "stream": None,
            "event": "gpu_idle_gap",
            "before_kernel": before,
            "after_kernel": after,
            "overlap_nvtx": overlap_nvtx,
        })
    return windows


def _make_window(
    ev: TimelineEvent,
    nvtx_events: list[TimelineEvent],
    gpu_compute_events: list[TimelineEvent],
) -> dict:
    """Build one evidence window dict from a single TimelineEvent."""
    ev_end = ev.start_ns + ev.dur_ns
    before = _find_kernel_before(ev.start_ns, gpu_compute_events)
    after = _find_kernel_after(ev_end, gpu_compute_events)
    overlap_nvtx = _overlapping_nvtx_labels(ev.start_ns, ev_end, nvtx_events)
    w: dict = {
        "start_ms": round(ev.start_ns / 1e6, 3),
        "end_ms": round(ev_end / 1e6, 3),
        "duration_ms": round(ev.dur_ns / 1e6, 3),
        "device": ev.device_id,
        "stream": ev.stream_id,
        "event": ev.name,
        "event_category": ev.category,
        "before_kernel": before,
        "after_kernel": after,
        "overlap_nvtx": overlap_nvtx,
        "correlation_id": ev.correlation_id,
    }
    # Memcpy-specific fields from TimelineEvent.extra
    if ev.category == "gpu_memcpy":
        w["copy_kind"] = ev.extra.get("copy_kind")   # 1=H2D 2=D2H 8=D2D etc.
        w["size_bytes"] = ev.extra.get("size_bytes")
    return w


def _find_kernel_before(ts_ns: int, gpu_events: list[TimelineEvent]) -> str | None:
    """Return the name of the most recent GPU kernel ending at or before ts_ns."""
    best: TimelineEvent | None = None
    for ev in gpu_events:
        ev_end = ev.start_ns + ev.dur_ns
        if ev_end <= ts_ns:
            if best is None or ev_end > best.start_ns + best.dur_ns:
                best = ev
    return best.name if best else None


def _find_kernel_after(ts_ns: int, gpu_events: list[TimelineEvent]) -> str | None:
    """Return the name of the first GPU kernel starting at or after ts_ns."""
    best: TimelineEvent | None = None
    for ev in gpu_events:
        if ev.start_ns >= ts_ns:
            if best is None or ev.start_ns < best.start_ns:
                best = ev
    return best.name if best else None


def _overlapping_nvtx_labels(
    start_ns: int,
    end_ns: int,
    nvtx_events: list[TimelineEvent],
    limit: int = 5,
) -> list[str]:
    """Return NVTX range labels whose time window overlaps [start_ns, end_ns]."""
    labels = []
    for ev in nvtx_events:
        ev_end = ev.start_ns + ev.dur_ns
        # Overlap: not (ev ends before start OR ev starts after end)
        if ev_end > start_ns and ev.start_ns < end_ns:
            if ev.name and ev.name not in labels:
                labels.append(ev.name)
        if len(labels) >= limit:
            break
    return labels


def _finding_id(finding: NsysFinding, idx: int) -> str:
    return f"{finding.category}_{idx:03d}"


def _extract_pct_from_finding(finding: NsysFinding) -> float:
    """Best-effort extraction of percentage from finding evidence strings."""
    import re
    for ev in finding.evidence:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", ev)
        if m:
            return float(m.group(1))
    return 0.0
