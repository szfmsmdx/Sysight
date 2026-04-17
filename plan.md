# Sysight Analyzer — 架构设计与实现 Plan

---

## ⚑ 设计意图

> analyzer 输出是 agent loop 的输入。每个 `NsysFinding` 由独立 agent 解决。
>
> - 精准第一：宁可 confidence=0，不伪造高置信结论
> - 证据链完整：每个 finding 必须有可审计的数字证据
> - 两层分开：analyzer core（deterministic）；analyzer agent（LLM 调工具）
> - 不要全仓扫：finding-aware scope
> - **分层输出**：analyzer core 输出 `TaskDraft`（deterministic 调查草稿）；analyzer agent / LLM investigator 将 `TaskDraft` 升级为 `OptimizeTask`（验证过的代码定位任务）；optimizer 消费 `OptimizeTask`
> - core 不直接输出 `OptimizeTask`，除非不需要 LLM 参与
> - 大 profile 用 SQL 聚合，不用内存截断

---

## 一、模块布局

```
sysight/analyzer/
  analyzer.py       scan_repo, build_repo_index, lookup_by_symbol,
                    AnalysisScope, derive_analysis_scope,
                    build_callsite_index, search_calls, get_callsite_context
  repo.py           re-export 入口
  cli.py            命令行入口
  __main__.py       python -m sysight.analyzer 入口

  nsys/
    __init__.py     analyze_nsys() 主流程（T1-T8 编排）+ repo mapping
    models.py       所有 dataclass（唯一数据结构来源）
    extract.py      T1/T2/T3 + interval math
    classify.py     T4 瓶颈分类 + NsysFinding 规则引擎
    classify_sql.py T4 深度 SQL 分析（六项）
    render.py       NsysDiag → 终端文本报告
    text.py         format_table / pad_display（Unicode 宽字符）

  scanners/
    base.py         CallSiteFacts / FileFacts / FunctionFacts / BaseScanner
    python.py       PythonScanner（AST + callsite 收集）
    cpp.py          CppScanner（CUDA kernel 扫描）
```

---

## 二、T1：输入解析（`extract.py:resolve_profile_input`）

1. 显式 `sqlite_path` 非空且文件存在 → `ProfileInput(action_required=False)`；否则 `action_required=True`。
2. `.nsys-rep` 后缀：查同名 sibling `.sqlite`；存在且非空 → ok；否则 `action_required=True`（提示 `nsys export`）。
3. `profile_path` 本身是非空文件 → 直接使用。
4. 其余 → `action_required=True`。

---

## 三、T2：Schema 探测（`extract.py:inspect_schema`）

1. `sqlite_master WHERE type='table'` 读所有表名。
2. `PRAGMA table_info(tbl)` 读各表列名；跳过 `_SKIP_COLUMN_READ`（`OSRT_CALLCHAINS`, `SAMPLING_CALLCHAINS`, `NVTX_PAYLOAD_SCHEMAS`）。
3. 遍历 `_CAPABILITY_TABLES` 字典（`capability → [表名前缀]`）：精确匹配优先，否则前缀扫描；若 capability 在 `_REQUIRES_STRING_IDS` 但 `StringIds` 不存在则 warning，仍记录。
4. `_detect_nsys_version()`：扫描 `META_DATA*` 表，找含 "Nsight Systems"/"version"/"exporter" 字段，返回 `version_hint`。
5. 不属于 `_KNOWN_PREFIXES` 的表名汇为 unknown_tables，添加 warning。
6. 返回 `SchemaInfo`。

---

## 四、T3：事件提取（`extract.py:extract_trace`）

### 4.1 时间范围
从 kernel 表（或 runtime 表）查 `MIN(start)` / `MAX(end)` 得 `trace_start_ns / trace_end_ns / duration_ns`。

### 4.2 各提取器

**`_extract_kernels`**：
- 有 `StringIds` 则 JOIN 取 name；否则 `CAST(shortName AS TEXT)`。
- `_is_comm_kernel(name)`：name 小写含 `nccl/allreduce/allgather/reducescatter/broadcast/sendrecv/reduce` 任一 → `category="gpu_comm"`，否则 `"gpu_compute"`。
- 保留 `deviceId`, `streamId`, `correlationId`, `globalTid`（`_sel_col` 处理列缺失时返回 NULL）。

**`_extract_memcpy`**：
- `copyKind` 映射 `{1:"HtoD", 2:"DtoH", 8:"DtoD", 10:"HtoH"}`，name=`memcpy_HtoD` 等。
- `extra={"size_bytes": bytes, "copy_kind": kind}`，`category="gpu_memcpy"`。

**`_extract_memset`**：同 memcpy，name=`"memset"`，`copy_kind=-1`。

**`_extract_runtime`**：JOIN StringIds 读 API name；`category="cuda_api"`；保留 `correlationId`。

**`_extract_sync`**：
- 检查列名：优先 `type`，其次 `syncType`，都无则 `0 AS sync_type`。
- `category="sync_wait"`，`extra={"sync_type": int}`。

**`_extract_nvtx`**：
- textId 变体：`LEFT JOIN StringIds ON n.textId = s.id`，`COALESCE(n.text, s.value)`；否则直接用 `text` 列。
- 过滤 `text IS NOT NULL AND end > start`；`category="nvtx"`。

**`_extract_cpu_samples`**（最复杂）：
- 检测 `COMPOSITE_EVENTS` 用 `timestamp` 还是 `start` 列。
- 有 `StringIds` 且 `SAMPLING_CALLCHAINS` 存在：
  - 一次大 JOIN：`COMPOSITE_EVENTS ce JOIN SAMPLING_CALLCHAINS sc ON ce.id=sc.id LEFT JOIN StringIds s ON sc.symbol=s.id`，`ORDER BY ce.timestamp, sc.stackDepth`（depth=0 = 叶帧 innermost）。
  - Python `itertools.groupby` 按 `(ts, globalTid)` 分组；每组收集所有帧符号；depth=0 帧为 `leaf_symbol = event.name`；全帧存 `extra["callstack"]`。
- 否则：只提取时间戳，`name="cpu_sample"`，无 callstack。
- `is_sample=True`，`dur_ns=0`（不参与 interval 计算）。

**`_extract_cudnn`**：`CUDNN_EVENTS`，`dur=max(0, end-start)`，`category="cuda_api"`。

**`_extract_osrt`**：`OSRT_API`，JOIN StringIds 读 nameId，`category="osrt"`。

---

## 五、Interval Math（`extract.py`）

- **`union_intervals`**：排序后合并重叠区间，返回不重叠列表。
- **`total_covered`**：sum(end-start) of merged intervals。
- **`union_ns`**：`total_covered(union_intervals(...))`。
- **`intersect_intervals`**：先各自 union，双指针求交集。
- **`intersect_total`**：`total_covered(intersect_intervals(a, b))`。
- **`find_gaps`**：在 `[trace_start, trace_end]` 内，找 union 后区间之间大于 `min_gap_ns` 的空隙，返回 `[(gap_start, gap_end)]`。

---

## 六、T4：瓶颈分类（`classify.py:classify_bottlenecks`）

### 6.1 时间分解
- `is_sample=True` 的事件单独归入 `cpu_sample`，不进入 interval 计算。
- `gpu_active_ns = union(gpu_compute + gpu_comm + gpu_memcpy intervals)`。
- `gpu_idle_ns = max(0, total_ns - gpu_active_ns)`。
- `compute_active_ns`, `comm_active_ns`, `memcpy_active_ns`, `sync_active_ns` 各自 union。
- `inclusive_ns = sum(all dur_ns)`（不去重叠，可超 total_ns）。

### 6.2 BottleneckLabel
- `gpu_compute`：`pct_of_trace=compute/total`，`pct_of_gpu_active=compute/gpu_active`，confidence=0.9。
- `gpu_comm`：同上，confidence=0.9。
- `gpu_memcpy`：confidence=0.85。
- `gpu_idle`：`pct_of_trace=idle/total`，`pct_of_gpu_active=None`，confidence=0.95。
- `sync_wait`：`pct_of_trace=sync/total`，confidence=0.85。
- 按 `pct_of_trace` 降序。

### 6.3 Top Events（`_compute_top_events`）
合并所有非 cpu_sample 事件，按名字聚合，返回 Top-20 `EventStat`（count, total_ns, max_ns, avg_ns, `inclusive_pct = total_ns / trace.duration_ns`）。

### 6.4 Per-device（`_compute_per_device`）
按 `device_id` 分组（None→0）；只有多设备时生成 `DeviceBreakdown`（active_ns, idle_ns, pct_active, top_event_names[:3]）。

### 6.5 Finding 生成器

**`_finding_gpu_idle`**：`idle_pct >= 0.05` 触发；`>= 0.40` critical，否则 warning。`find_gaps` 找 Top-5 空闲段，格式化为 `"空闲段 A–Bms（Cms）"` 填 evidence。

**`_finding_gpu_compute`**：`pct = compute/gpu_active`；`>= 0.60` critical，`>= 0.30`（隐含）warning（实际阈值在代码中为 0 < pct 触发，severity 按百分比判断）。Top-5 内核名填 evidence + `related_hotspots`。

**`_finding_gpu_comm`**：`pct_of_active >= 0.15` 或 `pct_of_trace >= 0.10` 才触发。同阈值判 critical/warning。evidence 含通信时长、占比、Top-5 通信内核名。

**`_finding_comm_overlap`**：
- `overlap_ns = intersect_total(comm_intervals, compute_intervals)`。
- `overlap_pct = overlap_ns / comm_active_ns`；`>= 0.80` → info（良好）；`< 0.20` → critical；其余 warning。
- evidence：NCCL 活跃时长、重叠时长/百分比、暴露时长。

**`_finding_gpu_memcpy`**：`pct_of_trace >= 0.05` 触发。HtoD/DtoH 次数 + 总字节 + 近似带宽（bytes / dur_ns * 1e9 GB/s）填 evidence。

**`_finding_sync_wait`**：`total_sync_ns = CUPTI_sync_ns + cuda_api_sync_ns`（`_SYNC_API_NAMES` 集合过滤 cuda_api 事件）。`pct >= 0.03` 或有同步 API 调用时触发；Top-5 API 按耗时汇总填 evidence。

**`_finding_host_launch`**：`avg_kernel_ns < 100µs AND overhead_pct >= 0.05` 才触发。通过 `correlationId` 关联 API launch 与 kernel 计算 `avg_launch_ns`。evidence 含内核总数、平均内核时长、平均启动延迟、总启动开销。

**`_finding_cpu_hotspot`**：Counter 统计叶帧，过滤 `_SKIP`（"cpu_sample", "unknown", "", "__GI___libc_start_main", "start_thread", "clone", "futex_wait"）。Top 符号占比 `>= 0.05` 触发；`>= 0.20` → warning，否则 info。evidence：Top-5 符号及采样次数/百分比。

**`_finding_tiny_kernels`**：内核总数 `>= 1000` 且平均时长 `< 10µs` 才触发。统计短于 10µs 内核比例；Top-5 短内核名（Counter most_common）填 evidence。

全部 findings 按 `(severity_rank={critical:0,warning:1,info:2}, -confidence)` 排序。

最后调用 `run_deep_sql_analysis(findings, trace)`（仅有 `sqlite_path` 时，异常非致命）。

---

## 七、T4 深度 SQL 分析（`classify_sql.py`）

### 7.1 `_sql_top_kernels`
- `COALESCE(demangledName, shortName)` JOIN StringIds，`GROUP BY kernel_name ORDER BY total_ms DESC LIMIT 15`。
- 统计：invocations, total_ms, avg_ms, min_ms, max_ms。
- `_fmt_table` 生成等宽对齐表格存入 `finding.evidence`（render.py 识别 divider 行后原样缩进输出）。
- category=`sql_top_kernels`，severity=info，confidence=0.95。

### 7.2 `_sql_gpu_idle_gaps`
- **聚合**：`LAG(end) OVER (PARTITION BY deviceId, streamId ORDER BY start)` 计算相邻内核间隙，WHERE `gap > min_gap_ns`（1ms），汇总 gap_count / total_idle_ms，分三档（< 5ms / 5-50ms / > 50ms）。
- **Top 间隙列表**：同 LAG CTE，保留 before_kernel / after_kernel（JOIN StringIds），`ORDER BY gap_ns DESC LIMIT 15`。
- **CPU API 归因**（Top-5 间隙）：`r.start < ge AND r.end > gs` 查 runtime API，JOIN StringIds，Top-3 按耗时汇总，格式 `[CPU: apiName Xms]` 追加 evidence。
- **NVTX 重叠标注**（Top-5 间隙）：NVTX_EVENTS `end > gs AND start < ge`，处理 textId 变体，DISTINCT label 最多 4 个，格式 `[NVTX: l1, l2]` 追加 evidence。
- severity：total_idle_ms > 50ms 或 gaps_large > 0 → warning；否则 info；confidence=0.90。

### 7.3 `_sql_memory_bandwidth`
- 按 `copyKind` GROUP BY，统计 op_count, total_mb, avg_kb, total_dur_ms, `avg_bw_gbps = SUM(bytes)/(SUM(dur_ns)/1e9)/1e9`，`peak_bw_gbps = MAX(bytes/(dur_ns/1e9)/1e9)`。
- H2D 均带宽 < 5GB/s → warning（提示 pin memory）；D2D < 50GB/s → warning（提示跨 NUMA）。
- severity：total_dur_ms > 100ms → warning，否则 info。

### 7.4 `_sql_nccl_breakdown`
- OR-连接 LIKE 条件过滤 NCCL 内核（`nccl/allreduce/allgather/reducescatter/broadcast/sendrecv/reduce`）。
- 按 `streamId` GROUP BY，统计 op_count, total_ms, avg_ms, max_ms, min_ms，Top-20 stream。
- 若多 stream 且最大/最小差 > 30% → 追加 `⚠️ Stream 负载不均衡` 行。
- category=`sql_nccl_breakdown`，severity=info，confidence=0.90。

### 7.5 `_sql_sync_cost`
- 若有 `ENUM_CUPTI_SYNC_TYPE` 表：LEFT JOIN 解析类型名；否则 `CAST(syncType AS TEXT)`。
- 按类型 GROUP BY，统计 call_count, total_ms, avg_ms, max_ms。
- `sync_pct = total_sync_ms / trace_ms * 100`；`> 10%` → warning，否则 info；confidence=0.90。

### 7.6 `_sql_nvtx_hotspots`
- textId 变体处理（JOIN StringIds + COALESCE 或直接 text）。
- 过滤 `label IS NOT NULL AND label != '' AND end > start`。
- 按 label GROUP BY，统计 call_count, total_ms, avg_ms, max_ms + 占 trace 百分比列，Top-15。
- category=`sql_nvtx_hotspots`，severity=info，confidence=0.85。
- **注**：render.py 中 `sql_nvtx_hotspots` **不**在 `_SQL_CATEGORIES` 辅助证据集合，作为 Host/CPU 域主要 finding 展示（有 Qn 编号）。

---

## 八、T6–T7：Repository Mapping（`nsys/__init__.py`）

### 8.1 入口（`_map_hotspots_to_repo`）
1. `derive_repo_scope(trace, repo_root)` 推导扫描范围（见 8.2）。
2. 覆盖 scope 的 `mode / max_files / max_file_bytes / include_runfiles`（来自 request）。
3. `scan_repo(repo_root, scope)` → `dict[str, FileFacts]`。
4. `build_repo_index(files)` → 符号查找索引。
5. **CPU sample 热点路径**：
   - 过滤 `ev.is_sample=True`，按 `ev.name` Counter，Top-N（默认 20）。
   - `lookup_by_symbol(symbol, index)` 查匹配；`>= threshold`（0.4）：取 best match，调 `callers_of / callees_of`，生成 `MappedHotspot(match_reason="symbol")`；否则 `match_confidence=0.0, match_reason="none"`。
6. **EvidenceLink 路径**：对每个 finding 的 `related_hotspots[:3]`，`lookup_by_symbol`，命中则 `EvidenceLink(link_type="symbol", confidence=m.confidence * finding.confidence)`。
7. **NVTX 路径（补充）**：`_map_nvtx_regions_to_repo(trace, files)` 结果追加到 hotspots。

### 8.2 `derive_repo_scope`
遍历所有 trace events：
- `category in {"gpu_compute", "gpu_comm"}` → name 加 `seed_kernels`。
- `category="cuda_api"` 且 `extra["source_file"]` 非空 → 加 `seed_files`。
- `category="nvtx"` 且 name 非空且不以 `"NCCL"` 大写开头 → 加 `seed_symbols`。

返回 `RepoScope(mode="targeted", seed_files, seed_symbols, seed_kernels, include_extensions={.py,.cpp,.cc,.cxx,.c,.h,.hpp,.cu,.cuh}, follow_imports_depth=1, max_files=500, max_file_bytes=512_000, include_gpu_related=True)`。

### 8.3 `_map_nvtx_regions_to_repo`
**目的**：在无 CPU backtrace 时，利用 NVTX region 名反查源码的 `nvtx.range_push` 调用点，补充代码定位。

1. 过滤 `category="nvtx"` 且 `name` 非空的事件。
2. Counter 统计 region 名；`_is_nvtx_noise(name)` 过滤噪音（见 8.4）。
3. `build_callsite_index(files)` 建立 `call_name → [CallSiteFacts]` 索引。
4. 收集 NVTX 调用点：`call_name in _NVTX_CALL_NAMES`（`range_push`, `range`, `mark`, `torch.cuda.nvtx.range_push`, `nvtx.range_push`, `range_start`）或以 `range_push`/`range` 结尾，汇总为 `nvtx_callsites`。
5. 对 Counter.most_common(20) 中每个 region_name：
   - 遍历 `nvtx_callsites`，拼接 `" ".join(cs.args_repr) + " " + (cs.source_line or "")`；`region_lower in text.lower()` 或 `region_name in text` 命中 → 取该 callsite（break）。
   - 无命中 → continue。
   - `(path, line, region_name)` 三元组去重（set seen）。
6. 构造 `MappedHotspot`：
   - `sample.frame = SourceFrame(symbol=region_name, source_file=matched_cs.path, source_line=matched_cs.line)`
   - `sample.pct = count / total_nvtx_events`
   - `repo_file=matched_cs.path`，`function=matched_cs.enclosing_function`
   - `match_confidence=0.60`，`match_reason="nvtx_region"`
   - `callers=[]`, `callees=[]`, `alternatives=[]`

### 8.4 `_is_nvtx_noise`
判断 NVTX region 名是否为框架/系统自动注入（无对应用户 callsite）：

1. `not name or len(name) < 2` → True。
2. `name in _NVTX_NOISE` → True。黑名单：`Holding GIL`, `Waiting for GIL`, `GIL`, `PyEval`, `cudaLaunchKernel`, `cudaMemcpy`, `cudaStreamSynchronize`, `ncclAllReduce`, `ncclBroadcast`, `ncclReduceScatter`, `cublasGemmEx`, `cublasGemmStridedBatchedEx`。
3. 正则 `^(iter|step|epoch|batch)_?\d+$`（IGNORECASE）→ True（如 `iter_42`，框架按迭代计数自动注入，无静态 `range_push("iter_42")` callsite）。
4. 正则 `^0x[0-9a-fA-F]+$` → True（十六进制地址，无法映射到源码）。
5. 否则 → False（用户主动打标，保留）。

---

## 九、T8：TaskDraft 生成（`__init__.py:_generate_task_drafts`）

1. 预先收集 nvtx_events 和 gpu_compute_events 列表（循环外一次性过滤）。
2. 遍历 findings，`seen_categories` set 去重，只处理 `_TASK_DRAFT_TEMPLATES` 中有的 category。
3. `_extract_pct_from_finding`：正则 `(\d+(?:\.\d+)?)\s*%` 扫 evidence 字符串，取第一个匹配的浮点数填 hypothesis 模板（无匹配返回 0.0）。
4. **`evidence_windows`**（`_build_evidence_windows`）：
   - `sync_wait`：取 `category="sync_wait" and dur_ns>0`，`dur_ns` 降序 Top-5，各自 `_make_window`。
   - `gpu_memcpy_hotspot/h2d/d2h`：取 `category="gpu_memcpy" and dur_ns>0`，Top-5。
   - `gpu_idle`（`_build_idle_windows`）：`find_gaps(gpu_compute_intervals, trace_start, trace_end, 1ms)` 找间隙，按时长降序取 Top-N，手动查 before/after kernel 和 NVTX 重叠，生成 `event="gpu_idle_gap"` 的 dict。
   - `gpu_comm_hotspot`：取 `category="gpu_comm" and dur_ns>0`，Top-5。
   - **`_make_window(ev, nvtx_events, gpu_compute_events)`**：
     - `before_kernel = _find_kernel_before(ev.start_ns, gpu_compute_events)`：遍历所有 gpu_compute，找 `ev_end <= ts_ns` 且 `ev_end` 最大的内核名。
     - `after_kernel = _find_kernel_after(ev_end, gpu_compute_events)`：找 `start_ns >= ts_ns` 且 `start_ns` 最小的内核名。
     - `overlap_nvtx = _overlapping_nvtx_labels(start_ns, end_ns, nvtx_events, limit=5)`：NVTX events 中 `ev_end > start_ns AND ev.start_ns < end_ns`（时间重叠），收集不重复 label，最多 5 个。
     - dict 字段：`start_ms, end_ms, duration_ms, device, stream, event, before_kernel, after_kernel, overlap_nvtx`。
5. **`search_specs`**：直接从 `_SEARCH_SPECS` 字典按 category 取预置列表（每条含 `pattern`, `kind="rg"`, `rationale`）。
6. **`candidate_callsites`**（`_find_task_candidate_callsites`）：
   - `derive_analysis_scope(category, repo_files)` → scope。
   - `search_calls(callsite_index, scope, limit=limit*4)` 取候选（limit=5 → 搜 20）。
   - `_is_task_candidate_noise(category, cs.source_line)`：`gpu_memcpy*` 类别下 source_line 含 `.to(tl.`（Triton dtype cast）→ 排除。
   - 取前 5 个候选的 `cs.id`（`"<path>:<line>:<col>:<call_name>"`）。

---

## 十、Callsite Index（`analyzer.py` + `scanners/`）

### 10.1 `CallSiteFacts`（`scanners/base.py`）
```python
@dataclass
class CallSiteFacts:
    id: str                 # "<path>:<line>:<col>:<call_name>"
    path: str
    line: int
    col: int | None
    end_line: int | None
    end_col: int | None
    call_name: str          # "to", "cuda", "copy_", "pin_memory"
    full_call_name: str | None  # "batch.to", "torch.cuda.synchronize"
    receiver: str | None    # "batch", "x", "self.data"
    args_repr: list[str]    # 位置参数的源码字符串
    keywords: dict[str, str]  # {"device": "cuda", "non_blocking": "False"}
    enclosing_function: str | None
    loop_depth: int         # 0 = 不在循环内
    source_line: str        # 原始源码行
```

### 10.2 `PythonScanner._collect_callsites`（`scanners/python.py`）
手动 DFS 追踪 loop nesting：
- `For`, `AsyncFor`, `While` → body 以 `depth+1` 递归；else/orelse 不加深度。
- `If`, `With`, `AsyncWith`, `Try`, `TryStar`(Python 3.11+) → body/handlers 以当前 `depth` 递归。
- 其余语句：`ast.walk(stmt)` 收集 `ast.Call` 节点：
  - `ast.Attribute`：`call_name=func.attr`，`receiver=_node_repr(func.value)`，`full_call_name=f"{receiver}.{call_name}"`。
  - `ast.Name`：`call_name=func.id`，`receiver=None`。
  - 其余（lambda、subscript 等）：跳过。
- `args_repr = [_node_repr(a) for a in node.args]`。
- `keywords = {kw.arg: _node_repr(kw.value) for kw in node.keywords if kw.arg is not None}`（排除 `**kwargs` 展开）。
- `source_line = src_lines[line-1].rstrip()`。
- `id = f"{rel_path}:{line}:{col}:{call_name}"`。
- 外层 `try: ... except RecursionError: pass`（防极深嵌套崩溃）。

### 10.3 `build_callsite_index`（`analyzer.py`）
遍历所有 `FileFacts.callsites`，按 `cs.call_name` 建倒排：`defaultdict(list)` → `dict[str, list[CallSiteFacts]]`。

### 10.4 `derive_analysis_scope`（`analyzer.py`）
- 从 `_SCOPE_PATTERNS` 取 finding_type 对应的 `{file_keywords, func_keywords, call_names, call_keywords}`（无匹配则用 `_DEFAULT_SCOPE_PATTERN = {file_keywords: {"train","trainer"}, ...}`）。
- 遍历 `language="python"` 的文件，路径按 `/`, `_`, `.` 拆词（小写），与 `file_kw` 做集合交集，命中加入 `selected`。
- 过滤 `.runfiles` 路径分段（防止打包路径误匹配）。
- 无命中 → fallback：全部 Python 文件（最多 `max_files=50`）。
- 返回 `AnalysisScope(finding_type, priority_file_patterns, priority_func_patterns, call_names, call_keywords, seed_files, selected_files, excluded_patterns=[], reason, max_files, max_candidates=100)`。

### 10.5 `search_calls`（`analyzer.py`）
- `search_names = names or scope.call_names`；`search_kw = set(keywords or scope.call_keywords)`。
- 对每个 `name in search_names`，从 index 取 `[CallSiteFacts]`，过滤 `cs.path in scope_set`（scope 有文件限制时）。
- **评分**（`CallSiteCandidate.score`）：
  - `+3`：文件在 `scope.selected_files`。
  - `+2`：`cs.loop_depth >= 1`（循环内调用，更可能是热路径）。
  - `+1 × count(kw)`：`cs.keywords` 中含 `search_kw` 的键数量。
  - 额外 bonus：`cs.enclosing_function`（小写）含 `_TRAIN_ITER_FRAGMENTS` 任意片段（`train_one_iter`, `train_iter_impl`, `iter_impl`, `train_iter`, `train_step`, `train_batch`, `forward_backward`, `step`）→ 语义加成（训练迭代内的调用优先）。
- 按 score 降序，去重同 `cs.id`，取 Top-limit。

---

## 十一、Render（`render.py`）

### 11.1 Section 顺序
`render_nsys_terminal(diag, verbose)` 依次：
1. `_render_header`：Profile 路径、SQLite、状态（✅/❌/⚠️）、生成时间。
2. `_render_capture_quality`：逐项 ✅/⬜（GPU 内核 / memcpy / NCCL / NVTX 标注 / CPU 采样 / 多设备 / 代码仓库映射）。
3. `_render_summary`：Trace 时长 + "任意 GPU 活跃/空闲" + `_bar(pct)` ASCII 进度条；多设备附 平均/最差/最好/偏差（pp）。
4. `_render_investigation_queue`：`_build_investigation_queue` 按 `_DOMAIN_PRIORITY` 排序，只列有 finding 的 domain，格式 `N. domain_label` + 发现摘要 + 排查提示（**数字序号，无 medal emoji**）。
5. `_render_bottlenecks`：`BottleneckLabel` 对齐表格（Union % / Inclusive % / Union 时长 / GPU 活跃 % / 置信度）；Inclusive > 100% 加 `⚠️`；附 Top-10 事件表格。
6. `_render_findings`：按 `_DOMAIN_ORDER` 分组：
   - 主要 finding 用 `Qn.` + `_SEV_ICON`（🔴/🟡/🔵）+ `[严重/警告/提示]`；evidence 最多 3 条（verbose 时 5 条）。
   - SQL 类 finding（`_SQL_CATEGORIES` = `{sql_gpu_idle_gaps, sql_memory_bandwidth, sql_sync_cost, sql_nccl_breakdown, sql_top_kernels}`）归 "辅助证据" 块，不单独 Qn 编号。
   - `_render_evidence_block`：判断 `evidence[1]` 含 `─` 则视为表格格式（`_fmt_table` 输出），保留表头 + 分隔线 + 最多 limit 条数据；否则每行加 `· ` 前缀。
7. `_render_code_localization`：
   - 无 hotspots → 给出 `--no-repo` 提示 + 各 task 的 rg 搜索入口。
   - 有 hotspots 时，分两个子表：
     - **CPU 采样热点**：`match_reason != "nvtx_region" AND match_confidence > 0 AND repo_file 非空`，列 文件/函数/行号/置信度。
     - **NVTX region 调用点**：`match_reason == "nvtx_region"`（同样要求 confidence > 0 且 repo_file 非空），列 文件/函数/行号/NVTX region 名。
   - 两子表均空 → 提示配置 `--backtrace cpu`。
8. `_render_task_drafts`：每个 TaskDraft 显示 hypothesis / verification_metric / id / inferred_by；候选调用点（去 `.runfiles/` 前缀）；搜索入口（`rg -n "..." <repo>`，标注"需人工核实，非定论"）；Top-3 证据窗口（含前序/后序内核、NVTX 重叠）。
9. `_render_warnings`：Union % vs Inclusive % 定义 + repo_warnings（`💡` 前缀）。

### 11.2 `format_table`（`text.py`）
- `display_width`：用 `unicodedata.east_asian_width` 计算终端宽度（中文/全角 = 2 格），跳过 combining 字符（Cc/Cf）。
- `clip_display`：按终端宽度（非字节）截断。
- `pad_display`：用终端宽度补空格，保证列对齐。
- 输出格式：`[header_line, divider_line(─), data_line, ...]`；divider 行的 `─` 是 render.py 识别"表格 evidence"的标志。

### 11.3 警告图标统一
全报告中警告类信息统一使用 `⚠️`（含 variation selector），不使用 `⚠`。

---

## 十二、关键数据结构（`models.py`）

### `TimelineEvent`
```python
category: str       # "gpu_compute"|"gpu_comm"|"gpu_memcpy"|"cuda_api"|"nvtx"|
                    # "cpu_sample"|"sync_wait"|"osrt"
name: str
start_ns: int
dur_ns: int         # cpu_sample 时为 0
is_sample: bool     # True → 不参与 interval-union
device_id: int | None
stream_id: int | None
correlation_id: int | None  # CUDA CPU↔GPU 关联键
global_tid: int | None
extra: dict         # cpu_sample: {"callstack":[...]};
                    # memcpy: {"size_bytes":..., "copy_kind":...}
```

### `BottleneckLabel`
两个百分比字段明确区分分母：
- `pct_of_trace`（分母=wall time）：用于 gpu_idle / sync_wait。
- `pct_of_gpu_active`（分母=gpu_active_ns，可为 None）：用于 gpu_compute / gpu_comm。

### `EvidenceLink`
```python
link_type: str      # "correlation_id"|"nvtx_to_api_to_kernel"|
                    # "cpu_callstack"|"file_line"|"symbol"
confidence: float   # 上限见第十三节
inferred_by: str    # "deterministic"|"llm_investigated"
```

### `TaskDraft`
```python
candidate_callsites: list[str]   # "<path>:<line>:<col>:<call_name>"
target_locations: list[dict]     # Core 阶段为 []
evidence_windows: list[dict]     # {"start_ms","end_ms","duration_ms","device",
                                 #  "stream","event","before_kernel",
                                 #  "after_kernel","overlap_nvtx"}
search_specs: list[dict]         # {"pattern":str,"kind":"rg","rationale":str}
```

### `OptimizeTask`
```python
# LLM investigator 将 TaskDraft 升级为 OptimizeTask；optimizer 消费 OptimizeTask
finding_id: str                             # NsysFinding.stable_id（基于 content hash，非 index）
target_locations: list[TargetLocation]      # 强类型；非空才允许 optimizer 执行
rejected_candidates: list[RejectedCandidate]  # LLM 必须填；不得静默忽略候选
confidence_breakdown: ConfidenceBreakdown | None  # 强类型；.composite() 计算最终置信度
inferred_by: str = "llm_investigated"
```

### `TargetLocation` / `RejectedCandidate` / `ConfidenceBreakdown`
- **`TargetLocation`**：每条必须至少有一个 anchor（`callsite_id` / `nvtx_region` / `correlation_id` / `cpu_sample_pct`），LLM 不得纯猜位置
- **`RejectedCandidate`**：标注"为什么排除"，确保审计链完整
- **`ConfidenceBreakdown.composite()`**：deterministic_finding 是 ceiling anchor；LLM 代码语义最多加 +0.15

---

## 十三、EvidenceLink 置信度上限

| link_type | 上限 | 原因 |
|-----------|------|------|
| `correlation_id` | 0.95 | CUDA API 与 kernel 同 correlationId，数学确定 |
| `nvtx_to_api_to_kernel` | 0.85 | NVTX 时间窗口包含 API launch + correlationId 到 kernel |
| `cpu_callstack` | 0.80 | CPU sample 调用栈，采样有误差 |
| `file_line` | 0.85 | SourceFrame 有文件名+行号，lookup 命中 |
| `symbol` | 0.70 | 仅符号名匹配 |
| `nvtx_region` | 0.60 | region 名字符串匹配，不如 CPU backtrace 直接 |
| `llm_investigated`（EvidenceLink）| 0.50 | 纯 LLM 语义判断，无 callsite 事实 |
| `OptimizeTask.confidence` | 不限 | 综合 deterministic + callsite + LLM 验证 |

> 硬性规则：
> 1. 同一 event 的多条 EvidenceLink，只输出最高优先级一条，其余进 `alternatives`。
> 2. `inferred_by="llm_investigated"` 的 EvidenceLink 不得用于时间计算。
> 3. 新增 link_type 时，上限不得超过其证据的确定性。

---

## 十四、测试状态

| 文件 | 场景 | 状态 |
|------|------|------|
| `test_analyzer.py` | 所有 baseline 测试 | ✅ 通过（83 tests） |
| `test_callsite.py` | CallSiteFacts / search_calls / get_callsite_context / NVTX region 映射 / 噪音过滤 | ✅ 通过 |
| `test_nsys_correctness.py` | per-device / EvidenceLink / cpu_callstack 全栈帧 | ✅ 通过 |
| `test_nsys_render.py` | 表格对齐 / TaskDraft rg 命令渲染 / 证据窗口渲染 | ✅ 通过 |
| `test_task_draft.py` | TaskDraft 生成 / evidence_windows / search_specs / OptimizeTask 新字段 | ✅ 通过 |

---

## 十五、架构约束备忘（不做的事）

| 能力 | 原因 |
|------|------|
| LLM 参与时间/overlap 计算 | 不确定性太高，会污染 findings |
| analyzer 生成代码 diff | 属于 optimizer 职责；输出 OptimizeTask，不是 patch |
| 全量事件无上限加载内存 | 大 profile 用 SQL 聚合，内存只保留 top events |
| full repo scan 为默认 | finding-aware scope |
| shell 调用（c++filt 等） | 违反 security；demangle 用纯 Python |
| search_symbols("to") 作为 callsite 搜索 | symbol search 搜不到 `.to()` 等调用点 |
| LLM 直接接收全文件 | 只传 callsite context（函数 + callers + 上下文行） |
| EvidenceLink 里写 next_step | finding 写 next_step，evidence_link 只写证据 |
| TaskDraft 猜代码位置 | Core 不猜；LLM investigator 用工具确认后才填 target_locations |
| search_specs 作为定论 | 明确标注"非定论"；LLM investigator 必须核实 |

---

*最后更新：2026-04-17*
