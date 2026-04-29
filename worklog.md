## v0.5 — 2026-04-29

- **memory 组件重构**：`sysight/memory/` 迁移到 `sysight/analyzer/memory/`，memory 逻辑与 analyzer 包内聚；原顶层 `sysight/memory/experience.md`、`workspace.md` 删除，运行时落盘路径统一为 `.sysight/memory/`。
- **SKILL.txt 规则加固**：SOP 步骤 3 增加"active variant 类文件必须显式确认"约束，防止落到 decoy 文件；`for` 循环定界原则补充"循环行与循环体内独立问题行需各自输出 finding"；C7 排查规则补充"热路径函数需从入口逐行覆盖，不能只报末尾显眼操作"。
- **benchmark 得分**：case_4=69%、case_5=94%、case_6=100%，三 case 平均 88%。

## v0.4 — 2026-04-28

- **Analyzer 架构继续收口**：`sysight/analyzer/cli.py` 完成第一轮小拆分，`scanner` 子命令迁移到 `sysight/analyzer/scanner/scanner_cli.py`，`nsys-sql` 子命令迁移到 `sysight/analyzer/nsys/sql_cli.py`，主入口继续保持兼容。
- **SQL 深分析模块化**：`sysight/analyzer/nsys/classify_sql.py` 从单体文件继续按职责拆分为 `sql_compute.py`、`sql_memory.py`、`sql_comm.py`、`sql_sync.py`、`sql_root_cause.py`、`sql_profile.py`、`sql_nvtx.py`；`classify_sql.py` 收敛为 facade + orchestrator，外部导入面保持稳定。
- **NVTX / Profile / Root-Cause 保真**：保留 NVTX→Kernel attribution、GIL NVTX host anchor、profile health、root cause 反模式检测等高价值逻辑；修复 split 后 `_NCCL_KEYWORDS` 丢失导致的回归，并补齐无 `StringIds` 但有 `demangledName` 时的 kernel 命名归因。
- **Prompt 与文档对齐**：更新 `sysight/analyzer/SKILL.txt`，继续弱化盲扫式措辞，收敛到 Evidence-Driven Top-Down Trace；同步更新 `sysight/analyzer/README.md` 说明当前 analyzer 结构与定位职责。
- **测试瘦身与加固**：删除低价值 surface tests，合并重复 legacy 字段断言，清理无用 helper/import；补充 `test/test_nsys_sql_analysis.py` focused tests，覆盖 facade re-export、NCCL same-stream regression、NVTX GIL anchor 与 demangled-name attribution。
- **验证**：本轮聚焦 SQL 单测与完整 `test/` 单测通过。

## v0.3 — 2026-04-23

- **nsys-bench 测评框架**：新增 `nsys-bench/` 基准测试套件（6 个 case），自动化 JSON 评分，覆盖 C1-C7 全类别
- **Codex 调查协议优化**：
  - 扩大预注入 SQL 数据（kernels / gaps / kernel-launch），减少 Codex 自主调用 CLI 次数
  - `workspace_memory_mode` 字段：首次 overwrite，后续 append，修复重复写入 bug
  - `experience_memory` 追加模式，跨 run 积累通用分析经验
  - 禁止 Codex 直接写文件，由 Python 进程负责 memory 落盘
- **TASK.txt 重构**：调查原则 + 定位策略 + 输出规则合并为单 section；C1-C7 改为表格含判断边界；删除冗余任务 section；主动探索原则去除方向枚举
- **scanner callsite 修复**：`for` 循环迭代器中的函数调用（如 `range(n)`）现在被正确索引
- **文件瘦身**：删除 PRINCIPLES.md、根目录临时 sqlite 文件；更新 README / SKILL.md；analyzer/README.md 改为当前架构
- 测试：123 tests，全部通过

## v0.2 — 2026-04-17

- **Agent-centric 重构**：analyzer 只提供工具层，repo 级代码定位全权交给 Codex agent
  - 删除 `OptimizationTask` / `LocationBundle` / `optimizer_handoff`；`analyze_nsys` 输出精简为 `bottlenecks/findings/hotspots/windows/investigation`
  - Stage 6 仅返回 `questions/anchors`，Codex 负责最终定位
  - `sysight nsys` 默认输出 profile 统计；`--report full` 同步等待 Codex 结果
- **nsys-sql CLI**：`sysight nsys-sql <cmd> <db>` 直接查询 nsys SQLite，子命令：`schema / kernels / gaps / sync / nvtx / memcpy / nccl / overlap / stream-concurrency / kernel-launch`
- **scanner CLI**：`sysight scanner <cmd> <repo>` 静态代码分析，子命令：`manifest / index / search / lookup / callers / callees / impact / trace / callsites`
- **SQL 深分析**：NVTX→Kernel 精确归因（correlationId sort-merge）、root cause 6 种反模式检测、profile health manifest、NVTX layer breakdown
- **callstack 可读性**：`stacks.py` 启发式从采样栈提取人类可读粗定位，过滤 GIL/runtime 包装层
- **诊断协议强化**：`stable_finding_id`（SHA1 content hash）、`TargetLocation / ConfidenceBreakdown` dataclass、evidence_windows 扩充相关字段
- 测试：88 → 129 tests，全部通过

## v0.1 — 2026-04-16

- `analyzer.py`：仓库三阶段分析（Discovery → Targeted Scan → Context Expansion）
- `cli.py`：CLI 入口，全中文输出
- `nsys/`：T1-T5 诊断流水线（profile 解析、schema 探测、事件提取、瓶颈分类、代码热点映射）
- `scanners/`：Python AST-based + C++/CUDA 静态分析
- Bug 修复：`globalTid` 列兼容旧版 nsys schema
- 测试：首批 unit tests 通过
