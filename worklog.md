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
