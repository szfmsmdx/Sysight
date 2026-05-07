# analyzer

## D：引导 LLM 先 SQL 后按需读代码

### 问题描述
日志显示 LLM 当前的实际行为是：Turn 2 读 memory（overview + experience），Turn 3-10 连续 `scanner_read` 24 个代码文件（几乎整棵代码树），Turn 11 才开始第一次 SQL 查询。这使得 context 在真正的 profile 分析开始前就已积累了大量无关代码。

### 根因
`analyze_system.md` 的 SOP 步骤 3 是"读取热路径验证假设"（`scanner_read` 核心循环），但没有说明**在读代码之前应先用 SQL 工具定向**。LLM 自然地把"验证假设"理解为"把所有热路径代码都读一遍"。

### 方案
在 SOP 步骤 1（形成假设）之后、步骤 3（读代码）之前，插入明确的 SQL 优先步骤：

**在 `analyze_system.md` 的"## 1. 形成假设"里补充**：形成初始假设后，**优先使用 SQL 工具（`nsys_sql_sync`、`nsys_sql_memcpy`、`nsys_sql_nvtx`、`nsys_sql_gaps`、`nsys_sql_nccl`）精确量化各假设的 profile 证据**，再根据 SQL 结果判断哪些假设值得进入代码验证。只有 SQL 证据指向的假设才需要 `scanner_read`；没有 profile 支撑的代码路径不应提前读取。

这样可以将 `scanner_read` 调用从 43 次降低到只读假设相关的 10-15 个文件，节约 30-40% 的 context 积累。

---

## F：积极驱逐已完成的代码 compaction

### 问题描述
当前 compaction 策略（level 2/3）只在 token 压力达到 90%/95% 阈值时被动触发，且压缩方式是对"旧的 tool results"做摘要。实际效果是：43 个文件的 `scanner_read` 结果大部分一直保留到最后一轮，在上下文中占用约 18,600 tokens（32.9%）。

### 方案
在 `AgentContext` 中增加一个**阶段感知驱逐**机制：当检测到 LLM 开始发起 SQL 查询（即 `nsys_sql_*` 工具首次被调用）时，视为"代码探索阶段结束"，主动将所有 `scanner_read` tool results 压缩为"文件摘要块"（保留文件名 + 行数 + 关键符号列表，丢弃全文内容）。

具体实现思路：
1. 在 `AgentContext.add_tool_result()` 中追踪当前"阶段"（代码探索 / SQL 分析 / 输出）
2. 当首个 `nsys_sql_*` 调用进入时，触发对所有现存 `scanner_read` results 的批量压缩
3. 压缩后的摘要块：`{"compressed": true, "files_read": [...], "note": "代码已读取，关键符号见 session_progress"}`
4. 这类摘要块不在后续 compaction 中被保护，可以被进一步淘汰