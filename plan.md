# Sysight 开发计划

## 上下文压缩策略调研（2026-05-07）

### 调研背景

当前 Sysight 的上下文压缩方案为基于 turn 计数的二元压缩（full/compact），存在以下问题：
- 无 token 感知——不知道实际用了多少 token，也不知道何时该触发
- 二元切换过于粗暴——一旦超过 1 个 turn，完整代码行直接丢失
- 无恢复机制——压缩后模型无法重新获取丢失的细节
- 无渐进层级——没有"先轻量清理、再重度压缩"的梯度

调研目标：广泛调研 OpenAI Codex CLI、Anthropic Claude Code、DeepSeek-TUI 等主流 AI coding 工具的上下文压缩策略，为 Sysight 的压缩方案优化提供参考。

---

### 一、Sysight 当前方案

核心实现在 `sysight/agent/context.py`：

- **策略**：基于 turn 的二元压缩（full / compact），无渐进层级
- **触发条件**：`(request_turn - stored.turn) > keep_recent_turns_full`（默认 `keep_recent_turns_full=1`，即只保留最近 1 个 turn 的完整结果）
- **压缩方式**：按工具类型定制 summary（`scanner_read` 保留行内容、`scanner_files` 保留路径列表等），附带 SHA256 校验
- **问题**：
  - 无 token 感知——不知道实际用了多少 token，也不知道何时该触发
  - 二元切换过于粗暴——一旦超过 1 个 turn，完整代码行直接丢失
  - 无恢复机制——压缩后模型无法重新获取丢失的细节
  - 无渐进层级——没有"先轻量清理、再重度压缩"的梯度

---

### 二、Claude Code：5 级渐进压缩管线

来源：harrisonsec.com 对 Claude Code 源码的深度逆向分析（`src/services/compact/`，3960+ 行 TypeScript，5 个文件）

```
消息历史
  │
  ├─ Level 1: Tool Result Budget（零成本）
  │   50K 字符阈值 → 持久化到磁盘 + 2KB 预览
  │
  ├─ Level 2: History Snip（零成本）
  │   清理重复包装器、冗余簿记
  │
  ├─ Level 3: Microcompact（零 API 调用）
  │   双路径：冷缓存→直接修改消息 / 热缓存→cache_edits API
  │
  ├─ Level 4: Context Collapse（零成本，可逆）
  │   90% 利用率触发，类似数据库 View，原始消息不删除
  │
  └─ Level 5: Autocompact（一次 API 调用，不可逆）
       Fork 子 agent 做全量摘要，CoT scratchpad → 只保留 <summary>
```

#### 核心设计亮点

**Level 1 — Tool Result Budget**
- 单次工具输出超过 50K 字符时，不截断，而是持久化完整输出到磁盘，上下文只保留 2KB 预览
- 模型后续可通过 `Read` 工具重新读取完整文件
- 这解决了大文件读取后压缩导致代码行丢失的问题

**Level 3 — Microcompact 双路径设计**
- **冷缓存路径**：用户离开太久，缓存已过期 → 直接修改消息内容为 `[Old tool result content cleared]`
- **热缓存路径**：用户活跃，缓存温热 → 使用 `cache_edits` API 在服务端原地删除引用，不破坏缓存
- 只对白名单中的高容量工具做 microcompact，防止意外清除结构化状态

**Level 4 — Context Collapse（非破坏性）**
- 类似数据库 View——底层消息数组不变，API 请求看到的是过滤/摘要投影
- 可逆——可以回滚
- 激活时抑制 Autocompact，避免两者竞争

**Level 5 — Autocompact + 恢复机制**
- 两阶段 CoT：`<analysis>` 逐条分析 → `<summary>` 结构化摘要（9 个标准章节）
- 只保留 `<summary>`，丢弃推理过程
- **Post-Compaction Recovery**：自动恢复最近 5 个已读文件（≤5K tokens 每个）、所有已激活 skills、MCP 指令等
- **Circuit Breaker**：连续 3 次 autocompact 失败后停止，防止无限循环（修复前每天浪费 ~250K API 调用）

**可定制压缩 Prompt**
- 用户可通过 `settings.json` 的 `compactPrompt` 自定义压缩指令
- 推荐策略：CLAUDE.md 中的规则放系统 prompt（不受压缩影响），会话上下文可以丢失

**Token 估算**
- 使用 API 返回的 `usage` 数据作为锚点，只估算增量部分，误差 <5%（vs 纯客户端估算的 30%+）

---

### 三、OpenAI Codex CLI

来源：OpenAI 官方文档、GitHub Issues、Reddit 社区讨论

- **服务端压缩**（Codex 模型）：通过 `compact_threshold` 和 `/responses/compact` 端点，服务端自动管理
- **本地压缩**（非 Codex 模型）：客户端用另一个 LLM 调用做摘要
- **v0.54-v0.55 关键改进**：逐字重放用户消息，只摘要中间轮次，防止递归的有损摘要（"ghost history" 问题）
- **扩展压缩 Prompt**（GitHub Issue #14347）：10 条规则的完整方案，包含 PreCompact/PostCompact hooks
- **v0.118 回归**：压缩静默丢弃所有工具输出和 assistant reasoning，导致 2x 更多问题
- 支持 `/compact` 手动触发

**关键教训**：递归摘要（对摘要再做摘要）会导致信息指数级衰减，Codex 的解决方案是用户消息逐字保留。

---

### 四、DeepSeek-TUI

来源：GitHub Issues（#541, #528）、Verdent AI 分析

- **"Cache-maximal" 哲学**：把压缩视为紧急回退机制，而非日常维护
- 当前行为：`token_threshold: 50000`，`message_threshold: 50`，不区分模型
- **v0.9.0 计划改进**：模型感知的阈值
  - V4 系列：提升到 ~500K tokens（V4 的 1M 上下文便宜，CSA+HCA 混合注意力使长上下文 FLOPs 仅为 V3.2 的 27%，KV cache 仅为 10%）
  - V3/旧模型：保持现有阈值
- `auto_compact = false` 默认关闭，手动 `/compact`
- 配套策略：重新读取活跃文件而非摘要（`working_set` 文件保持驻留）

**关键洞察**：不同模型的上下文经济学完全不同。对 V4 这种"长上下文便宜"的模型，过早压缩反而浪费了架构优势。

---

### 五、对比总结

| 维度 | Sysight（当前） | Claude Code | Codex CLI | DeepSeek-TUI |
|------|:------:|:------:|:------:|:------:|
| **压缩层级** | 1 层（二元） | 5 层渐进 | 2 层（服务端/本地） | 1 层 + 手动 |
| **触发机制** | turn 计数 | token 利用率% | token 阈值 | token 阈值 |
| **大文件处理** | 全量→压缩丢失 | 持久化磁盘+预览 | 服务端管理 | 重新读取 |
| **缓存感知** | ❌ | ✅ 双路径 | ❌ | ✅ V4 优化 |
| **可逆性** | ❌ | ✅ Level 4 | ❌ | ❌ |
| **恢复机制** | ❌ | ✅ 自动恢复 | ❌ | ✅ 重新读取 |
| **防递归衰减** | ❌ | ✅ CoT 分离 | ✅ 用户消息保留 | ❌ |
| **Circuit Breaker** | ❌ | ✅ 3次上限 | ❌ | ❌ |
| **可定制** | ❌ | ✅ compactPrompt | ✅ hooks | ❌ |

---

### 六、对 Sysight 的优化建议

#### 阶段 1（低成本，高收益）

1. **引入 token 感知触发**：不再用纯 turn 计数，改为基于实际 token 使用率（参考 Claude Code 的 anchor + delta 估计算法，误差 <5%）
2. **大文件持久化**：`scanner_read` 返回超过阈值（如 50K 字符）时，写完整内容到磁盘，上下文只保留 2KB 预览 + 文件路径，模型需要时可重新读取
3. **保留最近 N 个 turn 的完整结果**：将 `keep_recent_turns_full` 从 1 提升到 3-5，或改为 token 预算制（如保留最近 20K tokens 的完整工具结果）

#### 阶段 2（中期，结构改进）

4. **渐进压缩层级**：
   - Level 1：工具结果 >50K 字符 → 磁盘持久化 + 预览
   - Level 2：旧工具结果 → 替换为结构化摘要（保留文件名、行号范围、关键发现）
   - Level 3：上下文 >90% → 触发 autocompact（子 agent 摘要 + 恢复机制）
5. **防递归衰减**：压缩时用户消息逐字保留，只摘要中间的工具调用轮次（参考 Codex v0.55）
6. **Post-compaction 恢复**：压缩后自动将最近读取的关键文件内容重新注入上下文

#### 阶段 3（长期，架构优化）

7. **可定制压缩 prompt**：允许配置压缩时的保留规则
8. **Circuit breaker**：连续压缩失败后停止，防止无限循环
9. **模型感知阈值**：不同模型使用不同的压缩触发点
