# Optimizer SOP

将 verified findings 转化为可执行的 patch。

## 流程

1. 排序 findings：按 priority > confidence > category_weight
2. 阅读源码：对每个 finding，用 scanner.read 读取目标文件，理解上下文
3. 三重检查（生成 patch 前）：
   - 改动只影响目标代码路径？
   - 不会破坏现有测试？
   - 改动是 minimal 的？
4. 生成 PatchCandidate：
   - 精确的 old_span_start / old_span_end
   - 计算 old_span_hash
   - replacement 是完整、可直接替换的代码
