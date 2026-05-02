# Evidence-Driven Investigation SOP

按以下顺序逐类调查，每类先形成假设再通过工具验证：

## 分类优先级

| 优先级 | 类别 | 关注点 |
|--------|------|--------|
| 1 | C1 Host Scheduling | GPU idle time, step-level gaps |
| 2 | C7 Python Pipeline | DataLoader, preprocessing bottlenecks |
| 3 | C2 Kernel Launch Overhead | API→GPU latency, many small kernels |
| 4 | C5 Compute Inefficiency | Kernel occupancy, precision, math efficiency |
| 5 | C3 Synchronization | cudaDeviceSynchronize, implicit sync |
| 6 | C4 Memory Copy | D2H/H2D bandwidth, pageable memory |
| 7 | C6 Communication | NCCL overhead, compute/comm overlap |

## 调查流程

1. 阅读预注入的 profile summary，识别 top bottlenecks
2. 对每个 bottleneck 形成假设：什么问题？在哪里？
3. 用 scanner 工具定位到源码 file/function/line
4. 回看 profile evidence 验证假设
5. 输出 LocalizedFinding（含 evidence_refs）
6. 顺手记录 repo 结构发现 → memory_updates
