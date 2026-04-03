# Sysight

[English](./README.md)

面向 Nsight Systems profile 分析的轻量级、Torch-first 后端。

`Sysight` 将 Nsight Systems profile 转换成结构化结论、代码定位线索和下一步优化建议，重点服务于 PyTorch 训练与推理场景。项目保持轻量：唯一的输入只需要 `.sqlite` / `.nsys-rep` 文件，不需要源代码、项目工作区或编译环境。

正常使用方式是直接对 agent 说一句话，把 profile 文件路径给它，后面的事情 agent 自动完成。CLI 是 agent 底层调用的执行后端，不是主要的用户交互界面。

## 功能

- 面向训练和推理场景的 Torch-first 分析
- 基于 SQLite 的分析流水线，强调低上下文和 skill 化拆分
- 面向 agent 的后端，重点输出简洁结论、代码定位线索和可执行的下一步建议
- 内置 kernel hotspot、iteration timing、idle gaps、launch overhead、memory transfers、NCCL anomalies、计算/通信 overlap、NVTX attribution、代码定位、MFU、theoretical FLOPs 等分析能力
- 代码定位线索：从可疑时间窗口出发，经 CPU 线程 → 采样调用栈 → Python 文件/函数逐层缩小范围
- 输出 Markdown 和 JSON，方便交给后续调查步骤

## 快速开始

1. 把 Nsight Systems 的 `.sqlite` / `.sqlite3` / `.nsys-rep` 文件放到任意可访问路径下。

2. 直接对 agent 说：

```text
帮我分析一下 profiles/pnc_prof_0330_prediction_fakedata.sqlite
```

或者给绝对路径：

```text
帮我分析一下 /abs/path/to/profile.sqlite
```

或者直接给 Nsight Systems 采集结果：

```text
帮我分析一下 /abs/path/to/profile.nsys-rep
```

3. agent 自动调用分析后端，在终端返回三段式简要报告：

- `结论`
- `问题`
- `下一步行动建议`

4. 当前支持的输入格式：

- `*.sqlite`
- `*.sqlite3`
- `*.nsys-rep`

5. 如果输入是 `*.nsys-rep`，Sysight 会先尝试自动导出同名的 `*.sqlite`：

```bash
nsys export --type=sqlite -o <same-stem>.sqlite --force-overwrite=true <file.nsys-rep>
```

这一步要求本机 `PATH` 上能找到 `nsys`。如果本机没有 `nsys`，先手动导出，或使用辅助脚本：

```bash
scripts/export-nsys-sqlite.sh path/to/profile.nsys-rep
```

6. 分析产物写入 `outputs/`：

```text
outputs/<profile>.report.md      # 完整 Markdown 报告
outputs/<profile>.findings.json  # 机器可读的结论 JSON
```

## 分析输出说明

报告分为五个部分：

| 部分 | 内容 |
|---|---|
| 执行摘要 | GPU 整体利用率、主要瓶颈、NCCL 占比 |
| Kernel 热点 | 按计算耗时排序的 top kernel，名称已截断便于阅读 |
| 问题窗口 | 可疑时间窗口，含 CPU 线程、NVTX 上下文和采样调用栈 |
| 行动建议 | 带优先级、定位到文件/函数的下一步建议 |
| NVTX 热点候选 | 候选 NVTX region 及其时间分解 |

问题窗口中的代码定位以 Python 文件 + 函数名为粒度，来源于 profile 中的 CPU 采样调用栈。

## CLI 参考

在仓库根目录手动运行，适用于调试或脚本化调用：

```bash
# 查看帮助
PYTHONPATH=src python3 -m sysight --help

# 分析 profile，写入报告和结论
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite

# 显式指定输出路径
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite \
    --markdown outputs/my.report.md \
    --findings outputs/my.findings.json

# 列出可用分析 skill
PYTHONPATH=src python3 -m sysight skill list

# 导出 .nsys-rep 为 .sqlite（需要 PATH 上有 nsys）
scripts/export-nsys-sqlite.sh path/to/profile.nsys-rep

# 通过 shell 封装脚本运行分析（自动处理路径解析和输出命名）
scripts/run-profile-analysis.sh path/to/profile.sqlite
scripts/run-profile-analysis.sh path/to/profile.sqlite <gpu_id>
```

## 仓库结构

```text
sysight/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── worklog.md
├── src/sysight/                 # 包根目录
│   ├── analysis/                # 分析核心
│   │   ├── skills/              # 各分析 skill 模块
│   │   ├── code_location.py     # 调用栈 → 代码位置推断
│   │   ├── report.py            # 报告格式化
│   │   └── summary.py           # 自动摘要生成
│   ├── cli.py                   # CLI 入口
│   └── profile.py               # Profile 加载器
├── scripts/                     # Shell 辅助脚本
│   ├── export-nsys-sqlite.sh    # nsys 导出封装
│   └── run-profile-analysis.sh  # 分析运行器
├── skills/                      # Agent skill 定义
├── profiles/                    # 本地 profile 样例
├── tests/                       # 单元测试
└── outputs/                     # 生成的分析产物
```

## 路线图

- [ ] 强化 NVTX 约定和 attribution 质量
- [ ] 深化代码定位：在 debug symbol 可用时解析到行号
- [ ] 支持优化前后 profile 对比，做效果验证
- [ ] 增加围绕 `ncu` 和 targeted micro-benchmark 的 investigation workflow
- [ ] 补齐测试和工程化工作

## 致谢

- [nsys-ai](https://github.com/GindaChen/nsys-ai) 提供了 profile 读取、skill 化分析和 agent-oriented evidence 组织方面的重要上游思路
- NVIDIA Nsight Systems 提供 timeline、CUDA、NCCL 和 NVTX profiling 数据
- NVIDIA Nsight Compute 用于更细粒度的 kernel 级分析
- PyTorch 是这个项目目前主要面向的执行环境

## 许可证

[MIT](./LICENSE)
