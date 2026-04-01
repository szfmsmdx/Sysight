# Sysight

[简体中文版](./README.zh-CN.md)

Lightweight, Torch-first agent backend for Nsight Systems profile analysis.

`Sysight` is a small local analysis stack for turning Nsight Systems profiles into structured findings, code-location hints, and next-step optimization guidance for PyTorch workloads.

The project is aimed at agent-driven workflows first: the long-term goal is not to ship another heavy profile viewer, but to provide a compact backend that an agent can call to profile, analyze, and iterate on performance issues.

The normal usage is conversational: put a profile file into the project, then ask the agent to analyze it. The CLI exists as the backend the agent calls under the hood.

## Features

- Torch-first analysis for training and inference profiles
- SQLite-based analysis pipeline with low-context, skill-oriented decomposition
- Upstream-inspired analysis coverage without carrying over the heavy web-first surface area
- Built-in skills for kernel hotspots, iteration timing, idle gaps, launch overhead, memory transfers, NCCL anomalies, overlap, NVTX attribution, code-location hints, MFU, and theoretical FLOPs
- Markdown and JSON outputs that are easy to hand off to a follow-up agent or investigation step
- Early multi-worker orchestration demo for running several analysis tasks in parallel

## Current Status

The repository is currently focused on the analysis backend: lightweight profile inspection, skill-oriented analysis, MFU/FLOPs helpers, and report generation on top of Nsight Systems data. The goal is to stay close to the useful analysis functionality in `nsys-ai` without inheriting its heavier web/TUI/chat surface by default. Automated profiling around user projects, a more natural agent-facing interaction loop, and deeper investigation workflows are still roadmap items.

## Quick Start

1. Put your Nsight Systems `.sqlite` / `.sqlite3` / `.nsys-rep` file into the project.

2. Tell the agent directly:

```text
帮我分析一下 test/basemodel_8gpu.sqlite
```

or:

```text
帮我分析一下 /abs/path/to/profile.sqlite
```

3. The agent will run the analysis backend automatically and return a short terminal report with three sections:

- `结论`
- `问题`
- `下一步行动建议`

4. At the same time, it will generate a full Markdown report under `outputs/`, for example:

```text
outputs/<profile>.report.md
```

## Agent Workflow

The intended product direction is agent-first.

In this repository, the main path is:

1. The user provides a profile path in natural language
2. The agent invokes the analysis backend automatically
3. The agent returns `结论 / 问题 / 下一步行动建议`
4. Sysight writes a complete `report.md` report for deeper follow-up

The CLI is therefore an internal execution surface, not the primary user interface.

## Backend Commands

When debugging the backend itself, run commands directly from the repository root:

```bash
PYTHONPATH=src python3 -m sysight --help
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite
PYTHONPATH=src python3 -m sysight skill list
```

## Repository Layout

```text
sysight/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── worklog.md
├── src/sysight/                 # Public package entrypoint
├── scripts/                     # Dev utilities and orchestration demos
├── skills/                      # Internal skill notes and workflow docs
├── test/                        # Local-only test inputs
├── outputs/                     # Generated analysis artifacts
└── workspace/                   # User project workspace
```

## Roadmap

- [ ] Add a profiling stage that can wrap real Torch entrypoints and produce standard Nsight Systems captures
- [ ] Improve project understanding for user codebases, including entrypoints, modules, and data paths
- [ ] Strengthen NVTX conventions and attribution quality
- [ ] Add investigation workflows built around `ncu`, targeted benchmarks, and controlled reproductions
- [ ] Support before/after profile comparison for optimization validation
- [ ] Make multi-agent orchestration more automatic and less demo-oriented
- [ ] Explore a safe loop for proposing optimizations, validating them, and feeding the result back into the agent
- [ ] Add tests and release hygiene for the local project workflow

## Acknowledgements

- [nsys-ai](https://github.com/GindaChen/nsys-ai) for the upstream ideas around profile reading, skill-based analysis, and agent-oriented evidence organization
- NVIDIA Nsight Systems for timeline, CUDA, NCCL, and NVTX profiling data
- NVIDIA Nsight Compute for kernel-level investigation
- PyTorch as the primary execution environment this project is built around

## License

[MIT](./LICENSE)
