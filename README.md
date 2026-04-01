# Sysight

[简体中文版](./README.zh-CN.md)

Lightweight, Torch-first agent backend for Nsight Systems profile analysis.

`Sysight` is a small local analysis stack for turning Nsight Systems profiles into structured findings, code-location hints, and next-step optimization guidance for PyTorch workloads.

The project is aimed at agent-driven workflows first: the long-term goal is not to ship another heavy profile viewer, but to provide a compact backend that an agent can call to profile, analyze, and iterate on performance issues.

The normal usage is conversational: put a profile file into the project, then ask the agent to analyze it. The CLI exists as the backend the agent calls under the hood.

## Features

- Torch-first analysis for training and inference profiles
- SQLite-based analysis pipeline with low-context, skill-oriented decomposition
- Agent-oriented analysis backend focused on concise findings, attribution hints, and follow-up actions
- Built-in skills for kernel hotspots, iteration timing, idle gaps, launch overhead, memory transfers, NCCL anomalies, overlap, NVTX attribution, code-location hints, MFU, and theoretical FLOPs
- Markdown and JSON outputs that are easy to hand off to a follow-up agent or investigation step
- Early multi-worker orchestration demo for running several analysis tasks in parallel

## Current Status

The repository is currently focused on the analysis backend: lightweight profile inspection, skill-oriented analysis, MFU/FLOPs helpers, and report generation on top of Nsight Systems data. Automated profiling around user projects, a more natural agent-facing interaction loop, and deeper investigation workflows are still roadmap items.

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

or hand it an Nsight Systems capture directly:

```text
帮我分析一下 test/basemodel_8gpu.nsys-rep
```

3. The agent will run the analysis backend automatically and return a short terminal report with three sections:

- `结论`
- `问题`
- `下一步行动建议`

4. Accepted input files are:

- `*.sqlite`
- `*.sqlite3`
- `*.nsys-rep`

5. If the input is `*.nsys-rep`, Sysight will first try to export a sibling `*.sqlite` automatically by running:

```bash
nsys export --type=sqlite -o <same-stem>.sqlite --force-overwrite=true <file.nsys-rep>
```

This requires `nsys` on `PATH`. If `nsys` is not available, export manually first or use the helper script:

```bash
scripts/export-nsys-sqlite.sh path/to/profile.nsys-rep
```

6. At the same time, Sysight will generate analysis artifacts, for example:

```text
<profile>.sqlite                 # only when input was .nsys-rep and conversion was needed
outputs/<profile>.report.md
outputs/<profile>.findings.json
```

## Agent Workflow

The intended product direction is agent-first.

At the moment, the agent workflow is split into two paths:

1. `profile-only`: the user provides a `.sqlite` / `.sqlite3` / `.nsys-rep` path, and Sysight runs analysis directly
2. `workspace-aware`: the user also provides a `workspace/` plus a `program.md` contract so the agent can interpret findings with project entrypoints, performance goals, important paths, and constraints

The normal main path in this repository is therefore:

1. The user provides a profile path in natural language
2. The agent optionally resolves the workflow mode from `profile + workspace + program.md`
3. The agent invokes the analysis backend automatically
4. The agent returns `结论 / 问题 / 下一步行动建议`
5. Sysight writes a complete `report.md` report for deeper follow-up

The CLI is therefore an internal execution surface, not the primary user interface.

### Recommended Path

- Start with `profile-only` when the user only has a profile artifact and wants a fast analysis report.
- Upgrade to `workspace-aware` when the user can also provide `workspace/program.md` and wants stronger project-aware attribution.
- Keep the current priority simple: `analyze`/`report` first, deeper profiling and optimization loops later.

### Workspace Contract

If the user wants workspace-aware analysis, place a `program.md` file under `workspace/`. A starter template now lives at `workspace/program.md`.

The current template asks the user to fill in:

- task and project background
- framework / stack
- entry command
- performance goal
- important paths
- constraints
- success criteria
- output contract

## Backend Commands

When debugging the backend itself, run commands directly from the repository root:

```bash
PYTHONPATH=src python3 -m sysight --help
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite --workspace workspace --program program.md
PYTHONPATH=src python3 -m sysight route --profile path/to/profile.sqlite --workspace workspace --program program.md
PYTHONPATH=src python3 -m sysight skill list
PYTHONPATH=src python3 -m sysight skill run workflow_router path/to/profile.sqlite --workspace workspace --program program.md
scripts/export-nsys-sqlite.sh path/to/profile.nsys-rep
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
└── workspace/                   # User workspace, including program.md contract template
```

## Roadmap

- [ ] Add a profiling stage that can wrap real Torch entrypoints and produce standard Nsight Systems captures
- [ ] Strengthen workspace-aware analysis around `program.md`, project entrypoints, modules, and data paths
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
