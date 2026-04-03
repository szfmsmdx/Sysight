# Sysight

[简体中文版](./README.zh-CN.md)

Lightweight, Torch-first agent backend for Nsight Systems profile analysis.

`Sysight` turns Nsight Systems profiles into structured findings, code-location hints, and next-step optimization guidance for PyTorch workloads.  The project is kept intentionally small: the only input it needs is a `.sqlite` / `.nsys-rep` file; no source code, no project workspace, no build environment.

The normal usage is conversational: hand the agent a profile file and ask it to analyze.  The CLI is the execution backend the agent calls under the hood.

## Features

- Torch-first analysis for training and inference profiles
- SQLite-based analysis pipeline with low-context, skill-oriented decomposition
- Agent-oriented backend focused on concise findings, code-location hints, and actionable next steps
- Built-in skills for kernel hotspots, iteration timing, idle gaps, launch overhead, memory transfers, NCCL anomalies, compute/communication overlap, NVTX attribution, code-location hints, MFU, and theoretical FLOPs
- Code-location hints that narrow a suspicious time window down to CPU thread → sampled call stack → Python file / function
- Markdown and JSON outputs ready to hand off to a follow-up investigation step

## Quick Start

1. Put your Nsight Systems `.sqlite` / `.sqlite3` / `.nsys-rep` file anywhere accessible.

2. Tell the agent directly:

```text
帮我分析一下 profiles/pnc_prof_0330_prediction_fakedata.sqlite
```

or with an absolute path:

```text
帮我分析一下 /abs/path/to/profile.sqlite
```

or hand it an Nsight Systems capture directly:

```text
帮我分析一下 /abs/path/to/profile.nsys-rep
```

3. The agent runs the analysis backend automatically and returns a short three-section terminal report:

- `结论`
- `问题`
- `下一步行动建议`

4. Accepted input formats:

- `*.sqlite`
- `*.sqlite3`
- `*.nsys-rep`

5. If the input is `*.nsys-rep`, Sysight will try to export a sibling `*.sqlite` automatically:

```bash
nsys export --type=sqlite -o <same-stem>.sqlite --force-overwrite=true <file.nsys-rep>
```

This requires `nsys` on `PATH`.  If `nsys` is not available, export manually first, or use the helper script:

```bash
scripts/export-nsys-sqlite.sh path/to/profile.nsys-rep
```

6. Analysis artifacts are written to `outputs/`:

```text
outputs/<profile>.report.md      # full Markdown report
outputs/<profile>.findings.json  # machine-readable findings
```

## Analysis Output

The report is structured around five sections:

| Section | Contents |
|---|---|
| 执行摘要 | Overall GPU utilisation, top bottleneck, NCCL share |
| Kernel 热点 | Top kernels by compute time with truncated names |
| 问题窗口 | Suspicious time windows with CPU thread, NVTX context, and sampled call stack |
| 行动建议 | Prioritised, file/function-level next steps |
| NVTX 热点候选 | Candidate NVTX regions with timing breakdown |

Code-location hints in problem windows are reported at the granularity of Python file + function name, derived from sampled CPU call stacks in the profile.

## CLI Reference

Run directly from the repository root for debugging or scripted use:

```bash
# Show help
PYTHONPATH=src python3 -m sysight --help

# Analyse a profile and write report + findings
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite

# Specify output paths explicitly
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite \
    --markdown outputs/my.report.md \
    --findings outputs/my.findings.json

# List available analysis skills
PYTHONPATH=src python3 -m sysight skill list

# Export .nsys-rep to .sqlite (requires nsys on PATH)
scripts/export-nsys-sqlite.sh path/to/profile.nsys-rep

# Run analysis via the shell wrapper (handles path resolution and output naming)
scripts/run-profile-analysis.sh path/to/profile.sqlite
scripts/run-profile-analysis.sh path/to/profile.sqlite <gpu_id>
```

## Repository Layout

```text
sysight/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── worklog.md
├── src/sysight/                 # Package root
│   ├── analysis/                # Analysis core
│   │   ├── skills/              # Per-skill analysis modules
│   │   ├── code_location.py     # Call-stack → code location inference
│   │   ├── report.py            # Report formatting
│   │   └── summary.py           # Auto-commentary generation
│   ├── cli.py                   # CLI entrypoint
│   └── profile.py               # Profile loader
├── scripts/                     # Shell helpers
│   ├── export-nsys-sqlite.sh    # nsys export wrapper
│   └── run-profile-analysis.sh  # Analysis runner
├── skills/                      # Agent skill definitions
├── profiles/                    # Local profile samples
├── tests/                       # Unit tests
└── outputs/                     # Generated analysis artifacts
```

## Roadmap

- [ ] Strengthen NVTX conventions and attribution quality
- [ ] Deepen code-location hints: resolve line numbers from debug symbols when available
- [ ] Add before/after profile comparison for optimisation validation
- [ ] Add investigation workflows built around `ncu` and targeted micro-benchmarks
- [ ] Add tests and release hygiene

## Acknowledgements

- [nsys-ai](https://github.com/GindaChen/nsys-ai) for the upstream ideas around profile reading, skill-based analysis, and agent-oriented evidence organisation
- NVIDIA Nsight Systems for timeline, CUDA, NCCL, and NVTX profiling data
- NVIDIA Nsight Compute for kernel-level investigation
- PyTorch as the primary execution environment this project is built around

## License

[MIT](./LICENSE)
