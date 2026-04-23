# Sysight

Sysight is an AI-assisted GPU performance diagnosis pipeline.  
It combines deterministic profile analysis with agent-driven code localization to identify actionable performance bottlenecks in GPU training and inference workloads.

---

## Architecture

Sysight consists of three peer modules:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  analyzer   │───▶│  optimizer  │───▶│  executor   │
└─────────────┘    └─────────────┘    └─────────────┘
  profile +          patch proposal     apply changes
  code locate
```

- **analyzer**: Parses nsys profiles, classifies bottlenecks, and uses an AI agent to locate exact file/function/line in the target repo.
- **optimizer**: Consumes analyzer findings and proposes code-level fixes. *(planned)*
- **executor**: Applies optimizer proposals to the repo. *(planned)*

Current scope: **analyzer**.

---

## Design Background

Traditional GPU profiling tools (Nsight Systems, nsys-ai) require manual interpretation of profile data and source code. Sysight automates this loop:

1. **Deterministic analysis** — the analyzer pipeline extracts structured evidence from nsys SQLite profiles (bottleneck classification, kernel statistics, sync events, NVTX regions).
2. **Agent-driven localization** — a Codex agent uses `scanner` and `nsys-sql` tools to locate the exact source line causing each bottleneck, guided by pre-injected profile data.
3. **Memory accumulation** — the agent accumulates per-workspace and cross-workspace experience across runs, reducing redundant investigation over time.

The analyzer is intentionally **not** a black-box LLM call. Profile-side evidence is extracted deterministically; the agent handles only the final code localization step.

---

## What the Analyzer Can Do

- Parse nsys `.sqlite` profiles: extract kernels, sync events, memcpy, NVTX regions, GPU idle gaps
- Classify bottlenecks into C1–C7 categories (Host Scheduling, Kernel Launch, Sync, Memory Copy, Compute, Communication, Framework Pipeline)
- Run SQL deep analysis: top kernels, NCCL breakdown, compute/comm overlap, profile health
- Summarize CPU callstacks and identify coarse locations
- Locate exact file/function/line via Codex agent + scanner tools
- Accumulate investigation memory across runs (`workspace.md`, `experience.md`)
- Expose structured JSON output for downstream optimizer consumption

---

## Quick Start

Install:

```bash
pip install -e .
```

Run profile analysis:

```bash
# Profile statistics only
sysight nsys /path/to/trace.sqlite

# Full analysis with Codex code localization
sysight nsys /path/to/trace.sqlite --repo-root /path/to/repo --report full

# JSON output
sysight nsys /path/to/trace.sqlite --json
```

Run from repo root without installing:

```bash
PYTHONPATH=src python3 -m sysight.analyzer.cli --help
```

For full CLI reference and component internals, see [`sysight/analyzer/README.md`](sysight/analyzer/README.md).

---

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s test -v
```

---

## Related Projects

- **[nsys-bench](https://github.com/szfmsmdx/nsys-bench)** — benchmark suite for evaluating GPU performance diagnosis agents; used to test and score Sysight analyzer outputs against ground-truth findings
- **[nsys-ai](https://github.com/siboehm/nsys-ai)** — AI-assisted Nsight Systems analysis toolkit; inspiration for the agent-centric design and SQL-based profile tooling in Sysight

## ToDo

- [ ] **optimizer**: consume analyzer JSON findings and propose code patches
- [ ] **executor**: apply optimizer proposals with safety checks
- [ ] Multi-profile comparison (rank0 vs rank1, run-to-run regression)
- [ ] Broader profile format support (`.nsys-rep` direct export, Perfetto)
- [ ] Richer C++ callstack resolution (DWARF / symbolizer integration)
