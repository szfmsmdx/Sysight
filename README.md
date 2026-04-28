# Sysight

Sysight is an end-to-end AI performance tuning Agent platform driven by Nsight Systems (nsys). By deeply extracting features from `.nsys-rep` and `.sqlite` performance trace files and combining them with LLM inference, the system automates the diagnosis of compute and communication bottlenecks, the generation of expert-level optimization strategies, and source-level (AST) precise cross-file localization. Concurrently, leveraging the Agent's autonomous refactoring and end-to-end Benchmark loop validation mechanisms, Sysight achieves a closed loop from "issue discovery" to "code fix validation". Currently, platform v0.1 has been deployed internally, aiming to drive autonomous performance maintenance of codebases via AI, replacing traditional high-barrier manual tuning workflows, and fully unleashing the productivity of AI Infra teams.

---

## Architecture

Sysight operates on a closed-loop Agent architecture consisting of three peer modules:

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                      Sysight Autonomous Tuning Loop                     │
 │                                                                         │
 │   ┌───────────────┐                             ┌───────────────┐       │
 │   │               │       1. Profiling          │               │       │
 │   │               ├────────────────────────────▶│  1. Analyzer  │       │
 │   │               │   (nsys-rep / sqlite)       │  (AI Agent)   │       │
 │   │               │                             │               │       │
 │   │ Target Repo   │◀────────────────────────────┤ • Extract     │       │
 │   │  (Workspace)  │       Code Localization     │ • Classify    │       │
 │   │               │       (AST / File Reads)    │ • Locate      │       │
 │   │               │                             └───────┬───────┘       │
 │   │               │                                     │               │
 │   │               │                             Structured Findings     │
 │   │               │                                  (JSON)             │
 │   │               │                                     │               │
 │   │               │                             ┌───────▼───────┐       │
 │   │               │       3. Execution          │               │       │
 │   │               │◀────────────────────────────┤  2. Optimizer │       │
 │   │               │   (Safe Patch & Verify)     │  (AI Agent)   │       │
 │   └───────▲───────┘                             │               │       │
 │           │                                     │ • Propose Fix │       │
 │           │                                     │ • Gen Patch   │       │
 │           │                                     └───────────────┘       │
 │           │                                                             │
 │           └─────────────────────────────────────────────────────────────┘
 │                         4. Benchmark Verification                       │
 └─────────────────────────────────────────────────────────────────────────┘
```

- **analyzer**: Parses nsys profiles deterministically, classifies bottlenecks, and dispatches a Codex AI agent to perform evidence-driven, top-down code localization to find the exact file, function, and line.
- **optimizer**: Consumes analyzer JSON findings and proposes code-level fixes via intelligent AST-aware patching. *(planned)*
- **executor**: Applies optimizer proposals to the repo, validating the fix through test runs or subsequent profiling. *(planned)*

*Current development scope is focused on the **analyzer**.*

---

## Design Philosophy

Traditional GPU profiling tools (like Nsight Systems or nsys-ai) require heavy manual interpretation of profile data to map hardware bottlenecks back to source code. Sysight automates this loop:

1. **Deterministic Extraction**: The analyzer pipeline extracts structured evidence from nsys SQLite profiles (bottleneck classification, kernel statistics, sync events, NVTX regions) without guessing.
2. **Evidence-Driven Agent Trace**: A Codex agent uses `scanner` (AST inspection) and `nsys-sql` tools to locate the exact source line causing each bottleneck. The agent avoids blind global scanning by following strict top-down tracing SOPs guided by pre-injected profile data.
3. **Continuous Memory**: The agent accumulates per-workspace (`workspace.md`) and cross-workspace (`experience.md`) experiences across runs, significantly reducing redundant investigation over time.

**Note**: The analyzer is intentionally **not** a black-box LLM call. Profile-side evidence is extracted exactly; the AI agent handles only the complex reasoning required for final code localization.

---

## Core Capabilities (Analyzer)

- **SQL Extraction**: Parses nsys `.sqlite` profiles to extract kernels, sync events, memcpy, NVTX regions, and GPU idle gaps.
- **C1-C7 Bottleneck Classification**: Identifies and classifies issues across C1 (Host Scheduling), C2 (Kernel Launch), C3 (Sync), C4 (Memory Copy), C5 (Compute), C6 (Communication), and C7 (Framework Pipeline).
- **Deep Profiling**: Runs SQL deep analysis for top kernels, NCCL breakdown, compute/comm overlap, and profile health.
- **Coarse Callstack Resolution**: Summarizes CPU callstacks and identifies initial hot paths.
- **Agent Code Localization**: Locates the exact file, function, and line number via a Codex subprocess and AST-aware `scanner` tools.
- **Memory Persistence**: Accumulates investigation logic to avoid repeated blind traces.
- **JSON Serialization**: Exposes fully structured JSON outputs designed for the downstream Optimizer agent.

---

## Quick Start

Install:

```bash
pip install -e .
```

Run profile analysis:

```bash
# Profile statistics only (Fast mode, no Agent)
sysight nsys /path/to/trace.sqlite --no-codex

# Full analysis with Codex Agent code localization
sysight nsys /path/to/trace.sqlite --repo-root /path/to/repo

# Output machine-readable structured JSON
sysight nsys /path/to/trace.sqlite --json
```

Run directly from the repository root without installing:

```bash
PYTHONPATH=src python3 -m sysight.analyzer.cli --help
```

For full CLI reference and internal mechanics, see [`sysight/analyzer/README.md`](sysight/analyzer/README.md).

---

## Testing & Validation

All logic is rigorously tested, specifically the deterministic extraction and parsing steps:

```bash
PYTHONPATH=src python3 -m unittest discover -s test -v
```

---

## Related Projects

- **[nsys-bench](https://github.com/szfmsmdx/nsys-bench)** — A benchmark suite for evaluating GPU performance diagnosis agents. This is used to test and score the accuracy of Sysight analyzer's findings against ground-truth files and line numbers.
- **[nsys-ai](https://github.com/siboehm/nsys-ai)** — AI-assisted Nsight Systems analysis toolkit, which provided initial inspiration for the agent-centric design and SQL-based profile tooling.

## Roadmap

- [ ] **optimizer**: Consume analyzer JSON findings and propose AST-aware code patches.
- [ ] **executor**: Safely apply optimizer proposals, run validations, and revert on failure.
- [ ] Multi-profile comparison (e.g., rank0 vs rank1 analysis, run-to-run regression testing).
- [ ] Broader profile format support (direct `.nsys-rep` parsing, Perfetto).
- [ ] Richer C++ callstack resolution via DWARF and native symbolizer integrations.