# Sysight

Sysight is an evidence-driven agent loop for Nsight Systems analysis. The repository currently focuses on `analyzer v0.5`: it turns `.sqlite` profile data and target-repo source code into structured bottleneck findings, source-level localization, and benchmarkable outputs for downstream `optimizer` and `executor` stages.

---

## Architecture

```mermaid
flowchart LR
    Trace["Nsight Systems trace<br/>.sqlite"] --> Extract["analyzer<br/>extract + classify"]
    Repo["target repo"] --> Locate["agent localization<br/>scanner + nsys-sql"]
    Extract --> Locate
    Locate --> Findings["structured findings<br/>JSON + readable report"]
    Findings --> Bench["benchmark loop<br/>benchmark.py / nsys-bench"]
    Bench --> Memory["memory<br/>workspace.md + experience.md"]
    Memory -. informs next run .-> Locate

planned path:
    findings --> optimizer --> executor --> benchmark
```

- `analyzer` is implemented today.
- `optimizer` and `executor` remain planned peer modules.
- The loop is deliberate: benchmark feedback and accumulated memory feed the next investigation pass.

---

## Current Scope

- deterministic profile analysis over Nsight Systems `.sqlite` traces
- agent-driven source localization with `scanner` and `nsys-sql`
- structured JSON and terminal report output for downstream consumers
- benchmark execution against `nsys-bench`
- persistent workspace and cross-workspace memory

The repository does **not** yet ship automated patch generation or patch execution in the main loop.

---

## Evidence-Driven Analyzer Loop

1. **Extract profile evidence** — parse kernels, memcpy, sync, NVTX, GPU idle gaps, and SQL-derived aggregates.
2. **Classify bottlenecks** — normalize findings into `C1`-`C7` categories.
3. **Localize code** — combine profile evidence with repo navigation tools to resolve file, function, and line.
4. **Persist memory** — append reusable workspace and experience notes.
5. **Benchmark the output** — score analyzer findings with `benchmark.py` against `nsys-bench` ground truth.

This keeps the LLM focused on code localization instead of asking it to infer raw profile semantics from scratch.

---

## Repository Layout

```text
sysight/analyzer/         analyzer pipeline, CLI, nsys-sql helpers, scanner tools
sysight/analyzer/memory/  memory component (workspace + experience store)
benchmark.py              run nsys-bench and score analyzer outputs
.sysight/bench-runs/      benchmark logs, per-run summaries, SOTA tracking
.sysight/memory/          runtime memory files (workspace.md, experience.md)
nsys-bench/               benchmark cases and ground truth
test/                     unit tests for analyzer, renderer, SQL CLI, and scanner
```

For analyzer internals and the full CLI surface, see `sysight/analyzer/README.md`.

---

## Quick Start

### Install

```bash
pip install -e .
```

### Analyze a profile

```bash
# Statistics only
sysight nsys /path/to/trace.sqlite --no-codex

# Full analysis with source localization
sysight nsys /path/to/trace.sqlite --repo-root /path/to/repo

# Machine-readable output
sysight nsys /path/to/trace.sqlite --repo-root /path/to/repo --json
```

### Repo-local execution

```bash
PYTHONPATH=src python3 -m sysight.analyzer.cli nsys /path/to/trace.sqlite --no-codex
PYTHONPATH=src python3 -m sysight.analyzer.cli nsys /path/to/trace.sqlite --repo-root /path/to/repo
PYTHONPATH=src python3 -m sysight.analyzer.cli nsys /path/to/trace.sqlite --json
```

### Useful low-level tools

```bash
# Inspect profile-side evidence
sysight nsys-sql kernels /path/to/trace.sqlite
sysight nsys-sql overlap /path/to/trace.sqlite
sysight nsys-sql stream-concurrency /path/to/trace.sqlite

# Inspect repo-side evidence
sysight scanner search /path/to/repo dispatch_experts
sysight scanner trace /path/to/repo SomeSymbol
```

---

## Benchmark Evidence

- `benchmark.py` runs the analyzer on `nsys-bench` and scores outputs against ground truth.
- Per-run artifacts live under `.sysight/bench-runs/<timestamp>/`.
- The current best recorded scores are tracked in `.sysight/bench-runs/sota.md`, and each score is backed by the corresponding `summary.txt` file in that run directory.

This keeps README claims tied to recorded benchmark artifacts instead of informal notes.

---

## Testing

```bash
PYTHONPATH=src python3 -m unittest discover -s test -v
```

---

## Roadmap

- `optimizer`: turn analyzer findings into patch proposals
- `executor`: apply and validate proposed changes safely
- multi-profile comparison and regression analysis
- broader trace ingestion and export workflows
- better C++ and native callstack readability

---

## Related Projects

- [nsys-bench](https://github.com/szfmsmdx/nsys-bench) — benchmark suite used to score Sysight analyzer outputs
- [nsys-ai](https://github.com/siboehm/nsys-ai) — earlier inspiration for agent-oriented Nsight analysis
