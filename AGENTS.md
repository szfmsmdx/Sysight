# Sysight Agent Instructions

Sysight is a repo-loop optimization pipeline with three peer modules:
- `analyzer`
- `optimizer`
- `executor`

Current scope is `analyzer v0.3`.

## Analyzer Goals

The analyzer must quickly:
- Quickly locate issues in analysis files
- Understand the codebase structure
- Quickly pinpoint code files and functions based on the issue area
- Trace the main file/function call chain within the codebase
- Return structured output to downstream modules

## Common Commands

Run commands from the project root:

```bash
cd /Users/szf/Desktop/Sysight

PYTHONPATH=src python3 -m unittest discover -s test -v

PYTHONPATH=src python3 analyzer.py /path/to/repo
PYTHONPATH=src python3 analyzer.py /path/to/repo --max-entries 1 --max-depth 3 --max-steps 30
PYTHONPATH=src python3 analyzer.py /path/to/repo --verbose
PYTHONPATH=src python3 analyzer.py /path/to/repo --json

PYTHONPATH=src python3 -m sysight.analyzer.core /path/to/repo
```

## Project Constraints

- Keep code simple.
- Keep the repo simple.
- Prefer Python stdlib first.
- Prefer explicit data flow over hidden state.
- Avoid large frameworks unless they are clearly necessary.
- Keep `analyzer`, `optimizer`, and `executor` fully decoupled.
- `analyzer` must not optimize or execute.
- Every new feature should be easy to plug in or remove.

## Code Style

- Prefer small files and small classes.
- Prefer composition over inheritance.
- Prefer plain dataclasses and narrow interfaces.
- Prefer clear names over clever abstractions.
- Avoid premature generalization.
- If logic grows too large, split by responsibility instead of creating one giant file.
- Keep parser, detector, tracer, and orchestrator responsibilities separate.
- Return structured data, not formatted prose, from core logic.

## Testing

- Every new analyzer feature needs at least one focused unit test.
- Tests should create tiny synthetic repos in temp directories.
- Do not depend on external repos for unit tests.
- Keep tests deterministic and fast.
- Document important static-analysis limitations in tests when relevant.
- Callstack readability is a release criterion for `nsys` work.
- If a change touches Stage 4 / Stage 6 / Stage 7, CPU hotspots, or callstack cleanup/rendering, you must run at least one real or focused rendered report check and verify the output is readable.
- Bad cases such as `PyEval_RestoreThread <- PyGILState_Ensure`, `launch <- cfunction_call <- _PyEval_EvalFrameDefault`, or `pthread_cond_timedwait <- ...` must not be shipped as the final coarse location by themselves.
- If the rendered output still only shows runtime wrappers / syscall leafs and no readable coarse location, the task is not done; continue improving or explicitly surface that Stage 6 / better trace signals are required.
- Before declaring an analyzer change done, run:

```bash
PYTHONPATH=src python3 -m unittest discover -s test -v
```

## Security

- Treat all repo input as untrusted.
- Do not execute target repo code during analysis.
- Do not import target repo modules into the analyzer process.
- Prefer static parsing over runtime execution.
- Be conservative with path resolution and file discovery.
- Keep file reads inside the chosen repo root.
- If shell execution is added later, isolate it to `executor`, not `analyzer`.

## Worklog

- Keep `worklog.md` short and high signal.
- Do not add a new version section like `# v0.1` or `# v0.2` unless the user explicitly says that version is completed.
- Before explicit version completion, keep notes under a temporary section such as `## Current`.
- When a version is explicitly declared complete by the user, record the version header, date, short summary, and key feature / module / file changes.
- Do not turn `worklog.md` into a full changelog.

## Nested Projects

`nsys-ai/` has its own `AGENTS.md`; follow the nested instructions when working under that subtree.
****