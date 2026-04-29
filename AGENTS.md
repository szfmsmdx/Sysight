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
- **Evidence-Driven Top-Down Trace**: Rely on profile evidence instead of exhaustive blind scans to avoid excessive API token costs.

## Project Constraints

- Keep code simple.
- Keep the repo simple.
- Prefer Python stdlib first.
- Prefer explicit data flow over hidden state.
- Avoid large frameworks unless they are clearly necessary.
- Keep `analyzer`, `optimizer`, and `executor` fully decoupled.
- `analyzer` must not optimize or execute.
- Every new feature should be easy to plug in or remove.
- **Maintain Universal SOPs**: When maintaining `SKILL.txt` or system prompts, preserve generalized "Evidence-Driven" top-down flows. NEVER pile up overfitting rules or exact line-number heuristics to game benchmark scores.

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
- Before explicit version completion, keep notes under a temporary section such as `## Current`.
- When a version is explicitly declared complete by the user, record the version header, date, short summary, and key feature / module / file changes.
- Do not turn `worklog.md` into a full changelog.

## Nested Projects

`nsys-ai/` has its own `AGENTS.md`; follow the nested instructions when working under that subtree.
****