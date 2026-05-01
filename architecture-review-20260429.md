# Sysight Self-Improving Expert Agent Blueprint

Date: 2026-04-29
Status: ideal architecture memo; allowed to diverge from current code

## Executive Decision

Sysight should not become "Codex plus a larger prompt" or "an MCP server with
some tools." The target system should be a deterministic-first performance
engineering expert that improves through a controlled learning loop:

```text
profile evidence
  -> typed analysis
  -> bounded agent investigation
  -> verified findings
  -> optimizer / executor / measurement
  -> run ledger
  -> learning candidates
  -> gated promotion into memory, skills, detectors, or plugins
```

The important Hermes lesson is not any single implementation detail. It is the
closed loop:

- remember compact facts that should always be available
- search larger history only when needed
- turn repeated procedures into skills
- refine skills from real use
- evolve prompts/tools only through evaluations and guardrails

Sysight should adopt that pattern, but with stricter engineering controls
because performance optimization can damage user repositories and benchmark
truth can be gamed.

## North Star

Sysight should become an accumulated expert for AI infrastructure performance:

- profile evidence is the source of truth
- LLMs investigate and propose, but the parent process owns decisions
- every accepted output has a typed schema and verifiable evidence refs
- experience becomes reusable only after passing promotion gates
- prompt rules are not the final home for lessons
- deterministic detectors absorb stable high-confidence lessons
- skills store reusable procedures
- plugins/tools expose stable capabilities
- MCP is an adapter, not the core architecture

## What Current Code Gets Right

Current Sysight already has the right foundation:

- deterministic Nsight SQLite schema inspection and trace extraction
- SQL summaries for kernels, sync, memcpy, NVTX, gaps, and launch overhead
- evidence windows and CPU hotspot extraction
- static repo scanner tools that avoid importing target code
- runtime artifacts under `.sysight/codex_runs/`
- a memory store that already has wiki pages, namespaces, signal pages, and
  `apply_memory_updates()`

These should be kept and hardened.

## What Must Change

The current system still has prototype boundaries:

1. **Parent does not own final findings.**
   `LocalizationResult` parses questions and anchors, but benchmark-critical
   findings are still recovered from raw child text.

2. **The prompt is becoming the product.**
   `sysight/analyzer/SKILL.txt` mixes universal SOP, output schema, category
   definitions, benchmark hints, and line-number policies. Each missed benchmark
   case creates pressure to add more prompt text.

3. **Memory writeback has split paths.**
   `shared.memory.store.apply_memory_updates()` is the richer path, but
   localization still uses `_flush_memory()` and flat legacy writes.

4. **Namespace isolation is fragile.**
   CLI passes `--memory-namespace`, but request normalization can silently drop
   it if not preserved.

5. **Agent transport is ad hoc.**
   Analyzer uses Codex CLI, optimizer hardcodes a placeholder Catpaw command,
   and both parse markdown-fenced JSON with regex.

6. **Executor is unsafe for production.**
   It uses `shell=True`, runs in the user working tree, warns on dirty state but
   continues, destructively reverts with `git restore .` and `git clean -f`,
   and auto-commits from one noisy metric comparison.

7. **SOTA and learning records are manual.**
   Scores, prompt versions, model versions, memory used, accepted findings, and
   missed findings are not tied together in one ledger.

## Hermes-Inspired Lessons To Import

Hermes-style self-improvement has several transferable ideas:

- **Small active memory.** Keep always-in-context memory compact and curated.
  Anything large should be searched on demand.
- **Session search.** Store full run history and retrieve specific past
  evidence only when the current task needs it.
- **Skills as procedural memory.** When a workflow repeats and has a stable
  procedure, promote it into a skill instead of leaving it as an anecdote.
- **Skill patching.** Skills should be patched from real failures and user
  corrections, not replaced wholesale on every run.
- **Evolution with gates.** Prompt, skill, tool, and detector changes should be
  evaluated against traces, tests, size limits, semantic-preservation checks,
  and human review before becoming active.
- **Provider independence.** Model backends are replaceable; project state,
  tools, and policy remain Sysight-owned.

Sysight should be stricter than Hermes in two places:

- no direct self-modification of production prompts, skills, detectors, or
  plugins without parent-side validation
- no execution or patch application outside an executor sandbox

## Target Architecture

```text
ProfileArtifact
  -> AnalyzerCore
  -> EvidenceBundle
  -> LocalizationPlanner
  -> AgentRuntime
  -> LocalizedFindingSet
  -> FindingVerifier
  -> OptimizerPlanner
  -> PatchPlan
  -> ExecutorSandbox
  -> MeasurementReport
  -> RunLedger
  -> LearningCoordinator
  -> KnowledgeRegistry
```

### Boundary Rules

- `analyzer` reads profiles and target repo source; it does not patch or execute
  target code.
- `optimizer` turns verified findings into optimization intent and patch
  candidates; it does not run target code.
- `executor` is the only module that runs target commands, and only in an
  isolated worktree, container, or remote sandbox.
- `agent` owns LLM/tool mechanics, not domain truth.
- `workflow` owns end-to-end run state.
- `knowledge` owns memory, skills, detectors, plugins, and promotion policy.
- MCP exposes selected capabilities to external clients; it does not decide
  what Sysight knows.

## Proposed Package Layout

```text
sysight/
  analyzer/
    nsys/
      extract.py
      classify.py
      classify_sql.py
      windows.py
      localization.py          # EvidenceBundle -> localization tasks
      verify.py                # localized finding validation
      models.py
    pipeline.py
  optimizer/
    planner.py                 # FindingSet -> OptimizationIntent
    patch_builder.py           # OptimizationIntent -> PatchCandidate
    validators.py
    models.py
  executor/
    sandbox.py                 # worktree/container lifecycle
    runner.py                  # allowlisted command execution
    measure.py                 # repeated metric measurement
    patch_apply.py
    models.py
  agent/
    schema.py
    runner.py                  # AgentRunner Protocol
    loop.py                    # explicit tool-calling loop
    validation.py              # schema repair / retry policy
    artifacts.py
    prompts/
      loader.py
      fragments/
        common_role.md
        evidence_driven_sop.md
        localization_task.md
        optimizer_task.md
        output_schema_localized_findings.md
        memory_policy.md
        safety_read_only.md
        benchmark_hints.md
    tools/
      registry.py
      contracts.py
      scanner.py
      nsys_sql.py
      memory.py
      artifact.py
    backends/
      codex_cli.py
      openai_responses.py
      replay.py
      dry_run.py
    mcp/
      server.py                # adapter only
  workflow/
    state.py
    coordinator.py
    policies.py
  knowledge/
    ledger.py
    memory_store.py
    experience_store.py
    skill_store.py
    detector_registry.py
    plugin_registry.py
    promotion.py
    reflection.py
    evals.py
  shared/
    scanner/
    memory/
    repo/
      paths.py
      git.py
  eval/
    benchmark.py
    graders.py
    sota.py
```

This layout intentionally separates domain behavior from agent mechanics and
separates learning storage from runtime execution.

## Core Data Contracts

Use dataclasses where the deterministic analyzer is simple. Use Pydantic or a
similar schema layer at agent and cross-module boundaries where validation,
JSON Schema, and fixture stability matter.

### EvidenceBundle

```python
class EvidenceBundle(BaseModel):
    run_id: str
    profile: ProfileRef
    profile_hash: str
    schema: SchemaSummary
    bottlenecks: BottleneckSummary
    deterministic_findings: list[DeterministicFinding]
    windows: list[EvidenceWindowRef]
    sql_artifacts: dict[str, ArtifactRef]
    source_hints: list[SourceHint]
    warnings: list[str]
```

The bundle contains compact summaries and artifact refs. Large SQL output should
be read through tools or artifacts instead of being stuffed into prompts.

### LocalizedFindingSet

```python
class LocalizedFinding(BaseModel):
    finding_id: str
    category: Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    title: str
    priority: Literal["high", "medium", "low"]
    evidence_refs: list[str]
    file: str | None
    function: str | None
    line: int | None
    confidence: Literal["confirmed", "probable", "unresolved"]
    rationale: str
    suggestion: str

class LocalizedFindingSet(BaseModel):
    summary: str
    findings: list[LocalizedFinding]
    rejected_candidates: list[RejectedFinding]
    memory_updates: list[MemoryUpdate]
```

No benchmark or downstream module should parse raw `localization.output` once
this exists.

### AgentTask / AgentRunResult

```python
class AgentTask(BaseModel):
    run_id: str
    task_id: str
    task_type: Literal["localize", "optimize", "review_patch", "reflect_run"]
    instructions_ref: ArtifactRef
    input_refs: list[ArtifactRef]
    allowed_tools: list[str]
    output_schema: str
    budgets: AgentBudgets
    policy: AgentPolicy

class AgentRunResult(BaseModel):
    run_id: str
    task_id: str
    backend: str
    model: str
    status: Literal["ok", "schema_error", "tool_error", "timeout", "provider_error"]
    output: dict
    tool_calls: list[ToolCallRecord]
    usage: UsageRecord | None
    artifacts: list[ArtifactRef]
    errors: list[AgentError]
```

The domain code should call `AgentRunner.run(task)`. It should never call Codex
CLI, OpenAI SDK, LangChain, PydanticAI, or MCP directly.

## Knowledge System

Sysight needs multiple knowledge tiers. Do not collapse them into one memory
file or one prompt.

| Tier | Purpose | Storage | Loaded by default? | Write policy |
|---|---|---|---:|---|
| Active memory | critical compact facts | `.sysight/memory/active.md` or generated brief | yes, bounded | parent only |
| Workspace wiki | repo/case structure and stable observations | `.sysight/memory/wiki/workspaces/<ns>/` | brief only | parent only |
| Experience wiki | reusable performance lessons | `.sysight/memory/wiki/experiences/` | retrieved | parent only |
| Session/run history | exact traces, prompts, tool calls, outputs | `.sysight/runs/runs.sqlite` + artifacts | no | automatic |
| Skills | reusable procedures | `.sysight/skills/<name>/SKILL.md` | discovered, loaded on demand | candidate -> reviewed |
| Detectors | deterministic analyzers | source code + tests | runtime code | PR/review only |
| Plugins/tools | new capabilities | manifest + implementation | via registry | trust-gated |
| Benchmark notes | temporary hints | quarantined docs | opt-in | never production default |

### Promotion Rules

A lesson may move between tiers only through explicit gates:

```text
raw observation
  -> learning candidate
  -> verified candidate
  -> one of:
       active memory
       experience page
       skill patch
       detector proposal
       plugin/tool proposal
       rejected / archived
```

Promotion decision table:

| Signal | Destination |
|---|---|
| one-off run detail | run ledger only |
| stable repo structure | workspace wiki |
| reusable but non-procedural lesson | experience wiki |
| repeated workflow with steps | skill |
| rule can be checked from profile/source deterministically | detector |
| capability needs new callable operation | plugin/tool |
| benchmark-only hint | benchmark notes, excluded from production prompt |

Important: **a high benchmark score is not enough to promote a rule.** A rule
must be general, evidence-backed, and safe outside that benchmark case.

## Learning Loop

The learning loop is a first-class workflow stage.

```text
RunLedger + BenchmarkResult + UserFeedback
  -> Reflection
  -> LearningCandidate[]
  -> CandidateVerifier
  -> PromotionPlanner
  -> Review / Tests
  -> KnowledgeRegistry update
```

### Reflection

After each completed run, Sysight creates a structured reflection:

```python
class RunReflection(BaseModel):
    run_id: str
    what_worked: list[str]
    what_failed: list[str]
    missed_findings: list[MissedFinding]
    false_positives: list[FalsePositive]
    useful_tool_paths: list[ToolTraceSummary]
    memory_used: list[MemoryRef]
    candidate_lessons: list[LearningCandidate]
```

Reflection is not automatically trusted. It is a proposal generator.

### LearningCandidate

```python
class LearningCandidate(BaseModel):
    candidate_id: str
    source_run_id: str
    kind: Literal["memory", "experience", "skill", "detector", "plugin", "prompt"]
    title: str
    content: str
    evidence_refs: list[str]
    scope: Literal["workspace", "global", "benchmark", "unsafe"]
    expected_benefit: str
    risks: list[str]
    proposed_tests: list[str]
```

### Verification

Each candidate must pass the gates for its kind.

Memory / experience gates:

- exact dedup
- namespace correctness
- injection / secret / destructive instruction scan
- compactness check
- evidence refs exist
- no benchmark-only rule in global experience

Skill gates:

- valid `SKILL.md` frontmatter
- clear trigger conditions
- procedure is general, not case-id-specific
- referenced scripts/templates exist
- helper scripts pass syntax checks
- size budget
- regression task or benchmark demonstrates value

Detector gates:

- synthetic unit test with tiny repo/profile
- no target repo execution
- no target module import
- deterministic output
- clear false-positive boundary
- category and evidence refs validated

Plugin/tool gates:

- manifest declares name, trust level, input schema, output schema, permissions,
  read/write behavior, and sandbox needs
- tests cover success, invalid args, path traversal, and permission denial
- high-risk tools are disabled by default

Prompt gates:

- production prompt excludes benchmark hints by default
- prompt fragment has version hash
- size budget
- semantic-preservation check
- benchmark/eval improvement without increased false positives

## Agent Runtime

### Runner Interface

```python
class AgentRunner(Protocol):
    def run(self, task: AgentTask) -> AgentRunResult:
        ...
```

Backends:

- `CodexCliRunner`: compatibility path for current workflow
- `OpenAIResponsesRunner`: direct API backend with tool calls and structured output
- `ReplayRunner`: replays fixture runs for regression tests
- `DryRunRunner`: deterministic unit-test backend

### Explicit Tool Loop

1. Build `AgentTask`.
2. Select allowed tools from `ToolRegistry`.
3. Call backend with compact instructions and schemas.
4. Receive tool calls.
5. Validate tool name, arguments, path containment, budgets, and permissions.
6. Execute via registry, not shell prompt text.
7. Record tool inputs and outputs as artifacts.
8. Continue until final output or stop condition.
9. Validate final output against schema.
10. Run domain verifier.
11. Persist run ledger and learning candidates.

Stop conditions:

- max LLM turns
- max tool calls
- max wall time
- max prompt/output tokens
- repeated identical tool calls
- path policy violation
- schema failure after retry budget
- unsafe tool request

### ToolRegistry

Tools should be typed Python functions with JSON I/O:

```text
scanner.files(repo, ext, pattern)
scanner.search(repo, query, ext, fixed)
scanner.read(repo, path, start, end)
scanner.symbols(repo, path)
scanner.callers(repo, symbol)
scanner.callees(repo, path, symbol)
scanner.trace(repo, symbol, depth)
nsys_sql.kernels(sqlite, limit)
nsys_sql.sync(sqlite)
nsys_sql.memcpy(sqlite)
nsys_sql.nccl(sqlite)
nsys_sql.overlap(sqlite)
memory.search(query, namespace, scope)
memory.read(path)
artifact.read(ref)
ledger.search(query)
```

Analyzer tools are read-only. Optimizer may propose patches but cannot apply
them. Executor gets a separate high-trust tool policy.

## MCP Strategy

MCP should not be the system core.

Internal first:

```text
ToolRegistry -> AgentRuntime
SkillRegistry -> Prompt/Procedure Loader
PluginRegistry -> CapabilityRegistry
```

External adapter later:

```text
CapabilityRegistry -> MCP server -> Codex / Claude / IDE / other clients
```

MCP can expose:

- read-only analyzer tools
- memory search/read
- artifact read
- stable prompt templates
- selected skills as prompts/resources

MCP should not expose by default:

- patch application
- shell execution
- benchmark execution
- memory writes
- skill/plugin installation

If those are needed, use a separate high-trust executor MCP server with explicit
approvals, sandboxing, and audit logs.

### Can New Skills And Plugins Still Be Inserted With MCP?

Yes, if Sysight owns the registry.

The insertion path should be:

```text
new skill/plugin package
  -> manifest parse
  -> security scan
  -> schema validation
  -> trust classification
  -> registry install
  -> optional MCP list_changed notification
```

MCP clients then discover the updated tool/prompt/resource list. The protocol is
only the exposure layer. The real extensibility comes from Sysight's registry
and promotion policy.

## Prompt Architecture

Do not keep one giant `SKILL.txt`.

Use versioned fragments:

```text
sysight/agent/prompts/fragments/
  common_role.md
  evidence_driven_sop.md
  localization_task.md
  optimizer_task.md
  output_schema_localized_findings.md
  memory_policy.md
  safety_read_only.md
  benchmark_hints.md
```

Production prompt includes:

- common role
- evidence-driven SOP
- task-specific instructions
- output schema
- allowed tools
- compact memory brief
- read-only safety policy

Production prompt excludes:

- benchmark-specific hints
- exact line-number heuristics for benchmark scoring
- "always report" rules not backed by profile evidence
- memory write commands for child agents

Benchmark mode may opt into `benchmark_hints.md`, but run ledger must record
that the prompt was benchmark-mode.

## Analyzer Flow

### A1. Ingest Profile

Input:

- `.sqlite` path or `.nsys-rep` requiring export
- optional `repo_root`
- optional memory namespace

Output:

- `ProfileArtifact`
- schema info
- profile hash
- extraction warnings

### A2. Build EvidenceBundle

Run deterministic steps:

- bottleneck classification
- SQL summaries
- evidence windows
- CPU hotspot summary
- device/rank summary

Output:

- compact `EvidenceBundle`
- SQL artifacts
- deterministic findings

### A3. Plan Localization

The parent creates bounded localization tasks:

- group by category and evidence cluster
- include relevant windows and top rows
- include expected search path
- include allowed tools and budgets
- require `LocalizedFindingSet` output

This prevents one child agent from doing unbounded "find everything" work.

### A4. Agent Localization

Agent may:

- map profile evidence to source file/function/line
- call read-only scanner/nsys/memory tools
- cite evidence refs
- return memory update suggestions

Agent must not:

- write files
- execute target repo code
- import target modules
- update memory directly
- decide scoring policy
- return untyped markdown as the final product

### A5. Verify Findings

Parent verifies:

- file path is inside repo
- file exists
- line exists
- function exists or enclosing symbol is resolvable
- category is valid
- evidence refs exist
- status/confidence is consistent
- duplicates are merged
- unresolved findings remain explicit

### A6. Write Memory Candidates

Parent converts agent suggestions into `LearningCandidate` records. Only after
candidate verification does it call `apply_memory_updates()`.

## Optimizer Flow

Split optimizer into three stages.

### O1. Optimization Planning

Input:

- verified findings
- code snippets around confirmed lines
- project constraints
- metric objective

Output:

- `OptimizationIntent[]`

The planner decides whether a finding is actionable. It does not generate a
patch yet.

### O2. Patch Candidate Generation

Input:

- one `OptimizationIntent`
- exact source spans
- local style context

Output:

- `PatchCandidate`
- expected metric movement
- tests to run
- risk notes

Patch candidates should prefer:

```text
file path + old span hash + replacement text + rationale
```

Unified diff can be stored as an artifact, but parent must verify that it
matches the expected file/span.

### O3. Patch Validation

Parent validates:

- patch applies to expected base
- only allowed files touched
- no dependency addition unless policy allows it
- static syntax checks pass
- rationale links to finding id
- validation plan exists

Only validated candidates go to executor.

## Executor Flow

Executor is a separate trust boundary.

### E1. Create Sandbox

Use one of:

- git worktree per patch for local development
- Docker/container for production or untrusted repos
- remote runner for GPU benchmarking

Never run executor in the user's dirty working tree.

### E2. Apply Patch

Rules:

- one candidate per branch/worktree
- record base commit and patch hash
- reject if patch touches files outside plan
- reject if old span hash does not match

### E3. Run Validation Commands

No default `shell=True`.

Use typed commands:

```python
CommandSpec(argv=["python3", "-m", "pytest", "..."], cwd="repo", timeout_s=600)
```

Shell syntax requires `shell_allowed: true` in a trusted config, not in LLM
output.

### E4. Measure

Run:

- baseline N times
- patched N times
- warmup discarded if configured
- typed metric parser
- confidence interval or minimum threshold for noisy metrics

### E5. Commit Or Export

Default:

- do not auto-commit to user's branch
- produce patch branch/worktree and report
- allow `--commit-accepted` only when explicitly requested and clean

## Run Ledger

Use SQLite under `.sysight/runs/runs.sqlite`.

Tables:

```text
runs(run_id, created_at, status, profile_hash, repo_root, repo_commit,
     model_policy, prompt_version, memory_namespace)
stages(run_id, stage, status, started_at, ended_at, error)
artifacts(run_id, artifact_id, kind, path, sha256, size_bytes)
agent_runs(run_id, task_id, backend, model, status, prompt_tokens,
           output_tokens, elapsed_ms)
tool_calls(run_id, task_id, call_id, tool_name, args_json, result_ref,
           elapsed_ms, status)
findings(run_id, finding_id, category, file, line, confidence, source,
         accepted, reject_reason)
benchmark_results(run_id, case_name, score, total, matched_ids_json)
patches(run_id, patch_id, finding_id, status, branch, diff_ref)
measurements(run_id, patch_id, metric_name, value, unit, run_index)
learning_candidates(candidate_id, run_id, kind, status, title, content_hash)
promotion_decisions(candidate_id, decision, reviewer, reason, created_at)
memory_updates(run_id, path, status, content_hash)
```

The ledger must explain:

- which model and prompt version were used
- which memory was in the brief
- which tools were called
- which findings were accepted or rejected
- which benchmark truth was matched or missed
- which candidates were promoted
- why SOTA changed

## SOTA And Evaluation

SOTA should be generated from ledger rows, not manually maintained.

Evaluation artifacts:

- per-run summary
- per-case scores
- matched/missed finding ids
- false positives
- model/prompt/memory config
- candidate lessons generated from misses

Benchmark learning policy:

- A benchmark miss may create a candidate.
- A candidate must be generalized before promotion.
- Exact case-specific line-number tricks stay quarantined.
- Production prompt never includes benchmark-only hints by default.

## Skill And Plugin Lifecycle

### Skill Package

```text
.sysight/skills/<name>/
  SKILL.md
  manifest.json
  references/
  scripts/
  tests/
```

Manifest:

```json
{
  "name": "nsys-d2h-sync-triage",
  "version": "0.1.0",
  "kind": "skill",
  "scope": "global",
  "trust": "internal",
  "triggers": ["D2H count", "tiny D2H", ".item"],
  "permissions": ["scanner.read", "nsys_sql.memcpy", "memory.search"],
  "tests": ["tests/test_skill_nsys_d2h_sync.py"]
}
```

### Plugin Package

```text
.sysight/plugins/<name>/
  plugin.json
  src/
  tests/
  README.md
```

Plugin manifest must include:

- name and version
- provided tools
- input/output schemas
- permissions
- whether it reads, writes, executes, or accesses network
- sandbox requirements
- trust level
- tests

Trust levels:

- `builtin`: shipped with Sysight
- `internal`: created in this repo and reviewed
- `workspace`: local to one target repo/case
- `community`: installed from outside, disabled until reviewed
- `quarantined`: visible but not callable

## Security Policy

Analyzer:

- read-only target repo access
- no target code execution
- no target imports
- path containment on every file read
- no memory writes by child agents

Optimizer:

- no target execution
- no imports
- patch proposals only
- allowed file set
- old span hash required

Executor:

- isolated worktree/container/remote runner
- allowlisted command specs
- no destructive clean in user tree
- repeated measurements
- explicit commit/export policy

Knowledge:

- block prompt injection and credential exfiltration in memory/skills/plugins
- keep untrusted plugins quarantined
- record provenance for every promoted item
- support rollback of memory, skill, detector, and plugin changes

## Model Policy

Use config, not hardcoded model strings:

```toml
[models.localize]
provider = "openai"
model = "gpt-5.4"
reasoning_effort = "medium"
temperature = 0
max_tool_calls = 30

[models.optimize]
provider = "openai"
model = "gpt-5.4"
reasoning_effort = "high"
temperature = 0
max_tool_calls = 20

[models.reflect]
provider = "openai"
model = "gpt-5.4-mini"
reasoning_effort = "low"
temperature = 0
```

Do not hardcode "latest" in source. Record the resolved model in the run ledger.

## Migration Plan

### Phase 0: Stop The Bleeding

Goal: make current behavior structurally trustworthy.

- preserve `memory_namespace` in normalized requests
- parse child `findings` into typed parent fields
- make benchmark consume typed `localized_findings`
- replace localization `_flush_memory()` with parent-side `apply_memory_updates()`
- disable direct memory write commands in analyzer prompt
- add tests for malformed child JSON, namespace isolation, and memory writeback

### Phase 1: Run Ledger And Typed Artifacts

Goal: make every result explainable.

- add `.sysight/runs/runs.sqlite`
- record prompt hash, model, memory namespace, tool calls, artifacts, score
- generate SOTA from ledger
- keep existing `.sysight/bench-runs/` as exported views

### Phase 2: Learning Candidates

Goal: learn without polluting production knowledge.

- add `RunReflection`
- generate `LearningCandidate[]` after benchmark/user feedback
- implement candidate dedup and safety checks
- store candidates in ledger
- no automatic promotion yet

### Phase 3: Knowledge Registry

Goal: promote experience through explicit gates.

- add memory/experience/skill/detector/plugin registries
- implement promotion policy
- add skill validation and size checks
- add detector proposal format and required tests
- add rollback metadata

### Phase 4: Agent Core

Goal: remove direct domain dependency on Codex CLI.

- add `AgentTask`, `AgentRunResult`, and `AgentRunner`
- move Codex subprocess to `agent/backends/codex_cli.py`
- add replay and dry-run backends
- move prompt building to versioned fragments
- keep localization behavior equivalent

### Phase 5: Direct API Backend

Goal: use structured outputs and typed tool calls.

- add OpenAI Responses backend
- add explicit tool loop
- register scanner/nsys_sql/memory/artifact tools
- record every tool call in ledger
- add replay fixtures for regression tests

### Phase 6: Optimizer Refactor

Goal: make optimizer a typed planner, not a free-form patch generator.

- split optimization intent and patch generation
- require old span hash and allowed files
- validate patch candidates before executor
- remove hardcoded external agent command
- add synthetic repo tests

### Phase 7: Executor Sandbox

Goal: safe enough for real repos.

- create worktree per patch
- remove default `shell=True`
- add allowlisted `CommandSpec`
- repeat metrics
- never destructively clean user tree
- export report and patch branch by default

### Phase 8: MCP Adapter

Goal: expose stable Sysight capabilities externally.

- add read-only MCP server over `CapabilityRegistry`
- expose tools/resources/prompts only after schema stabilizes
- keep MCP out of core analyzer tests
- make executor MCP separate and high-trust

### Phase 9: Optional Evolution Engine

Goal: evolve prompts, skills, tools, and detectors with evidence.

- build eval datasets from run ledger
- propose variants from execution traces
- run benchmark and unit gates
- enforce size and semantic-preservation checks
- require human review or PR before activation

## Strong Recommendation

Do not rebuild Sysight around LangChain as the core.
Do not start with MCP.
Do not let Codex CLI or any other agent own product truth.
Do not let benchmark misses directly mutate production prompt rules.

Build:

```text
typed schemas
+ deterministic evidence bundle
+ Sysight-owned AgentRunner
+ ToolRegistry / CapabilityRegistry
+ RunLedger
+ LearningCandidate pipeline
+ gated KnowledgeRegistry
+ sandboxed executor
+ optional MCP adapter
```

This gives Sysight the path to become an expert that gets stronger with use
without sacrificing correctness, security, reproducibility, or benchmark
honesty.

## External Signals Used

- Hermes Agent: closed learning loop, skill creation/refinement, session search,
  provider independence, multi-surface runtime.
- Hermes memory docs: compact active memory plus searchable session history.
- Hermes skills docs: skills as procedural memory, agent-managed skill patching,
  external skill directories, hub/security scanning.
- Hermes self-evolution: trace-driven skill/prompt/tool evolution with tests,
  size limits, semantic preservation, and human review.
- MCP docs: tools are model-controlled, prompts are user-controlled templates,
  and `listChanged` supports dynamic discovery; therefore MCP fits as an
  adapter over Sysight's own registry, not as the core.
