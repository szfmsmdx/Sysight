"""LEARN stage — post-session, 0-2 LLM calls → ledger + experience wiki.

1. Ledger (deterministic): record runs + findings + patches → SQLite
2. Benchmark (deterministic): score against truth, generate candidates
3. Session Worklog (1 LLM, optional): append to workspace wiki
4. Experience Extraction (1 LLM, optional): patterns from successful patches
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LearnResult:
    run_id: str = ""
    worklog: str = ""
    experiences: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_learn(
    run_id: str,
    knowledge,
    provider=None,
    benchmark_truth: str | None = None,
    skip_summary: bool = False,
    skip_experience: bool = False,
) -> LearnResult:
    """Post-session learning.

    Args:
        run_id: The run to process
        knowledge: WikiRepository + RunLedger
        provider: LLMProvider for summary/experience (optional)
        benchmark_truth: Path to benchmark ground truth JSON (optional)
        skip_summary: Skip LLM worklog generation
        skip_experience: Skip LLM experience extraction
    """
    errors: list[str] = []
    worklog = ""
    experiences: list[str] = []

    # 1. Deterministic: benchmark scoring
    if benchmark_truth:
        try:
            _run_benchmark(run_id, benchmark_truth, knowledge)
        except Exception as e:
            errors.append(f"benchmark failed: {e}")

    # 2. LLM: session worklog
    if provider and not skip_summary:
        try:
            worklog = _generate_worklog(run_id, provider, knowledge)
            if worklog:
                from sysight.wiki.ledger import RunLedger
                ledger = RunLedger()
                ledger.init()
                ns = ledger.recent_session(run_id)
                if ns:
                    knowledge.append_worklog(ns.get("memory_namespace", "default"), worklog)
        except Exception as e:
            errors.append(f"worklog failed: {e}")

    # 3. LLM: experience extraction
    if provider and not skip_experience:
        try:
            experiences = _extract_experiences(run_id, provider, knowledge)
            for exp in experiences:
                _save_experience(exp, knowledge)
        except Exception as e:
            errors.append(f"experience extraction failed: {e}")

    return LearnResult(run_id=run_id, worklog=worklog, experiences=experiences, errors=errors)


def _run_benchmark(run_id: str, truth_path: str, knowledge) -> dict:
    """Score findings against benchmark truth."""
    import json

    truth = json.loads(Path(truth_path).read_text())
    truth_ids = {f["finding_id"] for f in truth.get("findings", [])}

    from sysight.wiki.ledger import RunLedger
    ledger = RunLedger()
    record = ledger.recent_session(run_id)
    if not record:
        return {"score": 0, "total": len(truth_ids), "matched": []}

    # Get findings for this run
    import sqlite3
    db = ledger._db_path
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT finding_id FROM findings WHERE run_id = ? AND accepted = 1",
        (run_id,),
    ).fetchall()
    conn.close()

    found_ids = {r["finding_id"] for r in rows}
    matched = truth_ids & found_ids
    missed = truth_ids - found_ids

    score = len(matched) / len(truth_ids) if truth_ids else 0
    ledger.record_benchmark(run_id, Path(truth_path).stem, {
        "score": score, "total": len(truth_ids), "matched_ids": list(matched),
    })

    # Generate candidates for missed findings
    for m in missed:
        ledger.record_candidate({
            "candidate_id": f"bench-{run_id}-{m}",
            "run_id": run_id, "kind": "memory", "scope": "benchmark",
            "title": f"Missed: {m}", "content_hash": "",
        })

    return {"score": score, "total": len(truth_ids), "matched": list(matched), "missed": list(missed)}


def _generate_worklog(run_id: str, provider, knowledge) -> str:
    """Generate a session worklog using LLM."""
    from sysight.agent.loop import AgentLoop, AgentTask
    from sysight.agent.prompts.loader import PromptLoader

    loader = PromptLoader()
    system = loader.build_system_prompt("learn")
    user = f"Summarize the results of run {run_id}. What worked? What failed? Keep it brief."

    loop = AgentLoop(provider, None, _learn_policy())
    task = AgentTask(
        run_id=run_id, task_id=f"{run_id}-worklog",
        task_type="learn", system_prompt=system, user_prompt=user,
        max_turns=3, max_wall_seconds=60,
    )
    result = loop.run(task)
    return result.raw_content or ""


def _extract_experiences(run_id: str, provider, knowledge) -> list[str]:
    """Extract reusable patterns from successful patches."""
    from sysight.agent.loop import AgentLoop, AgentTask
    from sysight.agent.prompts.loader import PromptLoader

    loader = PromptLoader()
    system = loader.build_system_prompt("learn")
    user = f"Extract reusable performance lessons from run {run_id}. Output one lesson per line starting with '- '."

    loop = AgentLoop(provider, None, _learn_policy())
    task = AgentTask(
        run_id=run_id, task_id=f"{run_id}-experience",
        task_type="learn", system_prompt=system, user_prompt=user,
        max_turns=3, max_wall_seconds=60,
    )
    result = loop.run(task)
    text = result.raw_content or ""
    return [line.strip("- ").strip() for line in text.splitlines() if line.startswith("- ")]


def _save_experience(text: str, knowledge) -> None:
    """Save an experience to the wiki."""
    slug = text[:50].lower().replace(" ", "-").replace("/", "-")
    knowledge.write_page(
        f"experiences/{slug}.md",
        text, title=text[:80], scope="workspace",
    )


def _learn_policy():
    from sysight.tools.registry import ToolPolicy
    return ToolPolicy(allowed_tools=set(), read_only=True)


from pathlib import Path
