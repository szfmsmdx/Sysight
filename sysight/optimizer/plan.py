"""Orchestration logic for sysight.optimizer.

Reads findings.json and repo context, dispatches to LLM (via CLI or internal API)
and parses the resulting patch_plan.json.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sysight.shared.memory.store import build_memory_brief
from .models import PatchPlan


logger = logging.getLogger(__name__)


def _skill_path() -> Path:
    return Path(__file__).parent / "SKILL.txt"


def build_optimizer_prompt(
    repo_root: str,
    findings_path: str,
    memory_dir: str = "",
    namespace: str | None = None,
) -> str:
    """Build the prompt for the optimizer agent."""
    skill_text = _skill_path().read_text(encoding="utf-8")

    # Load findings
    try:
        findings_text = Path(findings_path).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Could not read findings.json: %s", e)
        findings_text = "{}"

    # Build memory brief
    memory_brief = ""
    if memory_dir:
        memory_brief = build_memory_brief(
            memory_dir,
            repo_root=repo_root,
            namespace=namespace,
        )

    prompt = (
        f"{skill_text}\\n\\n"
        f"==== CONTEXT ====\n"
        f"Repo Root: {repo_root}\\n"
        f"Memory:\\n{memory_brief}\\n"
        f"==== FINDINGS ====\n"
        f"Please analyze these findings and generate a PatchPlan:\n"
        f"```json\n{findings_text}\n```\n"
    )

    return prompt


def parse_patch_plan_output(stdout: str) -> PatchPlan | None:
    """Extract the PatchPlan JSON from agent output."""
    matches = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", stdout))
    if not matches:
        return None

    # Take the last JSON block found
    last_json_str = matches[-1].group(1)
    try:
        data = json.loads(last_json_str)
        return PatchPlan.from_dict(data)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON block from agent output: %s", e)
        return None


def run_optimizer_agent(
    repo_root: str,
    findings_path: str,
    memory_dir: str = "",
    namespace: str | None = None,
    mock_stdout: str | None = None,
) -> PatchPlan | None:
    """Run the optimizer agent (mockable for tests) and return the PatchPlan."""
    
    prompt = build_optimizer_prompt(
        repo_root=repo_root,
        findings_path=findings_path,
        memory_dir=memory_dir,
        namespace=namespace,
    )

    # If mocking (e.g. tests or benchmark run), just parse the mock
    if mock_stdout is not None:
        return parse_patch_plan_output(mock_stdout)

    # Real agent execution (e.g. spawning Claude Code CLI or similar)
    # For now, we simulate the subprocess call similar to analyzer/nsys/localization.py
    logger.info("Executing optimizer agent via CLI...")
    
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_path = f.name

    try:
        # NOTE: Using a placeholder agent CLI call here. Adapt to actual CLI args.
        # e.g. ["codex", "--prompt-file", prompt_path] or ["catpaw", ...]
        cmd = ["catpaw", "--prompt-file", prompt_path, "--model", "claude-3-5-sonnet-20240620"]
        logger.debug("Running: %s", " ".join(cmd))
        
        # We allow a long timeout since generating patches takes time
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=600,
        )
        if res.returncode != 0:
            logger.error("Optimizer agent failed (exit %d):\\n%s", res.returncode, res.stderr)
            return None
            
        stdout = res.stdout
    except Exception as e:
        logger.error("Error executing optimizer agent: %s", e)
        return None
    finally:
        Path(prompt_path).unlink(missing_ok=True)

    return parse_patch_plan_output(stdout)
