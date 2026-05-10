from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sysight.agent.provider import LLMConfig, LLMResponse
from sysight.agent.providers.replay import ReplayProvider
from sysight.pipeline.agent_loop import _optimizer_inputs_from_analyze, _refresh_profile, run_agent_loop
from sysight.pipeline.analyze import AnalyzeResult
from sysight.types.repo_setup import RepoSetup
from sysight.tools.registry import ToolRegistry


class MemoryStub:
    def __init__(self, root: Path):
        self.root = root

    def workspace_namespace(self, repo_root: str | None = None, namespace: str | None = None) -> str:
        if namespace:
            return namespace
        return Path(repo_root or "default").name

    def read_page(self, path: str) -> str | None:
        target = self.root / "wiki" / path
        return target.read_text(encoding="utf-8") if target.exists() else None

    def write_page(self, path: str, content: str, **kwargs) -> Path:
        target = self.root / "wiki" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def append_page(self, path: str, content: str) -> Path:
        target = self.root / "wiki" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(content + "\n")
        return target

    def replace_in_page(self, path: str, old: str, new: str) -> Path:
        target = self.root / "wiki" / path
        text = target.read_text(encoding="utf-8")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return target

    def append_worklog(self, namespace: str, entry: str) -> Path:
        return self.append_page(f"workspaces/{namespace}/worklog.md", entry)


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_outer_loop_accepts_trial_and_records_worklog(self):
        repo = self.root / "repo"
        repo.mkdir()
        (repo / "run.py").write_text(
            "VALUE = 10\n"
            "print(f'iteration {VALUE} ms')\n",
            encoding="utf-8",
        )
        profile = self.root / "profile.sqlite"
        profile.write_text("", encoding="utf-8")
        _git(repo, "init")
        _git(repo, "add", "run.py")
        _git(
            repo,
            "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid",
            "commit", "-m", "init",
        )

        analyze = ReplayProvider(LLMConfig(provider="replay", model="replay"))
        analyze.load_fixtures([LLMResponse(content=json.dumps({
            "summary": "synthetic profile",
            "measurement_plan": {
                "run_command": ["python", "run.py"],
                "metrics": [{
                    "name": "iteration_ms",
                    "regex": "iteration (\\d+) ms",
                    "primary": True,
                    "lower_is_better": True,
                }],
            },
            "findings": [{
                "finding_id": "C5:test",
                "category": "C5",
                "title": "Synthetic slow value",
                "file_path": "run.py",
                "line": 1,
                "description": "VALUE controls printed iteration time",
            }],
        }))])

        optimize = ReplayProvider(LLMConfig(provider="replay", model="replay"))
        optimize.load_fixtures([LLMResponse(content=json.dumps({
            "action": "patch",
            "summary": "lower the synthetic iteration time",
            "patches": [{
                "patch_id": "patch-1",
                "finding_ids": ["C5:test"],
                "file_path": "run.py",
                "old_span_start": 1,
                "old_span_end": 1,
                "replacement": "VALUE = 5",
                "rationale": "synthetic speedup",
            }],
        }))])

        learn = ReplayProvider(LLMConfig(provider="replay", model="replay"))
        learn.load_fixtures([LLMResponse(content=json.dumps({
            "summary": "recorded synthetic speedup",
            "memory_updates": [],
        }))])

        def provider_factory(stage: str):
            return {"analyze": analyze, "optimize": optimize, "learn": learn}.get(stage)

        memory = MemoryStub(self.root / ".sysight" / "memory")
        result = run_agent_loop(
            str(profile),
            str(repo),
            ToolRegistry(),
            provider_factory,
            memory,
            max_loops=1,
            max_trials=1,
            refresh_profiles=False,
        )

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(result.iterations[0].accepted_count, 1)
        self.assertEqual(result.iterations[0].metric_before, 10.0)
        self.assertEqual(result.iterations[0].metric_after, 5.0)
        self.assertTrue((Path(result.run_dir) / "worklog.csv").exists())
        self.assertTrue((self.root / ".sysight" / "worklog" / "agent_loop.csv").exists())
        self.assertTrue(memory.read_page("workspaces/repo/worklog.md"))

    def test_profile_refresh_failure_writes_status(self):
        repo = self.root / "repo"
        repo.mkdir()
        profile = self.root / "profile.sqlite"
        profile.write_text("", encoding="utf-8")
        profiles_dir = self.root / "profiles"

        refreshed = _refresh_profile(
            repo,
            profile,
            RepoSetup(minimal_run=[]),
            profiles_dir,
            2,
            profile_command="definitely_missing_sysight_profile_command",
        )

        self.assertIsNone(refreshed)
        status = profiles_dir / "iter-002.profile_refresh.json"
        self.assertTrue(status.exists())
        data = json.loads(status.read_text(encoding="utf-8"))
        self.assertIn("error", data)

    def test_optimizer_inputs_fallback_when_analyze_empty(self):
        analyze = AnalyzeResult(run_id="empty")
        setup = RepoSetup(
            minimal_run=["python", "scripts/sysight_run.py"],
            metric_name="",
        )

        findings, plan = _optimizer_inputs_from_analyze(analyze, setup)

        self.assertTrue(plan.is_valid())
        self.assertEqual(plan.run_command, ["python", "scripts/sysight_run.py"])
        self.assertEqual(plan.primary_metric.name, "iteration_ms")
        self.assertEqual(len(findings.findings), 1)
        self.assertEqual(findings.findings[0].finding_id, "C0:end_to_end")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    return proc.stdout


if __name__ == "__main__":
    unittest.main()
