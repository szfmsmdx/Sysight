from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sysight.agent.provider import LLMConfig, LLMResponse
from sysight.agent.providers.replay import ReplayProvider
from sysight.pipeline.optimize import run_optimize_trials
from sysight.tools.registry import ToolRegistry
from sysight.types.findings import LocalizedFinding, LocalizedFindingSet
from sysight.types.optimization import MeasurementPlan, MetricSpec
from sysight.types.repo_setup import RepoSetup


class TestOptimizeTrials(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")

    def test_accepts_improving_trial_and_commits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            (repo / "run.py").write_text(
                "VALUE = 10\n"
                "print(f'iteration {VALUE} ms')\n",
                encoding="utf-8",
            )
            _git(repo, "init")
            _git(repo, "add", "run.py")
            _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-m", "init")

            provider = ReplayProvider(LLMConfig(provider="replay", model="replay"))
            provider.load_fixtures([
                LLMResponse(content=json.dumps({
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
                        "validation_commands": [["python", "-m", "py_compile", "run.py"]],
                    }],
                }))
            ])

            findings = LocalizedFindingSet(
                run_id="run-test",
                findings=[
                    LocalizedFinding(
                        finding_id="C5:test",
                        category="C5",
                        title="Synthetic slow value",
                        file_path="run.py",
                        line=1,
                        description="VALUE controls printed iteration time",
                    )
                ],
            )
            plan = MeasurementPlan(
                run_command=["python", "run.py"],
                metrics=[
                    MetricSpec(
                        name="iteration_ms",
                        regex=r"iteration (\d+) ms",
                        lower_is_better=True,
                        primary=True,
                    )
                ],
            )

            result = run_optimize_trials(
                findings,
                plan,
                str(repo),
                ToolRegistry(),
                provider,
                repo_setup=RepoSetup(minimal_run=["python", "run.py"]),
                run_dir=root / "out",
                max_trials=1,
            )

            self.assertEqual(result.errors, [])
            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.rejected_count, 0)
            self.assertEqual(result.best_measurement.primary_value, 5.0)
            self.assertTrue(result.best_commit)
            self.assertEqual(len(result.trials), 1)
            self.assertEqual(result.trials[0].status, "accepted")
            self.assertTrue((root / "out" / "worklog.md").exists())

    def test_rejected_trial_records_measurement_errors(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            (repo / "run.py").write_text(
                "VALUE = 10\n"
                "print(f'iteration {VALUE} ms')\n",
                encoding="utf-8",
            )
            _git(repo, "init")
            _git(repo, "add", "run.py")
            _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-m", "init")

            provider = ReplayProvider(LLMConfig(provider="replay", model="replay"))
            provider.load_fixtures([LLMResponse(content=json.dumps({
                "action": "patch",
                "summary": "break the run to force a measurement failure",
                "patches": [{
                    "patch_id": "patch-break",
                    "finding_ids": ["C1:test"],
                    "file_path": "run.py",
                    "old_span_start": 2,
                    "old_span_end": 2,
                    "replacement": "raise SystemExit(1)",
                    "rationale": "synthetic failure case",
                }],
            }))])

            findings = LocalizedFindingSet(
                run_id="run-test-fail",
                findings=[
                    LocalizedFinding(
                        finding_id="C1:test",
                        category="C1",
                        title="Synthetic failing patch",
                        file_path="run.py",
                        line=2,
                        description="replace the print line",
                    )
                ],
            )
            plan = MeasurementPlan(
                run_command=["python", "run.py"],
                metrics=[MetricSpec(name="iteration_ms", regex=r"iteration (\d+) ms", primary=True)],
            )

            result = run_optimize_trials(
                findings,
                plan,
                str(repo),
                ToolRegistry(),
                provider,
                repo_setup=RepoSetup(minimal_run=["python", "run.py"]),
                run_dir=root / "out",
                max_trials=1,
            )

            self.assertEqual(result.accepted_count, 0)
            self.assertEqual(result.rejected_count, 1)
            self.assertEqual(result.trials[0].status, "rejected")
            self.assertTrue(result.trials[0].measurement_errors)
            self.assertIn("after measurement failed", result.trials[0].reason)

    def test_baseline_falls_back_to_warmup_command(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            (repo / "run.py").write_text(
                "VALUE = 10\n"
                "print(f'iteration {VALUE} ms')\n",
                encoding="utf-8",
            )
            _git(repo, "init")
            _git(repo, "add", "run.py")
            _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-m", "init")

            provider = ReplayProvider(LLMConfig(provider="replay", model="replay"))
            provider.load_fixtures([LLMResponse(content=json.dumps({
                "action": "patch",
                "summary": "lower value after command fallback",
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

            findings = LocalizedFindingSet(
                run_id="run-fallback",
                findings=[LocalizedFinding(
                    finding_id="C5:test",
                    category="C5",
                    title="Synthetic slow value",
                    file_path="run.py",
                    line=1,
                    description="VALUE controls printed iteration time",
                )],
            )
            plan = MeasurementPlan(
                run_command=["python", "missing.py"],
                metrics=[MetricSpec(name="iteration_ms", regex=r"iteration (\d+) ms", primary=True)],
            )

            result = run_optimize_trials(
                findings,
                plan,
                str(repo),
                ToolRegistry(),
                provider,
                repo_setup=RepoSetup(minimal_run=["python", "run.py"]),
                run_dir=root / "out",
                max_trials=1,
            )

            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.best_measurement.primary_value, 5.0)
            self.assertTrue(any("fallback" in e for e in result.errors))


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    return proc.stdout


if __name__ == "__main__":
    unittest.main()
