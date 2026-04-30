"""Tests for sysight.shared.findings — Finding model and extraction utilities."""

import json
import tempfile
import unittest
from pathlib import Path

from sysight.shared.findings import (
    Finding,
    extract_findings,
    write_findings_json,
    find_latest_analysis_output,
    find_latest_nsys_artifact,
)


class TestFindingModel(unittest.TestCase):
    def test_roundtrip(self):
        f = Finding(
            id="f001",
            category="C1",
            title="DataLoader worker=0",
            file="src/data/factory.py",
            function="build_loader",
            line=28,
            description="worker count is 0",
            suggestion="set loader_workers=4",
            evidence=["GPU idle 99%"],
            priority="high",
        )
        d = f.to_dict()
        f2 = Finding.from_dict(d)
        self.assertEqual(f2.id, "f001")
        self.assertEqual(f2.file, "src/data/factory.py")
        self.assertEqual(f2.line, 28)
        self.assertEqual(f2.evidence, ["GPU idle 99%"])


class TestExtractFromCodexOutput(unittest.TestCase):
    """Test extraction from codex last_message.txt format."""

    def _make_codex_output(self, tmpdir: Path) -> Path:
        data = {
            "summary": "GPU idle caused by data pipeline",
            "findings": [
                {
                    "category": "C1",
                    "title": "DataLoader worker=0",
                    "priority": "high",
                    "evidence": ["GPU idle 99.1%"],
                    "file": "src/data/factory.py",
                    "function": "build_loader_parts",
                    "line": 28,
                    "description": "worker count is 0",
                    "suggestion": "set loader_workers=4",
                },
                {
                    "category": "C4",
                    "title": "No pinned memory",
                    "priority": "high",
                    "evidence": ["pin_memory=false"],
                    "file": "src/data/factory.py",
                    "function": "build_loader_parts",
                    "line": 29,
                    "description": "pin_memory disabled",
                    "suggestion": "enable pin_memory",
                },
            ],
            "memory_updates": [],
        }
        path = tmpdir / "last_message.txt"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_extract_from_codex_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_codex_output(Path(tmpdir))
            findings = extract_findings(path)
            self.assertEqual(len(findings), 2)
            self.assertEqual(findings[0].category, "C1")
            self.assertEqual(findings[0].file, "src/data/factory.py")
            self.assertEqual(findings[0].line, 28)
            self.assertEqual(findings[1].category, "C4")

    def test_extract_from_fenced_json(self):
        """Codex output may wrap JSON in ```json``` fences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = self._make_codex_output(Path(tmpdir)).read_text()
            fenced = f"Here is my analysis:\n```json\n{raw}\n```\nDone."
            path = Path(tmpdir) / "fenced_output.txt"
            path.write_text(fenced, encoding="utf-8")
            findings = extract_findings(path)
            self.assertEqual(len(findings), 2)


class TestExtractFromNsysArtifact(unittest.TestCase):
    """Test extraction from .sysight/nsys/*.json artifact."""

    def _make_nsys_artifact(self, tmpdir: Path) -> Path:
        data = {
            "status": "ok",
            "findings": [
                {"category": "gpu_idle", "severity": "critical", "title": "GPU Idle"},
            ],
            "localization": {
                "status": "ok",
                "questions": [
                    {
                        "question_id": "q001",
                        "category": "C1",
                        "title": "DataLoader worker=0",
                        "file_path": "src/data/factory.py",
                        "function": "build_loader",
                        "line": 28,
                        "rationale": "worker=0 causes serial loading",
                        "suggestion": "set workers=4",
                    },
                ],
                "anchors": [
                    {
                        "window_id": "w001",
                        "category": "C4",
                        "event_name": "cudaMemcpy",
                        "file_path": "src/pipeline/stages.py",
                        "function": "transfer",
                        "line": 16,
                        "rationale": "sync H2D",
                        "suggestion": "use non_blocking",
                    },
                ],
            },
        }
        path = tmpdir / "nsys_artifact.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_extract_prefers_questions_over_anchors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_nsys_artifact(Path(tmpdir))
            findings = extract_findings(path)
            # Should get questions first (more detailed)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].id, "q001")
            self.assertEqual(findings[0].category, "C1")

    def test_extract_falls_back_to_anchors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "localization": {
                    "questions": [],
                    "anchors": [
                        {
                            "window_id": "w001",
                            "category": "C4",
                            "file_path": "src/x.py",
                            "function": "foo",
                            "line": 10,
                            "rationale": "test",
                            "suggestion": "fix",
                        },
                    ],
                },
            }
            path = Path(tmpdir) / "artifact.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            findings = extract_findings(path)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].id, "w001")

    def test_no_localization_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"status": "ok", "findings": [{"category": "gpu_idle"}]}
            path = Path(tmpdir) / "no_localization.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            findings = extract_findings(path)
            self.assertEqual(len(findings), 0)


class TestExtractFromPremadeFindingsJson(unittest.TestCase):
    """Test extraction from a pre-made list-of-findings JSON."""

    def test_list_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {
                    "id": "f001",
                    "category": "C1",
                    "title": "test",
                    "file": "a.py",
                    "function": "foo",
                    "line": 1,
                    "description": "desc",
                    "suggestion": "fix",
                    "evidence": [],
                    "priority": "high",
                },
            ]
            path = Path(tmpdir) / "findings.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            findings = extract_findings(path)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].id, "f001")


class TestWriteFindingsJson(unittest.TestCase):
    def test_write_and_read(self):
        findings = [
            Finding(id="f1", category="C1", title="t", file="a.py",
                    function="foo", line=1, description="d", suggestion="s",
                    evidence=[], priority="high"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_findings_json(findings, Path(tmpdir) / "out.json")
            self.assertTrue(out.exists())
            reloaded = extract_findings(out)
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded[0].id, "f1")


class TestAutoDiscovery(unittest.TestCase):
    def test_no_dirs_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(find_latest_analysis_output(tmpdir))
            self.assertIsNone(find_latest_nsys_artifact(tmpdir))

    def test_discovers_analysis_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = Path(tmpdir) / ".sysight" / "analysis-runs" / "run-001"
            runs.mkdir(parents=True)
            (runs / "last_message.txt").write_text('{"findings":[]}', encoding="utf-8")
            from sysight.shared.findings import find_latest_analysis_output
            result = find_latest_analysis_output(tmpdir)
            self.assertIsNotNone(result)

    def test_discovers_nsys_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nsys = Path(tmpdir) / ".sysight" / "nsys"
            nsys.mkdir(parents=True)
            (nsys / "profile.json").write_text('{}', encoding="utf-8")
            result = find_latest_nsys_artifact(tmpdir)
            self.assertIsNotNone(result)


class TestOptimizerResolveFindings(unittest.TestCase):
    """Test the optimizer's resolve_findings() entry point."""

    def test_resolve_from_findings_path(self):
        from sysight.optimizer.plan import resolve_findings

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a codex-style output
            data = {
                "findings": [
                    {
                        "category": "C7",
                        "title": "Python loop in hot path",
                        "file": "src/data/transforms.py",
                        "function": "StackC.image",
                        "line": 26,
                        "description": "per-column Python loop",
                        "suggestion": "vectorize",
                        "evidence": ["GPU idle"],
                        "priority": "high",
                    },
                ],
            }
            path = Path(tmpdir) / "findings.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            findings = resolve_findings(findings_path=str(path))
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].category, "C7")
            self.assertEqual(findings[0].file, "src/data/transforms.py")

    def test_resolve_auto_discover(self):
        import os
        from sysight.optimizer.plan import resolve_findings

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .sysight/analysis-runs/run-xxx/last_message.txt
            runs = Path(tmpdir) / ".sysight" / "analysis-runs" / "run-001"
            runs.mkdir(parents=True)
            data = {
                "findings": [
                    {
                        "category": "C1",
                        "title": "test",
                        "file": "a.py",
                        "function": "foo",
                        "line": 1,
                        "description": "desc",
                        "suggestion": "fix",
                        "evidence": [],
                        "priority": "high",
                    },
                ],
            }
            (runs / "last_message.txt").write_text(json.dumps(data), encoding="utf-8")

            # auto-discover uses cwd — must chdir into tmpdir
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                findings = resolve_findings()
            finally:
                os.chdir(orig)
            self.assertEqual(len(findings), 1)

    def test_resolve_returns_empty_when_nothing_found(self):
        import os
        from sysight.optimizer.plan import resolve_findings

        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                findings = resolve_findings()
            finally:
                os.chdir(orig)
            self.assertEqual(len(findings), 0)


if __name__ == "__main__":
    unittest.main()
