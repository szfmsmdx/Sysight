"""Tests for sysight.wiki.brief — build_memory_brief."""

import tempfile
import unittest
from pathlib import Path

from sysight.wiki.store import WikiRepository
from sysight.wiki.brief import build_memory_brief


class TestBuildMemoryBrief(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / ".sysight" / "memory"
        self.repo = WikiRepository(root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_brief(self):
        brief = build_memory_brief(self.repo, namespace="nonexistent")
        self.assertIn("Memory Brief", brief)

    def test_brief_includes_overview(self):
        self.repo.write_page(
            "workspaces/ns1/overview.md",
            "# My Model\nEntry: train.py\nTests: pytest",
            title="My Model Overview",
        )
        brief = build_memory_brief(self.repo, namespace="ns1")
        self.assertIn("My Model", brief)

    def test_brief_includes_experiences(self):
        self.repo.write_page(
            "wiki/experiences/d2h-sync.md",
            "# D2H Sync",
            title="D2H Implicit Sync",
            category="C3",
            scope="global",
        )
        brief = build_memory_brief(self.repo)
        self.assertIn("D2H", brief)

    def test_brief_not_exceeds_200_lines(self):
        long_content = "\n".join(f"line {i}" for i in range(300))
        self.repo.write_page(
            "workspaces/ns1/overview.md",
            long_content,
            title="Long Overview",
        )
        brief = build_memory_brief(self.repo, namespace="ns1")
        lines = brief.split("\n")
        self.assertLessEqual(len(lines), 200)


if __name__ == "__main__":
    unittest.main()
