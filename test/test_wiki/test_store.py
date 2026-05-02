"""Tests for sysight.wiki.store — WikiRepository."""

import tempfile
import unittest
from pathlib import Path

from sysight.wiki.store import WikiRepository


class TestWikiRepository(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / ".sysight" / "memory"
        self.repo = WikiRepository(root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_read_page(self):
        self.repo.write_page(
            "workspaces/test_ns/overview.md",
            "# Test Repo\nEntry: train.py",
            title="Test Repo Overview",
        )
        content = self.repo.read_page("workspaces/test_ns/overview.md")
        self.assertIsNotNone(content)
        self.assertIn("# Test Repo", content)
        self.assertIn("title: Test Repo Overview", content)

    def test_read_nonexistent(self):
        self.assertIsNone(self.repo.read_page("nonexistent.md"))

    def test_write_page_creates_parent_dirs(self):
        self.repo.write_page("wiki/deep/nested/page.md", "content", title="Deep")
        self.assertTrue((self.root / "wiki" / "deep" / "nested" / "page.md").exists())

    def test_append_worklog(self):
        self.repo.append_worklog("test_ns", "Fixed loader bug, +10% throughput")
        wl = self.repo.read_page("workspaces/test_ns/worklog.md")
        self.assertIsNotNone(wl)
        self.assertIn("Fixed loader bug", wl)

    def test_list_experiences(self):
        self.repo.write_page(
            "wiki/experiences/d2h-sync.md",
            "# D2H Sync\nCheck .item() calls",
            title="D2H Implicit Sync",
            category="C3",
            tags=["d2h", "sync"],
            scope="global",
        )
        self.repo.write_page(
            "wiki/experiences/nccl-fusion.md",
            "# NCCL Fusion\nBatch small all-reduces",
            title="NCCL Fusion",
            category="C6",
            scope="global",
        )
        exps = self.repo.list_experiences()
        self.assertEqual(len(exps), 2)

    def test_list_experiences_filtered(self):
        self.repo.write_page(
            "wiki/experiences/d2h-sync.md",
            "content",
            title="D2H",
            category="C3",
            scope="global",
        )
        exps = self.repo.list_experiences(category="C3")
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0]["category"], "C3")

    def test_workspace_namespace_from_name(self):
        ns = self.repo.workspace_namespace(repo_root="/home/user/my-model")
        self.assertEqual(ns, "my-model")

    def test_workspace_namespace_explicit(self):
        ns = self.repo.workspace_namespace(namespace="bench/case_1")
        self.assertEqual(ns, "bench/case_1")

    def test_path_containment(self):
        with self.assertRaises(ValueError):
            self.repo._resolve_path("../../../etc/passwd")


if __name__ == "__main__":
    unittest.main()
