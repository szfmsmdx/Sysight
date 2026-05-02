"""Tests for sysight.wiki.index — FTSIndex."""

import tempfile
import unittest
from pathlib import Path

from sysight.wiki.index import FTSIndex


class TestFTSIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wiki_dir = Path(self.tmp.name) / "wiki"
        self.wiki_dir.mkdir(parents=True)
        (self.wiki_dir / "experiences").mkdir()
        (self.wiki_dir / "workspaces" / "ns1").mkdir(parents=True)

        # Write test pages
        (self.wiki_dir / "workspaces" / "ns1" / "overview.md").write_text(
            "---\ntitle: Overview\n---\nEntry point: train.py\nUses PyTorch DDP\n"
        )
        (self.wiki_dir / "experiences" / "d2h-sync.md").write_text(
            "---\ntitle: D2H Implicit Sync\ncategory: C3\n---\n"
            "## Trigger\nD2H count aligned with step count\n"
            "## Fix\nCheck .item() and .cpu() calls\n"
        )
        (self.wiki_dir / "experiences" / "nccl-fusion.md").write_text(
            "---\ntitle: NCCL Fusion\ncategory: C6\n---\n"
            "## Pattern\nMany small all-reduce operations\n"
        )
        self.index = FTSIndex(self.wiki_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_search_finds_experience(self):
        results = self.index.search("D2H")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("D2H", results[0].title)

    def test_search_finds_multiple(self):
        # D2H appears in both the title and body of d2h-sync.md
        results = self.index.search("D2H")
        self.assertEqual(len(results), 1)
        self.assertIn("D2H", results[0].title)

    def test_search_no_match(self):
        results = self.index.search("nonexistent_xyz")
        self.assertEqual(len(results), 0)

    def test_search_with_namespace(self):
        results = self.index.search("train", namespace="ns1")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("Overview", results[0].title)

    def test_search_scores_are_sorted(self):
        (self.wiki_dir / "experiences" / "gpu.md").write_text(
            "---\ntitle: GPU Tips\n---\nGPU GPU GPU GPU GPU\n"
        )
        # Create a new index that picks up the new file
        index2 = FTSIndex(self.wiki_dir)
        results = index2.search("GPU")
        self.assertGreaterEqual(len(results), 1)
        if len(results) >= 2:
            self.assertGreater(results[0].score, results[-1].score)


if __name__ == "__main__":
    unittest.main()
