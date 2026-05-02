"""Tests for sysight.wiki.promotion — CandidateValidator."""

import unittest

from sysight.wiki.promotion import (
    LearningCandidate,
    PromotionDecision,
    CandidateValidator,
)


class TestCandidateValidator(unittest.TestCase):
    def test_valid_global_experience(self):
        c = LearningCandidate(
            candidate_id="cand-001",
            source_run_id="run-001",
            kind="experience",
            title="D2H Sync Pattern",
            content="Check .item() calls that cause D2H sync",
            evidence_refs=["run-001:ev_001"],
            scope="global",
        )
        valid, issues = CandidateValidator.validate(c)
        self.assertTrue(valid, msg="; ".join(issues))

    def test_benchmark_cannot_promote_to_global(self):
        c = LearningCandidate(
            candidate_id="cand-002",
            kind="experience",
            title="Case-specific hint",
            content="For case_5, always check line 42",
            evidence_refs=["ev_001"],
            scope="benchmark",
        )
        valid, issues = CandidateValidator.validate(c)
        self.assertFalse(valid)

    def test_empty_content_rejected(self):
        c = LearningCandidate(
            candidate_id="cand-003",
            kind="memory",
            content="",
            scope="workspace",
        )
        valid, _ = CandidateValidator.validate(c)
        self.assertFalse(valid)

    def test_invalid_scope_rejected(self):
        c = LearningCandidate(
            candidate_id="cand-004",
            kind="memory",
            content="some content",
            scope="unsafe",
        )
        valid, _ = CandidateValidator.validate(c)
        self.assertFalse(valid)

    def test_invalid_kind_rejected(self):
        c = LearningCandidate(
            candidate_id="cand-005",
            kind="invalid_kind",
            content="content",
            scope="workspace",
        )
        valid, _ = CandidateValidator.validate(c)
        self.assertFalse(valid)

    def test_memory_without_evidence_warns(self):
        c = LearningCandidate(
            candidate_id="cand-006",
            kind="memory",
            content="Pattern description",
            scope="workspace",
            evidence_refs=[],
        )
        valid, issues = CandidateValidator.validate(c)
        self.assertFalse(valid)
        self.assertTrue(any("evidence" in i.lower() for i in issues))

    def test_promote_valid_candidate(self):
        c = LearningCandidate(
            candidate_id="cand-007",
            kind="experience",
            title="NCCL Fusion",
            content="Batch small all-reduces",
            evidence_refs=["run-001:ev_002"],
            scope="global",
        )
        decision = CandidateValidator.promote(c, reviewer="test")
        self.assertEqual(decision.decision, "promoted")
        self.assertEqual(decision.target_tier, "experience")

    def test_promote_invalid_candidate(self):
        c = LearningCandidate(
            candidate_id="cand-008",
            kind="experience",
            content="benchmark-only hint",
            evidence_refs=["ev_001"],
            scope="benchmark",
        )
        decision = CandidateValidator.promote(c, reviewer="test")
        self.assertEqual(decision.decision, "rejected")

    def test_promotion_routing_by_kind(self):
        for kind, tier in [
            ("memory", "active_memory"),
            ("experience", "experience"),
            ("skill", "skill"),
            ("detector", "detector"),
            ("plugin", "plugin"),
        ]:
            with self.subTest(kind=kind):
                c = LearningCandidate(
                    candidate_id=f"cand-{kind}",
                    kind=kind,
                    content="content",
                    evidence_refs=["ev_001"],
                    scope="workspace",
                )
                d = CandidateValidator.promote(c)
                self.assertEqual(d.target_tier, tier)


if __name__ == "__main__":
    unittest.main()
