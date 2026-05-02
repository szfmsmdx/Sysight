"""Gated promotion for learning candidates.

A lesson may move between knowledge tiers only through explicit gates.
Benchmark-only hints must never be promoted to global experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LearningCandidate:
    candidate_id: str
    source_run_id: str = ""
    kind: str = ""                # "memory" | "experience" | "skill" | "detector" | "prompt"
    title: str = ""
    content: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    scope: str = "workspace"       # "workspace" | "global" | "benchmark" | "unsafe"
    expected_benefit: str = ""
    risks: list[str] = field(default_factory=list)
    proposed_tests: list[str] = field(default_factory=list)


@dataclass
class PromotionDecision:
    candidate_id: str
    decision: str                 # "promoted" | "rejected" | "quarantined"
    target_tier: str = ""         # "active_memory" | "experience" | "skill" | "detector" | "plugin"
    reviewer: str = ""
    reason: str = ""
    created_at: str = ""


class CandidateValidator:
    """Validates and promotes learning candidates through explicit gates."""

    @staticmethod
    def validate(candidate: LearningCandidate) -> tuple[bool, list[str]]:
        """Validate a candidate against promotion gates. Returns (valid, issues)."""
        issues: list[str] = []

        # Gate 1: No benchmark-only rules in global scope
        if candidate.scope == "benchmark" and not candidate.kind == "prompt":
            issues.append("benchmark-scope candidates cannot be promoted to global")
            return False, issues

        # Gate 2: Scope must be valid
        if candidate.scope not in ("workspace", "global", "benchmark"):
            issues.append(f"invalid scope: {candidate.scope}")
            return False, issues

        # Gate 3: Must have content
        if not candidate.content.strip():
            issues.append("empty content")
            return False, issues

        # Gate 4: Must have evidence refs (except for skills/plugins)
        if candidate.kind in ("memory", "experience") and not candidate.evidence_refs:
            issues.append("memory/experience candidates require evidence refs")

        # Gate 5: Kind must be valid
        valid_kinds = {"memory", "experience", "skill", "detector", "plugin", "prompt"}
        if candidate.kind not in valid_kinds:
            issues.append(f"invalid kind: {candidate.kind}")
            return False, issues

        return len(issues) == 0, issues

    @staticmethod
    def promote(candidate: LearningCandidate, reviewer: str = "system") -> PromotionDecision:
        """Promote a validated candidate to the appropriate tier."""
        valid, issues = CandidateValidator.validate(candidate)
        if not valid:
            return PromotionDecision(
                candidate_id=candidate.candidate_id,
                decision="rejected",
                reason="; ".join(issues),
                reviewer=reviewer,
            )

        # Route to appropriate tier based on kind
        tier_map = {
            "memory": "active_memory",
            "experience": "experience",
            "skill": "skill",
            "detector": "detector",
            "plugin": "plugin",
            "prompt": "prompt",
        }

        return PromotionDecision(
            candidate_id=candidate.candidate_id,
            decision="promoted",
            target_tier=tier_map.get(candidate.kind, "experience"),
            reviewer=reviewer,
            reason=f"passed all gates for {candidate.kind}",
        )
