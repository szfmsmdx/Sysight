"""Skill registry — discover and load reusable procedures.

Skills are stored as SKILL.md files under .sysight/skills/<name>/.
Each skill has a manifest.json declaring triggers, permissions, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillManifest:
    name: str
    version: str = "0.1.0"
    kind: str = "skill"
    scope: str = "global"
    trust: str = "internal"
    triggers: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)


@dataclass
class Skill:
    manifest: SkillManifest
    body: str = ""
    path: str = ""


class SkillRegistry:
    """Discovers and loads skills from the filesystem."""

    def __init__(self, skills_dir: str | Path | None = None):
        self._skills_dir = Path(skills_dir) if skills_dir else Path.cwd() / ".sysight" / "skills"
        self._skills: dict[str, Skill] = {}

    def discover(self) -> list[str]:
        """Discover all skill directories containing SKILL.md."""
        names = []
        if not self._skills_dir.is_dir():
            return names
        for d in sorted(self._skills_dir.iterdir()):
            if d.is_dir() and (d / "SKILL.md").exists():
                names.append(d.name)
        return names

    def load(self, name: str) -> Skill | None:
        """Load a skill by name."""
        skill_dir = self._skills_dir / name
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return None

        body = skill_file.read_text(encoding="utf-8")

        # Parse manifest if present
        manifest = SkillManifest(name=name)
        manifest_file = skill_dir / "manifest.json"
        if manifest_file.exists():
            import json
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                manifest = SkillManifest(
                    name=data.get("name", name),
                    version=data.get("version", "0.1.0"),
                    kind=data.get("kind", "skill"),
                    scope=data.get("scope", "global"),
                    trust=data.get("trust", "internal"),
                    triggers=data.get("triggers", []),
                    permissions=data.get("permissions", []),
                    tests=data.get("tests", []),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        skill = Skill(manifest=manifest, body=body, path=str(skill_file))
        self._skills[name] = skill
        return skill

    def get(self, name: str) -> Skill | None:
        """Get a loaded skill, loading if necessary."""
        if name not in self._skills:
            return self.load(name)
        return self._skills[name]
