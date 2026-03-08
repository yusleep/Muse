"""Parse SKILL.md files into structured skill records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Skill:
    """A single loaded skill with metadata and body text."""

    name: str
    description: str
    body: str
    stages: list[str] = field(default_factory=lambda: ["*"])
    disciplines: list[str] = field(default_factory=lambda: ["*"])
    languages: list[str] = field(default_factory=lambda: ["*"])
    priority: int = 50
    source_path: str = ""

    @property
    def token_estimate(self) -> int:
        return len(self.body.encode("utf-8")) // 4


def _parse_skill_md(text: str, source_path: str = "") -> Skill | None:
    """Parse a SKILL.md file with YAML front matter and Markdown body."""

    text = text.strip()
    if not text.startswith("---"):
        return None

    end = text.find("---", 3)
    if end == -1:
        return None

    front_matter_raw = text[3:end].strip()
    body = text[end + 3 :].strip()

    try:
        metadata: dict[str, Any] = yaml.safe_load(front_matter_raw) or {}
    except yaml.YAMLError:
        return None

    if not isinstance(metadata, dict):
        return None

    name = str(metadata.get("name", "")).strip()
    if not name:
        return None

    applies_to = metadata.get("applies_to", {})
    if not isinstance(applies_to, dict):
        applies_to = {}

    stages = _normalize_filter_values(applies_to.get("stages", ["*"]))
    disciplines = _normalize_filter_values(applies_to.get("disciplines", ["*"]))
    languages = _normalize_filter_values(applies_to.get("languages", ["*"]))

    return Skill(
        name=name,
        description=str(metadata.get("description", "")),
        body=body,
        stages=stages,
        disciplines=disciplines,
        languages=languages,
        priority=int(metadata.get("priority", 50)),
        source_path=source_path,
    )


def _normalize_filter_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return ["*"]


class SkillLoader:
    """Scan directories for SKILL.md files and load parsed skills."""

    def __init__(self, dirs: list[str | Path] | None = None) -> None:
        if dirs is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            dirs = [
                project_root / "skills" / "public",
                project_root / "skills" / "custom",
            ]
        self._dirs = [Path(directory) for directory in dirs]

    def load_all(self) -> list[Skill]:
        skills: list[Skill] = []
        seen_names: set[str] = set()

        for scan_dir in self._dirs:
            if not scan_dir.is_dir():
                continue
            for skill_dir in sorted(scan_dir.iterdir()):
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.is_file():
                    continue
                try:
                    text = skill_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                skill = _parse_skill_md(text, source_path=str(skill_file))
                if skill is None:
                    continue
                if skill.name in seen_names:
                    skills = [existing for existing in skills if existing.name != skill.name]
                seen_names.add(skill.name)
                skills.append(skill)

        skills.sort(key=lambda skill: skill.priority, reverse=True)
        return skills
