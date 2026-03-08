"""Match loaded skills to context and render prompt injection blocks."""

from __future__ import annotations

import re

from muse.skills.loader import Skill, SkillLoader

_DEFAULT_TOKEN_BUDGET = 4000
_INJECTION_HEADER = "\n\n--- DOMAIN KNOWLEDGE (from skills) ---\n"
_INJECTION_FOOTER = "\n--- END DOMAIN KNOWLEDGE ---\n"


def _matches(skill_values: list[str], target: str) -> bool:
    """Return ``True`` when a skill filter list matches the target string."""

    if "*" in skill_values:
        return True
    target_lower = target.lower().strip()
    target_compact = _compact_text(target)
    target_acronym = _acronym(target)
    for value in skill_values:
        value_lower = value.lower().strip()
        value_compact = _compact_text(value)
        value_acronym = _acronym(value)
        if value_lower == target_lower:
            return True
        if value_compact and target_compact and (
            value_compact in target_compact or target_compact in value_compact
        ):
            return True
        if value_acronym and target_compact and value_acronym == target_compact:
            return True
        if target_acronym and value_compact and target_acronym == value_compact:
            return True
        if value_acronym and target_acronym and value_acronym == target_acronym:
            return True
    return False


def _compact_text(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", text.lower()))


def _acronym(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) <= 1:
        return ""
    return "".join(word[0] for word in words if word)


class SkillRegistry:
    """Hold loaded skills and resolve them by stage/discipline/language."""

    def __init__(
        self,
        skills: list[Skill] | None = None,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self._skills = list(skills) if skills else []
        self._token_budget = token_budget

    @classmethod
    def from_loader(
        cls,
        loader: SkillLoader | None = None,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> "SkillRegistry":
        loader = loader or SkillLoader()
        return cls(skills=loader.load_all(), token_budget=token_budget)

    @property
    def all_skills(self) -> list[Skill]:
        return list(self._skills)

    def get_for_context(
        self,
        *,
        stage: str = "*",
        discipline: str = "*",
        language: str = "*",
    ) -> list[Skill]:
        matched: list[Skill] = []
        for skill in self._skills:
            if (
                _matches(skill.stages, stage)
                and _matches(skill.disciplines, discipline)
                and _matches(skill.languages, language)
            ):
                matched.append(skill)
        matched.sort(key=lambda skill: skill.priority, reverse=True)
        return matched

    def render_for_prompt(
        self,
        *,
        stage: str = "*",
        discipline: str = "*",
        language: str = "*",
    ) -> str:
        matched = self.get_for_context(
            stage=stage,
            discipline=discipline,
            language=language,
        )
        if not matched:
            return ""

        parts: list[str] = []
        remaining = self._token_budget
        for skill in matched:
            cost = skill.token_estimate
            if cost > remaining:
                if remaining >= 100:
                    chars = remaining * 4
                    truncated_body = skill.body[:chars].rsplit("\n", 1)[0]
                    parts.append(f"### {skill.name}\n{truncated_body}\n[truncated]")
                break
            parts.append(f"### {skill.name}\n{skill.body}")
            remaining -= cost

        if not parts:
            return ""
        return _INJECTION_HEADER + "\n\n".join(parts) + _INJECTION_FOOTER

    def inject_into_prompt(self, system_prompt: str, **context: str) -> str:
        block = self.render_for_prompt(**context)
        if not block:
            return system_prompt
        return system_prompt + block
