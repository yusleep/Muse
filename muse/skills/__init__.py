"""Skills system for injecting domain knowledge into LLM prompts."""

from .loader import Skill, SkillLoader
from .registry import SkillRegistry

__all__ = ["Skill", "SkillLoader", "SkillRegistry"]
