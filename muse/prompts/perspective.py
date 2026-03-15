"""Prompts for Phase 5 perspective discovery."""

from __future__ import annotations

import json
from typing import Any


def perspective_personas_prompt(
    topic: str,
    discipline: str,
    references: list[dict[str, Any]],
) -> tuple[str, str]:
    system = (
        "Generate 3-5 expert personas for a thesis literature review.\n"
        "Return JSON with key: personas (list).\n"
        "Each persona must include: name, expertise, focus_area.\n"
        "Keep personas complementary and grounded in the provided references."
    )
    user = json.dumps(
        {
            "topic": topic,
            "discipline": discipline,
            "references": references[:12],
        },
        ensure_ascii=False,
    )
    return system, user


def perspective_dialogues_prompt(
    topic: str,
    discipline: str,
    personas: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> tuple[str, str]:
    system = (
        "Simulate short pairwise dialogues between the selected reviewers.\n"
        "Return JSON with keys: dialogues (list), research_questions (list), search_queries (list).\n"
        "Each dialogue item should summarize one pair and surface distinct follow-up research questions.\n"
        "Focus on gaps, tensions, and under-explored validation angles."
    )
    user = json.dumps(
        {
            "topic": topic,
            "discipline": discipline,
            "personas": personas,
            "references": references[:12],
        },
        ensure_ascii=False,
    )
    return system, user
