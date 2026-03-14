"""Prompt helpers for subsection argument planning."""

from __future__ import annotations

import json
from typing import Any


def argument_plan_prompt(
    subtask_title: str,
    subtask_description: str,
    reference_briefs: list[dict[str, Any]],
    *,
    language: str = "zh",
) -> tuple[str, str]:
    system = (
        "You are an academic argument-planning assistant. Build a structured argument plan before drafting.\n"
        "Return JSON with keys: core_claim, evidence_chain, logical_flow, paragraph_count.\n"
        "Each evidence_chain item must include claim, source, specific_finding.\n"
        "Use only source values that appear in reference_briefs."
    )
    user = json.dumps(
        {
            "subtask_title": subtask_title,
            "subtask_description": subtask_description,
            "reference_briefs": reference_briefs,
            "language": language,
        },
        ensure_ascii=False,
    )
    return system, user
