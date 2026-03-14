"""Prompts for chapter-level reference analysis."""

from __future__ import annotations

import json
from typing import Any


def ref_analysis_prompt(
    chapter_title: str,
    subtasks: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> tuple[str, str]:
    system = (
        "You are an academic literature analyst. For the given thesis chapter, select the most relevant "
        "references and explain how they should be cited.\n"
        "Return JSON with keys: key_references (list) and evidence_gaps (list).\n"
        "Each key_references item must include: ref_id, relevance, key_finding, how_to_cite.\n"
        "Use only ref_id values from the provided references."
    )
    user = json.dumps(
        {
            "chapter_title": chapter_title,
            "subtasks": [
                {
                    "title": str(subtask.get("title", "")),
                    "description": str(subtask.get("description", "")),
                }
                for subtask in subtasks
                if isinstance(subtask, dict)
            ],
            "references": references[:50],
        },
        ensure_ascii=False,
    )
    return system, user
