from __future__ import annotations

import json
from typing import Any


def outline_gen_prompt(
    topic: str,
    discipline: str,
    language: str,
    lit_summary: str,
    topic_analysis: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "Generate a thesis outline as JSON with keys: chapters (list). Each chapter must include "
        "chapter_id, chapter_title, target_words, complexity, subsections (list of {title}). "
        "Use the topic_analysis to create a discipline-specific, non-generic structure. "
        "For CS/systems topics include: background, related work, system design, evaluation, conclusion. "
        "For social science topics include: literature review, theory, methods, findings, discussion."
    )
    user = json.dumps(
        {
            "topic": topic,
            "discipline": discipline,
            "language": language,
            "literature_summary": lit_summary,
            "topic_analysis": topic_analysis,
        },
        ensure_ascii=False,
    )
    return system, user
