from __future__ import annotations

import json


def topic_analysis_prompt(topic: str, discipline: str, lit_summary: str) -> tuple[str, str]:
    system = (
        "Analyze this research topic and return JSON with keys: "
        "research_gaps (list), core_concepts (list), methodology_domain (string), "
        "suggested_contributions (list). Be specific to the discipline."
    )
    user = json.dumps(
        {
            "topic": topic,
            "discipline": discipline,
            "literature_summary": lit_summary,
        },
        ensure_ascii=False,
    )
    return system, user
