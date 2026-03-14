from __future__ import annotations

import json
from typing import Any


_PERSONA_CONFIG: dict[str, dict[str, Any]] = {
    "logic": {
        "title": "argument and structure reviewer",
        "dimensions": ("logic", "structure", "balance"),
        "avoid": "citation coverage, terminology consistency, and sentence-level polish",
    },
    "citation": {
        "title": "scholarship and citation reviewer",
        "dimensions": ("citation", "coverage", "depth"),
        "avoid": "argument ordering, terminology consistency, and sentence-level polish",
    },
    "readability": {
        "title": "readability reviewer",
        "dimensions": ("style", "term_consistency", "redundancy"),
        "avoid": "evidence sufficiency and high-level argument structure",
    },
}


def persona_dimensions(persona: str) -> tuple[str, ...]:
    config = _PERSONA_CONFIG.get(persona)
    if not config:
        raise ValueError(f"Unsupported reviewer persona: {persona}")
    return tuple(config["dimensions"])


def _history_context(review_history: list[dict[str, Any]], iteration: int) -> str:
    if not review_history:
        return ""
    previous = review_history[-1]
    previous_iteration = previous.get("iteration", "?")
    previous_scores = previous.get("scores", {})
    if not isinstance(previous_scores, dict):
        previous_scores = {}
    previous_notes = str(previous.get("notes_summary", "")).strip() or "No prior notes recorded."
    return "\n\n".join(
        [
            f"Previous review round (iteration {previous_iteration})",
            f"Previous scores: {json.dumps(previous_scores, ensure_ascii=False)}",
            f"Previous key issues: {previous_notes}",
            f"Current review iteration: {iteration}",
            "Check whether the previous issues are fixed before you issue new critique.",
            "Set is_recurring to true if an older issue is still unresolved.",
        ]
    )


def reviewer_persona_prompt(
    persona: str,
    *,
    merged_text: str,
    review_history: list[dict[str, Any]] | None = None,
    iteration: int = 1,
) -> tuple[str, str]:
    config = _PERSONA_CONFIG.get(persona)
    if not config:
        raise ValueError(f"Unsupported reviewer persona: {persona}")

    dimensions = ", ".join(config["dimensions"])
    score_schema = ", ".join(f'"{dimension}": N' for dimension in config["dimensions"])
    history_context = _history_context(review_history or [], iteration)
    system = "\n\n".join(
        part
        for part in (
            f"You are the {config['title']} for a full merged thesis draft.",
            f"Review only these dimensions: {dimensions}.",
            f"You must not output scores for any dimension outside: {dimensions}.",
            f"Do not focus on {config['avoid']}.",
            f'Return JSON like {{"scores": {{{score_schema}}}, "review_notes": [{{"section": "...", "severity": 1-5, "instruction": "...", "lens": "{persona}", "is_recurring": true/false}}]}}.',
            history_context,
        )
        if part
    )
    user = json.dumps({"text": merged_text}, ensure_ascii=False)
    return system, user
