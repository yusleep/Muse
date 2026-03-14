from __future__ import annotations

import json

from muse.prompts.chapter_review import review_boundary_for_lens, review_rubric_for_lens


def _base_global_system_prompt() -> str:
    return (
        "You are a strict thesis reviewer for the full merged thesis draft. "
        "Return JSON with keys: scores (object) and review_notes (list). "
        "scores values must be integers from 1-5. "
        'Use the review_notes item schema {"section": "...", "severity": 1-5, '
        '"instruction": "...", "lens": "...", "is_recurring": true/false}. '
        "Only include concrete revision instructions grounded in the merged draft."
    )


def global_review_prompt_for_lens(
    merged_text: str,
    lens: str,
) -> tuple[str, str]:
    system = "\n\n".join(
        [
            _base_global_system_prompt(),
            "Review the full merged thesis draft rather than a single chapter.",
            f"Primary review lens: {lens}",
            review_rubric_for_lens(lens),
            f"When you emit review_notes, set lens to '{lens}'.",
            review_boundary_for_lens(lens),
        ]
    )
    user = json.dumps({"text": merged_text}, ensure_ascii=False)
    return system, user
