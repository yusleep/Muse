from __future__ import annotations

import json


def chapter_review_prompt(chapter_title: str, merged_text: str) -> tuple[str, str]:
    system = (
        "You are a strict thesis reviewer. Return JSON with keys: scores (object) and review_notes (list). "
        "scores keys: coherence, logic, citation, term_consistency, balance, redundancy; values 1-5."
    )
    user = json.dumps({"chapter_title": chapter_title, "text": merged_text}, ensure_ascii=False)
    return system, user
