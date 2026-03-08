from __future__ import annotations

import json


def polish_prompt(language: str, format_standard: str, chapter_title: str, text: str) -> tuple[str, str]:
    system = (
        "Polish the academic thesis chapter for consistency and clarity. "
        "Do not alter core claims. Return JSON with keys: final_text, polish_notes (list)."
    )
    user = json.dumps(
        {
            "language": language,
            "format_standard": format_standard,
            "chapter_title": chapter_title,
            "text": text,
        },
        ensure_ascii=False,
    )
    return system, user
