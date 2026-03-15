"""Prompts for best-effort visual validation."""

from __future__ import annotations

import json
from typing import Any


def visual_check_prompt(page_summaries: list[dict[str, Any]]) -> tuple[str, str]:
    system = (
        "You are a thesis PDF visual checker.\n"
        "Review the provided page summaries and return JSON with key issues (list).\n"
        "Each issue must include: page, type, description, fix_suggestion.\n"
        "Focus on float drift, overflow, missing glyphs, large blank areas, and reference formatting."
    )
    user = json.dumps({"pages": page_summaries}, ensure_ascii=False)
    return system, user
