"""Graph-native chapter result schema."""

from __future__ import annotations

from typing import Any, TypedDict


class ChapterResult(TypedDict):
    chapter_id: str
    chapter_title: str
    merged_text: str
    quality_scores: dict[str, Any]
    iterations_used: int
