"""Merge chapter outputs back into the thesis-level graph state."""

from __future__ import annotations

from typing import Any


def _build_thesis_summary(chapter_results: list[dict[str, Any]]) -> str:
    lines = []
    for chapter in chapter_results:
        excerpt = " ".join(str(chapter.get("merged_text", "")).split()[:80])
        lines.append(f"[{chapter.get('chapter_title', '')}] {excerpt}".strip())
    return "\n\n".join(lines)


def build_merge_chapters_node(settings: Any, services: Any):
    def merge_chapters(state: dict[str, Any]) -> dict[str, Any]:
        chapter_lookup = state.get("chapters", {}) or {}
        chapter_results = [
            chapter_lookup[plan["chapter_id"]]
            for plan in state.get("chapter_plans", [])
            if plan.get("chapter_id") in chapter_lookup
        ]
        citation_uses: list[dict[str, Any]] = []
        for chapter in chapter_results:
            citation_uses.extend(chapter.get("citation_uses", []))

        thesis_summary = _build_thesis_summary(chapter_results)
        return {
            "paper_package": {
                "chapter_results": chapter_results,
                "thesis_summary": thesis_summary,
            },
            "citation_uses": citation_uses,
            "thesis_summary": thesis_summary,
            "final_text": "\n\n".join(ch.get("merged_text", "") for ch in chapter_results if ch.get("merged_text")),
        }

    return merge_chapters
