"""Cross-chapter terminology and citation consistency helpers."""

from __future__ import annotations

from typing import Any


def _excerpt(text: str, limit: int = 80) -> str:
    words = str(text or "").split()
    return " ".join(words[:limit]).strip()


class ConsistencyStore:
    def __init__(
        self,
        *,
        glossary: dict[str, str] | None = None,
        citation_counts: dict[str, int] | None = None,
        notation: dict[str, str] | None = None,
        chapter_summaries: dict[str, str] | None = None,
    ) -> None:
        self.glossary = dict(glossary or {})
        self.citation_counts = {
            str(key): int(value)
            for key, value in (citation_counts or {}).items()
            if str(key).strip()
        }
        self.notation = dict(notation or {})
        self.chapter_summaries = dict(chapter_summaries or {})

    def update_from_chapter(self, chapter_result: dict[str, Any]) -> None:
        if not isinstance(chapter_result, dict):
            return

        chapter_id = str(chapter_result.get("chapter_id", "")).strip()
        merged_text = str(chapter_result.get("merged_text", "")).strip()
        if chapter_id and merged_text:
            self.chapter_summaries[chapter_id] = _excerpt(merged_text)

        subtask_results = chapter_result.get("subtask_results", [])
        if not isinstance(subtask_results, list):
            subtask_results = []

        for subtask in subtask_results:
            if not isinstance(subtask, dict):
                continue

            glossary_additions = subtask.get("glossary_additions", {})
            if isinstance(glossary_additions, dict):
                for term, normalized in glossary_additions.items():
                    term_text = str(term).strip()
                    normalized_text = str(normalized).strip()
                    if term_text and normalized_text:
                        self.glossary[term_text] = normalized_text

            citations_used = subtask.get("citations_used", [])
            if isinstance(citations_used, list):
                for cite_key in citations_used:
                    cite_key_text = str(cite_key).strip()
                    if not cite_key_text:
                        continue
                    self.citation_counts[cite_key_text] = self.citation_counts.get(cite_key_text, 0) + 1

    def get_context_for_draft(self) -> dict[str, Any]:
        frequently_cited = [
            {"ref_id": ref_id, "count": count}
            for ref_id, count in sorted(
                self.citation_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:10]
        ]
        return {
            "glossary": dict(self.glossary),
            "citation_counts": dict(self.citation_counts),
            "frequently_cited": frequently_cited,
            "notation": dict(self.notation),
            "chapter_summaries": dict(self.chapter_summaries),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "glossary": dict(self.glossary),
            "citation_counts": dict(self.citation_counts),
            "notation": dict(self.notation),
            "chapter_summaries": dict(self.chapter_summaries),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ConsistencyStore":
        if not isinstance(data, dict):
            return cls()
        return cls(
            glossary=data.get("glossary", {}),
            citation_counts=data.get("citation_counts", {}),
            notation=data.get("notation", {}),
            chapter_summaries=data.get("chapter_summaries", {}),
        )
