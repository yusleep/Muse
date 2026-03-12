"""State schema for the LangGraph-native Muse flow."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


def _merge_dict(current: dict | None, new: dict | None) -> dict:
    result = dict(current or {})
    if new:
        result.update(new)
    return result


class MuseState(TypedDict, total=False):
    project_id: str
    topic: str
    discipline: str
    language: str
    format_standard: str
    output_format: str

    references: Annotated[list[dict[str, Any]], operator.add]
    search_queries: list[str]
    literature_summary: str

    outline: dict[str, Any]
    chapter_plans: list[dict[str, Any]]

    chapters: Annotated[dict[str, Any], _merge_dict]
    citation_uses: Annotated[list[dict[str, Any]], operator.add]
    citation_ledger: Annotated[dict[str, Any], _merge_dict]
    claim_text_by_id: Annotated[dict[str, str], _merge_dict]
    thesis_summary: str

    verified_citations: list[str]
    flagged_citations: list[dict[str, Any]]

    paper_package: Annotated[dict[str, Any], _merge_dict]
    final_text: str
    polish_notes: Annotated[list[str], operator.add]
    abstract_zh: str
    abstract_en: str
    keywords_zh: list[str]
    keywords_en: list[str]
    output_filepath: str
    export_artifacts: dict[str, Any]
    export_warnings: list[str]

    review_feedback: Annotated[list[dict[str, Any]], operator.add]
    rag_enabled: bool
    local_refs_count: int
