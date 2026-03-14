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
    paper_index_ready: bool
    indexed_papers: Annotated[dict[str, dict[str, Any]], _merge_dict]

    outline: dict[str, Any]
    chapter_plans: list[dict[str, Any]]
    current_chapter_index: int
    consistency_data: Annotated[dict[str, Any], _merge_dict]
    reflection_data: Annotated[dict[str, Any], _merge_dict]
    chapter_plan: dict[str, Any]
    subtask_results: list[dict[str, Any]]
    merged_text: str
    iteration: int
    max_iterations: int

    chapters: Annotated[dict[str, Any], _merge_dict]
    quality_scores: dict[str, Any]
    review_notes: Annotated[list[dict[str, Any]], operator.add]
    review_history: Annotated[list[dict[str, Any]], operator.add]
    review_iteration: int
    structural_iterations: int
    content_iterations: int
    line_iterations: int
    review_layer: str
    coherence_issues: list[dict[str, Any]]
    revision_instructions: Annotated[dict[str, str], _merge_dict]
    citation_uses: Annotated[list[dict[str, Any]], operator.add]
    citation_ledger: Annotated[dict[str, Any], _merge_dict]
    claim_text_by_id: Annotated[dict[str, str], _merge_dict]
    thesis_summary: str

    verified_citations: list[str]
    flagged_citations: list[dict[str, Any]]
    _citation_repair_attempted: bool

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
