"""Run-state schema helpers for Muse."""

from __future__ import annotations

from typing import Any, TypedDict

from .reference import CitationUse, FlaggedCitation, ReferenceRecord


class ThesisState(TypedDict):
    project_id: str
    topic: str
    discipline: str
    language: str
    format_standard: str
    current_stage: int
    outline_json: dict[str, Any]
    chapter_plans: list[dict[str, Any]]
    chapter_results: list[dict[str, Any]]
    current_chapter_index: int
    consistency_data: dict[str, Any]
    reflection_data: dict[str, Any]
    references: list[ReferenceRecord]
    search_queries: list[str]
    literature_summary: str
    quality_scores: dict[str, Any]
    review_notes: list[dict[str, Any]]
    review_history: list[dict[str, Any]]
    review_iteration: int
    structural_iterations: int
    content_iterations: int
    line_iterations: int
    review_layer: str
    coherence_issues: list[dict[str, Any]]
    citation_uses: list[CitationUse]
    claim_text_by_id: dict[str, str]
    verified_citations: list[str]
    flagged_citations: list[FlaggedCitation]
    _citation_repair_attempted: bool
    terminology_glossary: dict[str, str]
    thesis_summary: str
    polish_notes: list[str]
    final_text: str
    output_format: str
    output_filepath: str
    local_refs_count: int
    rag_enabled: bool
    stage1_status: str
    stage2_status: str
    stage3_status: str
    stage4_status: str
    stage5_status: str
    stage6_status: str
    hitl_feedback: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]


_REQUIRED_KEYS = {
    "project_id",
    "topic",
    "discipline",
    "language",
    "format_standard",
    "current_stage",
    "outline_json",
    "chapter_plans",
    "chapter_results",
    "current_chapter_index",
    "consistency_data",
    "reflection_data",
    "references",
    "search_queries",
    "literature_summary",
    "citation_uses",
    "claim_text_by_id",
    "verified_citations",
    "flagged_citations",
    "_citation_repair_attempted",
    "terminology_glossary",
    "thesis_summary",
    "polish_notes",
    "final_text",
    "output_format",
    "output_filepath",
    "stage1_status",
    "stage2_status",
    "stage3_status",
    "stage4_status",
    "stage5_status",
    "stage6_status",
    "hitl_feedback",
    "audit_events",
}


_DEFAULT_OPTIONAL_FIELDS: dict[str, Any] = {
    "search_queries": [],
    "literature_summary": "",
    "current_chapter_index": 0,
    "consistency_data": {},
    "reflection_data": {},
    "quality_scores": {},
    "review_notes": [],
    "review_history": [],
    "review_iteration": 1,
    "structural_iterations": 0,
    "content_iterations": 0,
    "line_iterations": 0,
    "review_layer": "",
    "coherence_issues": [],
    "citation_uses": [],
    "claim_text_by_id": {},
    "verified_citations": [],
    "flagged_citations": [],
    "_citation_repair_attempted": False,
    "terminology_glossary": {},
    "thesis_summary": "",
    "polish_notes": [],
    "final_text": "",
    "output_format": "markdown",
    "output_filepath": "",
    "export_artifacts": {},
    "export_warnings": [],
    "local_refs_count": 0,
    "rag_enabled": False,
    "abstract_zh": "",
    "keywords_zh": [],
    "abstract_en": "",
    "keywords_en": [],
    "stage1_status": "pending",
    "stage2_status": "pending",
    "stage3_status": "pending",
    "stage4_status": "pending",
    "stage5_status": "pending",
    "stage6_status": "pending",
    "hitl_feedback": [],
    "audit_events": [],
}


def new_thesis_state(
    project_id: str,
    topic: str,
    discipline: str,
    language: str,
    format_standard: str,
) -> ThesisState:
    return ThesisState(
        project_id=project_id,
        topic=topic,
        discipline=discipline,
        language=language,
        format_standard=format_standard,
        current_stage=1,
        outline_json={},
        chapter_plans=[],
        chapter_results=[],
        current_chapter_index=0,
        consistency_data={},
        reflection_data={},
        references=[],
        search_queries=[],
        literature_summary="",
        quality_scores={},
        review_notes=[],
        review_history=[],
        review_iteration=1,
        structural_iterations=0,
        content_iterations=0,
        line_iterations=0,
        review_layer="",
        coherence_issues=[],
        citation_uses=[],
        claim_text_by_id={},
        verified_citations=[],
        flagged_citations=[],
        _citation_repair_attempted=False,
        terminology_glossary={},
        thesis_summary="",
        polish_notes=[],
        final_text="",
        output_format="markdown",
        output_filepath="",
        export_artifacts={},
        export_warnings=[],
        stage1_status="pending",
        stage2_status="pending",
        stage3_status="pending",
        stage4_status="pending",
        stage5_status="pending",
        stage6_status="pending",
        hitl_feedback=[],
        audit_events=[],
    )


def hydrate_thesis_state(state: dict[str, Any]) -> dict[str, Any]:
    for key, default in _DEFAULT_OPTIONAL_FIELDS.items():
        state.setdefault(key, default.copy() if isinstance(default, (dict, list)) else default)
    return state


def validate_thesis_state(state: dict[str, Any]) -> None:
    missing = _REQUIRED_KEYS - set(state.keys())
    if missing:
        raise ValueError(f"missing required state keys: {sorted(missing)}")

    flagged = state.get("flagged_citations")
    if not isinstance(flagged, list):
        raise ValueError("flagged_citations must be a list")

    for idx, item in enumerate(flagged):
        if not isinstance(item, dict):
            raise ValueError(f"flagged_citations[{idx}] must be a dict")
        if "cite_key" not in item or "reason" not in item:
            raise ValueError(f"flagged_citations[{idx}] missing cite_key/reason")
