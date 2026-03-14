"""Initialize the graph-native Muse state."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_initialize_node(settings: Any, services: Any):
    def initialize(state: dict[str, Any]) -> dict[str, Any]:
        local_refs = list(getattr(services, "local_refs", []) or [])
        rag_index = getattr(services, "rag_index", None)
        result = {
            "project_id": state.get("project_id", ""),
            "topic": state.get("topic", ""),
            "discipline": state.get("discipline", "general"),
            "language": state.get("language", "zh"),
            "format_standard": state.get("format_standard", "GB/T 7714-2015"),
            "output_format": state.get("output_format", "markdown"),
            "references": local_refs,
            "search_queries": [],
            "literature_summary": state.get("literature_summary", ""),
            "outline": state.get("outline", {}),
            "chapter_plans": state.get("chapter_plans", []),
            "chapters": state.get("chapters", {}),
            "citation_uses": state.get("citation_uses", []),
            "citation_ledger": state.get("citation_ledger", {}),
            "claim_text_by_id": state.get("claim_text_by_id", {}),
            "thesis_summary": state.get("thesis_summary", ""),
            "verified_citations": state.get("verified_citations", []),
            "flagged_citations": state.get("flagged_citations", []),
            "paper_package": state.get("paper_package", {}),
            "final_text": state.get("final_text", ""),
            "polish_notes": state.get("polish_notes", []),
            "abstract_zh": state.get("abstract_zh", ""),
            "abstract_en": state.get("abstract_en", ""),
            "keywords_zh": state.get("keywords_zh", []),
            "keywords_en": state.get("keywords_en", []),
            "output_filepath": state.get("output_filepath", ""),
            "export_artifacts": state.get("export_artifacts", {}),
            "export_warnings": state.get("export_warnings", []),
            "review_feedback": state.get("review_feedback", []),
            "rag_enabled": rag_index is not None,
            "local_refs_count": len(local_refs),
        }
        paper_index = getattr(services, "paper_index", None)
        local_papers_dir = str(getattr(settings, "local_papers_dir", "") or "").strip()
        if paper_index is not None and local_papers_dir:
            try:
                indexed_papers = paper_index.ingest_local(Path(local_papers_dir))
            except Exception:  # noqa: BLE001
                indexed_papers = {}
            if indexed_papers:
                result["indexed_papers"] = indexed_papers
                result["paper_index_ready"] = True
        return result

    return initialize
