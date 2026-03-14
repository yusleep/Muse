"""Draft nodes and fan-out helpers for chapter writing."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import Send

from muse.graph.helpers.draft_support import write_subtasks

_log = logging.getLogger("muse.draft")


def build_chapter_draft_node(services: Any):
    def chapter_draft(state: dict[str, Any]) -> dict[str, Any]:
        chapter_plan = state.get("chapter_plan", {})
        ch_id = chapter_plan.get("chapter_id", "?")
        ch_title = chapter_plan.get("chapter_title", "?")
        n_subtasks = len(chapter_plan.get("subtask_plan", []))
        _log.info("[chapter %s] writing '%s' (%d subtasks)", ch_id, ch_title[:40], n_subtasks)
        subtask_results = write_subtasks(
            llm_client=getattr(services, "llm", None),
            state={
                "topic": state.get("topic", ""),
                "language": state.get("language", "zh"),
                "references": state.get("references", []),
            },
            chapter_title=chapter_plan.get("chapter_title", ""),
            subtask_plan=chapter_plan.get("subtask_plan", []),
            revision_instructions=state.get("revision_instructions", {}),
            previous=state.get("subtask_results", []),
            rag_index=getattr(services, "rag_index", None),
        )

        merged_text = "\n\n".join(item.get("output_text", "") for item in subtask_results)
        citation_uses: list[dict[str, Any]] = []
        claim_text_by_id: dict[str, str] = {}
        chapter_id = chapter_plan.get("chapter_id", "chapter")

        for subtask in subtask_results:
            for claim_index, claim in enumerate(subtask.get("key_claims", []), start=1):
                claim_id = f"{chapter_id}_{subtask['subtask_id']}_c{claim_index:02d}"
                claim_text_by_id[claim_id] = claim
                for cite_key in subtask.get("citations_used", []):
                    citation_uses.append(
                        {
                            "cite_key": cite_key,
                            "claim_id": claim_id,
                            "chapter_id": chapter_id,
                            "subtask_id": subtask["subtask_id"],
                        }
                    )

        return {
            "subtask_results": subtask_results,
            "merged_text": merged_text,
            "iteration": int(state.get("iteration", 0)) + 1,
            "citation_uses": citation_uses,
            "claim_text_by_id": claim_text_by_id,
        }

    return chapter_draft


def fan_out_chapters(state: dict[str, Any]) -> list[Send]:
    references = state.get("references", [])
    topic = state.get("topic", "")
    discipline = state.get("discipline", "")
    language = state.get("language", "zh")
    return [
        Send(
            "chapter_subgraph",
            {
                "chapter_plan": plan,
                "references": references,
                "topic": topic,
                "discipline": discipline,
                "language": language,
                "subtask_results": [],
                "merged_text": "",
                "quality_scores": {},
                "review_notes": [],
                "revision_instructions": {},
                "iteration": 0,
                "max_iterations": 3,
                "citation_uses": [],
                "claim_text_by_id": {},
            },
        )
        for plan in state.get("chapter_plans", [])
    ]
