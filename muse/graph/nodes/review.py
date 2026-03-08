"""Review and interrupt nodes for graph-native execution."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from muse.graph.helpers.review_state import build_revision_instructions
from muse.prompts.chapter_review import chapter_review_prompt


_REVIEW_LENSES = ["logic", "style", "citation", "structure"]


def build_chapter_review_node(services: Any):
    def chapter_review(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        chapter_title = state.get("chapter_plan", {}).get("chapter_title", "")
        merged_text = state.get("merged_text", "")
        packets: list[dict[str, Any]] = []

        if llm is not None:
            for lens in _REVIEW_LENSES:
                system, user = chapter_review_prompt(chapter_title, merged_text)
                system = f"{system} Focus primarily on {lens}."
                try:
                    payload = llm.structured(system=system, user=user, route="review", max_tokens=1800)
                except Exception:
                    payload = {}
                if isinstance(payload, dict):
                    packets.append(payload)

        scores: dict[str, int] = {}
        review_notes: list[dict[str, Any]] = []
        for packet in packets:
            packet_scores = packet.get("scores", {})
            if isinstance(packet_scores, dict):
                for key, value in packet_scores.items():
                    if isinstance(value, (int, float)):
                        scores[key] = min(scores.get(key, int(value)), int(value)) if key in scores else int(value)
            packet_notes = packet.get("review_notes", [])
            if isinstance(packet_notes, list):
                review_notes.extend(note for note in packet_notes if isinstance(note, dict))

        return {
            "quality_scores": scores,
            "review_notes": review_notes,
            "revision_instructions": build_revision_instructions(review_notes, min_severity=2),
        }

    return chapter_review


def build_interrupt_node(stage: str, *, auto_approve: bool):
    def interrupt_node(state: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "stage": stage,
            "project_id": state.get("project_id"),
            "ref_count": len(state.get("references", [])),
            "chapter_count": len(state.get("chapter_plans", [])),
        }
        if auto_approve:
            feedback = {"stage": stage, "approved": True, "auto_approve": True}
        else:
            feedback = interrupt(payload)
            if not isinstance(feedback, dict):
                feedback = {"stage": stage, "approved": bool(feedback)}
        return {"review_feedback": [feedback]}

    return interrupt_node
