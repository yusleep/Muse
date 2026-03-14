"""Review and interrupt nodes for graph-native execution."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from muse.graph.helpers.review_state import build_revision_instructions
from muse.prompts.adaptive_review import adaptive_review_prompt
from muse.prompts.chapter_review import chapter_review_prompt_for_lens
from muse.prompts.global_review import global_review_prompt_for_lens


_REVIEW_LENSES = ["logic", "style", "citation", "structure"]
_STAGE_INTERRUPT_META: dict[str, dict[str, Any]] = {
    "research": {
        "question": "References collected. How would you like to proceed?",
        "clarification_type": "risk_confirmation",
        "options": [
            {
                "label": "continue",
                "description": "Accept references and proceed to outline",
            },
            {
                "label": "add_keywords",
                "description": "Add more search keywords and re-search",
            },
            {
                "label": "add_manually",
                "description": "Provide additional references manually",
            },
        ],
    },
    "outline": {
        "question": "Outline generated. Choose a plan or customize.",
        "clarification_type": "approach_choice",
        "options": [
            {"label": "approve", "description": "Accept the proposed outline"},
            {"label": "revise", "description": "Request outline revisions with feedback"},
            {"label": "custom", "description": "Provide a custom outline structure"},
        ],
    },
    "draft": {
        "question": "Draft chapters complete. Review quality and decide next step.",
        "clarification_type": "suggestion",
        "options": [
            {
                "label": "approve",
                "description": "Accept draft and proceed to citation verification",
            },
            {
                "label": "auto_fix",
                "description": "Auto-fix flagged issues and re-draft",
            },
            {
                "label": "guide_revision",
                "description": "Provide specific revision guidance",
            },
        ],
    },
    "final": {
        "question": "Final thesis assembled. Confirm before export.",
        "clarification_type": "risk_confirmation",
        "options": [
            {"label": "accept", "description": "Accept and export the thesis"},
            {
                "label": "review_details",
                "description": "Show detailed quality report before accepting",
            },
            {
                "label": "remove_weak",
                "description": "Remove weakly-supported citations and re-polish",
            },
        ],
    },
}


def _subtask_results_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    direct_results = state.get("subtask_results", [])
    if isinstance(direct_results, list):
        results.extend(item for item in direct_results if isinstance(item, dict))

    chapters = state.get("chapters", {})
    if isinstance(chapters, dict):
        for chapter_data in chapters.values():
            if not isinstance(chapter_data, dict):
                continue
            chapter_results = chapter_data.get("subtask_results", [])
            if isinstance(chapter_results, list):
                results.extend(item for item in chapter_results if isinstance(item, dict))

    return results


def _build_self_assessment_notes(state: dict[str, Any]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for subtask in _subtask_results_from_state(state):
        subtask_id = str(subtask.get("subtask_id", "")).strip()
        if not subtask_id:
            continue

        confidence_value = subtask.get("confidence", 0.5)
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            confidence = 0.5

        weak_spots = subtask.get("weak_spots", [])
        if not isinstance(weak_spots, list):
            weak_spots = []
        needs_revision = bool(subtask.get("needs_revision", False))

        if confidence >= 0.4 and not needs_revision:
            continue

        weak_text = ", ".join(
            str(item).strip() for item in weak_spots[:3] if str(item).strip()
        ) or "overall quality"
        instruction = f"[自评标记] confidence={confidence:.2f}, 薄弱环节: {weak_text}"
        dedupe_key = (subtask_id, instruction)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        notes.append(
            {
                "subtask_id": subtask_id,
                "severity": 3,
                "instruction": instruction,
                "lens": "self_assessment",
            }
        )

    return notes


def _merge_review_packets(packets: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    scores: dict[str, int] = {}
    review_notes: list[dict[str, Any]] = []

    for packet in packets:
        packet_scores = packet.get("scores", {})
        if isinstance(packet_scores, dict):
            for key, value in packet_scores.items():
                if isinstance(value, (int, float)):
                    numeric_value = int(value)
                    scores[key] = min(scores.get(key, numeric_value), numeric_value)

        packet_notes = packet.get("review_notes", [])
        if isinstance(packet_notes, list):
            review_notes.extend(note for note in packet_notes if isinstance(note, dict))

    return scores, review_notes


def _normalize_global_review_notes(review_notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for note in review_notes:
        item = dict(note)
        item["is_recurring"] = bool(item.get("is_recurring", False))
        normalized.append(item)
    return normalized


def _review_notes_summary(review_notes: list[dict[str, Any]]) -> str:
    snippets: list[str] = []
    for note in review_notes[:5]:
        instruction = str(note.get("instruction", "")).strip()
        if instruction:
            snippets.append(instruction[:80])
    return "; ".join(snippets)


def build_chapter_review_node(services: Any):
    def chapter_review(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        chapter_title = state.get("chapter_plan", {}).get("chapter_title", "")
        merged_text = state.get("merged_text", "")
        packets: list[dict[str, Any]] = []

        if llm is not None:
            for lens in _REVIEW_LENSES:
                system, user = chapter_review_prompt_for_lens(chapter_title, merged_text, lens)
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

        auto_notes = _build_self_assessment_notes(state)
        review_notes = review_notes + auto_notes

        return {
            "quality_scores": scores,
            "review_notes": review_notes,
            "revision_instructions": build_revision_instructions(review_notes, min_severity=2),
        }

    return chapter_review


def build_global_review_node(services: Any):
    def global_review(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        merged_text = str(state.get("final_text", "") or "")
        review_history = state.get("review_history", [])
        if not isinstance(review_history, list):
            review_history = []

        iteration_value = state.get("review_iteration", 1)
        try:
            iteration = max(int(iteration_value), 1)
        except (TypeError, ValueError):
            iteration = 1

        packets: list[dict[str, Any]] = []
        if llm is not None:
            for lens in _REVIEW_LENSES:
                if iteration > 1 and review_history:
                    system, user = adaptive_review_prompt(
                        merged_text=merged_text,
                        lens=lens,
                        review_history=review_history,
                        iteration=iteration,
                    )
                else:
                    system, user = global_review_prompt_for_lens(
                        merged_text=merged_text,
                        lens=lens,
                    )
                try:
                    payload = llm.structured(system=system, user=user, route="review", max_tokens=1800)
                except Exception:
                    payload = {}
                if isinstance(payload, dict):
                    packets.append(payload)

        scores, review_notes = _merge_review_packets(packets)
        review_notes = _normalize_global_review_notes(review_notes)
        current_review_record = {
            "iteration": iteration,
            "scores": dict(scores),
            "notes_summary": _review_notes_summary(review_notes),
            "note_count": len(review_notes),
        }
        return {
            "quality_scores": scores,
            "review_notes": review_notes,
            "review_history": [current_review_record],
            "review_iteration": iteration + 1,
        }

    return global_review


def build_interrupt_node(stage: str, *, auto_approve: bool):
    def interrupt_node(state: dict[str, Any]) -> dict[str, Any]:
        meta = _STAGE_INTERRUPT_META.get(stage, {})
        payload = {
            "stage": stage,
            "project_id": state.get("project_id"),
            "ref_count": len(state.get("references", [])),
            "chapter_count": len(state.get("chapter_plans", [])),
            "question": meta.get("question", f"Stage '{stage}' complete. Approve?"),
            "clarification_type": meta.get("clarification_type", "risk_confirmation"),
            "options": meta.get("options", []),
            "context": _build_stage_context(stage, state),
        }
        if auto_approve:
            feedback = {"stage": stage, "approved": True, "auto_approve": True}
        else:
            feedback = interrupt(payload)
            if not isinstance(feedback, dict):
                feedback = {"stage": stage, "approved": bool(feedback)}
        return {"review_feedback": [feedback]}

    return interrupt_node


def _build_stage_context(stage: str, state: dict[str, Any]) -> str:
    """Build a concise human-readable context string for a stage interrupt."""

    parts: list[str] = []
    if stage == "research":
        references = state.get("references", [])
        parts.append(f"{len(references)} reference(s) found.")
        queries = state.get("search_queries", [])
        if queries:
            parts.append(f"Search queries used: {', '.join(queries[:5])}")
    elif stage == "outline":
        plans = state.get("chapter_plans", [])
        titles = [
            str(plan.get("chapter_title", "?"))
            for plan in plans
            if isinstance(plan, dict)
        ]
        parts.append(f"{len(plans)} chapter(s): {', '.join(titles)}")
    elif stage == "draft":
        chapters = state.get("chapters", {})
        parts.append(f"{len(chapters)} chapter(s) drafted.")
        scores: list[str] = []
        if isinstance(chapters, dict):
            for chapter_id, chapter in chapters.items():
                quality_scores = chapter.get("quality_scores", {}) if isinstance(chapter, dict) else {}
                numeric_scores = [
                    int(value)
                    for value in quality_scores.values()
                    if isinstance(value, (int, float))
                ]
                if numeric_scores:
                    scores.append(f"{chapter_id}={min(numeric_scores)}")
        if scores:
            parts.append(f"Min quality scores: {', '.join(scores)}")
    elif stage == "final":
        flagged = state.get("flagged_citations", [])
        verified = state.get("verified_citations", [])
        parts.append(f"{len(verified)} verified, {len(flagged)} flagged citation(s).")
    return " ".join(parts) if parts else ""
