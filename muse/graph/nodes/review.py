"""Review and interrupt nodes for graph-native execution."""

from __future__ import annotations

import json
from typing import Any

from langgraph.types import interrupt

from muse.graph.helpers.review_state import build_revision_instructions
from muse.prompts.adaptive_review import adaptive_review_prompt
from muse.prompts.chapter_review import chapter_review_prompt_for_lens
from muse.prompts.global_review import global_review_prompt_for_lens
from muse.prompts.layered_review import layered_review_prompt, layered_revision_prompt
from muse.prompts.review_judge import JUDGE_SYSTEM
from muse.prompts.reviewer_personas import persona_dimensions, reviewer_persona_prompt


_REVIEW_LENSES = ["logic", "style", "citation", "structure"]
_LAYER_SCORE_KEYS = {
    "structural": ("logic", "structure", "balance"),
    "content": ("citation", "coverage", "depth"),
    "line": ("style", "term_consistency", "redundancy"),
}
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


def _top_review_instructions(review_notes: list[dict[str, Any]], limit: int = 3) -> list[str]:
    ranked_notes = sorted(
        (note for note in review_notes if isinstance(note, dict)),
        key=lambda note: int(note.get("severity", 0)),
        reverse=True,
    )
    instructions: list[str] = []
    seen: set[str] = set()
    for note in ranked_notes:
        instruction = str(note.get("instruction", "")).strip()
        if not instruction or instruction in seen:
            continue
        seen.add(instruction)
        instructions.append(instruction)
        if len(instructions) >= limit:
            break
    return instructions


def _build_review_record(
    *,
    iteration: int,
    scores: dict[str, int],
    review_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "scores": dict(scores),
        "notes_summary": _review_notes_summary(review_notes),
        "note_count": len(review_notes),
        "top_instructions": _top_review_instructions(review_notes),
    }


def _update_reflection_data(
    state: dict[str, Any],
    *,
    current_review_record: dict[str, Any],
) -> dict[str, Any]:
    from muse.graph.helpers.reflection_bank import ReflectionBank

    review_history = state.get("review_history", [])
    if not isinstance(review_history, list):
        review_history = []
    updated_history = [item for item in review_history if isinstance(item, dict)] + [current_review_record]
    bank = ReflectionBank.from_dict(state.get("reflection_data", {}))
    bank.add_reflection(
        updated_history,
        chapter_id=str(state.get("project_id", "") or "thesis"),
    )
    return bank.to_dict()


def _sanitize_persona_packet(persona: str, payload: dict[str, Any]) -> dict[str, Any]:
    allowed_dimensions = set(persona_dimensions(persona))
    raw_scores = payload.get("scores", {})
    filtered_scores: dict[str, int] = {}
    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            if key in allowed_dimensions and isinstance(value, (int, float)):
                filtered_scores[key] = int(value)

    raw_notes = payload.get("review_notes", [])
    filtered_notes: list[dict[str, Any]] = []
    if isinstance(raw_notes, list):
        for note in raw_notes:
            if not isinstance(note, dict):
                continue
            item = dict(note)
            item.setdefault("lens", persona)
            item["is_recurring"] = bool(item.get("is_recurring", False))
            filtered_notes.append(item)

    return {"scores": filtered_scores, "review_notes": filtered_notes}


def _merge_persona_results(packets: list[dict[str, Any]]) -> dict[str, Any]:
    final_scores: dict[str, int] = {}
    unified_notes: list[dict[str, Any]] = []
    persona_floor_scores: dict[str, int] = {}

    for packet in packets:
        persona = str(packet.get("persona", "")).strip()
        result = packet.get("result", {})
        if not isinstance(result, dict):
            continue
        scores = result.get("scores", {})
        if isinstance(scores, dict):
            numeric_scores = []
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    final_scores[key] = int(value)
                    numeric_scores.append(int(value))
            if persona and numeric_scores:
                persona_floor_scores[persona] = min(numeric_scores)
        notes = result.get("review_notes", [])
        if isinstance(notes, list):
            unified_notes.extend(note for note in notes if isinstance(note, dict))

    unified_notes = _normalize_global_review_notes(unified_notes)
    unified_notes.sort(key=lambda note: int(note.get("severity", 0)), reverse=True)

    conflicts_resolved: list[dict[str, Any]] = []
    if persona_floor_scores:
        min_score = min(persona_floor_scores.values())
        max_score = max(persona_floor_scores.values())
        if max_score - min_score > 2:
            conflicts_resolved.append(
                {
                    "topic": "reviewer_strictness",
                    "scores": dict(persona_floor_scores),
                    "ruling": "Preserved the stricter persona signal while keeping the merged notes ordered by severity.",
                }
            )

    return {
        "final_scores": final_scores,
        "unified_notes": unified_notes,
        "conflicts_resolved": conflicts_resolved,
    }


def _layer_iteration_key(layer: str) -> str:
    return f"{layer}_iterations"


def _filter_scores_for_keys(scores: Any, allowed_keys: tuple[str, ...]) -> dict[str, int]:
    filtered: dict[str, int] = {}
    if not isinstance(scores, dict):
        return filtered
    allowed = set(allowed_keys)
    for key, value in scores.items():
        if key in allowed and isinstance(value, (int, float)):
            filtered[key] = int(value)
    return filtered


def _run_classic_global_review(
    state: dict[str, Any],
    *,
    llm: Any,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
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
    return scores, review_notes, _build_review_record(
        iteration=iteration,
        scores=scores,
        review_notes=review_notes,
    )


def _run_persona_global_review(
    state: dict[str, Any],
    *,
    llm: Any,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    merged_text = str(state.get("final_text", "") or "")
    review_history = state.get("review_history", [])
    if not isinstance(review_history, list):
        review_history = []

    iteration_value = state.get("review_iteration", 1)
    try:
        iteration = max(int(iteration_value), 1)
    except (TypeError, ValueError):
        iteration = 1

    persona_packets: list[dict[str, Any]] = []
    if llm is not None:
        for persona in ("logic", "citation", "readability"):
            system, user = reviewer_persona_prompt(
                persona,
                merged_text=merged_text,
                review_history=review_history,
                iteration=iteration,
            )
            try:
                payload = llm.structured(system=system, user=user, route="review", max_tokens=2000)
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            persona_packets.append(
                {
                    "persona": persona,
                    "result": _sanitize_persona_packet(persona, payload),
                }
            )

    judge_result = _merge_persona_results(persona_packets)
    if llm is not None:
        try:
            candidate = llm.structured(
                system=JUDGE_SYSTEM,
                user=json.dumps(persona_packets, ensure_ascii=False),
                route="review_judge",
                max_tokens=2000,
            )
        except Exception:
            candidate = None
        if isinstance(candidate, dict):
            judge_result = candidate

    raw_scores = judge_result.get("final_scores", {})
    scores = {
        key: int(value)
        for key, value in raw_scores.items()
        if isinstance(value, (int, float))
    } if isinstance(raw_scores, dict) else {}
    raw_notes = judge_result.get("unified_notes", [])
    review_notes = _normalize_global_review_notes(
        [note for note in raw_notes if isinstance(note, dict)]
        if isinstance(raw_notes, list)
        else []
    )
    return scores, review_notes, _build_review_record(
        iteration=iteration,
        scores=scores,
        review_notes=review_notes,
    )


def build_layered_review_node(services: Any, *, layer: str, route: str = "review"):
    allowed_keys = _LAYER_SCORE_KEYS[layer]
    iteration_key = _layer_iteration_key(layer)

    def layered_review(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        current_iterations = state.get(iteration_key, 0)
        try:
            current_iterations = max(int(current_iterations), 0)
        except (TypeError, ValueError):
            current_iterations = 0
        current_review_iteration = state.get("review_iteration", 1)
        try:
            current_review_iteration = max(int(current_review_iteration), 1)
        except (TypeError, ValueError):
            current_review_iteration = 1

        final_text = str(state.get("final_text", "") or "")
        system, user = layered_review_prompt(layer, final_text)
        try:
            payload = llm.structured(system=system, user=user, route=route, max_tokens=1800) if llm is not None else {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        scores = _filter_scores_for_keys(payload.get("scores", {}), allowed_keys)
        review_notes = _normalize_global_review_notes(
            [
                dict(note)
                for note in payload.get("review_notes", [])
                if isinstance(note, dict)
            ]
            if isinstance(payload.get("review_notes", []), list)
            else []
        )
        for note in review_notes:
            note.setdefault("lens", layer)
        record = _build_review_record(
            iteration=current_review_iteration,
            scores=scores,
            review_notes=review_notes,
        )
        record["layer"] = layer
        return {
            "quality_scores": scores,
            "review_notes": review_notes,
            iteration_key: current_iterations + 1,
            "review_layer": layer,
            "review_history": [record],
            "review_iteration": current_review_iteration + 1,
            "reflection_data": _update_reflection_data(
                state,
                current_review_record=record,
            ),
        }

    return layered_review


def build_global_revise_node(services: Any, *, layer: str, route: str = "writing_revision"):
    def revise(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        final_text = str(state.get("final_text", "") or "")
        review_notes = state.get("review_notes", [])
        if not isinstance(review_notes, list):
            review_notes = []
        if llm is None:
            return {"final_text": final_text, "review_layer": layer}

        system, user = layered_revision_prompt(layer, final_text, review_notes)
        try:
            payload = llm.structured(system=system, user=user, route=route, max_tokens=2800)
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        revised_text = str(payload.get("final_text", "") or final_text)
        return {"final_text": revised_text, "review_layer": layer}

    return revise


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


def build_global_review_node(services: Any, *, mode: str = "classic"):
    def global_review(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        iteration_value = state.get("review_iteration", 1)
        try:
            iteration = max(int(iteration_value), 1)
        except (TypeError, ValueError):
            iteration = 1

        if mode == "persona":
            scores, review_notes, current_review_record = _run_persona_global_review(
                state,
                llm=llm,
            )
        else:
            scores, review_notes, current_review_record = _run_classic_global_review(
                state,
                llm=llm,
            )
        return {
            "quality_scores": scores,
            "review_notes": review_notes,
            "review_history": [current_review_record],
            "review_iteration": iteration + 1,
            "reflection_data": _update_reflection_data(
                state,
                current_review_record=current_review_record,
            ),
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
