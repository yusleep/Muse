"""Chapter review state helpers shared by graph nodes and legacy shims."""

from __future__ import annotations

from typing import Mapping


def should_iterate(state: Mapping[str, object], threshold: int = 4) -> str:
    """Route chapter flow to either `revise` or `done`."""

    raw_scores = state.get("quality_scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    numeric_scores = [int(value) for value in raw_scores.values() if isinstance(value, (int, float))]
    min_score = min(numeric_scores) if numeric_scores else 0

    current_iteration = int(state.get("current_iteration", state.get("iteration", 0)))
    max_iterations = int(state.get("max_iterations", 3))

    if min_score >= threshold:
        return "done"
    if current_iteration >= max_iterations:
        return "done"
    return "revise"


def build_revision_instructions(
    review_notes: list[dict[str, object]],
    min_severity: int = 2,
) -> dict[str, str]:
    """Build subtask-specific revision instructions from chapter review notes."""

    instructions: dict[str, str] = {}
    for note in review_notes:
        if not isinstance(note, dict):
            continue
        subtask_id = note.get("subtask_id")
        instruction = note.get("instruction")
        severity = note.get("severity", 0)

        if not isinstance(subtask_id, str) or not subtask_id:
            continue
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        if not isinstance(severity, (int, float)) or int(severity) < min_severity:
            continue

        instructions[subtask_id] = instruction.strip()

    return instructions


def apply_chapter_review(
    state: dict[str, object],
    review: dict[str, object],
    score_threshold: int = 4,
    min_severity: int = 2,
) -> tuple[str, dict[str, object]]:
    """Apply review output to chapter state and return the next route."""

    scores = review.get("scores", {})
    notes = review.get("review_notes", [])

    if not isinstance(scores, dict):
        scores = {}
    if not isinstance(notes, list):
        notes = []

    state["quality_scores"] = scores
    state["review_notes"] = notes
    state["revision_instructions"] = build_revision_instructions(notes, min_severity=min_severity)
    state["current_iteration"] = int(state.get("current_iteration", 0)) + 1

    route = should_iterate(state, threshold=score_threshold)
    return route, state

