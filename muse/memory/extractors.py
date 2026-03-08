"""Memory extraction functions for specific trigger points."""

from __future__ import annotations

import re
from typing import Any

from muse.memory.store import MemoryEntry


def extract_from_initialize(
    state: dict[str, Any],
    run_id: str | None = None,
) -> list[MemoryEntry]:
    """Extract topic, discipline, language, and format preferences."""

    entries: list[MemoryEntry] = []
    topic = str(state.get("topic", "")).strip()
    discipline = str(state.get("discipline", "")).strip()
    language = str(state.get("language", "")).strip()
    format_standard = str(state.get("format_standard", "")).strip()

    if topic:
        entries.append(
            MemoryEntry(
                id="",
                key=f"topic:{_slugify(topic)}",
                category="fact",
                content=f"Research topic: {topic}",
                confidence=0.7,
                source_run=run_id,
            )
        )
    if discipline:
        entries.append(
            MemoryEntry(
                id="",
                key=f"discipline:{_slugify(discipline)}",
                category="fact",
                content=f"Academic discipline: {discipline}",
                confidence=0.7,
                source_run=run_id,
            )
        )
    if language:
        entries.append(
            MemoryEntry(
                id="",
                key=f"language_pref:{language}",
                category="user_pref",
                content=f"Preferred writing language: {language}",
                confidence=0.8,
                source_run=run_id,
            )
        )
    if format_standard:
        entries.append(
            MemoryEntry(
                id="",
                key=f"format_std:{_slugify(format_standard)}",
                category="user_pref",
                content=f"Citation format standard: {format_standard}",
                confidence=0.8,
                source_run=run_id,
            )
        )
    return entries


def extract_from_hitl_feedback(
    node_name: str,
    result: dict[str, Any],
    run_id: str | None = None,
) -> list[MemoryEntry]:
    """Extract feedback patterns from HITL review nodes."""

    entries: list[MemoryEntry] = []
    feedback_list = result.get("review_feedback", [])
    if not isinstance(feedback_list, list):
        return entries

    for feedback in feedback_list:
        if not isinstance(feedback, dict):
            continue
        notes = str(feedback.get("notes", "")).strip()
        if len(notes) < 15:
            continue

        category = "feedback_pattern"
        confidence = 0.6
        style_keywords = {
            "tone",
            "style",
            "formal",
            "informal",
            "concise",
            "verbose",
            "passive voice",
            "active voice",
            "academic",
        }
        lowered = notes.lower()
        if any(keyword in lowered for keyword in style_keywords):
            category = "writing_style"
            confidence = 0.7

        entries.append(
            MemoryEntry(
                id="",
                key=f"feedback:{node_name}:{_slugify(notes[:50])}",
                category=category,
                content=f"User feedback at {node_name}: {notes}",
                confidence=confidence,
                source_run=run_id,
            )
        )
    return entries


def extract_from_citation_subgraph(
    state: dict[str, Any],
    result: dict[str, Any],
    run_id: str | None = None,
) -> list[MemoryEntry]:
    """Extract verified citations as high-confidence citation memories."""

    entries: list[MemoryEntry] = []
    verified = result.get("verified_citations", [])
    if not isinstance(verified, list):
        return entries

    references = {
        reference.get("ref_id"): reference
        for reference in state.get("references", [])
        if isinstance(reference, dict) and reference.get("ref_id")
    }

    for cite_key in verified:
        if not isinstance(cite_key, str) or not cite_key.strip():
            continue
        reference = references.get(cite_key, {})
        doi = str(reference.get("doi", "") or "").strip()
        title = str(reference.get("title", cite_key) or cite_key).strip()
        year = reference.get("year", "")

        content = f"Verified citation: {title}"
        if year:
            content += f" ({year})"
        if doi:
            content += f" [DOI: {doi}]"

        entries.append(
            MemoryEntry(
                id="",
                key=f"cite:{cite_key}",
                category="citation",
                content=content,
                confidence=0.9,
                source_run=run_id,
            )
        )
    return entries


def extract_from_review(
    state: dict[str, Any],
    result: dict[str, Any],
    run_id: str | None = None,
) -> list[MemoryEntry]:
    """Extract recurring quality issues from review results."""

    del state
    entries: list[MemoryEntry] = []
    quality_scores = result.get("quality_scores", {})
    if not isinstance(quality_scores, dict):
        return entries

    for dimension, score in quality_scores.items():
        if isinstance(score, (int, float)) and score <= 2:
            entries.append(
                MemoryEntry(
                    id="",
                    key=f"quality_issue:{_slugify(str(dimension))}",
                    category="feedback_pattern",
                    content=f"Recurring quality issue: {dimension} scored {score}/5",
                    confidence=0.5,
                    source_run=run_id,
                )
            )
    return entries


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "_", text.lower())
    return slug[:60].strip("_")
