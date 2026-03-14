from __future__ import annotations

import json


_LAYER_REVIEW_SYSTEMS = {
    "structural": (
        "You are reviewing the full merged thesis draft and should only focus on the structural layer. "
        "Check argument order, information completeness, section balance, transitions, and cross-chapter coherence. "
        "Do not focus on citation volume, terminology polish, or sentence-level edits. "
        'Return JSON with keys: scores and review_notes. Scores must include "logic", "structure", and "balance".'
    ),
    "content": (
        "You are reviewing the full merged thesis draft and should only focus on the content layer. "
        "Check citation quality, topic coverage, technical depth, literature comparison, and cross-chapter evidence gaps. "
        "Do not focus on paragraph ordering or sentence-level polish. "
        'Return JSON with keys: scores and review_notes. Scores must include "citation", "coverage", and "depth".'
    ),
    "line": (
        "You are reviewing the full merged thesis draft and should only focus on the line layer. "
        "Check academic style, term consistency across chapters, and redundancy. "
        "Do not focus on paragraph ordering or citation coverage. "
        'Return JSON with keys: scores and review_notes. Scores must include "style", "term_consistency", and "redundancy".'
    ),
}


def layered_review_prompt(
    layer: str,
    merged_text: str,
) -> tuple[str, str]:
    system = _LAYER_REVIEW_SYSTEMS[layer]
    user = json.dumps({"text": merged_text}, ensure_ascii=False)
    return system, user


def layered_revision_prompt(
    layer: str,
    merged_text: str,
    review_notes: list[dict[str, object]],
) -> tuple[str, str]:
    system = (
        f"Revise the full merged thesis draft for the {layer} layer. "
        "Apply the review instructions faithfully and return JSON with key final_text."
    )
    user = json.dumps(
        {
            "text": merged_text,
            "review_notes": review_notes,
        },
        ensure_ascii=False,
    )
    return system, user
