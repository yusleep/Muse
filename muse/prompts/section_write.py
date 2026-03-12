from __future__ import annotations

import json
from typing import Any


def section_write_prompt(
    topic: str,
    chapter_title: str,
    subtask: dict[str, Any],
    refs: list[dict[str, Any]],
    language: str,
    previous_subsection: str = "",
    revision_instruction: str | None = None,
    local_context: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    system = (
        "Write one thesis subsection with citations. IMPORTANT: for citations_used, use ONLY ref_id values "
        "from the available_references list. Do not invent citation keys not in that list. "
        "Include specific technical details, mathematical notation where appropriate, and reference "
        "concrete experimental results. Return JSON with keys: text, citations_used, key_claims, "
        "transition_out, glossary_additions, self_assessment."
    )
    payload = {
        "topic": topic,
        "chapter_title": chapter_title,
        "subtask": subtask,
        "language": language,
        "available_references": refs,
        "allowed_refs": [r.get("ref_id") for r in refs if r.get("ref_id")],
        "previous_subsection": previous_subsection,
        "revision_instruction": revision_instruction,
    }
    if local_context:
        payload["local_context"] = local_context
    return system, json.dumps(payload, ensure_ascii=False)
