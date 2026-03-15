from __future__ import annotations

import json
from typing import Any


BASE_SECTION_WRITE_SYSTEM_PROMPT = (
    "Write one thesis subsection with citations. "
    "IMPORTANT: for citations_used, use ONLY ref_id values from the available_references list. "
    "Do not invent citation keys not in that list. "
    "SCOPE GUARD: Write ONLY about the topic defined in subtask.title. "
    "Do NOT include content that belongs to other subtasks. "
    "If a related topic is outside this subtask's scope, mention it briefly "
    "and note that it will be covered in a later section. "
    "If an argument_plan is provided, follow its logical_flow and make each paragraph execute one step "
    "of the evidence_chain. "
    "References marked source=local are author-provided core papers and should be prioritized when relevant. "
    "For references marked indexed=true, use get_paper_section when you need section-level evidence. "
    "Include specific technical details, mathematical notation where appropriate, "
    "and reference concrete experimental results. "
    "Return JSON with keys: text, citations_used (list of ref_id strings), key_claims (list), "
    "transition_out, glossary_additions (object), "
    "self_assessment (object with confidence, weak_spots, needs_revision)."
)


def section_write_system_prompt(prompt_variant: str | None = None) -> str:
    prompt = str(prompt_variant or "").strip()
    return prompt if prompt else BASE_SECTION_WRITE_SYSTEM_PROMPT


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
    system = section_write_system_prompt()
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
