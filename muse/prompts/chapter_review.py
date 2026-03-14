from __future__ import annotations

import json


_LENS_ORDER = ("logic", "style", "citation", "structure")
_SCORE_KEYS = "coherence, logic, citation, term_consistency, balance, redundancy"
_LENS_RUBRICS = {
    "logic": "\n".join(
        [
            "Logic rubric:",
            "1 point: The argument chain is broken, key claims lack support, or obvious fallacies remain.",
            "2 point: Multiple logical jumps remain and at least two conclusions are not adequately justified.",
            "3 point: The argument is mostly traceable but still contains one or two unsupported transitions.",
            "4 point: The argument is clear and well-supported, with only minor transitions needing tightening.",
            "5 point: The argument is rigorous, fully supported, and progresses through natural transitions.",
        ]
    ),
    "style": "\n".join(
        [
            "Style rubric:",
            "1 point: The prose is unclear, colloquial, repetitive, or not suitable for an academic thesis.",
            "2 point: The prose is understandable but has frequent awkward phrasing or inconsistent tone.",
            "3 point: The prose is readable with a mostly academic tone, though some sentences remain rough.",
            "4 point: The prose is polished and academic, with only minor wording issues remaining.",
            "5 point: The prose is precise, concise, and consistently professional throughout the chapter.",
        ]
    ),
    "citation": "\n".join(
        [
            "Citation rubric:",
            "1 point: Claims routinely lack citations or rely on clearly unsupported references.",
            "2 point: Several important claims are weakly supported or cite mismatched sources.",
            "3 point: Most claims are cited, but there are still noticeable gaps or imprecise support.",
            "4 point: Citations support the main claims well, with only limited room for stronger evidence.",
            "5 point: Citations are precise, sufficient, and consistently aligned with the claims they support.",
        ]
    ),
    "structure": "\n".join(
        [
            "Structure rubric:",
            "1 point: The chapter lacks a usable structure and ideas appear in a confusing order.",
            "2 point: The chapter has a partial structure, but sections are imbalanced or poorly sequenced.",
            "3 point: The chapter structure is understandable, though some ordering or section balance issues remain.",
            "4 point: The structure is coherent and well-paced, with only small organizational issues remaining.",
            "5 point: The structure is deliberate, balanced, and strongly supports the chapter's narrative arc.",
        ]
    ),
}
_LENS_BOUNDARIES = {
    "logic": "Do not focus on sentence-level polish unless it directly damages the reasoning.",
    "style": "Do not relitigate evidence coverage or chapter architecture unless the prose makes them unreadable.",
    "citation": "Do not focus on stylistic preferences or broad restructuring unless they directly affect citation fidelity.",
    "structure": "Do not spend review effort on line edits or minor wording unless they block structural clarity.",
}


def review_lenses() -> tuple[str, ...]:
    return _LENS_ORDER


def review_rubric_for_lens(lens: str) -> str:
    if lens not in _LENS_RUBRICS:
        raise ValueError(f"Unsupported review lens: {lens}")
    return _LENS_RUBRICS[lens]


def review_boundary_for_lens(lens: str) -> str:
    if lens not in _LENS_BOUNDARIES:
        raise ValueError(f"Unsupported review lens: {lens}")
    return _LENS_BOUNDARIES[lens]


def _base_system_prompt() -> str:
    return (
        "You are a strict thesis reviewer. "
        "Return JSON with keys: scores (object) and review_notes (list). "
        f"scores keys: {_SCORE_KEYS}; values 1-5. "
        "Each review_notes item must be an object with keys: "
        "subtask_id (string), severity (integer 1-5), instruction (string), lens (string). "
        "Only include concrete revision instructions tied to a subtask."
    )


def chapter_review_prompt_for_lens(
    chapter_title: str,
    merged_text: str,
    lens: str,
) -> tuple[str, str]:
    system = "\n\n".join(
        [
            _base_system_prompt(),
            f"Primary review lens: {lens}",
            review_rubric_for_lens(lens),
            f"When you emit review_notes, set lens to '{lens}'.",
            review_boundary_for_lens(lens),
        ]
    )
    user = json.dumps({"chapter_title": chapter_title, "text": merged_text}, ensure_ascii=False)
    return system, user


def chapter_review_prompt(chapter_title: str, merged_text: str) -> tuple[str, str]:
    system = "\n\n".join(
        [
            _base_system_prompt(),
            f"Available review lenses: {', '.join(_LENS_ORDER)}.",
            "If no specific lens is requested, evaluate the chapter from all listed lenses.",
            *(
                f"Lens {lens}:\n{review_rubric_for_lens(lens)}\n{review_boundary_for_lens(lens)}"
                for lens in _LENS_ORDER
            ),
        ]
    )
    user = json.dumps({"chapter_title": chapter_title, "text": merged_text}, ensure_ascii=False)
    return system, user
