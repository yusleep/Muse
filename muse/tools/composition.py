"""Composition tools for terminology, cross-references, transitions, and rewriting."""

import json
import re
from typing import Annotated
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool


def _services_from_runtime(runtime: ToolRuntime | None) -> Any:
    if runtime is not None:
        context = getattr(runtime, "context", None)
        if isinstance(context, dict) and context.get("services") is not None:
            return context["services"]

    from muse.tools._context import get_services

    return get_services()


@tool
def check_terminology(
    text: str,
    *,
    runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> str:
    """Scan text for terminology inconsistencies."""

    services = _services_from_runtime(runtime)
    llm = getattr(services, "llm", None)
    if llm is None:
        abbreviations = sorted(set(re.findall(r"\b[A-Z]{2,}\b", text)))
        return json.dumps(
            {
                "issues": [],
                "abbreviations_found": abbreviations,
            },
            ensure_ascii=False,
        )

    system = (
        "Scan the following text for terminology inconsistencies. "
        "Find cases where the same concept uses different terms or abbreviations. "
        "Return JSON with key 'issues': list of {term, variants, suggestion}."
    )
    try:
        result = llm.structured(
            system=system,
            user=text[:6000],
            route="review",
            max_tokens=1200,
        )
    except Exception:  # noqa: BLE001
        result = {"issues": []}

    return json.dumps(result if isinstance(result, dict) else {"issues": []}, ensure_ascii=False)


@tool
def align_cross_refs(text: str) -> str:
    """Extract and summarize cross-references found in the text."""

    patterns = {
        "figure": r"(?:Figure|Fig\.?|图)\s*(\d+[\.\d]*)",
        "table": r"(?:Table|Tab\.?|表)\s*(\d+[\.\d]*)",
        "section": r"(?:Section|Sec\.?|节|章)\s*(\d+[\.\d]*)",
        "equation": r"(?:Equation|Eq\.?|式|公式)\s*[\(（]?(\d+[\.\d]*)[\)）]?",
    }
    cross_refs: list[dict[str, Any]] = []
    for ref_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            cross_refs.append(
                {
                    "type": ref_type,
                    "number": match.group(1),
                    "position": match.start(),
                }
            )

    return json.dumps(
        {
            "cross_refs_found": cross_refs[:100],
            "dangling": [],
            "total_count": len(cross_refs),
        },
        ensure_ascii=False,
    )


@tool
def check_transitions(
    chapter_texts_json: str,
    *,
    runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> str:
    """Check transition quality between adjacent chapter excerpts."""

    try:
        chapters = json.loads(chapter_texts_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"transitions": [], "error": "invalid JSON"}, ensure_ascii=False)

    if not isinstance(chapters, list):
        chapters = []

    services = _services_from_runtime(runtime)
    llm = getattr(services, "llm", None)
    transitions: list[dict[str, Any]] = []
    for index in range(len(chapters) - 1):
        current = chapters[index] if isinstance(chapters[index], dict) else {}
        following = chapters[index + 1] if isinstance(chapters[index + 1], dict) else {}
        transition = {
            "from_chapter": current.get("chapter_id", f"ch{index + 1}"),
            "to_chapter": following.get("chapter_id", f"ch{index + 2}"),
            "quality": "unknown",
            "suggestion": "",
        }

        if llm is not None:
            system = (
                "Evaluate the transition between these two chapter segments. "
                "Return JSON: {quality: 'smooth'|'abrupt'|'missing', suggestion: str}"
            )
            user = json.dumps(
                {
                    "ending": current.get("ending", "")[:500],
                    "opening": following.get("opening", "")[:500],
                },
                ensure_ascii=False,
            )
            try:
                result = llm.structured(
                    system=system,
                    user=user,
                    route="review",
                    max_tokens=500,
                )
                if isinstance(result, dict):
                    transition["quality"] = str(result.get("quality", "unknown"))
                    transition["suggestion"] = str(result.get("suggestion", ""))
            except Exception:  # noqa: BLE001
                pass

        transitions.append(transition)

    return json.dumps({"transitions": transitions}, ensure_ascii=False)


@tool
def rewrite_passage(
    passage: str,
    instruction: str,
    context: str = "",
    *,
    runtime: Annotated[ToolRuntime, InjectedToolArg],
) -> str:
    """Rewrite a passage according to a focused instruction."""

    services = _services_from_runtime(runtime)
    llm = getattr(services, "llm", None)
    if llm is None:
        return passage

    system = f"Rewrite the following passage. Context: {context}. Return ONLY the rewritten text."
    user = json.dumps(
        {
            "instruction": instruction,
            "passage": passage,
        },
        ensure_ascii=False,
    )
    try:
        return llm.text(system=system, user=user, route="polish", max_tokens=2000)
    except Exception:  # noqa: BLE001
        return passage
