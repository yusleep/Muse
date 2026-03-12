"""Review tools for multi-lens chapter evaluation."""

import json
from typing import Annotated
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool

from muse.graph.helpers.review_state import build_revision_instructions
from muse.prompts.chapter_review import chapter_review_prompt
from muse.tools._context import AgentRuntimeContext

MuseToolRuntime = ToolRuntime[AgentRuntimeContext, Any]


def _services_from_runtime(runtime: MuseToolRuntime | None) -> Any:
    from muse.tools._context import get_services, services_from_runtime

    services = services_from_runtime(runtime)
    return services if services is not None else get_services()


@tool
def self_review(
    chapter_title: str,
    merged_text: str,
    lenses: str = "logic,style,citation,structure",
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Run a multi-lens quality review on a chapter draft."""

    services = _services_from_runtime(runtime)
    llm = getattr(services, "llm", None)
    lens_list = [lens.strip() for lens in lenses.split(",") if lens.strip()]
    if not lens_list:
        lens_list = ["logic", "style", "citation", "structure"]

    packets: list[dict[str, Any]] = []
    if llm is not None:
        for lens in lens_list:
            system, user = chapter_review_prompt(chapter_title, merged_text)
            system = f"{system} Focus primarily on {lens}."
            try:
                payload = llm.structured(
                    system=system,
                    user=user,
                    route="review",
                    max_tokens=1800,
                )
            except Exception:  # noqa: BLE001
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
                    if key in scores:
                        scores[key] = min(scores[key], int(value))
                    else:
                        scores[key] = int(value)
        packet_notes = packet.get("review_notes", [])
        if isinstance(packet_notes, list):
            review_notes.extend(note for note in packet_notes if isinstance(note, dict))

    revision_instructions = build_revision_instructions(review_notes, min_severity=2)
    return json.dumps(
        {
            "scores": scores,
            "review_notes": review_notes,
            "revision_instructions": revision_instructions,
        },
        ensure_ascii=False,
    )
