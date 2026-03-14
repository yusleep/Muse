"""Chapter-level reference analysis node."""

from __future__ import annotations

from typing import Any

from muse.graph.helpers.draft_support import _build_refs_snapshot
from muse.prompts.ref_analysis import ref_analysis_prompt


def build_ref_analysis_node(*, services: Any):
    def ref_analysis(state: dict[str, Any]) -> dict[str, Any]:
        chapter_plans = state.get("chapter_plans", [])
        references = state.get("references", [])
        llm = getattr(services, "llm", None)
        if not isinstance(chapter_plans, list) or not chapter_plans:
            return {}
        if not isinstance(references, list) or not references or llm is None:
            return {}

        refs_snapshot = _build_refs_snapshot(state=state, references=references)
        briefs: dict[str, Any] = {}
        for chapter_plan in chapter_plans:
            if not isinstance(chapter_plan, dict):
                continue
            chapter_id = str(chapter_plan.get("chapter_id", "")).strip()
            if not chapter_id:
                continue
            system, user = ref_analysis_prompt(
                str(chapter_plan.get("chapter_title", "")),
                chapter_plan.get("subtask_plan", [])
                if isinstance(chapter_plan.get("subtask_plan", []), list)
                else [],
                refs_snapshot,
            )
            try:
                payload = llm.structured(
                    system=system,
                    user=user,
                    route="default",
                    max_tokens=1500,
                )
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            key_references = payload.get("key_references", [])
            evidence_gaps = payload.get("evidence_gaps", [])
            briefs[chapter_id] = [
                dict(item)
                for item in key_references
                if isinstance(item, dict) and str(item.get("ref_id", "")).strip()
            ]
            briefs[f"{chapter_id}_gaps"] = [
                str(item).strip()
                for item in evidence_gaps
                if str(item).strip()
            ]

        return {"reference_briefs": briefs} if briefs else {}

    return ref_analysis
