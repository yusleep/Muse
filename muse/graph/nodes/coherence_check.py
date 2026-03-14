from __future__ import annotations

from typing import Any

from muse.prompts.coherence_check import coherence_check_prompt


def _safe_int(value: Any, default: int = 5) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_coherence_check_node(*, services: Any):
    def coherence_check(state: dict[str, Any]) -> dict[str, Any]:
        final_text = str(state.get("final_text", "") or "")
        if len(final_text) < 500:
            return {}

        llm = getattr(services, "llm", None)
        if llm is None:
            return {}

        system, user = coherence_check_prompt(final_text)
        try:
            payload = llm.structured(system=system, user=user, route="review", max_tokens=1500)
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        raw_issues = payload.get("issues", [])
        issues = [issue for issue in raw_issues if isinstance(issue, dict)] if isinstance(raw_issues, list) else []
        score = _safe_int(payload.get("coherence_score"), default=5)

        if score < 3 and issues:
            review_notes = []
            for issue in issues:
                description = str(issue.get("description", "") or "").strip()
                suggestion = str(issue.get("fix_suggestion", "") or "").strip()
                instruction = f"[连贯性] {description}"
                if suggestion:
                    instruction = f"{instruction}: {suggestion}"
                review_notes.append(
                    {
                        "section": str(issue.get("location", "") or ""),
                        "severity": 4,
                        "instruction": instruction,
                        "lens": "coherence",
                    }
                )
            return {
                "coherence_issues": issues,
                "review_notes": review_notes,
            }

        return {"coherence_issues": issues}

    return coherence_check
