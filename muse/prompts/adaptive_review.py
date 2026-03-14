from __future__ import annotations

import json
from typing import Any

from muse.prompts.global_review import global_review_prompt_for_lens


def adaptive_review_prompt(
    merged_text: str,
    lens: str,
    review_history: list[dict[str, Any]],
    iteration: int,
) -> tuple[str, str]:
    system, user = global_review_prompt_for_lens(merged_text, lens)

    history_context = ""
    if review_history:
        previous = review_history[-1]
        previous_iteration = previous.get("iteration", "?")
        previous_scores = previous.get("scores", {})
        if not isinstance(previous_scores, dict):
            previous_scores = {}
        previous_notes = str(previous.get("notes_summary", "")).strip() or "No prior notes recorded."
        history_context = "\n\n".join(
            [
                f"Previous review round (iteration {previous_iteration})",
                f"Previous scores: {json.dumps(previous_scores, ensure_ascii=False)}",
                f"Previous key issues: {previous_notes}",
                "For this round:",
                "1. First check whether the previously reported issues were actually fixed.",
                "2. Do not repeat resolved issues.",
                "3. If the same issue remains, escalate the severity instead of repeating it unchanged.",
                "4. Report newly discovered issues normally.",
                "5. Set is_recurring to true when an issue is still unresolved from the prior round.",
            ]
        )

    system = "\n\n".join(
        part
        for part in (
            system,
            f"Current review iteration: {iteration}",
            history_context,
        )
        if part
    )
    return system, user
