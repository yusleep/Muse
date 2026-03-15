"""Meta-prompts for prompt optimization."""

from __future__ import annotations

import json


def optimizer_meta_prompt(
    prompt_name: str,
    current_prompt: str,
    weaknesses: list[str],
) -> tuple[str, str]:
    system = (
        "You are a prompt optimizer for academic writing agents.\n"
        "Return JSON with key improved_prompt.\n"
        "Preserve the original schema/output contract while strengthening the prompt "
        "for the provided weaknesses."
    )
    user = json.dumps(
        {
            "prompt_name": prompt_name,
            "current_prompt": current_prompt,
            "weaknesses": weaknesses,
        },
        ensure_ascii=False,
    )
    return system, user
