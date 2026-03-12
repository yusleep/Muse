from __future__ import annotations

import json


def search_queries_prompt(topic: str, discipline: str, count: int) -> tuple[str, str]:
    system = (
        f"Generate {count} diverse English academic search queries for the given research topic. "
        "Cover: core concepts, methodology, sub-topics, related fields, and key debates. "
        "Return JSON with key: queries (list of strings)."
    )
    user = json.dumps({"topic": topic, "discipline": discipline}, ensure_ascii=False)
    return system, user
