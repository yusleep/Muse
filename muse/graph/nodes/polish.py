"""Polish helpers and node for graph-native execution."""

from __future__ import annotations

import json
from typing import Any


def _chapter_results_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    chapter_results = state.get("chapter_results")
    if isinstance(chapter_results, list):
        return chapter_results
    paper_package = state.get("paper_package", {})
    if isinstance(paper_package, dict):
        nested = paper_package.get("chapter_results")
        if isinstance(nested, list):
            return nested
    return []


def _polish_chapters(state: dict[str, Any], llm_client: Any) -> dict[str, Any]:
    chapter_results = _chapter_results_from_state(state)
    polished_chapters: list[str] = []
    all_notes: list[str] = []

    system = (
        "Polish the academic thesis chapter for consistency and clarity. "
        "Do not alter core claims. Return JSON with keys: final_text, polish_notes (list)."
    )

    for chapter in chapter_results:
        chapter_title = chapter.get("chapter_title", "")
        chapter_text = chapter.get("merged_text", "")
        if not chapter_text.strip():
            polished_chapters.append(chapter_text)
            continue

        user = json.dumps(
            {
                "language": state.get("language", "zh"),
                "format_standard": state.get("format_standard", "GB/T 7714-2015"),
                "chapter_title": chapter_title,
                "text": chapter_text,
            },
            ensure_ascii=False,
        )
        try:
            out = llm_client.structured(system=system, user=user, route="polish", max_tokens=4500)
            polished = out.get("final_text") if isinstance(out, dict) else None
            notes = out.get("polish_notes", []) if isinstance(out, dict) else []
            if isinstance(polished, str) and polished.strip():
                polished_chapters.append(polished)
            else:
                polished_chapters.append(chapter_text)
            if isinstance(notes, list):
                all_notes.extend(f"[{chapter_title}] {note}" for note in notes)
        except Exception:
            polished_chapters.append(chapter_text)

    return {
        "final_text": "\n\n".join(polished_chapters),
        "polish_notes": all_notes,
    }


def _generate_abstracts(state: dict[str, Any], llm_client: Any) -> dict[str, Any]:
    final_text = state.get("final_text", "")
    if not final_text.strip():
        return {}

    topic = state.get("topic", "")
    text_snippet = final_text[:8000]
    outputs: dict[str, Any] = {}

    try:
        zh_resp = llm_client.structured(
            system=(
                "你是一位学术论文摘要撰写专家。根据以下论文全文，生成一段300-500字的中文摘要和3-5个关键词。"
                "返回JSON，keys: abstract (string), keywords (list of strings)。"
            ),
            user=json.dumps({"topic": topic, "text": text_snippet}, ensure_ascii=False),
            route="polish",
            max_tokens=2000,
        )
        if isinstance(zh_resp, dict):
            outputs["abstract_zh"] = str(zh_resp.get("abstract", ""))
            keywords = zh_resp.get("keywords", [])
            outputs["keywords_zh"] = [str(keyword) for keyword in keywords] if isinstance(keywords, list) else []
    except Exception:
        pass

    try:
        en_resp = llm_client.structured(
            system=(
                "You are an academic abstract writer. Generate a 200-300 word English abstract "
                "and 3-5 keywords for the following thesis. "
                "Return JSON with keys: abstract (string), keywords (list of strings)."
            ),
            user=json.dumps({"topic": topic, "text": text_snippet}, ensure_ascii=False),
            route="polish",
            max_tokens=2000,
        )
        if isinstance(en_resp, dict):
            outputs["abstract_en"] = str(en_resp.get("abstract", ""))
            keywords = en_resp.get("keywords", [])
            outputs["keywords_en"] = [str(keyword) for keyword in keywords] if isinstance(keywords, list) else []
    except Exception:
        pass

    return outputs


def _run_polish(state: dict[str, Any], llm_client: Any) -> dict[str, Any]:
    outputs = _polish_chapters(state, llm_client)
    outputs.update(_generate_abstracts({**state, **outputs}, llm_client))
    return outputs


def build_polish_node(services: Any):
    def polish(state: dict[str, Any]) -> dict[str, Any]:
        temp_state = {
            "topic": state.get("topic", ""),
            "language": state.get("language", "zh"),
            "format_standard": state.get("format_standard", "GB/T 7714-2015"),
            "paper_package": state.get("paper_package", {}),
            "chapter_results": state.get("paper_package", {}).get("chapter_results", []),
            "final_text": state.get("final_text", ""),
        }
        return _run_polish(temp_state, getattr(services, "llm", None))

    return polish
