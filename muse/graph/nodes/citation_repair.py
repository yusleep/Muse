"""Repair flagged citation markers before polish."""

from __future__ import annotations

import re
from typing import Any


_LATEX_CITE_PATTERN = re.compile(r"\\cite\{([^}]*)\}")


def _clean_latex_cite_block(match: re.Match[str], flagged_keys: set[str]) -> str:
    raw_keys = [part.strip() for part in match.group(1).split(",")]
    kept_keys = [key for key in raw_keys if key and key not in flagged_keys]
    if not kept_keys:
        return ""
    return r"\cite{" + ",".join(kept_keys) + "}"


def _remove_citations(text: str, flagged_keys: set[str]) -> str:
    cleaned = str(text or "")
    normalized_keys = {
        str(item).strip() for item in flagged_keys if str(item).strip()
    }
    for key in sorted(normalized_keys, key=len, reverse=True):
        cleaned = re.sub(rf"\[\s*{re.escape(key)}\s*\]", "", cleaned)

    cleaned = _LATEX_CITE_PATTERN.sub(
        lambda match: _clean_latex_cite_block(match, normalized_keys),
        cleaned,
    )
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\\cite\{\s*\}", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def build_citation_repair_node():
    def citation_repair(state: dict[str, Any]) -> dict[str, Any]:
        flagged = state.get("flagged_citations", [])
        if not isinstance(flagged, list) or not flagged:
            return {"_citation_repair_attempted": True}

        flagged_keys = {
            str(entry.get("cite_key", "")).strip()
            for entry in flagged
            if isinstance(entry, dict) and str(entry.get("cite_key", "")).strip()
        }
        if not flagged_keys:
            return {"_citation_repair_attempted": True}

        repaired_chapters: dict[str, dict[str, Any]] = {}
        chapters = state.get("chapters", {})
        if isinstance(chapters, dict):
            for chapter_id, chapter_data in chapters.items():
                if not isinstance(chapter_data, dict):
                    continue
                chapter_copy = dict(chapter_data)
                if "merged_text" in chapter_copy:
                    chapter_copy["merged_text"] = _remove_citations(
                        str(chapter_copy.get("merged_text", "")),
                        flagged_keys,
                    )
                repaired_chapters[str(chapter_id)] = chapter_copy

        citation_uses = state.get("citation_uses", [])
        if not isinstance(citation_uses, list):
            citation_uses = []
        filtered_uses = [
            dict(item)
            for item in citation_uses
            if isinstance(item, dict)
            and str(item.get("cite_key", "")).strip() not in flagged_keys
        ]

        return {
            "chapters": repaired_chapters,
            "final_text": _remove_citations(str(state.get("final_text", "")), flagged_keys),
            "citation_uses": filtered_uses,
            "_citation_repair_attempted": True,
        }

    return citation_repair
