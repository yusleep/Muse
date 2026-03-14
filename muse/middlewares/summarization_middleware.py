"""Codex-style local compaction middleware for large graph state payloads."""

from __future__ import annotations

import json
from typing import Any

COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary "
    "for another LLM that will resume the task. Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue"
)

SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary "
    "of its thinking process. Use this to build on the work that has already been "
    "done and avoid duplicating work."
)

_BYTES_PER_TOKEN = 4
_DEFAULT_PRESERVE_KEYS = (
    "project_id",
    "topic",
    "discipline",
    "language",
    "format_standard",
    "output_format",
    "references",
    "search_queries",
    "literature_summary",
    "outline",
    "chapter_plans",
    "chapters",
    "dispatch_history",
    "citation_uses",
    "citation_ledger",
    "claim_text_by_id",
    "verified_citations",
    "flagged_citations",
    "paper_package",
    "final_text",
    "abstract_zh",
    "abstract_en",
    "keywords_zh",
    "keywords_en",
    "output_filepath",
    "export_artifacts",
    "evolution_report",
    "review_feedback",
    "rag_enabled",
    "local_refs_count",
)


def estimate_tokens(text: str) -> int:
    """Estimate token count via Codex CLI's 4 bytes/token heuristic."""

    return len(text.encode("utf-8")) // _BYTES_PER_TOKEN


class SummarizationMiddleware:
    """Compact oversized state into a preserved-key subset plus summary."""

    def __init__(
        self,
        llm: Any,
        context_window: int,
        threshold_ratio: float = 0.9,
        recent_tokens: int = 20_000,
        preserve_keys: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._context_window = context_window
        self._threshold_ratio = threshold_ratio
        self._recent_tokens = recent_tokens
        self._preserve_keys = (
            tuple(preserve_keys) if preserve_keys is not None else _DEFAULT_PRESERVE_KEYS
        )

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        del config
        if self._llm is None:
            return state

        serialized = json.dumps(state, ensure_ascii=False, default=str)
        token_count = estimate_tokens(serialized)
        threshold = int(self._context_window * self._threshold_ratio)
        if token_count <= threshold:
            return state

        try:
            summary_text = self._llm.text(
                system=COMPACTION_PROMPT,
                user=serialized[: self._recent_tokens * _BYTES_PER_TOKEN],
                route="default",
                max_tokens=2000,
            )
        except Exception:
            # Compaction is best-effort; if LLM call fails, proceed with original state.
            return state

        compacted = {
            key: state[key] for key in self._preserve_keys if key in state
        }
        compacted["_compaction_summary"] = f"{SUMMARY_PREFIX}\n\n{summary_text}"
        return compacted

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        del state, config
        return result
