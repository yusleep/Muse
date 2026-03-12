"""MemoryMiddleware for automatic memory injection and extraction."""

from __future__ import annotations

import logging
from typing import Any

from muse.memory.extractors import (
    extract_from_citation_subgraph,
    extract_from_hitl_feedback,
    extract_from_initialize,
)
from muse.memory.prompt import select_memories
from muse.memory.store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)

_DEFAULT_EXTRACTION_TRIGGERS = frozenset(
    {
        "initialize",
        "review_refs",
        "approve_outline",
        "review_draft",
        "citation_subgraph",
    }
)


class MemoryMiddleware:
    """Middleware that injects and extracts memories around node invocations."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        token_budget: int = 2000,
        extraction_triggers: frozenset[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self._store = store
        self._token_budget = token_budget
        self._triggers = extraction_triggers or _DEFAULT_EXTRACTION_TRIGGERS
        self._enabled = enabled

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def before_invoke(
        self,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject relevant memories into config and state for downstream nodes."""

        if not self._enabled:
            return state

        memory_text = select_memories(
            self._store,
            min_confidence=0.1,
            token_budget=self._token_budget,
        )
        if not memory_text:
            return state

        configurable = config.get("configurable", {})
        configurable["memory_context"] = memory_text
        config["configurable"] = configurable

        updated_state = dict(state)
        updated_state["memory_context"] = memory_text
        logger.debug("Injected %d chars of memory context", len(memory_text))
        return updated_state

    async def after_invoke(
        self,
        state: dict[str, Any],
        result: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract memories from the result at configured trigger nodes."""

        if not self._enabled:
            return result

        node_name = config.get("configurable", {}).get("node_name", "")
        if node_name not in self._triggers:
            return result

        run_id = state.get("project_id") or config.get("configurable", {}).get("thread_id")
        extracted = self._extract_memories(node_name, state, result, run_id)
        for entry in extracted:
            try:
                self._store.upsert(entry)
                logger.debug("Extracted memory: [%s] %s", entry.category, entry.key)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to store extracted memory '%s': %s", entry.key, exc)
        return result

    def _extract_memories(
        self,
        node_name: str,
        state: dict[str, Any],
        result: dict[str, Any],
        run_id: str | None,
    ) -> list[MemoryEntry]:
        if node_name == "initialize":
            return extract_from_initialize(state, run_id)
        if node_name in ("review_refs", "approve_outline", "review_draft"):
            return extract_from_hitl_feedback(node_name, result, run_id)
        if node_name == "citation_subgraph":
            return extract_from_citation_subgraph(state, result, run_id)
        return []
