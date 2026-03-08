"""MemoryMiddleware for automatic memory injection and extraction."""

from __future__ import annotations

import logging
import re
from typing import Any

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
        entries: list[MemoryEntry] = []
        if node_name == "initialize":
            entries.extend(self._extract_from_initialize(state, run_id))
        elif node_name in ("review_refs", "approve_outline", "review_draft"):
            entries.extend(self._extract_from_hitl(node_name, state, result, run_id))
        elif node_name == "citation_subgraph":
            entries.extend(self._extract_from_citations(state, result, run_id))
        return entries

    def _extract_from_initialize(
        self,
        state: dict[str, Any],
        run_id: str | None,
    ) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        topic = str(state.get("topic", "")).strip()
        discipline = str(state.get("discipline", "")).strip()
        language = str(state.get("language", "")).strip()

        if topic:
            entries.append(
                MemoryEntry(
                    id="",
                    key=f"topic:{_slugify(topic)}",
                    category="fact",
                    content=f"Research topic: {topic}",
                    confidence=0.7,
                    source_run=run_id,
                )
            )
        if discipline:
            entries.append(
                MemoryEntry(
                    id="",
                    key=f"discipline:{_slugify(discipline)}",
                    category="fact",
                    content=f"Academic discipline: {discipline}",
                    confidence=0.7,
                    source_run=run_id,
                )
            )
        if language:
            entries.append(
                MemoryEntry(
                    id="",
                    key=f"language_pref:{language}",
                    category="user_pref",
                    content=f"Writing language: {language}",
                    confidence=0.8,
                    source_run=run_id,
                )
            )
        return entries

    def _extract_from_hitl(
        self,
        node_name: str,
        state: dict[str, Any],
        result: dict[str, Any],
        run_id: str | None,
    ) -> list[MemoryEntry]:
        del state
        entries: list[MemoryEntry] = []
        feedback_list = result.get("review_feedback", [])
        if not isinstance(feedback_list, list):
            return entries

        for feedback in feedback_list:
            if not isinstance(feedback, dict):
                continue
            notes = str(feedback.get("notes", "")).strip()
            if not notes or len(notes) < 10:
                continue
            entries.append(
                MemoryEntry(
                    id="",
                    key=f"feedback:{node_name}:{_slugify(notes[:40])}",
                    category="feedback_pattern",
                    content=f"User feedback at {node_name}: {notes}",
                    confidence=0.6,
                    source_run=run_id,
                )
            )
        return entries

    def _extract_from_citations(
        self,
        state: dict[str, Any],
        result: dict[str, Any],
        run_id: str | None,
    ) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        verified = result.get("verified_citations", [])
        if not isinstance(verified, list):
            return entries

        references = {
            reference.get("ref_id"): reference
            for reference in state.get("references", [])
            if isinstance(reference, dict) and reference.get("ref_id")
        }

        for cite_key in verified:
            if not isinstance(cite_key, str):
                continue
            reference = references.get(cite_key, {})
            doi = reference.get("doi", "")
            title = reference.get("title", cite_key)
            content = f"Verified citation: {title}"
            if doi:
                content += f" (DOI: {doi})"
            entries.append(
                MemoryEntry(
                    id="",
                    key=f"cite:{cite_key}",
                    category="citation",
                    content=content,
                    confidence=0.9,
                    source_run=run_id,
                )
            )
        return entries


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "_", text.lower())
    return slug[:60].strip("_")
