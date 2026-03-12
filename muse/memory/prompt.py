"""Format memory entries for system prompt injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from muse.memory.store import MemoryEntry, MemoryStore

_BYTES_PER_TOKEN = 4

_MEMORY_HEADER = (
    "## Remembered Context\n"
    "The following are remembered facts and preferences from previous sessions.\n"
    "Use these to maintain consistency and respect user preferences.\n"
)

_CATEGORY_LABELS = {
    "user_pref": "User Preferences",
    "writing_style": "Writing Style",
    "citation": "Verified Citations",
    "feedback_pattern": "Feedback Patterns",
    "fact": "Domain Knowledge",
}


def format_memory(entries: list[MemoryEntry]) -> str:
    """Convert memory entries into a formatted prompt section."""

    if not entries:
        return ""

    groups: dict[str, list[MemoryEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.category, []).append(entry)

    lines = [_MEMORY_HEADER]
    for category, label in _CATEGORY_LABELS.items():
        group_entries = groups.get(category, [])
        if not group_entries:
            continue
        lines.append(f"\n### {label}")
        for entry in sorted(group_entries, key=lambda item: item.confidence, reverse=True):
            lines.append(f"- [{_confidence_marker(entry.confidence)}] {entry.content}")
    return "\n".join(lines)


def truncate_to_budget(text: str, token_budget: int) -> str:
    """Truncate formatted memory text to fit within the token budget."""

    max_bytes = token_budget * _BYTES_PER_TOKEN
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    last_newline = truncated.rfind("\n")
    if last_newline > 0:
        truncated = truncated[:last_newline]
    return truncated + "\n... (memory truncated to fit token budget)"


def select_memories(
    store: MemoryStore,
    *,
    categories: list[str] | None = None,
    min_confidence: float = 0.1,
    token_budget: int = 2000,
) -> str:
    """Query, format, and truncate memory context in one call."""

    entries: list[MemoryEntry] = []
    if categories:
        for category in categories:
            entries.extend(store.query(category=category, min_confidence=min_confidence))
    else:
        entries.extend(store.query(min_confidence=min_confidence))

    if not entries:
        return ""

    seen_keys: set[str] = set()
    unique: list[MemoryEntry] = []
    for entry in entries:
        if entry.key in seen_keys:
            continue
        seen_keys.add(entry.key)
        unique.append(entry)

    formatted = format_memory(unique)
    return truncate_to_budget(formatted, token_budget)


def _confidence_marker(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"
