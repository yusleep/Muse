# Phase 5: Memory System

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persistent cross-session memory for user preferences, writing styles, verified citations, and feedback patterns.

**Architecture:** SQLite-based MemoryStore with category/confidence filtering. MemoryMiddleware auto-injects relevant memories into system prompts (2000-token budget) and auto-extracts new memories from interactions.

**Tech Stack:** SQLite, Python 3.10

**Depends on:** Phase 0-B (Middleware framework)

---

## Task 1: Create MemoryEntry dataclass and MemoryStore (`muse/memory/store.py`)

**Files:**
- `muse/memory/__init__.py` (create)
- `muse/memory/store.py` (create)
- `tests/test_memory_store.py` (create)

**What to do:**

Create the `MemoryEntry` dataclass and `MemoryStore` class backed by SQLite. The store supports CRUD operations with filtering by category and confidence threshold. Database is created lazily at `~/.muse/memory.sqlite` by default.

Create `muse/memory/__init__.py`:

```python
"""Persistent memory system for Muse."""
```

Create `muse/memory/store.py`:

```python
"""SQLite-backed memory store for persistent cross-session memory.

Each memory is a categorized text snippet with a confidence score.
Memories are used to personalize LLM prompts across runs.

Database location: ``~/.muse/memory.sqlite`` (configurable).
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Valid memory categories
CATEGORIES = frozenset({
    "user_pref",        # User preferences (style, format, language)
    "writing_style",    # Writing style observations
    "citation",         # Verified citation DOIs and metadata
    "feedback_pattern", # Recurring feedback patterns from HITL
    "fact",             # Factual knowledge about the user's domain
})


@dataclass
class MemoryEntry:
    """A single memory record.

    Attributes:
        id: Unique identifier (UUID4 hex).
        key: Short descriptor / dedup key (e.g., "prefers_formal_tone").
        category: One of CATEGORIES.
        content: The memory text injected into prompts.
        confidence: Score 0.0-1.0. Higher = more trusted.
        source_run: Run ID that created/updated this memory, or None.
        created_at: UTC timestamp of creation.
        updated_at: UTC timestamp of last update.
    """
    id: str
    key: str
    category: str
    content: str
    confidence: float
    source_run: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'. Must be one of: {', '.join(sorted(CATEGORIES))}"
            )
        self.confidence = max(0.0, min(1.0, self.confidence))


class MemoryStore:
    """SQLite-backed persistent memory store.

    Usage::

        store = MemoryStore()                     # uses ~/.muse/memory.sqlite
        store = MemoryStore("/path/to/mem.db")    # custom path
        store = MemoryStore(":memory:")            # in-memory (for tests)

        store.upsert(MemoryEntry(...))
        memories = store.query(category="user_pref", min_confidence=0.3)
        store.delete("memory-id")
    """

    _SCHEMA = """\
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        key TEXT NOT NULL,
        category TEXT NOT NULL,
        content TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 0.5,
        source_run TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
    CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
    CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_dir = Path.home() / ".muse"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "memory.sqlite"
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(self._SCHEMA)
        self._conn.commit()

    def upsert(self, entry: MemoryEntry) -> None:
        """Insert or update a memory entry.

        If an entry with the same ``key`` exists, it is updated.
        Otherwise, a new entry is inserted.
        """
        now = datetime.now(timezone.utc).isoformat()
        existing = self._find_by_key(entry.key)
        if existing is not None:
            self._conn.execute(
                """\
                UPDATE memories
                SET content = ?, confidence = ?, category = ?,
                    source_run = ?, updated_at = ?
                WHERE id = ?
                """,
                (entry.content, entry.confidence, entry.category,
                 entry.source_run, now, existing.id),
            )
        else:
            entry_id = entry.id or uuid.uuid4().hex
            self._conn.execute(
                """\
                INSERT INTO memories (id, key, category, content, confidence,
                                      source_run, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_id, entry.key, entry.category, entry.content,
                 entry.confidence, entry.source_run, now, now),
            )
        self._conn.commit()

    def query(
        self,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Query memories by category and confidence threshold.

        Results are sorted by confidence (descending), then updated_at (descending).
        """
        conditions = ["confidence >= ?"]
        params: list[Any] = [min_confidence]

        if category is not None:
            conditions.append("category = ?")
            params.append(category)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        rows = self._conn.execute(
            f"""\
            SELECT id, key, category, content, confidence,
                   source_run, created_at, updated_at
            FROM memories
            WHERE {where_clause}
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a single memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def update_confidence(self, memory_id: str, delta: float) -> None:
        """Adjust confidence by *delta*. Clamped to [0.0, 1.0]."""
        entry = self.get(memory_id)
        if entry is None:
            return
        new_conf = max(0.0, min(1.0, entry.confidence + delta))
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
            (new_conf, now, memory_id),
        )
        self._conn.commit()

    def set_confidence(self, memory_id: str, value: float) -> None:
        """Set confidence to an absolute value. Clamped to [0.0, 1.0]."""
        value = max(0.0, min(1.0, value))
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
            (value, now, memory_id),
        )
        self._conn.commit()

    def decay_old_memories(self, days: int = 90, factor: float = 0.9) -> int:
        """Multiply confidence by *factor* for memories not updated in *days*.

        Returns the number of memories decayed.
        """
        cutoff = datetime.now(timezone.utc)
        # Calculate cutoff as ISO string
        from datetime import timedelta
        threshold = (cutoff - timedelta(days=days)).isoformat()
        cursor = self._conn.execute(
            """\
            UPDATE memories
            SET confidence = confidence * ?, updated_at = updated_at
            WHERE updated_at < ? AND confidence > 0.01
            """,
            (factor, threshold),
        )
        self._conn.commit()
        return cursor.rowcount

    def count(self, *, category: str | None = None) -> int:
        """Count memories, optionally filtered by category."""
        if category:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE category = ?", (category,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _find_by_key(self, key: str) -> MemoryEntry | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE key = ?", (key,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            key=row["key"],
            category=row["category"],
            content=row["content"],
            confidence=row["confidence"],
            source_run=row["source_run"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
```

Create `tests/test_memory_store.py`:

```python
"""Tests for MemoryStore (muse.memory.store)."""

from __future__ import annotations

import pytest

from muse.memory.store import CATEGORIES, MemoryEntry, MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


class TestMemoryEntry:
    def test_valid_category(self):
        e = MemoryEntry(id="1", key="k", category="user_pref", content="c", confidence=0.5)
        assert e.category == "user_pref"

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Invalid category"):
            MemoryEntry(id="1", key="k", category="bad", content="c", confidence=0.5)

    def test_confidence_clamped_high(self):
        e = MemoryEntry(id="1", key="k", category="fact", content="c", confidence=1.5)
        assert e.confidence == 1.0

    def test_confidence_clamped_low(self):
        e = MemoryEntry(id="1", key="k", category="fact", content="c", confidence=-0.2)
        assert e.confidence == 0.0


class TestMemoryStoreUpsert:
    def test_insert_new(self, store):
        entry = MemoryEntry(id="a1", key="formal_tone", category="user_pref",
                            content="User prefers formal academic tone", confidence=0.8)
        store.upsert(entry)
        assert store.count() == 1

    def test_upsert_updates_existing_key(self, store):
        e1 = MemoryEntry(id="a1", key="tone", category="user_pref",
                         content="formal", confidence=0.5)
        store.upsert(e1)

        e2 = MemoryEntry(id="a2", key="tone", category="user_pref",
                         content="very formal", confidence=0.9)
        store.upsert(e2)

        assert store.count() == 1
        results = store.query()
        assert results[0].content == "very formal"
        assert results[0].confidence == 0.9

    def test_different_keys_both_stored(self, store):
        store.upsert(MemoryEntry(id="a1", key="k1", category="fact", content="c1", confidence=0.5))
        store.upsert(MemoryEntry(id="a2", key="k2", category="fact", content="c2", confidence=0.5))
        assert store.count() == 2


class TestMemoryStoreQuery:
    def test_filter_by_category(self, store):
        store.upsert(MemoryEntry(id="1", key="k1", category="user_pref", content="c1", confidence=0.5))
        store.upsert(MemoryEntry(id="2", key="k2", category="citation", content="c2", confidence=0.5))
        results = store.query(category="user_pref")
        assert len(results) == 1
        assert results[0].key == "k1"

    def test_filter_by_min_confidence(self, store):
        store.upsert(MemoryEntry(id="1", key="k1", category="fact", content="c1", confidence=0.2))
        store.upsert(MemoryEntry(id="2", key="k2", category="fact", content="c2", confidence=0.8))
        results = store.query(min_confidence=0.5)
        assert len(results) == 1
        assert results[0].key == "k2"

    def test_sorted_by_confidence_desc(self, store):
        store.upsert(MemoryEntry(id="1", key="k1", category="fact", content="c1", confidence=0.3))
        store.upsert(MemoryEntry(id="2", key="k2", category="fact", content="c2", confidence=0.9))
        store.upsert(MemoryEntry(id="3", key="k3", category="fact", content="c3", confidence=0.6))
        results = store.query()
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_limit(self, store):
        for i in range(10):
            store.upsert(MemoryEntry(id=f"m{i}", key=f"k{i}", category="fact",
                                     content=f"c{i}", confidence=0.5))
        results = store.query(limit=3)
        assert len(results) == 3

    def test_empty_store_returns_empty(self, store):
        assert store.query() == []


class TestMemoryStoreGet:
    def test_get_existing(self, store):
        store.upsert(MemoryEntry(id="x1", key="k1", category="fact", content="c1", confidence=0.5))
        entry = store.get("x1")
        assert entry is not None
        assert entry.content == "c1"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None


class TestMemoryStoreDelete:
    def test_delete_existing(self, store):
        store.upsert(MemoryEntry(id="x1", key="k1", category="fact", content="c1", confidence=0.5))
        assert store.delete("x1") is True
        assert store.count() == 0

    def test_delete_missing_returns_false(self, store):
        assert store.delete("nonexistent") is False


class TestMemoryStoreConfidence:
    def test_update_confidence_increments(self, store):
        store.upsert(MemoryEntry(id="x1", key="k1", category="fact", content="c1", confidence=0.5))
        store.update_confidence("x1", delta=0.1)
        entry = store.get("x1")
        assert entry is not None
        assert abs(entry.confidence - 0.6) < 0.001

    def test_update_confidence_clamped_at_1(self, store):
        store.upsert(MemoryEntry(id="x1", key="k1", category="fact", content="c1", confidence=0.95))
        store.update_confidence("x1", delta=0.2)
        entry = store.get("x1")
        assert entry is not None
        assert entry.confidence == 1.0

    def test_update_confidence_clamped_at_0(self, store):
        store.upsert(MemoryEntry(id="x1", key="k1", category="fact", content="c1", confidence=0.1))
        store.update_confidence("x1", delta=-0.5)
        entry = store.get("x1")
        assert entry is not None
        assert entry.confidence == 0.0

    def test_set_confidence_absolute(self, store):
        store.upsert(MemoryEntry(id="x1", key="k1", category="fact", content="c1", confidence=0.5))
        store.set_confidence("x1", 0.0)
        entry = store.get("x1")
        assert entry is not None
        assert entry.confidence == 0.0

    def test_decay_old_memories(self, store):
        # Insert memory with old updated_at
        store.upsert(MemoryEntry(id="old", key="old_k", category="fact",
                                 content="old", confidence=1.0))
        # Manually set updated_at to 100 days ago
        from datetime import datetime, timedelta, timezone
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        store._conn.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?", (old_date, "old")
        )
        store._conn.commit()

        decayed = store.decay_old_memories(days=90, factor=0.9)
        assert decayed == 1
        entry = store.get("old")
        assert entry is not None
        assert abs(entry.confidence - 0.9) < 0.001


class TestMemoryStoreCount:
    def test_count_all(self, store):
        store.upsert(MemoryEntry(id="1", key="k1", category="fact", content="c", confidence=0.5))
        store.upsert(MemoryEntry(id="2", key="k2", category="user_pref", content="c", confidence=0.5))
        assert store.count() == 2

    def test_count_by_category(self, store):
        store.upsert(MemoryEntry(id="1", key="k1", category="fact", content="c", confidence=0.5))
        store.upsert(MemoryEntry(id="2", key="k2", category="fact", content="c", confidence=0.5))
        store.upsert(MemoryEntry(id="3", key="k3", category="user_pref", content="c", confidence=0.5))
        assert store.count(category="fact") == 2
        assert store.count(category="user_pref") == 1
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_memory_store.py -x` -- fails because files do not exist.
2. GREEN: Create files. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 5 minutes.

---

## Task 2: Create memory formatting (`muse/memory/prompt.py`)

**Files:**
- `muse/memory/prompt.py` (create)
- `tests/test_memory_prompt.py` (create)

**What to do:**

Create the functions that format memory entries for system prompt injection. `format_memory()` converts a list of `MemoryEntry` objects into a prompt-friendly string. `truncate_to_budget()` ensures the formatted output stays within a token budget (estimated at 4 bytes per token, matching the Codex CLI heuristic used in the design doc).

Create `muse/memory/prompt.py`:

```python
"""Format memory entries for system prompt injection.

Functions:
    format_memory     -- Convert MemoryEntry list to prompt-ready string
    truncate_to_budget -- Trim to token budget (4 bytes/token heuristic)
    select_memories   -- Query and format in one call
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from muse.memory.store import MemoryEntry, MemoryStore

# Codex CLI heuristic: 1 token ~ 4 bytes of text
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
    """Convert memory entries into a formatted system prompt section.

    Entries are grouped by category, sorted by confidence (descending).
    Each entry shows key and content.

    Returns:
        Formatted string ready for system prompt injection.
        Empty string if no entries.
    """
    if not entries:
        return ""

    # Group by category
    groups: dict[str, list[MemoryEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.category, []).append(entry)

    lines = [_MEMORY_HEADER]

    for category, label in _CATEGORY_LABELS.items():
        group_entries = groups.get(category, [])
        if not group_entries:
            continue
        lines.append(f"\n### {label}")
        for entry in sorted(group_entries, key=lambda e: e.confidence, reverse=True):
            confidence_marker = _confidence_marker(entry.confidence)
            lines.append(f"- [{confidence_marker}] {entry.content}")

    return "\n".join(lines)


def truncate_to_budget(text: str, token_budget: int) -> str:
    """Truncate formatted memory text to fit within *token_budget*.

    Uses the 4 bytes/token heuristic (same as Codex CLI local compaction).
    Truncates at the last complete line that fits within the budget.

    Args:
        text: Formatted memory text.
        token_budget: Maximum number of tokens.

    Returns:
        Truncated text. May be shorter than input.
    """
    max_bytes = token_budget * _BYTES_PER_TOKEN
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    # Truncate at line boundary
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
    """Query the store and return formatted, truncated memory text.

    This is the all-in-one helper for MemoryMiddleware.

    Args:
        store: The MemoryStore instance.
        categories: Limit to these categories. None = all.
        min_confidence: Minimum confidence threshold.
        token_budget: Max tokens for the memory section.

    Returns:
        Formatted memory string, truncated to budget.
        Empty string if no memories found.
    """
    entries: list[MemoryEntry] = []
    if categories:
        for cat in categories:
            entries.extend(store.query(category=cat, min_confidence=min_confidence))
    else:
        entries.extend(store.query(min_confidence=min_confidence))

    if not entries:
        return ""

    # Deduplicate by key (in case multiple category queries overlap)
    seen_keys: set[str] = set()
    unique: list[MemoryEntry] = []
    for entry in entries:
        if entry.key not in seen_keys:
            seen_keys.add(entry.key)
            unique.append(entry)

    formatted = format_memory(unique)
    return truncate_to_budget(formatted, token_budget)


def _confidence_marker(confidence: float) -> str:
    """Visual confidence indicator."""
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"
```

Create `tests/test_memory_prompt.py`:

```python
"""Tests for memory formatting (muse.memory.prompt)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from muse.memory.store import MemoryEntry, MemoryStore
from muse.memory.prompt import (
    format_memory,
    select_memories,
    truncate_to_budget,
)


def _entry(key, category="fact", content="content", confidence=0.5):
    return MemoryEntry(
        id=key,
        key=key,
        category=category,
        content=content,
        confidence=confidence,
    )


class TestFormatMemory:
    def test_empty_returns_empty(self):
        assert format_memory([]) == ""

    def test_single_entry(self):
        result = format_memory([_entry("k1", content="User prefers APA")])
        assert "User prefers APA" in result
        assert "Remembered Context" in result

    def test_groups_by_category(self):
        entries = [
            _entry("k1", category="user_pref", content="prefers formal"),
            _entry("k2", category="citation", content="DOI verified"),
        ]
        result = format_memory(entries)
        assert "User Preferences" in result
        assert "Verified Citations" in result
        assert "prefers formal" in result
        assert "DOI verified" in result

    def test_confidence_markers(self):
        entries = [
            _entry("k1", confidence=0.9, content="high confidence fact"),
            _entry("k2", confidence=0.5, content="medium confidence fact"),
            _entry("k3", confidence=0.1, content="low confidence fact"),
        ]
        result = format_memory(entries)
        assert "[high]" in result
        assert "[medium]" in result
        assert "[low]" in result

    def test_sorted_by_confidence_within_group(self):
        entries = [
            _entry("k1", confidence=0.3, content="low"),
            _entry("k2", confidence=0.9, content="high"),
        ]
        result = format_memory(entries)
        high_pos = result.index("high")
        low_pos = result.index("low")
        # "high" should appear before "low" (sorted desc)
        assert high_pos < low_pos


class TestTruncateToBudget:
    def test_within_budget_unchanged(self):
        text = "short text"
        assert truncate_to_budget(text, 1000) == text

    def test_truncated_at_line_boundary(self):
        text = "line1\nline2\nline3\nline4\nline5"
        # 4 bytes per token, budget=3 tokens = 12 bytes
        result = truncate_to_budget(text, 3)
        assert "truncated" in result
        assert len(result) < len(text) + 50  # some overhead for truncation message

    def test_large_budget_no_change(self):
        text = "x" * 100
        result = truncate_to_budget(text, 10000)
        assert result == text


class TestSelectMemories:
    @pytest.fixture
    def store(self):
        s = MemoryStore(":memory:")
        s.upsert(_entry("pref1", category="user_pref", content="prefers GB/T 7714", confidence=0.8))
        s.upsert(_entry("style1", category="writing_style", content="formal academic", confidence=0.6))
        s.upsert(_entry("cite1", category="citation", content="10.1234/verified", confidence=0.9))
        s.upsert(_entry("low1", category="fact", content="low conf fact", confidence=0.05))
        yield s
        s.close()

    def test_returns_all_above_threshold(self, store):
        result = select_memories(store, min_confidence=0.1)
        assert "prefers GB/T 7714" in result
        assert "formal academic" in result
        assert "10.1234/verified" in result
        assert "low conf fact" not in result

    def test_filter_by_categories(self, store):
        result = select_memories(store, categories=["user_pref"])
        assert "prefers GB/T 7714" in result
        assert "formal academic" not in result

    def test_respects_token_budget(self, store):
        result = select_memories(store, token_budget=5)
        # Very small budget should truncate
        assert len(result) < 200

    def test_empty_store_returns_empty(self):
        store = MemoryStore(":memory:")
        result = select_memories(store)
        assert result == ""
        store.close()
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_memory_prompt.py -x` -- fails because files do not exist.
2. GREEN: Create files. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 4 minutes.

---

## Task 3: Create MemoryMiddleware (`muse/memory/middleware.py`)

**Files:**
- `muse/memory/middleware.py` (create)
- `tests/test_memory_middleware.py` (create)

**What to do:**

Create the `MemoryMiddleware` class that follows the Middleware protocol defined in Phase 0-B. It has two hooks:

- `before_invoke`: Query relevant memories from the store, format them, and inject into the config dict under a `memory_context` key for `MuseChatModel` to include in system prompts.
- `after_invoke`: At certain nodes (configurable), extract new memories from the interaction result using a simple heuristic (no LLM call in Phase 5 MVP -- LLM extraction is a future enhancement).

Create `muse/memory/middleware.py`:

```python
"""MemoryMiddleware for automatic memory injection and extraction.

Follows the Middleware protocol (Phase 0-B):
    before_invoke(state, config) -> config
    after_invoke(state, result, config) -> result

before_invoke:
    Queries MemoryStore for relevant memories, formats them, and injects
    into config["configurable"]["memory_context"] for MuseChatModel.

after_invoke:
    At configured extraction triggers, extracts new memories from the
    result and upserts them into the store.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from muse.memory.prompt import select_memories
from muse.memory.store import CATEGORIES, MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)

# Nodes after which memory extraction runs
_DEFAULT_EXTRACTION_TRIGGERS = frozenset({
    "initialize",           # extract topic/discipline facts
    "review_refs",          # extract user preferences from HITL feedback
    "approve_outline",      # extract structural preferences
    "review_draft",         # extract feedback patterns
    "citation_subgraph",    # extract verified citations
})


class MemoryMiddleware:
    """Middleware that injects and extracts memories around node invocations.

    Args:
        store: The MemoryStore instance (must be initialized).
        token_budget: Max tokens for injected memory context.
        extraction_triggers: Set of node names that trigger memory extraction.
        enabled: Global enable/disable switch.
    """

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
        """Inject relevant memories into config for the LLM."""
        if not self._enabled:
            return config

        memory_text = select_memories(
            self._store,
            min_confidence=0.1,
            token_budget=self._token_budget,
        )
        if not memory_text:
            return config

        configurable = config.get("configurable", {})
        configurable["memory_context"] = memory_text
        config["configurable"] = configurable

        logger.debug("Injected %d chars of memory context", len(memory_text))
        return config

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
            except Exception as exc:
                logger.warning("Failed to store extracted memory '%s': %s", entry.key, exc)

        return result

    def _extract_memories(
        self,
        node_name: str,
        state: dict[str, Any],
        result: dict[str, Any],
        run_id: str | None,
    ) -> list[MemoryEntry]:
        """Heuristic memory extraction based on node type.

        This is a rule-based MVP. Future enhancement: use LLM for extraction.
        """
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
        """Extract topic/discipline facts from the initialize node."""
        entries: list[MemoryEntry] = []
        topic = state.get("topic", "").strip()
        discipline = state.get("discipline", "").strip()
        language = state.get("language", "").strip()

        if topic:
            entries.append(MemoryEntry(
                id="",
                key=f"topic:{_slugify(topic)}",
                category="fact",
                content=f"Research topic: {topic}",
                confidence=0.7,
                source_run=run_id,
            ))
        if discipline:
            entries.append(MemoryEntry(
                id="",
                key=f"discipline:{_slugify(discipline)}",
                category="fact",
                content=f"Academic discipline: {discipline}",
                confidence=0.7,
                source_run=run_id,
            ))
        if language:
            entries.append(MemoryEntry(
                id="",
                key=f"language_pref:{language}",
                category="user_pref",
                content=f"Writing language: {language}",
                confidence=0.8,
                source_run=run_id,
            ))
        return entries

    def _extract_from_hitl(
        self,
        node_name: str,
        state: dict[str, Any],
        result: dict[str, Any],
        run_id: str | None,
    ) -> list[MemoryEntry]:
        """Extract preferences from HITL feedback nodes."""
        entries: list[MemoryEntry] = []
        feedback_list = result.get("review_feedback", [])
        if not isinstance(feedback_list, list):
            return entries

        for fb in feedback_list:
            if not isinstance(fb, dict):
                continue
            notes = fb.get("notes", "").strip()
            if not notes or len(notes) < 10:
                continue

            key = f"feedback:{node_name}:{_slugify(notes[:40])}"
            entries.append(MemoryEntry(
                id="",
                key=key,
                category="feedback_pattern",
                content=f"User feedback at {node_name}: {notes}",
                confidence=0.6,
                source_run=run_id,
            ))
        return entries

    def _extract_from_citations(
        self,
        state: dict[str, Any],
        result: dict[str, Any],
        run_id: str | None,
    ) -> list[MemoryEntry]:
        """Extract verified citations from citation subgraph results."""
        entries: list[MemoryEntry] = []
        verified = result.get("verified_citations", [])
        if not isinstance(verified, list):
            return entries

        references = {
            ref.get("ref_id"): ref
            for ref in state.get("references", [])
            if isinstance(ref, dict) and ref.get("ref_id")
        }

        for cite_key in verified:
            if not isinstance(cite_key, str):
                continue
            ref = references.get(cite_key, {})
            doi = ref.get("doi", "")
            title = ref.get("title", cite_key)
            content = f"Verified citation: {title}"
            if doi:
                content += f" (DOI: {doi})"

            entries.append(MemoryEntry(
                id="",
                key=f"cite:{cite_key}",
                category="citation",
                content=content,
                confidence=0.9,
                source_run=run_id,
            ))
        return entries


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug for use as memory key."""
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "_", text.lower())
    return slug[:60].strip("_")
```

Create `tests/test_memory_middleware.py`:

```python
"""Tests for MemoryMiddleware (muse.memory.middleware)."""

from __future__ import annotations

import asyncio

import pytest

from muse.memory.middleware import MemoryMiddleware
from muse.memory.store import MemoryEntry, MemoryStore


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def mw(store):
    return MemoryMiddleware(store, token_budget=2000)


class TestBeforeInvoke:
    def test_injects_memory_context(self, store, mw):
        store.upsert(MemoryEntry(
            id="m1", key="tone", category="user_pref",
            content="User prefers formal tone", confidence=0.8,
        ))
        config = {"configurable": {"thread_id": "t1"}}
        result = _run(mw.before_invoke({}, config))
        assert "memory_context" in result["configurable"]
        assert "formal tone" in result["configurable"]["memory_context"]

    def test_no_memories_no_injection(self, store, mw):
        config = {"configurable": {}}
        result = _run(mw.before_invoke({}, config))
        assert "memory_context" not in result.get("configurable", {})

    def test_disabled_middleware_no_injection(self, store):
        store.upsert(MemoryEntry(
            id="m1", key="tone", category="user_pref",
            content="formal", confidence=0.8,
        ))
        mw = MemoryMiddleware(store, enabled=False)
        config = {"configurable": {}}
        result = _run(mw.before_invoke({}, config))
        assert "memory_context" not in result.get("configurable", {})


class TestAfterInvokeInitialize:
    def test_extracts_topic_and_discipline(self, store, mw):
        state = {
            "topic": "Deep Learning for NLP",
            "discipline": "Computer Science",
            "language": "zh",
        }
        config = {"configurable": {"node_name": "initialize", "thread_id": "t1"}}
        _run(mw.after_invoke(state, {}, config))

        memories = store.query()
        keys = {m.key for m in memories}
        assert any("deep_learning" in k for k in keys)
        assert any("computer_science" in k for k in keys)
        assert any("language_pref" in k for k in keys)

    def test_skips_non_trigger_nodes(self, store, mw):
        state = {"topic": "Topic"}
        config = {"configurable": {"node_name": "search"}}
        _run(mw.after_invoke(state, {}, config))
        assert store.count() == 0


class TestAfterInvokeHITL:
    def test_extracts_feedback(self, store, mw):
        state = {}
        result = {
            "review_feedback": [
                {"stage": "review_draft", "notes": "Please use more formal citations in chapter 3"}
            ]
        }
        config = {"configurable": {"node_name": "review_draft", "thread_id": "t1"}}
        _run(mw.after_invoke(state, result, config))

        memories = store.query(category="feedback_pattern")
        assert len(memories) == 1
        assert "formal citations" in memories[0].content

    def test_ignores_short_notes(self, store, mw):
        result = {"review_feedback": [{"notes": "ok"}]}
        config = {"configurable": {"node_name": "review_draft"}}
        _run(mw.after_invoke({}, result, config))
        assert store.count() == 0


class TestAfterInvokeCitations:
    def test_extracts_verified_citations(self, store, mw):
        state = {
            "references": [
                {"ref_id": "@smith2024deep", "title": "Deep Learning Survey", "doi": "10.1234/test"},
            ]
        }
        result = {"verified_citations": ["@smith2024deep"]}
        config = {"configurable": {"node_name": "citation_subgraph", "thread_id": "t1"}}
        _run(mw.after_invoke(state, result, config))

        memories = store.query(category="citation")
        assert len(memories) == 1
        assert "Deep Learning Survey" in memories[0].content
        assert "10.1234/test" in memories[0].content
        assert memories[0].confidence == 0.9

    def test_no_verified_no_extraction(self, store, mw):
        result = {"verified_citations": []}
        config = {"configurable": {"node_name": "citation_subgraph"}}
        _run(mw.after_invoke({}, result, config))
        assert store.count() == 0


class TestMiddlewareDisabled:
    def test_disabled_skips_extraction(self, store):
        mw = MemoryMiddleware(store, enabled=False)
        state = {"topic": "Topic"}
        config = {"configurable": {"node_name": "initialize"}}
        _run(mw.after_invoke(state, {}, config))
        assert store.count() == 0
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_memory_middleware.py -x` -- fails because files do not exist.
2. GREEN: Create files. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 5 minutes.

---

## Task 4: Add confidence lifecycle

**Files:**
- `muse/memory/lifecycle.py` (create)
- `tests/test_memory_lifecycle.py` (create)

**What to do:**

Create lifecycle functions that manage memory confidence over time:
- `confirm_memory(store, key)`: Increment confidence by 0.1 (capped at 1.0)
- `deny_memory(store, key)`: Set confidence to 0.0, then auto-delete
- `run_decay(store)`: Decay memories not updated in 90 days by factor 0.9
- `cleanup_dead_memories(store)`: Delete memories with confidence <= 0.01

Create `muse/memory/lifecycle.py`:

```python
"""Memory confidence lifecycle management.

Provides functions for:
- Confirming memories (increment confidence)
- Denying memories (zero confidence + delete)
- Time-based decay (multiply by factor after N days)
- Cleanup of dead memories (confidence near zero)
"""

from __future__ import annotations

import logging
from typing import Any

from muse.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def confirm_memory(store: MemoryStore, key: str, *, delta: float = 0.1) -> bool:
    """Increment confidence for a memory identified by *key*.

    Returns True if the memory was found and updated.
    """
    entries = store.query()
    for entry in entries:
        if entry.key == key:
            store.update_confidence(entry.id, delta)
            logger.debug("Confirmed memory '%s': confidence += %.2f", key, delta)
            return True
    return False


def deny_memory(store: MemoryStore, key: str) -> bool:
    """Set memory confidence to zero and delete it.

    This is the "user explicitly denied this memory" action.
    Returns True if the memory was found and deleted.
    """
    entries = store.query(min_confidence=0.0)
    for entry in entries:
        if entry.key == key:
            store.delete(entry.id)
            logger.info("Denied and deleted memory '%s'", key)
            return True
    return False


def run_decay(
    store: MemoryStore,
    *,
    days: int = 90,
    factor: float = 0.9,
) -> int:
    """Decay stale memories that have not been updated in *days*.

    Multiplies their confidence by *factor*.
    Returns the number of memories decayed.
    """
    count = store.decay_old_memories(days=days, factor=factor)
    if count:
        logger.info("Decayed %d memories (>%d days old, factor=%.2f)", count, days, factor)
    return count


def cleanup_dead_memories(
    store: MemoryStore,
    *,
    threshold: float = 0.01,
) -> int:
    """Delete memories with confidence at or below *threshold*.

    Returns the number of memories deleted.
    """
    dead = store.query(min_confidence=0.0, limit=1000)
    deleted = 0
    for entry in dead:
        if entry.confidence <= threshold:
            store.delete(entry.id)
            deleted += 1
    if deleted:
        logger.info("Cleaned up %d dead memories (confidence <= %.3f)", deleted, threshold)
    return deleted


def run_maintenance(store: MemoryStore) -> dict[str, int]:
    """Run full maintenance cycle: decay + cleanup.

    Returns a summary dict with counts.
    """
    decayed = run_decay(store)
    cleaned = cleanup_dead_memories(store)
    return {"decayed": decayed, "cleaned": cleaned}
```

Create `tests/test_memory_lifecycle.py`:

```python
"""Tests for memory lifecycle management (muse.memory.lifecycle)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from muse.memory.lifecycle import (
    cleanup_dead_memories,
    confirm_memory,
    deny_memory,
    run_decay,
    run_maintenance,
)
from muse.memory.store import MemoryEntry, MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


class TestConfirmMemory:
    def test_increments_confidence(self, store):
        store.upsert(MemoryEntry(id="m1", key="tone", category="user_pref",
                                 content="formal", confidence=0.5))
        assert confirm_memory(store, "tone") is True
        entry = store.get("m1")
        assert entry is not None
        assert abs(entry.confidence - 0.6) < 0.001

    def test_capped_at_1(self, store):
        store.upsert(MemoryEntry(id="m1", key="tone", category="user_pref",
                                 content="formal", confidence=0.95))
        confirm_memory(store, "tone")
        entry = store.get("m1")
        assert entry is not None
        assert entry.confidence == 1.0

    def test_returns_false_for_missing_key(self, store):
        assert confirm_memory(store, "nonexistent") is False

    def test_custom_delta(self, store):
        store.upsert(MemoryEntry(id="m1", key="tone", category="user_pref",
                                 content="formal", confidence=0.5))
        confirm_memory(store, "tone", delta=0.2)
        entry = store.get("m1")
        assert entry is not None
        assert abs(entry.confidence - 0.7) < 0.001


class TestDenyMemory:
    def test_deletes_memory(self, store):
        store.upsert(MemoryEntry(id="m1", key="wrong_fact", category="fact",
                                 content="incorrect", confidence=0.5))
        assert deny_memory(store, "wrong_fact") is True
        assert store.count() == 0

    def test_returns_false_for_missing(self, store):
        assert deny_memory(store, "nonexistent") is False


class TestRunDecay:
    def test_decays_old_memories(self, store):
        store.upsert(MemoryEntry(id="old", key="old_fact", category="fact",
                                 content="stale", confidence=1.0))
        # Set updated_at to 100 days ago
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        store._conn.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?", (old_date, "old")
        )
        store._conn.commit()

        count = run_decay(store, days=90, factor=0.9)
        assert count == 1
        entry = store.get("old")
        assert entry is not None
        assert abs(entry.confidence - 0.9) < 0.001

    def test_does_not_decay_recent(self, store):
        store.upsert(MemoryEntry(id="new", key="new_fact", category="fact",
                                 content="fresh", confidence=1.0))
        count = run_decay(store, days=90, factor=0.9)
        assert count == 0


class TestCleanupDeadMemories:
    def test_deletes_low_confidence(self, store):
        store.upsert(MemoryEntry(id="dead", key="k1", category="fact",
                                 content="dead", confidence=0.005))
        store.upsert(MemoryEntry(id="alive", key="k2", category="fact",
                                 content="alive", confidence=0.5))
        deleted = cleanup_dead_memories(store, threshold=0.01)
        assert deleted == 1
        assert store.count() == 1
        assert store.get("alive") is not None

    def test_no_dead_no_deletion(self, store):
        store.upsert(MemoryEntry(id="m1", key="k1", category="fact",
                                 content="alive", confidence=0.5))
        deleted = cleanup_dead_memories(store)
        assert deleted == 0


class TestRunMaintenance:
    def test_combined_maintenance(self, store):
        # Old memory that will decay below threshold
        store.upsert(MemoryEntry(id="old", key="k_old", category="fact",
                                 content="old", confidence=0.005))
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        store._conn.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?", (old_date, "old")
        )
        store._conn.commit()

        # Fresh memory that survives
        store.upsert(MemoryEntry(id="new", key="k_new", category="fact",
                                 content="fresh", confidence=0.8))

        result = run_maintenance(store)
        assert isinstance(result, dict)
        assert "decayed" in result
        assert "cleaned" in result
        # The old memory should have been decayed and then cleaned
        assert store.get("new") is not None
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_memory_lifecycle.py -x` -- fails because files do not exist.
2. GREEN: Create files. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 4 minutes.

---

## Task 5: Add extraction triggers

**Files:**
- `muse/memory/extractors.py` (create)
- `tests/test_memory_extractors.py` (create)

**What to do:**

Create specialized extractor functions for each trigger point. These are called by `MemoryMiddleware._extract_memories()` and can be enhanced independently. This task refactors the extraction logic from the middleware into standalone, testable functions.

Create `muse/memory/extractors.py`:

```python
"""Memory extraction functions for specific trigger points.

Each extractor analyzes a specific node's state/result and returns
candidate MemoryEntry objects. The middleware orchestrates calling
the right extractor based on node_name.

Extraction strategy (MVP):
    - Rule-based heuristics (no LLM call)
    - Future: LLM-assisted extraction with structured prompts
"""

from __future__ import annotations

import re
from typing import Any

from muse.memory.store import MemoryEntry


def extract_from_initialize(
    state: dict[str, Any],
    run_id: str | None = None,
) -> list[MemoryEntry]:
    """Extract topic, discipline, and language preferences."""
    entries: list[MemoryEntry] = []

    topic = str(state.get("topic", "")).strip()
    discipline = str(state.get("discipline", "")).strip()
    language = str(state.get("language", "")).strip()
    format_std = str(state.get("format_standard", "")).strip()

    if topic:
        entries.append(MemoryEntry(
            id="", key=f"topic:{_slugify(topic)}",
            category="fact",
            content=f"Research topic: {topic}",
            confidence=0.7, source_run=run_id,
        ))
    if discipline:
        entries.append(MemoryEntry(
            id="", key=f"discipline:{_slugify(discipline)}",
            category="fact",
            content=f"Academic discipline: {discipline}",
            confidence=0.7, source_run=run_id,
        ))
    if language:
        entries.append(MemoryEntry(
            id="", key=f"language_pref:{language}",
            category="user_pref",
            content=f"Preferred writing language: {language}",
            confidence=0.8, source_run=run_id,
        ))
    if format_std:
        entries.append(MemoryEntry(
            id="", key=f"format_std:{_slugify(format_std)}",
            category="user_pref",
            content=f"Citation format standard: {format_std}",
            confidence=0.8, source_run=run_id,
        ))
    return entries


def extract_from_hitl_feedback(
    node_name: str,
    result: dict[str, Any],
    run_id: str | None = None,
) -> list[MemoryEntry]:
    """Extract feedback patterns from HITL review nodes.

    Looks for ``review_feedback`` in the result with ``notes`` fields.
    Only extracts notes longer than 15 characters (short notes like
    "ok" or "approved" are not useful as memories).
    """
    entries: list[MemoryEntry] = []
    feedback_list = result.get("review_feedback", [])
    if not isinstance(feedback_list, list):
        return entries

    for fb in feedback_list:
        if not isinstance(fb, dict):
            continue
        notes = str(fb.get("notes", "")).strip()
        if len(notes) < 15:
            continue

        # Classify the feedback
        category = "feedback_pattern"
        confidence = 0.6

        # Detect style-related feedback
        style_keywords = {"tone", "style", "formal", "informal", "concise", "verbose",
                          "passive voice", "active voice", "academic"}
        if any(kw in notes.lower() for kw in style_keywords):
            category = "writing_style"
            confidence = 0.7

        key = f"feedback:{node_name}:{_slugify(notes[:50])}"
        entries.append(MemoryEntry(
            id="", key=key,
            category=category,
            content=f"User feedback at {node_name}: {notes}",
            confidence=confidence, source_run=run_id,
        ))
    return entries


def extract_from_citation_subgraph(
    state: dict[str, Any],
    result: dict[str, Any],
    run_id: str | None = None,
) -> list[MemoryEntry]:
    """Extract verified citations as citation memories.

    Each verified citation becomes a memory with high confidence (0.9).
    Includes DOI when available.
    """
    entries: list[MemoryEntry] = []
    verified = result.get("verified_citations", [])
    if not isinstance(verified, list):
        return entries

    references = {
        ref.get("ref_id"): ref
        for ref in state.get("references", [])
        if isinstance(ref, dict) and ref.get("ref_id")
    }

    for cite_key in verified:
        if not isinstance(cite_key, str) or not cite_key.strip():
            continue
        ref = references.get(cite_key, {})
        doi = str(ref.get("doi", "") or "").strip()
        title = str(ref.get("title", cite_key) or cite_key).strip()
        year = ref.get("year", "")

        content = f"Verified citation: {title}"
        if year:
            content += f" ({year})"
        if doi:
            content += f" [DOI: {doi}]"

        entries.append(MemoryEntry(
            id="", key=f"cite:{cite_key}",
            category="citation",
            content=content,
            confidence=0.9, source_run=run_id,
        ))
    return entries


def extract_from_review(
    state: dict[str, Any],
    result: dict[str, Any],
    run_id: str | None = None,
) -> list[MemoryEntry]:
    """Extract recurring quality patterns from review results.

    Looks at ``quality_scores`` and ``review_notes`` in the result
    to identify persistent quality issues.
    """
    entries: list[MemoryEntry] = []
    quality_scores = result.get("quality_scores", {})
    if not isinstance(quality_scores, dict):
        return entries

    # Flag consistently low-scoring dimensions
    for dimension, score in quality_scores.items():
        if isinstance(score, (int, float)) and score <= 2:
            key = f"quality_issue:{_slugify(str(dimension))}"
            entries.append(MemoryEntry(
                id="", key=key,
                category="feedback_pattern",
                content=f"Recurring quality issue: {dimension} scored {score}/5",
                confidence=0.5, source_run=run_id,
            ))

    return entries


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug for use as memory key."""
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "_", text.lower())
    return slug[:60].strip("_")
```

Create `tests/test_memory_extractors.py`:

```python
"""Tests for memory extraction functions (muse.memory.extractors)."""

from __future__ import annotations

import pytest

from muse.memory.extractors import (
    extract_from_citation_subgraph,
    extract_from_hitl_feedback,
    extract_from_initialize,
    extract_from_review,
)


class TestExtractFromInitialize:
    def test_extracts_topic(self):
        entries = extract_from_initialize({"topic": "Deep Learning"})
        assert any("Deep Learning" in e.content for e in entries)
        assert any(e.category == "fact" for e in entries)

    def test_extracts_discipline(self):
        entries = extract_from_initialize({"discipline": "Computer Science"})
        assert any("Computer Science" in e.content for e in entries)

    def test_extracts_language(self):
        entries = extract_from_initialize({"language": "zh"})
        assert any(e.category == "user_pref" for e in entries)
        assert any("zh" in e.content for e in entries)

    def test_extracts_format_standard(self):
        entries = extract_from_initialize({"format_standard": "GB/T 7714"})
        assert any("GB/T 7714" in e.content for e in entries)

    def test_empty_state(self):
        assert extract_from_initialize({}) == []

    def test_passes_run_id(self):
        entries = extract_from_initialize({"topic": "X"}, run_id="run_123")
        assert all(e.source_run == "run_123" for e in entries)


class TestExtractFromHITLFeedback:
    def test_extracts_long_notes(self):
        result = {
            "review_feedback": [
                {"notes": "Please use more formal language in the introduction section"}
            ]
        }
        entries = extract_from_hitl_feedback("review_draft", result)
        assert len(entries) == 1
        assert "formal language" in entries[0].content

    def test_skips_short_notes(self):
        result = {"review_feedback": [{"notes": "ok"}]}
        entries = extract_from_hitl_feedback("review_draft", result)
        assert len(entries) == 0

    def test_classifies_style_feedback(self):
        result = {
            "review_feedback": [
                {"notes": "Please adopt a more formal academic tone throughout the paper"}
            ]
        }
        entries = extract_from_hitl_feedback("review_draft", result)
        assert len(entries) == 1
        assert entries[0].category == "writing_style"
        assert entries[0].confidence == 0.7

    def test_handles_empty_feedback(self):
        assert extract_from_hitl_feedback("review_draft", {}) == []
        assert extract_from_hitl_feedback("review_draft", {"review_feedback": []}) == []

    def test_handles_non_dict_entries(self):
        result = {"review_feedback": ["string_entry", None, 42]}
        assert extract_from_hitl_feedback("review_draft", result) == []


class TestExtractFromCitationSubgraph:
    def test_extracts_verified_citation(self):
        state = {
            "references": [
                {"ref_id": "@smith2024dl", "title": "Deep Learning", "doi": "10.1234/test", "year": 2024}
            ]
        }
        result = {"verified_citations": ["@smith2024dl"]}
        entries = extract_from_citation_subgraph(state, result)
        assert len(entries) == 1
        assert entries[0].category == "citation"
        assert "Deep Learning" in entries[0].content
        assert "10.1234/test" in entries[0].content
        assert entries[0].confidence == 0.9

    def test_includes_year(self):
        state = {"references": [{"ref_id": "@a", "title": "T", "year": 2024}]}
        result = {"verified_citations": ["@a"]}
        entries = extract_from_citation_subgraph(state, result)
        assert "(2024)" in entries[0].content

    def test_no_verified_returns_empty(self):
        result = {"verified_citations": []}
        assert extract_from_citation_subgraph({}, result) == []

    def test_missing_reference_uses_cite_key(self):
        state = {"references": []}
        result = {"verified_citations": ["@unknown2024"]}
        entries = extract_from_citation_subgraph(state, result)
        assert len(entries) == 1
        assert "@unknown2024" in entries[0].content


class TestExtractFromReview:
    def test_flags_low_quality_dimensions(self):
        result = {"quality_scores": {"logic": 2, "style": 4, "citation": 1}}
        entries = extract_from_review({}, result)
        assert len(entries) == 2
        dims = {e.content for e in entries}
        assert any("logic" in d for d in dims)
        assert any("citation" in d for d in dims)

    def test_ignores_good_scores(self):
        result = {"quality_scores": {"logic": 4, "style": 5}}
        entries = extract_from_review({}, result)
        assert len(entries) == 0

    def test_empty_scores(self):
        assert extract_from_review({}, {}) == []
        assert extract_from_review({}, {"quality_scores": {}}) == []
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_memory_extractors.py -x` -- fails because files do not exist.
2. GREEN: Create files. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 4 minutes.

---

## Task 6: Integration test -- memory persists across simulated runs

**Files:**
- `tests/test_memory_integration.py` (create)

**What to do:**

Create an end-to-end integration test that simulates two thesis runs and verifies that memories from the first run are available in the second run. Uses a real SQLite file (in tmp_path) to verify persistence across `MemoryStore` instances.

Create `tests/test_memory_integration.py`:

```python
"""Integration tests for memory system persistence.

Tests the full lifecycle: extract -> persist -> reload -> inject.
Uses real SQLite (in tmp_path) to verify cross-session persistence.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from muse.memory.extractors import (
    extract_from_citation_subgraph,
    extract_from_hitl_feedback,
    extract_from_initialize,
)
from muse.memory.lifecycle import (
    confirm_memory,
    deny_memory,
    run_maintenance,
)
from muse.memory.middleware import MemoryMiddleware
from muse.memory.prompt import select_memories
from muse.memory.store import MemoryEntry, MemoryStore


def _run(coro):
    return asyncio.run(coro)


class TestCrossSessionPersistence:
    """Simulate two sessions sharing the same SQLite file."""

    def test_memories_persist_across_store_instances(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"

        # --- Session 1: Create memories ---
        store1 = MemoryStore(db_path)
        entries = extract_from_initialize({
            "topic": "Transformer Architectures",
            "discipline": "Computer Science",
            "language": "en",
        }, run_id="run_001")
        for e in entries:
            store1.upsert(e)
        assert store1.count() >= 3
        store1.close()

        # --- Session 2: Read memories from same DB ---
        store2 = MemoryStore(db_path)
        all_memories = store2.query()
        assert len(all_memories) >= 3

        # Verify specific memories survived
        contents = " ".join(m.content for m in all_memories)
        assert "Transformer Architectures" in contents
        assert "Computer Science" in contents

        # Verify injection works
        formatted = select_memories(store2, token_budget=2000)
        assert "Transformer Architectures" in formatted
        store2.close()

    def test_middleware_full_lifecycle(self, tmp_path):
        """Test: init -> extract -> close -> reopen -> inject."""
        db_path = tmp_path / "memory.sqlite"

        # --- Run 1: Extract memories during initialize ---
        store1 = MemoryStore(db_path)
        mw1 = MemoryMiddleware(store1)
        state = {
            "topic": "Graph Neural Networks",
            "discipline": "AI",
            "language": "zh",
        }
        config = {"configurable": {"node_name": "initialize", "thread_id": "run_001"}}
        _run(mw1.after_invoke(state, {}, config))
        assert store1.count() >= 2
        store1.close()

        # --- Run 2: Inject memories into config ---
        store2 = MemoryStore(db_path)
        mw2 = MemoryMiddleware(store2)
        config2 = {"configurable": {"thread_id": "run_002"}}
        result_config = _run(mw2.before_invoke({}, config2))

        memory_context = result_config["configurable"]["memory_context"]
        assert "Graph Neural Networks" in memory_context
        store2.close()


class TestConfidenceLifecycle:
    """Test the full confidence lifecycle: confirm -> decay -> cleanup."""

    def test_confirm_increases_trust(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"
        store = MemoryStore(db_path)
        store.upsert(MemoryEntry(
            id="m1", key="uses_apa", category="user_pref",
            content="User prefers APA citation style", confidence=0.5,
        ))

        # Simulate 3 confirmations
        for _ in range(3):
            confirm_memory(store, "uses_apa")
        entry = store.get("m1")
        assert entry is not None
        assert abs(entry.confidence - 0.8) < 0.001

        store.close()

    def test_deny_removes_memory(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"
        store = MemoryStore(db_path)
        store.upsert(MemoryEntry(
            id="m1", key="wrong_pref", category="user_pref",
            content="wrong assumption", confidence=0.6,
        ))
        deny_memory(store, "wrong_pref")
        assert store.count() == 0
        store.close()

    def test_decay_and_cleanup_cycle(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"
        store = MemoryStore(db_path)

        # Create a very old memory with low confidence
        store.upsert(MemoryEntry(
            id="old", key="old_fact", category="fact",
            content="outdated info", confidence=0.02,
        ))
        from datetime import datetime, timedelta, timezone
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        store._conn.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?", (old_date, "old")
        )
        store._conn.commit()

        # Create a fresh, high-confidence memory
        store.upsert(MemoryEntry(
            id="fresh", key="fresh_fact", category="fact",
            content="current info", confidence=0.9,
        ))

        result = run_maintenance(store)
        # Old memory should be decayed (0.02 * 0.9 = 0.018) and then cleaned (< 0.01)
        assert store.get("fresh") is not None
        store.close()


class TestCitationMemoryPipeline:
    """Test citation extraction -> persistence -> injection."""

    def test_verified_citations_remembered(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"

        # Run 1: Citation verification
        store1 = MemoryStore(db_path)
        state = {
            "references": [
                {"ref_id": "@vaswani2017attention", "title": "Attention Is All You Need",
                 "doi": "10.48550/arXiv.1706.03762", "year": 2017},
                {"ref_id": "@devlin2019bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                 "doi": "10.18653/v1/N19-1423", "year": 2019},
            ]
        }
        result = {"verified_citations": ["@vaswani2017attention", "@devlin2019bert"]}
        entries = extract_from_citation_subgraph(state, result, run_id="run_001")
        for e in entries:
            store1.upsert(e)
        assert store1.count(category="citation") == 2
        store1.close()

        # Run 2: Memories available
        store2 = MemoryStore(db_path)
        formatted = select_memories(store2, categories=["citation"])
        assert "Attention Is All You Need" in formatted
        assert "BERT" in formatted
        store2.close()


class TestFeedbackMemoryPipeline:
    """Test HITL feedback -> memory -> injection."""

    def test_feedback_remembered_across_sessions(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"

        # Run 1: User gives feedback
        store1 = MemoryStore(db_path)
        result = {
            "review_feedback": [
                {"notes": "Please use active voice consistently throughout the paper"}
            ]
        }
        entries = extract_from_hitl_feedback("review_draft", result, run_id="run_001")
        for e in entries:
            store1.upsert(e)
        store1.close()

        # Run 2: Feedback injected
        store2 = MemoryStore(db_path)
        formatted = select_memories(store2)
        assert "active voice" in formatted
        store2.close()
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_memory_integration.py -x` -- fails because files do not exist.
2. GREEN: Create the file. Run `python3 -m pytest tests/test_memory_integration.py -v` -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 4 minutes.
