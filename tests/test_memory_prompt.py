"""Tests for memory formatting (muse.memory.prompt)."""

from __future__ import annotations

import pytest

from muse.memory.prompt import format_memory, select_memories, truncate_to_budget
from muse.memory.store import MemoryEntry, MemoryStore


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
            _entry("k1", confidence=0.3, content="content-low"),
            _entry("k2", confidence=0.9, content="content-high"),
        ]
        result = format_memory(entries)
        assert result.index("content-high") < result.index("content-low")


class TestTruncateToBudget:
    def test_within_budget_unchanged(self):
        text = "short text"
        assert truncate_to_budget(text, 1000) == text

    def test_truncated_at_line_boundary(self):
        text = "line1\nline2\nline3\nline4\nline5"
        result = truncate_to_budget(text, 3)
        assert "truncated" in result
        assert len(result) < len(text) + 50

    def test_large_budget_no_change(self):
        text = "x" * 100
        assert truncate_to_budget(text, 10000) == text


class TestSelectMemories:
    @pytest.fixture
    def store(self):
        memory_store = MemoryStore(":memory:")
        memory_store.upsert(
            _entry("pref1", category="user_pref", content="prefers GB/T 7714", confidence=0.8)
        )
        memory_store.upsert(
            _entry("style1", category="writing_style", content="formal academic", confidence=0.6)
        )
        memory_store.upsert(
            _entry("cite1", category="citation", content="10.1234/verified", confidence=0.9)
        )
        memory_store.upsert(
            _entry("low1", category="fact", content="low conf fact", confidence=0.05)
        )
        yield memory_store
        memory_store.close()

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
        assert len(result) < 200

    def test_empty_store_returns_empty(self):
        store = MemoryStore(":memory:")
        assert select_memories(store) == ""
        store.close()
