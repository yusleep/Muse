"""Tests for MemoryStore (muse.memory.store)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from muse.memory.store import MemoryEntry, MemoryStore


@pytest.fixture
def store():
    memory_store = MemoryStore(":memory:")
    yield memory_store
    memory_store.close()


class TestMemoryEntry:
    def test_valid_category(self):
        entry = MemoryEntry(id="1", key="k", category="user_pref", content="c", confidence=0.5)
        assert entry.category == "user_pref"

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Invalid category"):
            MemoryEntry(id="1", key="k", category="bad", content="c", confidence=0.5)

    def test_confidence_clamped_high(self):
        entry = MemoryEntry(id="1", key="k", category="fact", content="c", confidence=1.5)
        assert entry.confidence == 1.0

    def test_confidence_clamped_low(self):
        entry = MemoryEntry(id="1", key="k", category="fact", content="c", confidence=-0.2)
        assert entry.confidence == 0.0


class TestMemoryStoreUpsert:
    def test_insert_new(self, store):
        entry = MemoryEntry(
            id="a1",
            key="formal_tone",
            category="user_pref",
            content="User prefers formal academic tone",
            confidence=0.8,
        )
        store.upsert(entry)
        assert store.count() == 1

    def test_upsert_updates_existing_key(self, store):
        store.upsert(MemoryEntry(id="a1", key="tone", category="user_pref", content="formal", confidence=0.5))
        store.upsert(
            MemoryEntry(id="a2", key="tone", category="user_pref", content="very formal", confidence=0.9)
        )
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
        confidences = [entry.confidence for entry in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_limit(self, store):
        for index in range(10):
            store.upsert(
                MemoryEntry(
                    id=f"m{index}",
                    key=f"k{index}",
                    category="fact",
                    content=f"c{index}",
                    confidence=0.5,
                )
            )
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
        store.upsert(MemoryEntry(id="old", key="old_k", category="fact", content="old", confidence=1.0))
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        store._conn.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?",
            (old_date, "old"),
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
