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
    memory_store = MemoryStore(":memory:")
    yield memory_store
    memory_store.close()


class TestConfirmMemory:
    def test_increments_confidence(self, store):
        store.upsert(MemoryEntry(id="m1", key="tone", category="user_pref", content="formal", confidence=0.5))
        assert confirm_memory(store, "tone") is True
        entry = store.get("m1")
        assert entry is not None
        assert abs(entry.confidence - 0.6) < 0.001

    def test_capped_at_1(self, store):
        store.upsert(MemoryEntry(id="m1", key="tone", category="user_pref", content="formal", confidence=0.95))
        confirm_memory(store, "tone")
        entry = store.get("m1")
        assert entry is not None
        assert entry.confidence == 1.0

    def test_returns_false_for_missing_key(self, store):
        assert confirm_memory(store, "nonexistent") is False

    def test_custom_delta(self, store):
        store.upsert(MemoryEntry(id="m1", key="tone", category="user_pref", content="formal", confidence=0.5))
        confirm_memory(store, "tone", delta=0.2)
        entry = store.get("m1")
        assert entry is not None
        assert abs(entry.confidence - 0.7) < 0.001


class TestDenyMemory:
    def test_deletes_memory(self, store):
        store.upsert(
            MemoryEntry(id="m1", key="wrong_fact", category="fact", content="incorrect", confidence=0.5)
        )
        assert deny_memory(store, "wrong_fact") is True
        assert store.count() == 0

    def test_returns_false_for_missing(self, store):
        assert deny_memory(store, "nonexistent") is False


class TestRunDecay:
    def test_decays_old_memories(self, store):
        store.upsert(MemoryEntry(id="old", key="old_fact", category="fact", content="stale", confidence=1.0))
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        store._conn.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?",
            (old_date, "old"),
        )
        store._conn.commit()

        count = run_decay(store, days=90, factor=0.9)
        assert count == 1
        entry = store.get("old")
        assert entry is not None
        assert abs(entry.confidence - 0.9) < 0.001

    def test_does_not_decay_recent(self, store):
        store.upsert(MemoryEntry(id="new", key="new_fact", category="fact", content="fresh", confidence=1.0))
        assert run_decay(store, days=90, factor=0.9) == 0


class TestCleanupDeadMemories:
    def test_deletes_low_confidence(self, store):
        store.upsert(MemoryEntry(id="dead", key="k1", category="fact", content="dead", confidence=0.005))
        store.upsert(MemoryEntry(id="alive", key="k2", category="fact", content="alive", confidence=0.5))
        deleted = cleanup_dead_memories(store, threshold=0.01)
        assert deleted == 1
        assert store.count() == 1
        assert store.get("alive") is not None

    def test_no_dead_no_deletion(self, store):
        store.upsert(MemoryEntry(id="m1", key="k1", category="fact", content="alive", confidence=0.5))
        assert cleanup_dead_memories(store) == 0


class TestRunMaintenance:
    def test_combined_maintenance(self, store):
        store.upsert(MemoryEntry(id="old", key="k_old", category="fact", content="old", confidence=0.005))
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        store._conn.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?",
            (old_date, "old"),
        )
        store._conn.commit()

        store.upsert(MemoryEntry(id="new", key="k_new", category="fact", content="fresh", confidence=0.8))

        result = run_maintenance(store)
        assert isinstance(result, dict)
        assert "decayed" in result
        assert "cleaned" in result
        assert store.get("new") is not None
