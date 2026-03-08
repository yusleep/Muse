"""Integration tests for memory system persistence."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from muse.memory.extractors import (
    extract_from_citation_subgraph,
    extract_from_hitl_feedback,
    extract_from_initialize,
)
from muse.memory.lifecycle import confirm_memory, deny_memory, run_maintenance
from muse.memory.middleware import MemoryMiddleware
from muse.memory.prompt import select_memories
from muse.memory.store import MemoryEntry, MemoryStore


def _run(coro):
    return asyncio.run(coro)


class TestCrossSessionPersistence:
    def test_memories_persist_across_store_instances(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"

        store_one = MemoryStore(db_path)
        entries = extract_from_initialize(
            {
                "topic": "Transformer Architectures",
                "discipline": "Computer Science",
                "language": "en",
            },
            run_id="run_001",
        )
        for entry in entries:
            store_one.upsert(entry)
        assert store_one.count() >= 3
        store_one.close()

        store_two = MemoryStore(db_path)
        memories = store_two.query()
        assert len(memories) >= 3

        contents = " ".join(memory.content for memory in memories)
        assert "Transformer Architectures" in contents
        assert "Computer Science" in contents

        formatted = select_memories(store_two, token_budget=2000)
        assert "Transformer Architectures" in formatted
        store_two.close()

    def test_middleware_full_lifecycle(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"

        store_one = MemoryStore(db_path)
        middleware_one = MemoryMiddleware(store_one)
        state = {
            "topic": "Graph Neural Networks",
            "discipline": "AI",
            "language": "zh",
        }
        config = {"configurable": {"node_name": "initialize", "thread_id": "run_001"}}
        _run(middleware_one.after_invoke(state, {}, config))
        assert store_one.count() >= 2
        store_one.close()

        store_two = MemoryStore(db_path)
        middleware_two = MemoryMiddleware(store_two)
        config_two = {"configurable": {"thread_id": "run_002"}}
        result_state = _run(middleware_two.before_invoke({}, config_two))
        assert "memory_context" in config_two["configurable"]
        assert "Graph Neural Networks" in config_two["configurable"]["memory_context"]
        assert "memory_context" in result_state
        store_two.close()


class TestConfidenceLifecycle:
    def test_confirm_increases_trust(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"
        store = MemoryStore(db_path)
        store.upsert(
            MemoryEntry(
                id="m1",
                key="uses_apa",
                category="user_pref",
                content="User prefers APA citation style",
                confidence=0.5,
            )
        )

        for _ in range(3):
            confirm_memory(store, "uses_apa")
        entry = store.get("m1")
        assert entry is not None
        assert abs(entry.confidence - 0.8) < 0.001
        store.close()

    def test_deny_removes_memory(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"
        store = MemoryStore(db_path)
        store.upsert(
            MemoryEntry(
                id="m1",
                key="wrong_pref",
                category="user_pref",
                content="wrong assumption",
                confidence=0.6,
            )
        )
        deny_memory(store, "wrong_pref")
        assert store.count() == 0
        store.close()

    def test_decay_and_cleanup_cycle(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"
        store = MemoryStore(db_path)
        store.upsert(
            MemoryEntry(
                id="old",
                key="old_fact",
                category="fact",
                content="outdated info",
                confidence=0.02,
            )
        )
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        store._conn.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?",
            (old_date, "old"),
        )
        store._conn.commit()

        store.upsert(
            MemoryEntry(
                id="fresh",
                key="fresh_fact",
                category="fact",
                content="current info",
                confidence=0.9,
            )
        )

        result = run_maintenance(store)
        assert isinstance(result, dict)
        assert "decayed" in result
        assert "cleaned" in result
        assert store.get("fresh") is not None
        store.close()


class TestCitationMemoryPipeline:
    def test_verified_citations_remembered(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"

        store_one = MemoryStore(db_path)
        state = {
            "references": [
                {
                    "ref_id": "@vaswani2017attention",
                    "title": "Attention Is All You Need",
                    "doi": "10.48550/arXiv.1706.03762",
                    "year": 2017,
                },
                {
                    "ref_id": "@devlin2019bert",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "doi": "10.18653/v1/N19-1423",
                    "year": 2019,
                },
            ]
        }
        result = {"verified_citations": ["@vaswani2017attention", "@devlin2019bert"]}
        for entry in extract_from_citation_subgraph(state, result, run_id="run_001"):
            store_one.upsert(entry)
        assert store_one.count(category="citation") == 2
        store_one.close()

        store_two = MemoryStore(db_path)
        formatted = select_memories(store_two, categories=["citation"])
        assert "Attention Is All You Need" in formatted
        assert "BERT" in formatted
        store_two.close()


class TestFeedbackMemoryPipeline:
    def test_feedback_remembered_across_sessions(self, tmp_path):
        db_path = tmp_path / "memory.sqlite"

        store_one = MemoryStore(db_path)
        result = {
            "review_feedback": [{"notes": "Please use active voice consistently throughout the paper"}]
        }
        for entry in extract_from_hitl_feedback("review_draft", result, run_id="run_001"):
            store_one.upsert(entry)
        store_one.close()

        store_two = MemoryStore(db_path)
        formatted = select_memories(store_two)
        assert "active voice" in formatted
        store_two.close()
