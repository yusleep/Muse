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
    memory_store = MemoryStore(":memory:")
    yield memory_store
    memory_store.close()


@pytest.fixture
def mw(store):
    return MemoryMiddleware(store, token_budget=2000)


class TestBeforeInvoke:
    def test_injects_memory_context(self, store, mw):
        store.upsert(
            MemoryEntry(
                id="m1",
                key="tone",
                category="user_pref",
                content="User prefers formal tone",
                confidence=0.8,
            )
        )
        state = {}
        config = {"configurable": {"thread_id": "t1"}}
        result_state = _run(mw.before_invoke(state, config))
        assert "memory_context" in config["configurable"]
        assert "formal tone" in config["configurable"]["memory_context"]
        assert "memory_context" in result_state

    def test_no_memories_no_injection(self, store, mw):
        config = {"configurable": {}}
        result_state = _run(mw.before_invoke({}, config))
        assert "memory_context" not in result_state
        assert "memory_context" not in config.get("configurable", {})

    def test_disabled_middleware_no_injection(self, store):
        store.upsert(
            MemoryEntry(
                id="m1",
                key="tone",
                category="user_pref",
                content="formal",
                confidence=0.8,
            )
        )
        middleware = MemoryMiddleware(store, enabled=False)
        config = {"configurable": {}}
        result_state = _run(middleware.before_invoke({}, config))
        assert "memory_context" not in result_state
        assert "memory_context" not in config.get("configurable", {})


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
        keys = {memory.key for memory in memories}
        assert any("deep_learning" in key for key in keys)
        assert any("computer_science" in key for key in keys)
        assert any("language_pref" in key for key in keys)

    def test_skips_non_trigger_nodes(self, store, mw):
        config = {"configurable": {"node_name": "search"}}
        _run(mw.after_invoke({"topic": "Topic"}, {}, config))
        assert store.count() == 0


class TestAfterInvokeHITL:
    def test_extracts_feedback(self, store, mw):
        result = {
            "review_feedback": [
                {"stage": "review_draft", "notes": "Please use more formal citations in chapter 3"}
            ]
        }
        config = {"configurable": {"node_name": "review_draft", "thread_id": "t1"}}
        _run(mw.after_invoke({}, result, config))

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
                {
                    "ref_id": "@smith2024deep",
                    "title": "Deep Learning Survey",
                    "doi": "10.1234/test",
                }
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
        middleware = MemoryMiddleware(store, enabled=False)
        config = {"configurable": {"node_name": "initialize"}}
        _run(middleware.after_invoke({"topic": "Topic"}, {}, config))
        assert store.count() == 0
