"""Regression tests for ReAct context schema wiring."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class ReactContextSchemaTests(unittest.TestCase):
    def test_citation_agent_passes_context_schema(self):
        from muse.graph.subgraphs.citation import CitationState, _build_react_citation_agent

        captured: dict[str, object] = {}

        def fake_create_agent(*, model, tools, middleware, state_schema, context_schema, name):
            captured["tools"] = tools
            captured["middleware"] = middleware
            captured["state_schema"] = state_schema
            captured["context_schema"] = context_schema
            captured["name"] = name
            return object()

        with patch("langchain.agents.create_agent", side_effect=fake_create_agent), patch(
            "muse.graph.subgraphs.citation._create_react_model",
            return_value=object(),
        ):
            agent = _build_react_citation_agent(services=SimpleNamespace(), settings=object())

        self.assertIsNotNone(agent)
        self.assertIs(captured["state_schema"], CitationState)
        self.assertIsNotNone(captured["context_schema"])

    def test_chapter_agent_passes_context_schema(self):
        from muse.graph.subgraphs.chapter import ChapterState, _build_react_chapter_agent

        captured: dict[str, object] = {}

        def fake_create_agent(*, model, tools, middleware, state_schema, context_schema, name):
            captured["state_schema"] = state_schema
            captured["context_schema"] = context_schema
            captured["tool_names"] = [tool.name for tool in tools]
            return object()

        with patch("langchain.agents.create_agent", side_effect=fake_create_agent), patch(
            "muse.graph.subgraphs.chapter._create_react_model",
            return_value=object(),
        ):
            agent = _build_react_chapter_agent(
                services=SimpleNamespace(paper_index=object()),
                settings=object(),
            )

        self.assertIsNotNone(agent)
        self.assertIs(captured["state_schema"], ChapterState)
        self.assertIsNotNone(captured["context_schema"])
        self.assertIn("get_paper_section", captured["tool_names"])

    def test_composition_agent_passes_context_schema(self):
        from muse.graph.subgraphs.composition import CompositionState, _build_react_composition_agent

        captured: dict[str, object] = {}

        def fake_create_agent(*, model, tools, middleware, state_schema, context_schema, name):
            captured["state_schema"] = state_schema
            captured["context_schema"] = context_schema
            return object()

        with patch("langchain.agents.create_agent", side_effect=fake_create_agent), patch(
            "muse.graph.subgraphs.composition._create_react_model",
            return_value=object(),
        ):
            agent = _build_react_composition_agent(services=SimpleNamespace(), settings=object())

        self.assertIsNotNone(agent)
        self.assertIs(captured["state_schema"], CompositionState)
        self.assertIsNotNone(captured["context_schema"])

    def test_runtime_context_accepts_object_style_services(self):
        from muse.tools.research import academic_search

        search_service = SimpleNamespace(
            search_multi_source=lambda topic, discipline, extra_queries=None: (
                [{"ref_id": "@object_ctx", "title": f"{topic} / {discipline}"}],
                [topic],
            )
        )
        runtime = SimpleNamespace(
            context=SimpleNamespace(services=SimpleNamespace(search=search_service)),
            state={"discipline": "Computer Science"},
        )

        result = json.loads(
            academic_search.func(
                query="react citation",
                max_results=3,
                runtime=runtime,
            )
        )

        self.assertEqual(result[0]["ref_id"], "@object_ctx")


if __name__ == "__main__":
    unittest.main()
