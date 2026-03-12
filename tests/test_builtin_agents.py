"""Tests for built-in sub-agent configurations."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from muse.agents.result import SubagentResult


class BuiltinAgentTests(unittest.TestCase):
    def test_builtin_registry_has_three_types(self):
        from muse.agents.builtins import BUILTIN_AGENT_FACTORIES

        self.assertEqual(set(BUILTIN_AGENT_FACTORIES.keys()), {"research", "writing", "bash"})

    def test_research_factory_returns_callable(self):
        from muse.agents.builtins import BUILTIN_AGENT_FACTORIES

        fn = BUILTIN_AGENT_FACTORIES["research"]("find papers on LLMs")
        self.assertTrue(callable(fn))

    def test_writing_factory_returns_callable(self):
        from muse.agents.builtins import BUILTIN_AGENT_FACTORIES

        fn = BUILTIN_AGENT_FACTORIES["writing"]("write introduction section")
        self.assertTrue(callable(fn))

    def test_bash_factory_returns_callable(self):
        from muse.agents.builtins import BUILTIN_AGENT_FACTORIES

        fn = BUILTIN_AGENT_FACTORIES["bash"]("compile thesis.tex")
        self.assertTrue(callable(fn))

    def test_factory_callable_returns_subagent_result(self):
        from muse.agents.builtins import BUILTIN_AGENT_FACTORIES

        fn = BUILTIN_AGENT_FACTORIES["research"]("find papers on transformers")
        result = fn()
        self.assertIsInstance(result, SubagentResult)
        self.assertEqual(result.status, "completed")
        self.assertTrue(any("research" in accomplishment for accomplishment in result.accomplishments))
        self.assertFalse(any("stub mode" in issue.lower() for issue in result.issues))

    def test_research_agent_resolves_services_at_run_time(self):
        from muse.agents.builtins import build_research_agent
        from muse.tools._context import set_services

        class _Search:
            def search_multi_source(self, topic, discipline, extra_queries=None):
                del discipline, extra_queries
                return (
                    [{"ref_id": "@paper2024", "title": f"Paper for {topic}", "year": 2024}],
                    [topic],
                )

        set_services(SimpleNamespace(search=None, llm=None, sandbox=None, settings=None))
        fn = build_research_agent("runtime topic")
        set_services(SimpleNamespace(search=_Search(), llm=None, sandbox=None, settings=None))

        result = fn()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.citations[0]["ref_id"], "@paper2024")
        self.assertIn("Paper for runtime topic", result.key_findings)

    def test_writing_agent_resolves_services_at_run_time(self):
        from muse.agents.builtins import build_writing_agent
        from muse.tools._context import set_services

        class _LLM:
            def text(self, *, system, user, route="default", max_tokens=2500):
                del system, route, max_tokens
                return f"drafted: {user}"

        set_services(SimpleNamespace(search=None, llm=None, sandbox=None, settings=None))
        fn = build_writing_agent("write runtime section")
        set_services(SimpleNamespace(search=None, llm=_LLM(), sandbox=None, settings=None))

        result = fn()

        self.assertEqual(result.status, "completed")
        self.assertIn("drafted: write runtime section", result.key_findings)

    def test_bash_agent_resolves_services_at_run_time(self):
        from muse.agents.builtins import build_bash_agent
        from muse.tools._context import set_services

        async def fake_shell_tool(sandbox, message, timeout=60):
            del timeout
            return f"{sandbox.workspace}|{message}"

        set_services(SimpleNamespace(search=None, llm=None, sandbox=None, settings=None))
        fn = build_bash_agent("pwd")
        set_services(
            SimpleNamespace(
                search=None,
                llm=None,
                sandbox=None,
                settings=SimpleNamespace(runs_dir="/tmp/runtime-services"),
            )
        )

        with patch("muse.agents.builtins.shell_tool", fake_shell_tool):
            result = fn()

        self.assertEqual(result.status, "completed")
        self.assertIn("/tmp/runtime-services", result.key_findings[0])

    def test_bash_agent_can_run_inside_existing_event_loop(self):
        from muse.agents.builtins import build_bash_agent
        from muse.tools._context import set_services

        async def fake_shell_tool(sandbox, message, timeout=60):
            del sandbox, timeout
            return f"ok:{message}"

        set_services(
            SimpleNamespace(
                search=None,
                llm=None,
                sandbox=None,
                settings=SimpleNamespace(runs_dir="/tmp/bash-loop"),
            )
        )
        fn = build_bash_agent("echo ok")

        async def invoke_in_loop():
            return fn()

        with patch("muse.agents.builtins.shell_tool", fake_shell_tool):
            result = asyncio.run(invoke_in_loop())

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.key_findings, ["ok:echo ok"])

    def test_agent_tool_profiles_defined(self):
        from muse.agents.builtins import AGENT_TOOL_PROFILES

        self.assertIn("research", AGENT_TOOL_PROFILES)
        self.assertIn("writing", AGENT_TOOL_PROFILES)
        self.assertIn("bash", AGENT_TOOL_PROFILES)

    def test_blocked_tools_include_spawn_and_clarification(self):
        from muse.agents.builtins import BLOCKED_TOOLS

        self.assertIn("spawn_subagent", BLOCKED_TOOLS)
        self.assertIn("ask_clarification", BLOCKED_TOOLS)

    def test_max_turns_defined(self):
        from muse.agents.builtins import AGENT_MAX_TURNS

        self.assertEqual(AGENT_MAX_TURNS["research"], 15)
        self.assertEqual(AGENT_MAX_TURNS["writing"], 25)
        self.assertEqual(AGENT_MAX_TURNS["bash"], 15)


if __name__ == "__main__":
    unittest.main()
