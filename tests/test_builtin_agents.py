"""Tests for built-in sub-agent configurations."""

from __future__ import annotations

import unittest

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
