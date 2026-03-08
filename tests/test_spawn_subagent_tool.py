"""Tests for spawn_subagent tool."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from muse.agents.result import SubagentResult


class SpawnSubagentToolTests(unittest.TestCase):
    def test_tool_schema_has_required_fields(self):
        from muse.tools.orchestration import spawn_subagent

        schema = spawn_subagent.args_schema.model_json_schema()
        props = schema.get("properties", {})
        self.assertIn("message", props)
        self.assertIn("agent_type", props)
        self.assertIn("wait", props)
        required = schema.get("required", [])
        self.assertIn("message", required)
        self.assertIn("agent_type", required)

    def test_agent_type_enum(self):
        from muse.tools.orchestration import spawn_subagent

        schema = spawn_subagent.args_schema.model_json_schema()
        agent_type = schema["properties"]["agent_type"]
        allowed = set(agent_type.get("enum", []))
        self.assertEqual(allowed, {"research", "writing", "bash"})

    def test_tool_invoke_with_executor(self):
        from muse.tools.orchestration import set_subagent_executor, spawn_subagent

        mock_executor = MagicMock()
        mock_executor.submit.return_value = "task_123"
        mock_executor.get_result.return_value = SubagentResult(
            status="completed",
            accomplishments=["done"],
        )
        set_subagent_executor(mock_executor)

        def dummy_factory(message):
            self.assertEqual(message, "find papers on LLMs")

            def run():
                return SubagentResult(status="completed", accomplishments=["done"])

            return run

        with patch(
            "muse.tools.orchestration._get_builtin_registry",
            return_value={"research": dummy_factory},
        ):
            result = spawn_subagent.invoke(
                {
                    "message": "find papers on LLMs",
                    "agent_type": "research",
                    "wait": True,
                }
            )

        mock_executor.submit.assert_called_once()
        self.assertIn("completed", result)
        set_subagent_executor(None)

    def test_tool_returns_result_summary(self):
        from muse.tools.orchestration import set_subagent_executor, spawn_subagent

        mock_executor = MagicMock()
        mock_executor.submit.return_value = "task_456"
        expected = SubagentResult(
            status="completed",
            accomplishments=["found 5 papers"],
            key_findings=["LLMs improve writing quality"],
        )
        mock_executor.get_result.return_value = expected
        set_subagent_executor(mock_executor)

        def dummy_factory(message):
            return lambda: expected

        with patch(
            "muse.tools.orchestration._get_builtin_registry",
            return_value={"research": dummy_factory},
        ):
            result = spawn_subagent.invoke(
                {"message": "search", "agent_type": "research", "wait": True}
            )

        self.assertIn("completed", result)
        self.assertIn("1", result)
        set_subagent_executor(None)

    def test_tool_returns_task_id_when_no_wait(self):
        from muse.tools.orchestration import set_subagent_executor, spawn_subagent

        mock_executor = MagicMock()
        mock_executor.submit.return_value = "task_789"
        set_subagent_executor(mock_executor)

        def dummy_factory(message):
            return lambda: SubagentResult(status="completed")

        with patch(
            "muse.tools.orchestration._get_builtin_registry",
            return_value={"writing": dummy_factory},
        ):
            result = spawn_subagent.invoke(
                {"message": "write intro", "agent_type": "writing", "wait": False}
            )

        self.assertIn("task_789", result)
        self.assertIn("writing", result)
        set_subagent_executor(None)

    def test_no_executor_returns_error(self):
        from muse.tools.orchestration import set_subagent_executor, spawn_subagent

        set_subagent_executor(None)
        result = spawn_subagent.invoke(
            {"message": "test", "agent_type": "research", "wait": True}
        )
        self.assertIn("SUBAGENT ERROR", result)
        self.assertIn("No SubagentExecutor", result)


if __name__ == "__main__":
    unittest.main()
