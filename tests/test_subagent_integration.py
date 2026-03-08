"""Integration tests for sub-agent delegation pipeline."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from muse.agents.executor import SubagentExecutor
from muse.agents.result import SubagentResult
from muse.middlewares.subagent_limit_middleware import SubagentLimitMiddleware
from muse.tools.orchestration import set_subagent_executor, spawn_subagent


class SubagentIntegrationTests(unittest.TestCase):
    def setUp(self):
        set_subagent_executor(None)

    def tearDown(self):
        set_subagent_executor(None)

    def test_spawn_and_collect_result(self):
        executor = SubagentExecutor(max_concurrent=2)
        set_subagent_executor(executor)

        def research_factory(message):
            def run():
                return SubagentResult(
                    status="completed",
                    accomplishments=[f"researched: {message}"],
                    key_findings=["finding1"],
                    files_created=[],
                    issues=[],
                    citations=[{"ref_id": "@test"}],
                )

            return run

        with patch(
            "muse.tools.orchestration._get_builtin_registry",
            return_value={"research": research_factory},
        ):
            result_str = spawn_subagent.invoke(
                {
                    "message": "find papers on LLMs",
                    "agent_type": "research",
                    "wait": True,
                }
            )

        executor.shutdown()
        self.assertIn("completed", result_str)

    def test_limit_middleware_caps_spawns(self):
        middleware = SubagentLimitMiddleware(max_concurrent=3)
        tool_calls = [
            {
                "name": "spawn_subagent",
                "id": f"tc_{idx}",
                "args": {"message": f"task {idx}", "agent_type": "research"},
            }
            for idx in range(5)
        ]
        filtered = middleware.filter_tool_calls(tool_calls)
        self.assertEqual(len(filtered), 3)

    def test_executor_runs_agent_to_completion(self):
        executor = SubagentExecutor(max_concurrent=1)

        def slow_agent():
            time.sleep(0.2)
            return SubagentResult(status="completed", accomplishments=["done"])

        task_id = executor.submit(agent_fn=slow_agent)
        status_before = executor.get_status(task_id)
        result = executor.get_result(task_id, timeout=5)
        executor.shutdown()

        self.assertEqual(status_before, "running")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "completed")

    def test_full_pipeline_spawn_collect_limit(self):
        executor = SubagentExecutor(max_concurrent=2)
        set_subagent_executor(executor)
        middleware = SubagentLimitMiddleware(max_concurrent=2)

        def research_factory(message):
            def run():
                return SubagentResult(
                    status="completed",
                    accomplishments=[f"researched: {message}"],
                )

            return run

        tool_calls = [
            {
                "name": "spawn_subagent",
                "id": f"tc_{idx}",
                "args": {"message": f"task {idx}", "agent_type": "research"},
            }
            for idx in range(4)
        ]
        filtered = middleware.filter_tool_calls(tool_calls)
        self.assertEqual(len(filtered), 2)

        with patch(
            "muse.tools.orchestration._get_builtin_registry",
            return_value={"research": research_factory},
        ):
            outputs = [
                spawn_subagent.invoke(
                    {
                        "message": call["args"]["message"],
                        "agent_type": call["args"]["agent_type"],
                        "wait": True,
                    }
                )
                for call in filtered
            ]

        executor.shutdown()
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all("completed" in output for output in outputs))


if __name__ == "__main__":
    unittest.main()
