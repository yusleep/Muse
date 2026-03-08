"""Tests for SubagentExecutor."""

from __future__ import annotations

import threading
import time
import unittest

from muse.agents.result import SubagentResult


class SubagentExecutorTests(unittest.TestCase):
    def _make_executor(self, **kwargs):
        from muse.agents.executor import SubagentExecutor

        return SubagentExecutor(**kwargs)

    def test_execute_returns_subagent_result(self):
        executor = self._make_executor(max_concurrent=2)

        def stub():
            return SubagentResult(
                status="completed",
                accomplishments=["found 3 papers"],
                key_findings=["key1"],
                files_created=[],
                issues=[],
                citations=[{"ref_id": "@a2024"}],
            )

        task_id = executor.submit(agent_fn=stub)
        result = executor.get_result(task_id, timeout=5)
        executor.shutdown()

        self.assertIsNotNone(result)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.accomplishments, ["found 3 papers"])
        self.assertEqual(result.key_findings, ["key1"])
        self.assertEqual(result.citations, [{"ref_id": "@a2024"}])

    def test_max_concurrent_limit_enforced(self):
        executor = self._make_executor(max_concurrent=3)
        concurrent_peak: list[int] = []
        counter_lock = threading.Lock()
        active = [0]

        def slow_task():
            with counter_lock:
                active[0] += 1
                concurrent_peak.append(active[0])
            time.sleep(0.3)
            with counter_lock:
                active[0] -= 1
            return SubagentResult(
                status="completed",
                accomplishments=[],
                key_findings=[],
                files_created=[],
                issues=[],
                citations=[],
            )

        task_ids = [executor.submit(agent_fn=slow_task) for _ in range(5)]
        for task_id in task_ids:
            executor.get_result(task_id, timeout=10)
        executor.shutdown()

        self.assertLessEqual(max(concurrent_peak), 3)

    def test_timeout_produces_timed_out_status(self):
        executor = self._make_executor(max_concurrent=1, default_timeout=0.1)

        def slow_task():
            time.sleep(2)
            return SubagentResult()

        task_id = executor.submit(agent_fn=slow_task)
        result = executor.get_result(task_id, timeout=0.1)
        executor.shutdown(wait=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.status, "timed_out")

    def test_exception_produces_failed_status(self):
        executor = self._make_executor(max_concurrent=2)

        def failing():
            raise ValueError("bad input")

        task_id = executor.submit(agent_fn=failing)
        result = executor.get_result(task_id, timeout=5)
        executor.shutdown()

        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertTrue(any("bad input" in issue for issue in result.issues))

    def test_get_status_returns_running_for_active_task(self):
        executor = self._make_executor(max_concurrent=1)
        started = threading.Event()

        def slow():
            started.set()
            time.sleep(2)
            return SubagentResult(
                status="completed",
                accomplishments=[],
                key_findings=[],
                files_created=[],
                issues=[],
                citations=[],
            )

        task_id = executor.submit(agent_fn=slow)
        started.wait(timeout=3)
        status = executor.get_status(task_id)
        executor.shutdown(wait=False)

        self.assertEqual(status, "running")


if __name__ == "__main__":
    unittest.main()
