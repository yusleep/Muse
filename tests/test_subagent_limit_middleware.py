"""Tests for SubagentLimitMiddleware."""

from __future__ import annotations

import unittest


class SubagentLimitMiddlewareTests(unittest.TestCase):
    def _make_middleware(self, max_concurrent=3):
        from muse.middlewares.subagent_limit_middleware import SubagentLimitMiddleware

        return SubagentLimitMiddleware(max_concurrent=max_concurrent)

    def _spawn_call(self, idx):
        return {
            "name": "spawn_subagent",
            "id": f"tc_{idx}",
            "args": {"message": f"task {idx}", "agent_type": "research"},
        }

    def _other_call(self, name, idx):
        return {"name": name, "id": f"tc_other_{idx}", "args": {}}

    def test_truncates_excess_spawn_calls(self):
        middleware = self._make_middleware(max_concurrent=3)
        calls = [self._spawn_call(idx) for idx in range(5)]
        filtered = middleware.filter_tool_calls(calls)
        spawn_filtered = [call for call in filtered if call["name"] == "spawn_subagent"]
        self.assertEqual(len(spawn_filtered), 3)

    def test_preserves_non_spawn_tool_calls(self):
        middleware = self._make_middleware(max_concurrent=1)
        calls = [
            self._other_call("write_section", 0),
            self._other_call("self_review", 1),
            self._spawn_call(0),
            self._spawn_call(1),
        ]
        filtered = middleware.filter_tool_calls(calls)
        non_spawn = [call for call in filtered if call["name"] != "spawn_subagent"]
        self.assertEqual(len(non_spawn), 2)

    def test_mixed_calls_truncate_only_spawn(self):
        middleware = self._make_middleware(max_concurrent=2)
        calls = [
            self._other_call("write_section", 0),
            self._spawn_call(0),
            self._other_call("write_section", 1),
            self._spawn_call(1),
            self._spawn_call(2),
            self._spawn_call(3),
        ]
        filtered = middleware.filter_tool_calls(calls)
        spawn_filtered = [call for call in filtered if call["name"] == "spawn_subagent"]
        other_filtered = [call for call in filtered if call["name"] != "spawn_subagent"]
        self.assertEqual(len(spawn_filtered), 2)
        self.assertEqual(len(other_filtered), 2)

    def test_under_limit_passes_all(self):
        middleware = self._make_middleware(max_concurrent=3)
        calls = [self._spawn_call(0), self._spawn_call(1)]
        filtered = middleware.filter_tool_calls(calls)
        self.assertEqual(len(filtered), 2)

    def test_zero_spawn_passes_through(self):
        middleware = self._make_middleware(max_concurrent=3)
        calls = [self._other_call("read_file", 0), self._other_call("edit_file", 1)]
        filtered = middleware.filter_tool_calls(calls)
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
