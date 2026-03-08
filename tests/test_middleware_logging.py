from __future__ import annotations

import json
import os
import tempfile
import unittest


class LoggingMiddlewareTests(unittest.TestCase):
    def test_import(self):
        from muse.middlewares.logging_middleware import LoggingMiddleware

        self.assertTrue(callable(LoggingMiddleware))

    def test_conforms_to_protocol(self):
        from muse.middlewares.base import Middleware
        from muse.middlewares.logging_middleware import LoggingMiddleware

        middleware = LoggingMiddleware()
        self.assertIsInstance(middleware, Middleware)

    def test_writes_jsonl_entry_on_after_invoke(self):
        import asyncio

        from muse.middlewares.logging_middleware import LoggingMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "nodes.jsonl")
            middleware = LoggingMiddleware(log_path=log_path)
            state = {"project_id": "run-1", "topic": "test"}
            config = {"configurable": {"thread_id": "t1"}}

            asyncio.run(middleware.before_invoke(state, config))
            result = {"references": [{"title": "A"}]}
            asyncio.run(middleware.after_invoke(state, result, config))

            self.assertTrue(os.path.exists(log_path))
            with open(log_path, encoding="utf-8") as handle:
                lines = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(lines), 1)
            entry = lines[0]
            self.assertIn("timestamp", entry)
            self.assertIn("latency_ms", entry)
            self.assertIn("node", entry)

    def test_tracks_latency(self):
        import asyncio
        import time

        from muse.middlewares.logging_middleware import LoggingMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "nodes.jsonl")
            middleware = LoggingMiddleware(log_path=log_path)

            asyncio.run(middleware.before_invoke({}, {}))
            time.sleep(0.05)
            asyncio.run(middleware.after_invoke({}, {}, {}))

            with open(log_path, encoding="utf-8") as handle:
                entry = json.loads(handle.readline())
            self.assertGreaterEqual(entry["latency_ms"], 40)

    def test_records_token_usage_from_state(self):
        import asyncio

        from muse.middlewares.logging_middleware import LoggingMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "nodes.jsonl")
            middleware = LoggingMiddleware(log_path=log_path, node_name="search")

            asyncio.run(middleware.before_invoke({}, {}))
            result = {"_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
            asyncio.run(middleware.after_invoke({}, result, {}))

            with open(log_path, encoding="utf-8") as handle:
                entry = json.loads(handle.readline())
            self.assertEqual(entry["node"], "search")
            self.assertEqual(entry["usage"]["prompt_tokens"], 100)

    def test_after_invoke_returns_result_unchanged(self):
        import asyncio

        from muse.middlewares.logging_middleware import LoggingMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "nodes.jsonl")
            middleware = LoggingMiddleware(log_path=log_path)

            asyncio.run(middleware.before_invoke({}, {}))
            result = {"key": "value"}
            out = asyncio.run(middleware.after_invoke({}, result, {}))
            self.assertEqual(out, result)

    def test_before_invoke_returns_state_unchanged(self):
        import asyncio

        from muse.middlewares.logging_middleware import LoggingMiddleware

        middleware = LoggingMiddleware()
        state = {"x": 1, "y": 2}
        out = asyncio.run(middleware.before_invoke(state, {}))
        self.assertEqual(out, state)

    def test_no_log_path_does_not_crash(self):
        import asyncio

        from muse.middlewares.logging_middleware import LoggingMiddleware

        middleware = LoggingMiddleware(log_path=None)
        asyncio.run(middleware.before_invoke({}, {}))
        out = asyncio.run(middleware.after_invoke({}, {"v": 1}, {}))
        self.assertEqual(out, {"v": 1})


if __name__ == "__main__":
    unittest.main()
