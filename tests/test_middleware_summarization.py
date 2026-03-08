from __future__ import annotations

import asyncio
import unittest


COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary "
    "for another LLM that will resume the task. Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue"
)

SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary "
    "of its thinking process. Use this to build on the work that has already been "
    "done and avoid duplicating work."
)


class SummarizationMiddlewareTests(unittest.TestCase):
    def test_import(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        self.assertTrue(callable(SummarizationMiddleware))

    def test_conforms_to_protocol(self):
        from muse.middlewares.base import Middleware
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        middleware = SummarizationMiddleware(llm=None, context_window=128_000)
        self.assertIsInstance(middleware, Middleware)

    def test_estimate_tokens_heuristic(self):
        from muse.middlewares.summarization_middleware import estimate_tokens

        text = "a" * 400
        self.assertEqual(estimate_tokens(text), 100)

    def test_estimate_tokens_unicode(self):
        from muse.middlewares.summarization_middleware import estimate_tokens

        text = "中" * 100
        self.assertEqual(estimate_tokens(text), 75)

    def test_no_compaction_below_threshold(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        call_log = []

        class FakeLLM:
            def text(self, **kwargs):
                call_log.append("llm_called")
                return "summary"

        middleware = SummarizationMiddleware(
            llm=FakeLLM(),
            context_window=128_000,
            threshold_ratio=0.9,
        )
        state = {"topic": "small state"}
        out = asyncio.run(middleware.before_invoke(state, {}))
        self.assertEqual(out, state)
        self.assertEqual(call_log, [])

    def test_compaction_triggered_above_threshold(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        call_log = []

        class FakeLLM:
            def text(self, **kwargs):
                call_log.append(kwargs)
                return "Compact summary of progress."

        middleware = SummarizationMiddleware(
            llm=FakeLLM(),
            context_window=200,
            threshold_ratio=0.9,
        )
        big_state = {"data": "x" * 1000}
        out = asyncio.run(middleware.before_invoke(big_state, {}))

        self.assertEqual(len(call_log), 1)
        self.assertIn("CONTEXT CHECKPOINT COMPACTION", call_log[0]["system"])
        self.assertIn("_compaction_summary", out)
        self.assertIn("Compact summary of progress.", out["_compaction_summary"])

    def test_compaction_preserves_recent_keys(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        class FakeLLM:
            def text(self, **kwargs):
                return "Summary."

        middleware = SummarizationMiddleware(
            llm=FakeLLM(),
            context_window=200,
            threshold_ratio=0.9,
            preserve_keys=["topic", "project_id"],
        )
        big_state = {
            "topic": "My Topic",
            "project_id": "run-1",
            "data": "x" * 1000,
        }
        out = asyncio.run(middleware.before_invoke(big_state, {}))
        self.assertEqual(out["topic"], "My Topic")
        self.assertEqual(out["project_id"], "run-1")

    def test_compaction_summary_has_prefix(self):
        from muse.middlewares.summarization_middleware import (
            SUMMARY_PREFIX,
            SummarizationMiddleware,
        )

        class FakeLLM:
            def text(self, **kwargs):
                return "The agent completed steps 1-3."

        middleware = SummarizationMiddleware(llm=FakeLLM(), context_window=200)
        big_state = {"data": "x" * 1000}
        out = asyncio.run(middleware.before_invoke(big_state, {}))
        summary = out["_compaction_summary"]
        self.assertTrue(summary.startswith(SUMMARY_PREFIX))

    def test_after_invoke_passthrough(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        middleware = SummarizationMiddleware(llm=None, context_window=128_000)
        result = {"ok": True}
        out = asyncio.run(middleware.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_no_llm_skips_compaction(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        middleware = SummarizationMiddleware(llm=None, context_window=200)
        big_state = {"data": "x" * 1000}
        out = asyncio.run(middleware.before_invoke(big_state, {}))
        self.assertNotIn("_compaction_summary", out)

    def test_compaction_prompt_constant(self):
        from muse.middlewares.summarization_middleware import COMPACTION_PROMPT

        self.assertIn("CONTEXT CHECKPOINT COMPACTION", COMPACTION_PROMPT)

    def test_summary_prefix_constant(self):
        from muse.middlewares.summarization_middleware import SUMMARY_PREFIX

        self.assertIn("Another language model", SUMMARY_PREFIX)

    def test_recent_tokens_budget(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        middleware = SummarizationMiddleware(
            llm=None, context_window=128_000, recent_tokens=20_000
        )
        self.assertEqual(middleware._recent_tokens, 20_000)


if __name__ == "__main__":
    unittest.main()
