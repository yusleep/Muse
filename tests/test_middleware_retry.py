from __future__ import annotations

import asyncio
import unittest

from muse.services.http import ProviderError


class RetryMiddlewareTests(unittest.TestCase):
    def test_import(self):
        from muse.middlewares.retry_middleware import RetryMiddleware

        self.assertTrue(callable(RetryMiddleware))

    def test_conforms_to_protocol(self):
        from muse.middlewares.base import Middleware
        from muse.middlewares.retry_middleware import RetryMiddleware

        middleware = RetryMiddleware()
        self.assertIsInstance(middleware, Middleware)

    def test_before_invoke_passthrough(self):
        from muse.middlewares.retry_middleware import RetryMiddleware

        middleware = RetryMiddleware()
        state = {"x": 1}
        out = asyncio.run(middleware.before_invoke(state, {}))
        self.assertEqual(out, state)

    def test_after_invoke_passthrough_on_success(self):
        from muse.middlewares.retry_middleware import RetryMiddleware

        middleware = RetryMiddleware()
        asyncio.run(middleware.before_invoke({}, {}))
        result = {"value": 42}
        out = asyncio.run(middleware.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_wrap_retries_on_transient_error(self):
        from muse.middlewares.base import MiddlewareChain
        from muse.middlewares.retry_middleware import RetryMiddleware

        call_count = 0

        def flaky_node(state):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("HTTP 503: temporarily unavailable")
            return {"ok": True}

        chain = MiddlewareChain([RetryMiddleware(max_retries=2, base_delay=0.01)])
        wrapped = chain.wrap(flaky_node)
        result = wrapped({})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count, 2)

    def test_wrap_gives_up_after_max_retries(self):
        from muse.middlewares.base import MiddlewareChain
        from muse.middlewares.retry_middleware import RetryMiddleware

        def always_fail(state):
            raise ProviderError("HTTP 503: down")

        chain = MiddlewareChain([RetryMiddleware(max_retries=2, base_delay=0.01)])
        wrapped = chain.wrap(always_fail)
        with self.assertRaises(ProviderError):
            wrapped({})

    def test_non_transient_error_not_retried(self):
        from muse.middlewares.base import MiddlewareChain
        from muse.middlewares.retry_middleware import RetryMiddleware

        call_count = 0

        def bad_node(state):
            nonlocal call_count
            call_count += 1
            raise ValueError("programming error, not transient")

        chain = MiddlewareChain([RetryMiddleware(max_retries=3, base_delay=0.01)])
        wrapped = chain.wrap(bad_node)
        with self.assertRaises(ValueError):
            wrapped({})
        self.assertEqual(call_count, 1)

    def test_retryable_keywords(self):
        from muse.middlewares.retry_middleware import is_transient_error

        self.assertTrue(is_transient_error(ProviderError("HTTP 429: rate limit")))
        self.assertTrue(is_transient_error(ProviderError("connection timed out")))
        self.assertTrue(is_transient_error(ProviderError("HTTP 502: bad gateway")))
        self.assertTrue(is_transient_error(ProviderError("HTTP 503: unavailable")))
        self.assertFalse(is_transient_error(ProviderError("HTTP 400: bad request")))
        self.assertFalse(is_transient_error(ValueError("something else")))


if __name__ == "__main__":
    unittest.main()
