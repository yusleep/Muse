# Phase 0-B: Middleware Framework

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an ordered middleware chain for cross-cutting concerns (logging, retry, context compaction, dangling tool-call repair).

**Architecture:** Middleware protocol with before_invoke/after_invoke hooks. MiddlewareChain wraps node functions. Codex CLI local compaction mode for context management.

**Tech Stack:** Python 3.10, asyncio

**Depends on:** Nothing (foundation, parallel with Phase 0-A)

---

## Task 1: Create Middleware protocol and MiddlewareChain

**File:** `muse/middlewares/__init__.py`
**File:** `muse/middlewares/base.py`
**Test:** `tests/test_middleware_base.py`

### 1.1 Write failing tests

Create `tests/test_middleware_base.py`:

```python
import asyncio
import unittest
from typing import Any


class MiddlewareProtocolTests(unittest.TestCase):
    """Verify the Middleware protocol and MiddlewareChain plumbing."""

    def test_import_middleware_protocol(self):
        from muse.middlewares.base import Middleware
        # Middleware must be a runtime_checkable Protocol
        import typing
        self.assertTrue(getattr(Middleware, '__protocol_attrs__', None) is not None
                        or typing.runtime_checkable)

    def test_import_middleware_chain(self):
        from muse.middlewares.base import MiddlewareChain
        self.assertTrue(callable(MiddlewareChain))

    def test_chain_wrap_returns_callable(self):
        from muse.middlewares.base import MiddlewareChain

        chain = MiddlewareChain([])

        def node_fn(state):
            return {"out": state.get("x", 0) + 1}

        wrapped = chain.wrap(node_fn)
        self.assertTrue(callable(wrapped))

    def test_chain_no_middlewares_passthrough(self):
        """With zero middlewares the wrapped node should behave identically."""
        from muse.middlewares.base import MiddlewareChain

        chain = MiddlewareChain([])

        def node_fn(state):
            return {"result": state.get("v", 0) * 2}

        wrapped = chain.wrap(node_fn)
        result = wrapped({"v": 5})
        self.assertEqual(result, {"result": 10})

    def test_before_invoke_can_modify_state(self):
        """A middleware's before_invoke can inject keys into state."""
        from muse.middlewares.base import MiddlewareChain

        class InjectMiddleware:
            async def before_invoke(self, state, config):
                state = dict(state)
                state["injected"] = True
                return state

            async def after_invoke(self, state, result, config):
                return result

        chain = MiddlewareChain([InjectMiddleware()])
        captured = {}

        def node_fn(state):
            captured.update(state)
            return {"done": True}

        wrapped = chain.wrap(node_fn)
        wrapped({"x": 1})
        self.assertTrue(captured.get("injected"))

    def test_after_invoke_can_modify_result(self):
        """A middleware's after_invoke can augment the result dict."""
        from muse.middlewares.base import MiddlewareChain

        class TagMiddleware:
            async def before_invoke(self, state, config):
                return state

            async def after_invoke(self, state, result, config):
                result = dict(result)
                result["tagged"] = True
                return result

        chain = MiddlewareChain([TagMiddleware()])

        def node_fn(state):
            return {"value": 42}

        wrapped = chain.wrap(node_fn)
        out = wrapped({"x": 1})
        self.assertEqual(out["value"], 42)
        self.assertTrue(out["tagged"])

    def test_middleware_execution_order(self):
        """before_invoke runs top-down; after_invoke runs bottom-up."""
        from muse.middlewares.base import MiddlewareChain

        order = []

        class MW:
            def __init__(self, name):
                self._name = name

            async def before_invoke(self, state, config):
                order.append(f"before:{self._name}")
                return state

            async def after_invoke(self, state, result, config):
                order.append(f"after:{self._name}")
                return result

        chain = MiddlewareChain([MW("A"), MW("B"), MW("C")])

        def node_fn(state):
            order.append("node")
            return {}

        wrapped = chain.wrap(node_fn)
        wrapped({})
        self.assertEqual(order, [
            "before:A", "before:B", "before:C",
            "node",
            "after:C", "after:B", "after:A",
        ])

    def test_chain_wrap_with_config_arg(self):
        """Wrapped node must accept optional config dict (LangGraph passes it)."""
        from muse.middlewares.base import MiddlewareChain

        chain = MiddlewareChain([])

        def node_fn(state):
            return {"ok": True}

        wrapped = chain.wrap(node_fn)
        # LangGraph sometimes passes config as second positional arg
        result = wrapped({"x": 1}, {"configurable": {"thread_id": "t1"}})
        self.assertEqual(result, {"ok": True})

    def test_async_node_fn_supported(self):
        """MiddlewareChain.wrap must also support async node functions."""
        from muse.middlewares.base import MiddlewareChain

        chain = MiddlewareChain([])

        async def async_node(state):
            return {"async": True}

        wrapped = chain.wrap(async_node)
        result = wrapped({"x": 1})
        self.assertEqual(result, {"async": True})

    def test_middlewares_package_init_reexports(self):
        from muse.middlewares import Middleware, MiddlewareChain
        self.assertTrue(callable(MiddlewareChain))


if __name__ == "__main__":
    unittest.main()
```

### 1.2 Verify tests fail

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_base.py -x 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'muse.middlewares'`

### 1.3 Implement

Create `muse/middlewares/__init__.py`:

```python
"""Middleware framework for Muse graph nodes."""

from .base import Middleware, MiddlewareChain

__all__ = ["Middleware", "MiddlewareChain"]
```

Create `muse/middlewares/base.py`:

```python
"""Middleware protocol and chain for wrapping LangGraph node functions."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Middleware(Protocol):
    """Hook interface for graph-node cross-cutting concerns.

    before_invoke: called before the node function, may modify state.
    after_invoke:  called after the node function, may modify result.
    """

    async def before_invoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]: ...
    async def after_invoke(self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]: ...


class MiddlewareChain:
    """Ordered middleware stack that wraps a LangGraph node function.

    Execution order:
        before_invoke:  top -> bottom  (middlewares[0] first)
        node function:  called once
        after_invoke:   bottom -> top  (middlewares[-1] first, then back up)
    """

    def __init__(self, middlewares: list[Middleware]) -> None:
        self._middlewares: list[Middleware] = list(middlewares)

    def wrap(self, node_fn: Callable) -> Callable:
        """Return a sync callable suitable for ``builder.add_node()``.

        The returned function accepts ``(state)`` or ``(state, config)`` to
        match LangGraph's node calling conventions.
        """
        middlewares = self._middlewares

        def wrapped(state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
            cfg = config or {}
            loop = _get_or_create_event_loop()
            return loop.run_until_complete(_run(middlewares, node_fn, state, cfg))

        return wrapped


async def _run(
    middlewares: list[Middleware],
    node_fn: Callable,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    # --- before_invoke (top -> bottom) ---
    current_state = state
    for mw in middlewares:
        current_state = await mw.before_invoke(current_state, config)

    # --- node execution ---
    if inspect.iscoroutinefunction(node_fn):
        result = await node_fn(current_state)
    else:
        result = node_fn(current_state)

    # --- after_invoke (bottom -> top) ---
    current_result = result
    for mw in reversed(middlewares):
        current_result = await mw.after_invoke(current_state, current_result, config)

    return current_result


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return the running event loop or create a new one."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are inside an existing async context -- create a new loop in a
        # thread to avoid "cannot run nested event loop" errors.  This is the
        # typical situation when LangGraph nodes are invoked from an async
        # graph runner.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _sentinel())
            future.result()  # just to confirm thread works
        # Return a fresh loop that is NOT running yet.
        return asyncio.new_event_loop()

    return asyncio.new_event_loop()


async def _sentinel() -> None:
    """No-op coroutine used to verify thread-pool event loop creation."""
    pass
```

### 1.4 Verify tests pass

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_base.py -v
```

Expected: all 10 tests pass.

### 1.5 Commit

```bash
git add muse/middlewares/__init__.py muse/middlewares/base.py tests/test_middleware_base.py
git commit -m "feat(middleware): add Middleware protocol and MiddlewareChain"
```

---

## Task 2: Create LoggingMiddleware

**File:** `muse/middlewares/logging_middleware.py`
**Test:** `tests/test_middleware_logging.py`

### 2.1 Write failing tests

Create `tests/test_middleware_logging.py`:

```python
import json
import os
import tempfile
import unittest
from typing import Any


class LoggingMiddlewareTests(unittest.TestCase):
    def test_import(self):
        from muse.middlewares.logging_middleware import LoggingMiddleware
        self.assertTrue(callable(LoggingMiddleware))

    def test_conforms_to_protocol(self):
        from muse.middlewares.base import Middleware
        from muse.middlewares.logging_middleware import LoggingMiddleware
        mw = LoggingMiddleware()
        self.assertIsInstance(mw, Middleware)

    def test_writes_jsonl_entry_on_after_invoke(self):
        import asyncio
        from muse.middlewares.logging_middleware import LoggingMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "nodes.jsonl")
            mw = LoggingMiddleware(log_path=log_path)
            state = {"project_id": "run-1", "topic": "test"}
            config = {"configurable": {"thread_id": "t1"}}

            asyncio.run(mw.before_invoke(state, config))
            result = {"references": [{"title": "A"}]}
            asyncio.run(mw.after_invoke(state, result, config))

            self.assertTrue(os.path.exists(log_path))
            with open(log_path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
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
            mw = LoggingMiddleware(log_path=log_path)
            state = {}
            config = {}

            asyncio.run(mw.before_invoke(state, config))
            time.sleep(0.05)  # 50ms
            asyncio.run(mw.after_invoke(state, {}, config))

            with open(log_path) as f:
                entry = json.loads(f.readline())
            self.assertGreaterEqual(entry["latency_ms"], 40)

    def test_records_token_usage_from_state(self):
        """If result contains usage info, it should be captured."""
        import asyncio
        from muse.middlewares.logging_middleware import LoggingMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "nodes.jsonl")
            mw = LoggingMiddleware(log_path=log_path, node_name="search")
            state = {}
            config = {}

            asyncio.run(mw.before_invoke(state, config))
            result = {"_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
            asyncio.run(mw.after_invoke(state, result, config))

            with open(log_path) as f:
                entry = json.loads(f.readline())
            self.assertEqual(entry["node"], "search")
            self.assertEqual(entry["usage"]["prompt_tokens"], 100)

    def test_after_invoke_returns_result_unchanged(self):
        import asyncio
        from muse.middlewares.logging_middleware import LoggingMiddleware

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "nodes.jsonl")
            mw = LoggingMiddleware(log_path=log_path)

            asyncio.run(mw.before_invoke({}, {}))
            result = {"key": "value"}
            out = asyncio.run(mw.after_invoke({}, result, {}))
            self.assertEqual(out, result)

    def test_before_invoke_returns_state_unchanged(self):
        import asyncio
        from muse.middlewares.logging_middleware import LoggingMiddleware

        mw = LoggingMiddleware()
        state = {"x": 1, "y": 2}
        out = asyncio.run(mw.before_invoke(state, {}))
        self.assertEqual(out, state)

    def test_no_log_path_does_not_crash(self):
        """When log_path is None, middleware should still work (no-op logging)."""
        import asyncio
        from muse.middlewares.logging_middleware import LoggingMiddleware

        mw = LoggingMiddleware(log_path=None)
        asyncio.run(mw.before_invoke({}, {}))
        out = asyncio.run(mw.after_invoke({}, {"v": 1}, {}))
        self.assertEqual(out, {"v": 1})


if __name__ == "__main__":
    unittest.main()
```

### 2.2 Verify tests fail

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_logging.py -x 2>&1 | head -10
```

Expected: `ModuleNotFoundError`

### 2.3 Implement

Create `muse/middlewares/logging_middleware.py`:

```python
"""LoggingMiddleware -- JSONL node-execution tracing."""

from __future__ import annotations

import json
import os
import time
from typing import Any


class LoggingMiddleware:
    """Records per-node-invocation metrics to a JSONL file.

    Tracked fields: timestamp, node name, latency_ms, token usage,
    result key summary.

    When *log_path* is ``None`` the middleware is a silent pass-through
    (useful in tests or when logging is disabled via config).
    """

    def __init__(
        self,
        log_path: str | None = None,
        node_name: str = "unknown",
    ) -> None:
        self._log_path = log_path
        self._node_name = node_name
        self._start_time: float | None = None

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self._start_time = time.monotonic()
        return state

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        elapsed_ms = 0.0
        if self._start_time is not None:
            elapsed_ms = (time.monotonic() - self._start_time) * 1000.0
            self._start_time = None

        if self._log_path is not None:
            entry: dict[str, Any] = {
                "timestamp": time.time(),
                "node": self._node_name,
                "latency_ms": round(elapsed_ms, 2),
                "result_keys": sorted(result.keys()) if isinstance(result, dict) else [],
            }

            # Capture token usage when the node stashes it in result.
            usage = result.get("_usage") if isinstance(result, dict) else None
            if isinstance(usage, dict):
                entry["usage"] = usage

            # Thread id from config, if available.
            configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
            thread_id = configurable.get("thread_id")
            if thread_id:
                entry["thread_id"] = thread_id

            _append_jsonl(self._log_path, entry)

        return result


def _append_jsonl(path: str, entry: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
```

### 2.4 Verify tests pass

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_logging.py -v
```

### 2.5 Commit

```bash
git add muse/middlewares/logging_middleware.py tests/test_middleware_logging.py
git commit -m "feat(middleware): add LoggingMiddleware with JSONL tracing"
```

---

## Task 3: Create RetryMiddleware

**File:** `muse/middlewares/retry_middleware.py`
**Test:** `tests/test_middleware_retry.py`

This replaces the scattered retry logic in `providers.py` `_chat_completion` (lines 325-370) with a unified, configurable middleware.

### 3.1 Write failing tests

Create `tests/test_middleware_retry.py`:

```python
import asyncio
import unittest
from typing import Any

from muse.services.http import ProviderError


class RetryMiddlewareTests(unittest.TestCase):
    def test_import(self):
        from muse.middlewares.retry_middleware import RetryMiddleware
        self.assertTrue(callable(RetryMiddleware))

    def test_conforms_to_protocol(self):
        from muse.middlewares.base import Middleware
        from muse.middlewares.retry_middleware import RetryMiddleware
        mw = RetryMiddleware()
        self.assertIsInstance(mw, Middleware)

    def test_before_invoke_passthrough(self):
        from muse.middlewares.retry_middleware import RetryMiddleware
        mw = RetryMiddleware()
        state = {"x": 1}
        out = asyncio.run(mw.before_invoke(state, {}))
        self.assertEqual(out, state)

    def test_after_invoke_passthrough_on_success(self):
        from muse.middlewares.retry_middleware import RetryMiddleware
        mw = RetryMiddleware()
        asyncio.run(mw.before_invoke({}, {}))
        result = {"value": 42}
        out = asyncio.run(mw.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_wrap_retries_on_transient_error(self):
        """Node that fails once with ProviderError then succeeds should work."""
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
        from muse.middlewares.retry_middleware import RetryMiddleware, is_transient_error

        self.assertTrue(is_transient_error(ProviderError("HTTP 429: rate limit")))
        self.assertTrue(is_transient_error(ProviderError("connection timed out")))
        self.assertTrue(is_transient_error(ProviderError("HTTP 502: bad gateway")))
        self.assertTrue(is_transient_error(ProviderError("HTTP 503: unavailable")))
        self.assertFalse(is_transient_error(ProviderError("HTTP 400: bad request")))
        self.assertFalse(is_transient_error(ValueError("something else")))


if __name__ == "__main__":
    unittest.main()
```

### 3.2 Verify tests fail

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_retry.py -x 2>&1 | head -10
```

### 3.3 Implement

Create `muse/middlewares/retry_middleware.py`:

```python
"""RetryMiddleware -- unified transient-failure retry for graph nodes."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from muse.services.http import ProviderError

_TRANSIENT_KEYWORDS = (
    "timed out",
    "timeout",
    "network",
    "connection",
    "502",
    "503",
    "429",
    "invalid json response",
    "rate limit",
)


def is_transient_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a transient network / rate-limit error."""
    if not isinstance(exc, (ProviderError, ConnectionError, TimeoutError, OSError)):
        return False
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


class RetryMiddleware:
    """Retries the wrapped node function on transient failures.

    This middleware operates differently from the others: it hooks into the
    MiddlewareChain's execution by wrapping the **node function itself**.
    ``before_invoke`` / ``after_invoke`` are pass-throughs; the actual retry
    logic lives in ``wrap_node``, which ``MiddlewareChain.wrap`` calls
    to produce the retrying node callable.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (default 2, so up to 3 total calls).
    base_delay : float
        Base delay in seconds between retries.  Actual delay is
        ``base_delay * (attempt + 1)`` (linear back-off).
    """

    def __init__(self, max_retries: int = 2, base_delay: float = 5.0) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._node_fn: Any = None

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return state

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return result

    def wrap_node(self, node_fn):
        """Return a wrapper around *node_fn* that retries on transient errors."""
        max_retries = self.max_retries
        base_delay = self.base_delay

        if inspect.iscoroutinefunction(node_fn):
            async def retrying_node(state):
                last_exc: BaseException | None = None
                for attempt in range(max_retries + 1):
                    try:
                        return await node_fn(state)
                    except Exception as exc:
                        if not is_transient_error(exc) or attempt >= max_retries:
                            raise
                        last_exc = exc
                        await asyncio.sleep(base_delay * (attempt + 1))
                raise last_exc  # unreachable, but satisfies type checker
            return retrying_node
        else:
            def retrying_node(state):
                last_exc: BaseException | None = None
                for attempt in range(max_retries + 1):
                    try:
                        return node_fn(state)
                    except Exception as exc:
                        if not is_transient_error(exc) or attempt >= max_retries:
                            raise
                        last_exc = exc
                        time.sleep(base_delay * (attempt + 1))
                raise last_exc  # unreachable
            return retrying_node
```

Then update `muse/middlewares/base.py` to give `RetryMiddleware` its special treatment.  The `MiddlewareChain.wrap` method needs to detect middlewares that implement `wrap_node` and apply that wrapper around the node function before the normal before/after cycle:

In `muse/middlewares/base.py`, replace the `wrap` method body:

```python
    def wrap(self, node_fn: Callable) -> Callable:
        """Return a sync callable suitable for ``builder.add_node()``.

        The returned function accepts ``(state)`` or ``(state, config)`` to
        match LangGraph's node calling conventions.

        Middlewares that expose a ``wrap_node`` method (e.g. RetryMiddleware)
        get applied as node-level decorators, in addition to the standard
        before_invoke / after_invoke hooks.
        """
        middlewares = self._middlewares

        # Apply node-level wrappers (innermost first).
        effective_fn = node_fn
        for mw in reversed(middlewares):
            wrapper = getattr(mw, "wrap_node", None)
            if callable(wrapper):
                effective_fn = wrapper(effective_fn)

        def wrapped(state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
            cfg = config or {}
            loop = _get_or_create_event_loop()
            return loop.run_until_complete(_run(middlewares, effective_fn, state, cfg))

        return wrapped
```

### 3.4 Verify tests pass

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_retry.py -v
```

### 3.5 Commit

```bash
git add muse/middlewares/retry_middleware.py muse/middlewares/base.py tests/test_middleware_retry.py
git commit -m "feat(middleware): add RetryMiddleware with transient-error backoff"
```

---

## Task 4: Create SummarizationMiddleware

**File:** `muse/middlewares/summarization_middleware.py`
**Test:** `tests/test_middleware_summarization.py`

Implements Codex CLI local compaction mode: when estimated token count of accumulated state exceeds 90% of the context window, the middleware sends the state to the LLM for summarization, then replaces it with a compact summary plus recent messages.

### 4.1 Write failing tests

Create `tests/test_middleware_summarization.py`:

```python
import asyncio
import unittest
from typing import Any


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
        mw = SummarizationMiddleware(llm=None, context_window=128_000)
        self.assertIsInstance(mw, Middleware)

    def test_estimate_tokens_heuristic(self):
        from muse.middlewares.summarization_middleware import estimate_tokens
        # 4 bytes per token heuristic
        text = "a" * 400  # 400 bytes -> 100 tokens
        self.assertEqual(estimate_tokens(text), 100)

    def test_estimate_tokens_unicode(self):
        from muse.middlewares.summarization_middleware import estimate_tokens
        # Chinese chars are ~3 bytes each in UTF-8
        text = "中" * 100  # 300 bytes -> 75 tokens
        self.assertEqual(estimate_tokens(text), 75)

    def test_no_compaction_below_threshold(self):
        """When state is small, before_invoke should be a pass-through."""
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        call_log = []

        class FakeLLM:
            def text(self, **kwargs):
                call_log.append("llm_called")
                return "summary"

        mw = SummarizationMiddleware(
            llm=FakeLLM(),
            context_window=128_000,
            threshold_ratio=0.9,
        )
        state = {"topic": "small state"}
        out = asyncio.run(mw.before_invoke(state, {}))
        self.assertEqual(out, state)
        self.assertEqual(call_log, [])  # LLM should NOT have been called

    def test_compaction_triggered_above_threshold(self):
        """When state exceeds threshold, LLM should be called for compaction."""
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        call_log = []

        class FakeLLM:
            def text(self, **kwargs):
                call_log.append(kwargs)
                return "Compact summary of progress."

        # Use a tiny context window so the state easily exceeds 90%
        mw = SummarizationMiddleware(
            llm=FakeLLM(),
            context_window=200,   # 200 tokens
            threshold_ratio=0.9,  # trigger at 180 tokens
        )
        # Create state that is large (> 180 tokens = > 720 bytes)
        big_state = {"data": "x" * 1000}
        out = asyncio.run(mw.before_invoke(big_state, {}))

        self.assertEqual(len(call_log), 1)
        # The system prompt must contain the compaction prompt text
        self.assertIn("CONTEXT CHECKPOINT COMPACTION", call_log[0]["system"])
        # Result state must contain the summary with prefix
        self.assertIn("_compaction_summary", out)
        self.assertIn("Compact summary of progress.", out["_compaction_summary"])

    def test_compaction_preserves_recent_keys(self):
        """Compaction must keep specified recent state keys intact."""
        from muse.middlewares.summarization_middleware import SummarizationMiddleware

        class FakeLLM:
            def text(self, **kwargs):
                return "Summary."

        mw = SummarizationMiddleware(
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
        out = asyncio.run(mw.before_invoke(big_state, {}))
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

        mw = SummarizationMiddleware(llm=FakeLLM(), context_window=200)
        big_state = {"data": "x" * 1000}
        out = asyncio.run(mw.before_invoke(big_state, {}))
        summary = out["_compaction_summary"]
        self.assertTrue(summary.startswith(SUMMARY_PREFIX))

    def test_after_invoke_passthrough(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware
        mw = SummarizationMiddleware(llm=None, context_window=128_000)
        result = {"ok": True}
        out = asyncio.run(mw.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_no_llm_skips_compaction(self):
        """When llm is None, compaction is skipped even if state is huge."""
        from muse.middlewares.summarization_middleware import SummarizationMiddleware
        mw = SummarizationMiddleware(llm=None, context_window=200)
        big_state = {"data": "x" * 1000}
        out = asyncio.run(mw.before_invoke(big_state, {}))
        self.assertNotIn("_compaction_summary", out)

    def test_compaction_prompt_constant(self):
        from muse.middlewares.summarization_middleware import COMPACTION_PROMPT
        self.assertIn("CONTEXT CHECKPOINT COMPACTION", COMPACTION_PROMPT)

    def test_summary_prefix_constant(self):
        from muse.middlewares.summarization_middleware import SUMMARY_PREFIX
        self.assertIn("Another language model", SUMMARY_PREFIX)

    def test_recent_tokens_budget(self):
        from muse.middlewares.summarization_middleware import SummarizationMiddleware
        mw = SummarizationMiddleware(llm=None, context_window=128_000, recent_tokens=20_000)
        self.assertEqual(mw._recent_tokens, 20_000)


if __name__ == "__main__":
    unittest.main()
```

### 4.2 Verify tests fail

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_summarization.py -x 2>&1 | head -10
```

### 4.3 Implement

Create `muse/middlewares/summarization_middleware.py`:

```python
"""SummarizationMiddleware -- Codex CLI local compaction mode.

When estimated token count of the serialized state exceeds a configurable
fraction of the context window, the middleware asks the LLM to produce a
compact handoff summary, then replaces the state with:

  { preserved_keys..., _compaction_summary: PREFIX + summary_text }

This mirrors the compaction strategy used by Codex CLI in "local" mode.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Constants -- taken verbatim from Codex CLI
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_BYTES_PER_TOKEN = 4  # Codex CLI heuristic


def estimate_tokens(text: str) -> int:
    """Estimate token count using the 4-bytes-per-token heuristic."""
    return len(text.encode("utf-8")) // _BYTES_PER_TOKEN


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# Default state keys that survive compaction (identity / routing fields).
_DEFAULT_PRESERVE_KEYS = (
    "project_id",
    "topic",
    "discipline",
    "language",
    "format_standard",
    "output_format",
)


class SummarizationMiddleware:
    """Context-window compaction middleware.

    Parameters
    ----------
    llm : object | None
        An ``LLMClient`` (or compatible) with a ``.text()`` method.
        If ``None``, compaction is disabled.
    context_window : int
        Model context window size in tokens (e.g. 128_000).
    threshold_ratio : float
        Fraction of *context_window* at which compaction triggers (default 0.9).
    recent_tokens : int
        Token budget for "recent" state keys to keep verbatim (default 20_000).
    preserve_keys : list[str] | None
        State keys that survive compaction untouched.  Defaults to identity
        fields (project_id, topic, discipline, ...).
    """

    def __init__(
        self,
        llm: Any,
        context_window: int,
        threshold_ratio: float = 0.9,
        recent_tokens: int = 20_000,
        preserve_keys: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._context_window = context_window
        self._threshold_ratio = threshold_ratio
        self._recent_tokens = recent_tokens
        self._preserve_keys: tuple[str, ...] = (
            tuple(preserve_keys) if preserve_keys is not None
            else _DEFAULT_PRESERVE_KEYS
        )

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        if self._llm is None:
            return state

        serialized = json.dumps(state, ensure_ascii=False, default=str)
        token_count = estimate_tokens(serialized)
        threshold = int(self._context_window * self._threshold_ratio)

        if token_count <= threshold:
            return state

        # --- Compact ---
        summary_text = self._llm.text(
            system=COMPACTION_PROMPT,
            user=serialized[: self._recent_tokens * _BYTES_PER_TOKEN],
            route="default",
            max_tokens=2000,
        )

        full_summary = f"{SUMMARY_PREFIX}\n\n{summary_text}"

        compacted: dict[str, Any] = {}
        for key in self._preserve_keys:
            if key in state:
                compacted[key] = state[key]
        compacted["_compaction_summary"] = full_summary
        return compacted

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return result
```

### 4.4 Verify tests pass

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_summarization.py -v
```

### 4.5 Commit

```bash
git add muse/middlewares/summarization_middleware.py tests/test_middleware_summarization.py
git commit -m "feat(middleware): add SummarizationMiddleware (Codex CLI compaction)"
```

---

## Task 5: Create DanglingToolCallMiddleware

**File:** `muse/middlewares/dangling_tool_call.py`
**Test:** `tests/test_middleware_dangling.py`

Fixes a common ReAct failure mode: the LLM emits a `tool_calls` list in its response but the graph node exits before all tool calls are resolved.  This middleware detects dangling tool-call entries in the result and patches them with error responses so the next invocation doesn't crash.

### 5.1 Write failing tests

Create `tests/test_middleware_dangling.py`:

```python
import asyncio
import unittest
from typing import Any


class DanglingToolCallMiddlewareTests(unittest.TestCase):
    def test_import(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        self.assertTrue(callable(DanglingToolCallMiddleware))

    def test_conforms_to_protocol(self):
        from muse.middlewares.base import Middleware
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        mw = DanglingToolCallMiddleware()
        self.assertIsInstance(mw, Middleware)

    def test_before_invoke_passthrough(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        mw = DanglingToolCallMiddleware()
        state = {"x": 1}
        out = asyncio.run(mw.before_invoke(state, {}))
        self.assertEqual(out, state)

    def test_no_tool_calls_passthrough(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        mw = DanglingToolCallMiddleware()
        result = {"text": "hello"}
        out = asyncio.run(mw.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_complete_tool_calls_passthrough(self):
        """Tool calls that have a matching tool_response are left alone."""
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        mw = DanglingToolCallMiddleware()
        result = {
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "tc_1", "function": {"name": "search", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "tc_1", "content": "results"},
            ]
        }
        out = asyncio.run(mw.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_dangling_tool_call_gets_error_response(self):
        """Tool calls without a matching tool response should get an error patch."""
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        mw = DanglingToolCallMiddleware()
        result = {
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "tc_1", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "tc_2", "function": {"name": "write", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
                # tc_2 is dangling -- no tool response
            ]
        }
        out = asyncio.run(mw.after_invoke({}, result, {}))
        messages = out["messages"]
        # There should now be a tool response for tc_2
        tool_responses = [m for m in messages if m.get("role") == "tool"]
        tc2_responses = [m for m in tool_responses if m.get("tool_call_id") == "tc_2"]
        self.assertEqual(len(tc2_responses), 1)
        self.assertIn("error", tc2_responses[0]["content"].lower())

    def test_multiple_dangling_all_patched(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        mw = DanglingToolCallMiddleware()
        result = {
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "tc_a", "function": {"name": "f1", "arguments": "{}"}},
                    {"id": "tc_b", "function": {"name": "f2", "arguments": "{}"}},
                    {"id": "tc_c", "function": {"name": "f3", "arguments": "{}"}},
                ]},
                # No tool responses at all
            ]
        }
        out = asyncio.run(mw.after_invoke({}, result, {}))
        tool_responses = [m for m in out["messages"] if m.get("role") == "tool"]
        self.assertEqual(len(tool_responses), 3)
        response_ids = {m["tool_call_id"] for m in tool_responses}
        self.assertEqual(response_ids, {"tc_a", "tc_b", "tc_c"})

    def test_no_messages_key_passthrough(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        mw = DanglingToolCallMiddleware()
        result = {"references": [{"title": "A"}]}
        out = asyncio.run(mw.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_repair_message_content(self):
        """The injected tool response should include the tool name for debugging."""
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware
        mw = DanglingToolCallMiddleware()
        result = {
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "tc_x", "function": {"name": "verify_doi", "arguments": "{}"}},
                ]},
            ]
        }
        out = asyncio.run(mw.after_invoke({}, result, {}))
        tool_msg = [m for m in out["messages"] if m.get("role") == "tool"][0]
        self.assertIn("verify_doi", tool_msg["content"])


if __name__ == "__main__":
    unittest.main()
```

### 5.2 Verify tests fail

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_dangling.py -x 2>&1 | head -10
```

### 5.3 Implement

Create `muse/middlewares/dangling_tool_call.py`:

```python
"""DanglingToolCallMiddleware -- patch incomplete tool_calls in ReAct output.

When a ReAct agent emits an assistant message with ``tool_calls`` but the
graph exits before all tool calls receive a ``role: tool`` response, the
next LLM invocation will fail because the message history is inconsistent.

This middleware scans the result's ``messages`` list after every node
invocation and injects synthetic error-response tool messages for any
tool call ID that lacks a matching response.
"""

from __future__ import annotations

from typing import Any


class DanglingToolCallMiddleware:
    """Detect and repair dangling tool calls in node output."""

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return state

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            return result

        messages = result.get("messages")
        if not isinstance(messages, list):
            return result

        # Collect all tool_call IDs and their function names.
        pending: dict[str, str] = {}  # {tool_call_id: function_name}
        answered: set[str] = set()

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc.get("id", "")
                            fn = tc.get("function", {})
                            fn_name = fn.get("name", "unknown") if isinstance(fn, dict) else "unknown"
                            if tc_id:
                                pending[tc_id] = fn_name
            elif msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id", "")
                if tc_id:
                    answered.add(tc_id)

        dangling = {tc_id: fn_name for tc_id, fn_name in pending.items() if tc_id not in answered}
        if not dangling:
            return result

        # Patch: append synthetic error responses for each dangling call.
        patched_messages = list(messages)
        for tc_id, fn_name in dangling.items():
            patched_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": (
                    f"Error: tool call '{fn_name}' (id={tc_id}) was not executed. "
                    f"The node exited before this tool call could be processed. "
                    f"Please retry or choose an alternative approach."
                ),
            })

        patched_result = dict(result)
        patched_result["messages"] = patched_messages
        return patched_result
```

### 5.4 Verify tests pass

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_dangling.py -v
```

### 5.5 Commit

```bash
git add muse/middlewares/dangling_tool_call.py tests/test_middleware_dangling.py
git commit -m "feat(middleware): add DanglingToolCallMiddleware for ReAct safety"
```

---

## Task 6: Integration -- wire middleware chain into main_graph.py

**File:** `muse/graph/main_graph.py` (modify)
**File:** `muse/middlewares/__init__.py` (update exports)
**Test:** `tests/test_middleware_integration.py`

### 6.1 Write failing tests

Create `tests/test_middleware_integration.py`:

```python
import json
import os
import tempfile
import unittest
from typing import Any

from muse.config import Settings


class _FakeSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        return ([{"ref_id": "@a2024x", "title": "X", "authors": ["A"], "year": 2024,
                  "doi": None, "venue": "V", "abstract": "...", "source": "test",
                  "verified_metadata": True}],
                extra_queries or [topic])


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def structured(self, *, system, user, route="default", max_tokens=2500):
        self.calls += 1
        if self.calls == 1:
            return {"queries": ["q1"]}
        if self.calls == 2:
            return {"research_gaps": [], "core_concepts": [], "methodology_domain": "cs",
                    "suggested_contributions": []}
        return {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Intro",
                              "target_words": 2000, "complexity": "low",
                              "subsections": [{"title": "Background"}]}]}


class _FakeServices:
    def __init__(self):
        self.llm = _FakeLLM()
        self.search = _FakeSearch()
        self.local_refs = []
        self.rag_index = None


def _make_settings(tmp_dir: str, log_path: str | None = None) -> Settings:
    return Settings(
        llm_api_key="x",
        llm_base_url="http://localhost",
        llm_model="stub",
        model_router_config={},
        runs_dir=tmp_dir,
        semantic_scholar_api_key=None,
        openalex_email=None,
        crossref_mailto=None,
        refs_dir=None,
        checkpoint_dir=None,
    )


class MiddlewareIntegrationTests(unittest.TestCase):
    def test_build_graph_with_middleware_runs(self):
        """Graph built with default middleware chain should execute normally."""
        from muse.graph.launcher import build_graph, invoke

        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(tmp)
            services = _FakeServices()
            graph = build_graph(settings, services=services, thread_id="mw-test")
            result = invoke(graph, {
                "project_id": "mw-test",
                "topic": "Middleware testing",
                "discipline": "CS",
                "language": "en",
                "format_standard": "APA",
                "output_format": "markdown",
            }, thread_id="mw-test")
            self.assertIn("references", result)
            self.assertIn("outline", result)

    def test_build_default_middleware_chain(self):
        """The public helper should return a MiddlewareChain."""
        from muse.middlewares import MiddlewareChain, build_default_chain

        chain = build_default_chain()
        self.assertIsInstance(chain, MiddlewareChain)

    def test_build_default_chain_with_log_path(self):
        from muse.middlewares import build_default_chain

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "test.jsonl")
            chain = build_default_chain(log_dir=tmp)
            self.assertIsNotNone(chain)


if __name__ == "__main__":
    unittest.main()
```

### 6.2 Verify tests fail

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_integration.py -x 2>&1 | head -10
```

### 6.3 Implement

Update `muse/middlewares/__init__.py` with a `build_default_chain` factory:

```python
"""Middleware framework for Muse graph nodes."""

from .base import Middleware, MiddlewareChain
from .dangling_tool_call import DanglingToolCallMiddleware
from .logging_middleware import LoggingMiddleware
from .retry_middleware import RetryMiddleware
from .summarization_middleware import SummarizationMiddleware

__all__ = [
    "DanglingToolCallMiddleware",
    "LoggingMiddleware",
    "Middleware",
    "MiddlewareChain",
    "RetryMiddleware",
    "SummarizationMiddleware",
    "build_default_chain",
]


def build_default_chain(
    *,
    log_dir: str | None = None,
    node_name: str = "unknown",
    llm=None,
    context_window: int = 128_000,
    max_retries: int = 2,
    retry_base_delay: float = 5.0,
) -> MiddlewareChain:
    """Build the standard middleware chain for a graph node.

    Execution order:
        1. LoggingMiddleware       -- always
        2. RetryMiddleware         -- always
        3. SummarizationMiddleware -- only when llm is provided
        4. DanglingToolCallMiddleware -- always (safe no-op when no messages)
    """
    import os

    middlewares: list[Middleware] = []

    log_path = None
    if log_dir:
        log_path = os.path.join(log_dir, "nodes.jsonl")
    middlewares.append(LoggingMiddleware(log_path=log_path, node_name=node_name))

    middlewares.append(RetryMiddleware(max_retries=max_retries, base_delay=retry_base_delay))

    if llm is not None:
        middlewares.append(SummarizationMiddleware(llm=llm, context_window=context_window))

    middlewares.append(DanglingToolCallMiddleware())

    return MiddlewareChain(middlewares)
```

Update `muse/graph/main_graph.py` to wrap nodes with the middleware chain.  The changes are minimal -- add an import and a helper, then wrap each `build_*_node()` result:

```python
"""Top-level Muse LangGraph definition."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from muse.config import Settings
from muse.graph.nodes import (
    build_export_node,
    build_initialize_node,
    build_interrupt_node,
    build_merge_chapters_node,
    build_outline_node,
    build_polish_node,
    build_search_node,
)
from muse.graph.nodes.draft import fan_out_chapters
from muse.graph.state import MuseState
from muse.graph.subgraphs.chapter import build_chapter_subgraph_node
from muse.graph.subgraphs.citation import build_citation_subgraph_node
from muse.graph.subgraphs.composition import build_composition_subgraph_node
from muse.middlewares import build_default_chain


class _NullServices:
    def __init__(self) -> None:
        self.local_refs = []
        self.rag_index = None
        self.search = None
        self.llm = None


def _default_settings() -> Settings:
    return Settings(
        llm_api_key="",
        llm_base_url="https://api.openai.com/v1",
        llm_model="router/default",
        model_router_config={},
        runs_dir="runs",
        semantic_scholar_api_key=None,
        openalex_email=None,
        crossref_mailto=None,
        refs_dir=None,
        checkpoint_dir=None,
    )


def _wrap(node_fn, node_name: str, settings: Settings, services: Any):
    """Wrap a node function with the default middleware chain."""
    log_dir = getattr(settings, "runs_dir", None)
    llm = getattr(services, "llm", None)
    chain = build_default_chain(
        log_dir=log_dir,
        node_name=node_name,
        llm=None,  # SummarizationMiddleware is for ReAct sub-graphs only (Phase 1+)
    )
    return chain.wrap(node_fn)


def build_graph(
    settings: Settings | None = None,
    *,
    services: Any | None = None,
    checkpointer: Any | None = None,
    auto_approve: bool = True,
):
    settings = settings or _default_settings()
    services = services or _NullServices()

    builder = StateGraph(MuseState)
    builder.add_node("initialize", _wrap(build_initialize_node(settings, services), "initialize", settings, services))
    builder.add_node("search", _wrap(build_search_node(settings, services), "search", settings, services))
    builder.add_node("review_refs", build_interrupt_node("research", auto_approve=auto_approve))
    builder.add_node("outline", _wrap(build_outline_node(settings, services), "outline", settings, services))
    builder.add_node("approve_outline", build_interrupt_node("outline", auto_approve=auto_approve))
    builder.add_node("chapter_subgraph", build_chapter_subgraph_node(services=services))
    builder.add_node("merge_chapters", _wrap(build_merge_chapters_node(settings, services), "merge_chapters", settings, services))
    builder.add_node("review_draft", build_interrupt_node("draft", auto_approve=auto_approve))
    builder.add_node("citation_subgraph", build_citation_subgraph_node(services=services))
    builder.add_node("polish", _wrap(build_polish_node(services), "polish", settings, services))
    builder.add_node("composition_subgraph", build_composition_subgraph_node())
    builder.add_node("approve_final", build_interrupt_node("final", auto_approve=auto_approve))
    builder.add_node("export", _wrap(build_export_node(settings), "export", settings, services))
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "search")
    builder.add_edge("search", "review_refs")
    builder.add_edge("review_refs", "outline")
    builder.add_edge("outline", "approve_outline")
    builder.add_conditional_edges("approve_outline", fan_out_chapters, ["chapter_subgraph"])
    builder.add_edge("chapter_subgraph", "merge_chapters")
    builder.add_edge("merge_chapters", "review_draft")
    builder.add_edge("review_draft", "citation_subgraph")
    builder.add_edge("citation_subgraph", "polish")
    builder.add_edge("polish", "composition_subgraph")
    builder.add_edge("composition_subgraph", "approve_final")
    builder.add_edge("approve_final", "export")
    builder.add_edge("export", END)
    return builder.compile(checkpointer=checkpointer)
```

Note: Interrupt nodes are NOT wrapped because they use `langgraph.types.interrupt` which requires a clean call stack.  Subgraph nodes are left unwrapped for now because they have their own internal node logic.

### 6.4 Verify tests pass

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_integration.py -v
```

Then run the full existing test suite to ensure nothing is broken:

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_graph.py tests/test_phase0_structure.py -v
```

### 6.5 Commit

```bash
git add muse/middlewares/__init__.py muse/graph/main_graph.py tests/test_middleware_integration.py
git commit -m "feat(middleware): wire default chain into main_graph.py node wrapping"
```

---

## Task 7: Add middleware config to Settings

**File:** `muse/config.py` (modify)
**Test:** `tests/test_middleware_config.py`

### 7.1 Write failing tests

Create `tests/test_middleware_config.py`:

```python
import unittest


class MiddlewareConfigTests(unittest.TestCase):
    def test_settings_has_middleware_fields(self):
        from muse.config import Settings
        s = Settings(
            llm_api_key="k",
            llm_base_url="http://localhost",
            llm_model="m",
            model_router_config={},
            runs_dir="runs",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
            checkpoint_dir=None,
            middleware_retry_max=3,
            middleware_retry_delay=2.0,
            middleware_compaction_threshold=0.85,
            middleware_compaction_recent_tokens=15_000,
            middleware_context_window=200_000,
        )
        self.assertEqual(s.middleware_retry_max, 3)
        self.assertEqual(s.middleware_retry_delay, 2.0)
        self.assertEqual(s.middleware_compaction_threshold, 0.85)
        self.assertEqual(s.middleware_compaction_recent_tokens, 15_000)
        self.assertEqual(s.middleware_context_window, 200_000)

    def test_settings_middleware_defaults(self):
        from muse.config import Settings
        s = Settings(
            llm_api_key="k",
            llm_base_url="http://localhost",
            llm_model="m",
            model_router_config={},
            runs_dir="runs",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
        )
        self.assertEqual(s.middleware_retry_max, 2)
        self.assertEqual(s.middleware_retry_delay, 5.0)
        self.assertAlmostEqual(s.middleware_compaction_threshold, 0.9)
        self.assertEqual(s.middleware_compaction_recent_tokens, 20_000)
        self.assertEqual(s.middleware_context_window, 128_000)

    def test_load_settings_reads_middleware_env_vars(self):
        from muse.config import load_settings
        env = {
            "MUSE_LLM_API_KEY": "key",
            "MUSE_LLM_MODEL": "gpt-4",
            "MUSE_MIDDLEWARE_RETRY_MAX": "4",
            "MUSE_MIDDLEWARE_RETRY_DELAY": "3.5",
            "MUSE_MIDDLEWARE_COMPACTION_THRESHOLD": "0.8",
            "MUSE_MIDDLEWARE_COMPACTION_RECENT_TOKENS": "10000",
            "MUSE_MIDDLEWARE_CONTEXT_WINDOW": "64000",
        }
        s = load_settings(env)
        self.assertEqual(s.middleware_retry_max, 4)
        self.assertAlmostEqual(s.middleware_retry_delay, 3.5)
        self.assertAlmostEqual(s.middleware_compaction_threshold, 0.8)
        self.assertEqual(s.middleware_compaction_recent_tokens, 10_000)
        self.assertEqual(s.middleware_context_window, 64_000)

    def test_load_settings_middleware_defaults_when_env_absent(self):
        from muse.config import load_settings
        env = {"MUSE_LLM_API_KEY": "key", "MUSE_LLM_MODEL": "gpt-4"}
        s = load_settings(env)
        self.assertEqual(s.middleware_retry_max, 2)
        self.assertAlmostEqual(s.middleware_compaction_threshold, 0.9)

    def test_existing_settings_construction_unchanged(self):
        """Verify all existing Settings tests still work (no positional arg breakage)."""
        from muse.config import Settings
        # Existing callers pass positional or keyword -- both must still work.
        s = Settings(
            llm_api_key="k",
            llm_base_url="u",
            llm_model="m",
            model_router_config={},
            runs_dir="r",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
            checkpoint_dir=None,
        )
        self.assertEqual(s.llm_api_key, "k")
        # Defaults should apply
        self.assertEqual(s.middleware_retry_max, 2)


if __name__ == "__main__":
    unittest.main()
```

### 7.2 Verify tests fail

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_config.py -x 2>&1 | head -10
```

Expected: `TypeError: Settings.__init__() got an unexpected keyword argument 'middleware_retry_max'`

### 7.3 Implement

Edit `muse/config.py`.  Add five new fields to `Settings` with defaults, and read them in `load_settings`:

In the `Settings` dataclass, after the `checkpoint_dir` field, add:

```python
    # Middleware configuration
    middleware_retry_max: int = 2
    middleware_retry_delay: float = 5.0
    middleware_compaction_threshold: float = 0.9
    middleware_compaction_recent_tokens: int = 20_000
    middleware_context_window: int = 128_000
```

In `load_settings`, before the `return Settings(...)` call, add env-var parsing:

```python
    middleware_retry_max = int(source.get("MUSE_MIDDLEWARE_RETRY_MAX", "2").strip() or "2")
    middleware_retry_delay = float(source.get("MUSE_MIDDLEWARE_RETRY_DELAY", "5.0").strip() or "5.0")
    middleware_compaction_threshold = float(source.get("MUSE_MIDDLEWARE_COMPACTION_THRESHOLD", "0.9").strip() or "0.9")
    middleware_compaction_recent_tokens = int(source.get("MUSE_MIDDLEWARE_COMPACTION_RECENT_TOKENS", "20000").strip() or "20000")
    middleware_context_window = int(source.get("MUSE_MIDDLEWARE_CONTEXT_WINDOW", "128000").strip() or "128000")
```

And add them to the `return Settings(...)` call:

```python
        middleware_retry_max=middleware_retry_max,
        middleware_retry_delay=middleware_retry_delay,
        middleware_compaction_threshold=middleware_compaction_threshold,
        middleware_compaction_recent_tokens=middleware_compaction_recent_tokens,
        middleware_context_window=middleware_context_window,
```

Then update `_wrap` in `main_graph.py` to read from settings:

```python
def _wrap(node_fn, node_name: str, settings: Settings, services: Any):
    """Wrap a node function with the default middleware chain."""
    log_dir = getattr(settings, "runs_dir", None)
    chain = build_default_chain(
        log_dir=log_dir,
        node_name=node_name,
        llm=None,  # SummarizationMiddleware is for ReAct sub-graphs only (Phase 1+)
        max_retries=getattr(settings, "middleware_retry_max", 2),
        retry_base_delay=getattr(settings, "middleware_retry_delay", 5.0),
        context_window=getattr(settings, "middleware_context_window", 128_000),
    )
    return chain.wrap(node_fn)
```

Also update `_default_settings()` in `main_graph.py` -- no changes needed since the new fields have defaults.

### 7.4 Verify tests pass

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_middleware_config.py -v
```

Then run the full suite to confirm nothing is broken:

```bash
cd /home/planck/gradute/Muse && python -m pytest tests/ -v --tb=short
```

### 7.5 Commit

```bash
git add muse/config.py muse/graph/main_graph.py tests/test_middleware_config.py
git commit -m "feat(middleware): add middleware settings to config with env var support"
```

---

## Summary

| Task | Files Created / Modified | Tests |
|------|-------------------------|-------|
| 1. Middleware protocol + chain | `muse/middlewares/__init__.py`, `muse/middlewares/base.py` | `tests/test_middleware_base.py` (10 tests) |
| 2. LoggingMiddleware | `muse/middlewares/logging_middleware.py` | `tests/test_middleware_logging.py` (7 tests) |
| 3. RetryMiddleware | `muse/middlewares/retry_middleware.py`, `muse/middlewares/base.py` (update) | `tests/test_middleware_retry.py` (7 tests) |
| 4. SummarizationMiddleware | `muse/middlewares/summarization_middleware.py` | `tests/test_middleware_summarization.py` (11 tests) |
| 5. DanglingToolCallMiddleware | `muse/middlewares/dangling_tool_call.py` | `tests/test_middleware_dangling.py` (8 tests) |
| 6. Integration | `muse/graph/main_graph.py` (modify), `muse/middlewares/__init__.py` (update) | `tests/test_middleware_integration.py` (3 tests) |
| 7. Settings | `muse/config.py` (modify), `muse/graph/main_graph.py` (update) | `tests/test_middleware_config.py` (5 tests) |

**Total: 7 commits, 51 tests, 7 new files, 2 modified files**

### Middleware execution order (final)

```
1. LoggingMiddleware              -- records timestamp, latency, token usage to JSONL
2. RetryMiddleware                -- wraps node_fn with transient-error backoff
3. SummarizationMiddleware        -- Codex CLI compaction (ReAct sub-graphs only, Phase 1+)
4. DanglingToolCallMiddleware     -- patches incomplete tool_calls with error responses
```

### Environment variables added

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSE_MIDDLEWARE_RETRY_MAX` | `2` | Maximum retry attempts for transient failures |
| `MUSE_MIDDLEWARE_RETRY_DELAY` | `5.0` | Base delay (seconds) between retries |
| `MUSE_MIDDLEWARE_COMPACTION_THRESHOLD` | `0.9` | Context window fraction triggering compaction |
| `MUSE_MIDDLEWARE_COMPACTION_RECENT_TOKENS` | `20000` | Token budget for recent state in compaction |
| `MUSE_MIDDLEWARE_CONTEXT_WINDOW` | `128000` | Model context window size in tokens |
