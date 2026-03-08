# Phase 3: Subagent Delegation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable agents to dynamically spawn sub-agents for parallel/independent tasks.

**Architecture:** SubagentExecutor manages lifecycle with ThreadPoolExecutor. spawn_subagent @tool for ReAct agents. SubagentLimitMiddleware truncates excess concurrent spawns.

**Tech Stack:** LangChain, asyncio, ThreadPoolExecutor, Python 3.10

**Depends on:** Phase 1 (ReAct sub-graphs), Phase 2 (HITL)

---

## Task 1: Create SubagentExecutor

**File:** `muse/agents/executor.py` (new file)

**Why:** The SubagentExecutor is the core runtime that manages sub-agent lifecycles.
It wraps `create_react_agent` invocations inside a `ThreadPoolExecutor` so that
parent agents can spawn child agents without blocking the event loop. It tracks
active tasks, enforces the concurrency limit, and collects results into a
standardized `SubagentResult`.

**TDD steps:**

1. Write `tests/test_subagent_executor.py` with these tests:
   - `test_execute_returns_subagent_result` -- execute a stub agent, verify the
     return type has `status`, `accomplishments`, `key_findings`, `files_created`,
     `issues`, `citations` fields.
   - `test_max_concurrent_limit_enforced` -- submit 5 tasks with `max_concurrent=3`,
     verify at most 3 run simultaneously (use a threading barrier to detect).
   - `test_timeout_produces_timed_out_status` -- submit a task with a 0.1s timeout
     against a stub that sleeps 2s, verify `status == "timed_out"`.
   - `test_exception_produces_failed_status` -- submit a task whose stub raises,
     verify `status == "failed"` and `issues` contains the error message.
   - `test_get_status_returns_running_for_active_task` -- submit a slow task, call
     `get_status` before completion, verify it returns `"running"`.
2. Run tests: all 5 fail.
3. Implement `SubagentExecutor`.
4. Run tests: all 5 pass.

**Implementation:**

```python
# muse/agents/executor.py
"""SubagentExecutor: manages sub-agent lifecycle with bounded concurrency."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from muse.agents.result import SubagentResult


class SubagentExecutor:
    """Manages sub-agent execution with bounded concurrency.

    Each sub-agent runs in a thread pool worker. The executor tracks active
    tasks and enforces a max_concurrent limit by blocking new submissions
    until a slot opens.
    """

    def __init__(
        self,
        *,
        agent_factory: Callable[..., Any] | None = None,
        max_concurrent: int = 3,
        default_timeout: float = 300.0,
    ) -> None:
        self._agent_factory = agent_factory
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent)
        self._tasks: dict[str, _TaskEntry] = {}
        self._lock = threading.Lock()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == "running")

    def submit(
        self,
        *,
        agent_fn: Callable[[], SubagentResult],
        timeout: float | None = None,
        task_id: str | None = None,
    ) -> str:
        """Submit a sub-agent task. Returns task_id."""
        task_id = task_id or f"subagent_{uuid.uuid4().hex[:8]}"
        effective_timeout = timeout if timeout is not None else self._default_timeout

        entry = _TaskEntry(task_id=task_id, status="running")
        with self._lock:
            self._tasks[task_id] = entry

        future = self._pool.submit(self._run_task, task_id, agent_fn, effective_timeout)
        entry.future = future
        return task_id

    def _run_task(
        self,
        task_id: str,
        agent_fn: Callable[[], SubagentResult],
        timeout: float,
    ) -> SubagentResult:
        try:
            result = agent_fn()
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].status = "completed"
                    self._tasks[task_id].result = result
            return result
        except TimeoutError:
            failed = SubagentResult(
                status="timed_out",
                accomplishments=[],
                key_findings=[],
                files_created=[],
                issues=[f"Task timed out after {timeout}s"],
                citations=[],
            )
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].status = "timed_out"
                    self._tasks[task_id].result = failed
            return failed
        except Exception as exc:
            failed = SubagentResult(
                status="failed",
                accomplishments=[],
                key_findings=[],
                files_created=[],
                issues=[f"{type(exc).__name__}: {exc}"],
                citations=[],
            )
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].status = "failed"
                    self._tasks[task_id].result = failed
            return failed

    def get_status(self, task_id: str) -> str:
        """Return task status: 'running', 'completed', 'failed', 'timed_out', or 'unknown'."""
        with self._lock:
            entry = self._tasks.get(task_id)
            return entry.status if entry else "unknown"

    def get_result(self, task_id: str, timeout: float | None = None) -> SubagentResult | None:
        """Block until task completes and return result, or None if unknown."""
        with self._lock:
            entry = self._tasks.get(task_id)
        if entry is None:
            return None
        if entry.future is not None:
            try:
                entry.future.result(timeout=timeout or self._default_timeout)
            except Exception:
                pass
        return entry.result

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool."""
        self._pool.shutdown(wait=wait)


class _TaskEntry:
    __slots__ = ("task_id", "status", "result", "future")

    def __init__(self, task_id: str, status: str) -> None:
        self.task_id = task_id
        self.status = status
        self.result: SubagentResult | None = None
        self.future: Future | None = None
```

**Test file:**

```python
# tests/test_subagent_executor.py
"""Tests for SubagentExecutor."""

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
        barrier = threading.Barrier(4, timeout=2)  # 4 = 3 workers + would-be 4th
        concurrent_peak = []
        counter_lock = threading.Lock()
        active = [0]

        def slow_task():
            with counter_lock:
                active[0] += 1
                concurrent_peak.append(active[0])
            time.sleep(0.3)
            with counter_lock:
                active[0] -= 1
            return SubagentResult(status="completed", accomplishments=[],
                                  key_findings=[], files_created=[], issues=[], citations=[])

        ids = [executor.submit(agent_fn=slow_task) for _ in range(5)]
        for tid in ids:
            executor.get_result(tid, timeout=10)
        executor.shutdown()

        # At most 3 should have been active at once (thread pool size)
        self.assertLessEqual(max(concurrent_peak), 3)

    def test_exception_produces_failed_status(self):
        executor = self._make_executor(max_concurrent=2)

        def failing():
            raise ValueError("bad input")

        task_id = executor.submit(agent_fn=failing)
        result = executor.get_result(task_id, timeout=5)
        executor.shutdown()

        self.assertIsNotNone(result)
        self.assertEqual(result.status, "failed")
        self.assertTrue(any("bad input" in i for i in result.issues))

    def test_get_status_returns_running_for_active_task(self):
        executor = self._make_executor(max_concurrent=1)
        started = threading.Event()

        def slow():
            started.set()
            time.sleep(2)
            return SubagentResult(status="completed", accomplishments=[],
                                  key_findings=[], files_created=[], issues=[], citations=[])

        task_id = executor.submit(agent_fn=slow)
        started.wait(timeout=3)
        status = executor.get_status(task_id)
        executor.shutdown(wait=False)

        self.assertEqual(status, "running")

    def test_unknown_task_status(self):
        executor = self._make_executor(max_concurrent=1)
        self.assertEqual(executor.get_status("nonexistent"), "unknown")
        executor.shutdown()


if __name__ == "__main__":
    unittest.main()
```

---

## Task 2: Create SubagentResult protocol

**File:** `muse/agents/result.py` (new file)

**Why:** A standardized data structure for sub-agent outputs. Every sub-agent --
regardless of type (research, writing, bash) -- returns a `SubagentResult` so the
parent agent can uniformly process outputs. This decouples the parent from the
internal details of each sub-agent type.

**TDD steps:**

1. Write `tests/test_subagent_result.py` with these tests:
   - `test_result_has_all_fields` -- construct a `SubagentResult`, verify all 6
     fields are accessible.
   - `test_result_defaults` -- verify default factory values (empty lists).
   - `test_result_to_dict` -- verify `to_dict()` returns a JSON-serializable dict.
   - `test_result_from_dict` -- verify `from_dict()` round-trips.
   - `test_result_summary_string` -- verify `summary()` returns a human-readable
     string mentioning status and accomplishment count.
2. Run tests: all 5 fail.
3. Implement `SubagentResult`.
4. Run tests: all 5 pass.

**Implementation:**

```python
# muse/agents/result.py
"""Standardized result type for sub-agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SubagentResult:
    """Uniform output from any sub-agent execution.

    Fields:
        status: Terminal state of the sub-agent.
        accomplishments: What the sub-agent achieved (human-readable).
        key_findings: Important discoveries or conclusions.
        files_created: Paths to files written by the sub-agent.
        issues: Problems encountered (errors, warnings).
        citations: Citation dicts produced or verified by the sub-agent.
    """

    status: Literal["completed", "failed", "timed_out"] = "completed"
    accomplishments: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "status": self.status,
            "accomplishments": list(self.accomplishments),
            "key_findings": list(self.key_findings),
            "files_created": list(self.files_created),
            "issues": list(self.issues),
            "citations": list(self.citations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubagentResult:
        """Deserialize from a dict."""
        return cls(
            status=data.get("status", "completed"),
            accomplishments=list(data.get("accomplishments", [])),
            key_findings=list(data.get("key_findings", [])),
            files_created=list(data.get("files_created", [])),
            issues=list(data.get("issues", [])),
            citations=list(data.get("citations", [])),
        )

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [f"status={self.status}"]
        if self.accomplishments:
            parts.append(f"{len(self.accomplishments)} accomplishment(s)")
        if self.key_findings:
            parts.append(f"{len(self.key_findings)} finding(s)")
        if self.files_created:
            parts.append(f"{len(self.files_created)} file(s)")
        if self.issues:
            parts.append(f"{len(self.issues)} issue(s)")
        if self.citations:
            parts.append(f"{len(self.citations)} citation(s)")
        return "SubagentResult(" + ", ".join(parts) + ")"
```

**`muse/agents/__init__.py`:**

```python
# muse/agents/__init__.py
"""Sub-agent execution framework."""
```

**Test file:**

```python
# tests/test_subagent_result.py
"""Tests for SubagentResult data structure."""

import json
import unittest

from muse.agents.result import SubagentResult


class SubagentResultTests(unittest.TestCase):
    def test_result_has_all_fields(self):
        r = SubagentResult(
            status="completed",
            accomplishments=["wrote intro"],
            key_findings=["key1"],
            files_created=["ch1.md"],
            issues=[],
            citations=[{"ref_id": "@x"}],
        )
        self.assertEqual(r.status, "completed")
        self.assertEqual(r.accomplishments, ["wrote intro"])
        self.assertEqual(r.key_findings, ["key1"])
        self.assertEqual(r.files_created, ["ch1.md"])
        self.assertEqual(r.issues, [])
        self.assertEqual(r.citations, [{"ref_id": "@x"}])

    def test_result_defaults(self):
        r = SubagentResult()
        self.assertEqual(r.status, "completed")
        self.assertEqual(r.accomplishments, [])
        self.assertEqual(r.key_findings, [])
        self.assertEqual(r.files_created, [])
        self.assertEqual(r.issues, [])
        self.assertEqual(r.citations, [])

    def test_result_to_dict(self):
        r = SubagentResult(status="failed", issues=["timeout"])
        d = r.to_dict()
        # Must be JSON-serializable
        json.dumps(d)
        self.assertEqual(d["status"], "failed")
        self.assertEqual(d["issues"], ["timeout"])

    def test_result_from_dict(self):
        original = SubagentResult(
            status="completed",
            accomplishments=["a"],
            key_findings=["f"],
            files_created=["p"],
            issues=["i"],
            citations=[{"ref_id": "@y"}],
        )
        d = original.to_dict()
        restored = SubagentResult.from_dict(d)
        self.assertEqual(restored.status, original.status)
        self.assertEqual(restored.accomplishments, original.accomplishments)
        self.assertEqual(restored.citations, original.citations)

    def test_result_summary_string(self):
        r = SubagentResult(
            status="completed",
            accomplishments=["a", "b"],
            key_findings=["f"],
            files_created=[],
            issues=["w"],
            citations=[],
        )
        s = r.summary()
        self.assertIn("completed", s)
        self.assertIn("2 accomplishment(s)", s)
        self.assertIn("1 finding(s)", s)
        self.assertIn("1 issue(s)", s)
        self.assertNotIn("file(s)", s)  # empty, should be omitted


if __name__ == "__main__":
    unittest.main()
```

---

## Task 3: Create `spawn_subagent` tool

**File:** `muse/tools/orchestration.py` (extend -- add to file from Phase 2 Task 1)

**Why:** ReAct agents need a tool to delegate subtasks to specialized sub-agents.
The `spawn_subagent` tool accepts a message, agent type, and wait flag. When
`wait=True` (default), the tool blocks until the sub-agent completes and returns
the result summary. The tool never calls `spawn_subagent` itself (no nesting).

**TDD steps:**

1. Write `tests/test_spawn_subagent_tool.py` with these tests:
   - `test_tool_schema_has_required_fields` -- verify `message`, `agent_type` are
     required; `wait` is optional with default `True`.
   - `test_agent_type_enum` -- only `research`, `writing`, `bash` allowed.
   - `test_tool_invoke_with_executor` -- inject a mock executor, invoke the tool,
     verify the executor's `submit` was called with the correct agent type.
   - `test_tool_returns_result_summary` -- verify the tool returns the
     `SubagentResult.summary()` string when `wait=True`.
   - `test_tool_returns_task_id_when_no_wait` -- verify the tool returns a task_id
     string when `wait=False`.
2. Run tests: all 5 fail.
3. Implement the tool.
4. Run tests: all 5 pass.

**Implementation (appended to `muse/tools/orchestration.py`):**

```python
# --- Append to muse/tools/orchestration.py ---

from muse.agents.result import SubagentResult  # noqa: E402 (top of file in practice)


class SpawnSubagentInput(BaseModel):
    """Input schema for spawn_subagent tool."""

    message: str = Field(description="Task description for the sub-agent")
    agent_type: Literal["research", "writing", "bash"] = Field(
        description="Type of sub-agent to spawn"
    )
    wait: bool = Field(
        default=True,
        description="If True, block until sub-agent completes. If False, return task_id immediately.",
    )


# Module-level reference; set by the runtime before tool is used.
_subagent_executor: Any = None


def set_subagent_executor(executor: Any) -> None:
    """Called by runtime to inject the SubagentExecutor instance."""
    global _subagent_executor
    _subagent_executor = executor


def get_subagent_executor() -> Any:
    """Return the currently configured executor (or None)."""
    return _subagent_executor


@tool(args_schema=SpawnSubagentInput)
def spawn_subagent(
    message: str,
    agent_type: str,
    wait: bool = True,
) -> str:
    """Spawn a sub-agent to handle an independent subtask.

    Use this to delegate work like deep literature search, independent
    section writing, or command execution. Sub-agents cannot spawn
    further sub-agents or interact with the human directly.
    """
    executor = get_subagent_executor()
    if executor is None:
        return (
            "[SUBAGENT ERROR] No SubagentExecutor configured. "
            "Cannot spawn sub-agent."
        )

    builtin_registry = _get_builtin_registry()
    agent_fn = builtin_registry.get(agent_type)
    if agent_fn is None:
        return f"[SUBAGENT ERROR] Unknown agent type: {agent_type}"

    task_fn = agent_fn(message)
    task_id = executor.submit(agent_fn=task_fn)

    if not wait:
        return f"Sub-agent spawned: task_id={task_id}, type={agent_type}"

    result = executor.get_result(task_id)
    if result is None:
        return f"[SUBAGENT ERROR] No result for task {task_id}"
    return result.summary()


def _get_builtin_registry() -> dict[str, Any]:
    """Lazy import to avoid circular dependencies."""
    try:
        from muse.agents.builtins import BUILTIN_AGENT_FACTORIES
        return BUILTIN_AGENT_FACTORIES
    except ImportError:
        return {}
```

**Test file:**

```python
# tests/test_spawn_subagent_tool.py
"""Tests for spawn_subagent tool."""

import unittest
from unittest.mock import MagicMock, patch

from muse.agents.result import SubagentResult


class SpawnSubagentToolTests(unittest.TestCase):
    def test_tool_schema_has_required_fields(self):
        from muse.tools.orchestration import spawn_subagent

        schema = spawn_subagent.args_json_schema
        props = schema.get("properties", {})
        self.assertIn("message", props)
        self.assertIn("agent_type", props)
        self.assertIn("wait", props)
        required = schema.get("required", [])
        self.assertIn("message", required)
        self.assertIn("agent_type", required)

    def test_agent_type_enum(self):
        from muse.tools.orchestration import spawn_subagent

        schema = spawn_subagent.args_json_schema
        at = schema["properties"]["agent_type"]
        allowed = set(at.get("enum", []))
        self.assertEqual(allowed, {"research", "writing", "bash"})

    def test_tool_invoke_with_executor(self):
        from muse.tools.orchestration import set_subagent_executor, spawn_subagent

        mock_executor = MagicMock()
        mock_executor.submit.return_value = "task_123"
        mock_executor.get_result.return_value = SubagentResult(
            status="completed", accomplishments=["done"]
        )
        set_subagent_executor(mock_executor)

        # Patch builtins to provide a simple factory
        def dummy_factory(msg):
            def run():
                return SubagentResult(status="completed", accomplishments=["done"])
            return run

        with patch("muse.tools.orchestration._get_builtin_registry",
                    return_value={"research": dummy_factory}):
            result = spawn_subagent.invoke(
                {"message": "find papers on LLMs", "agent_type": "research", "wait": True}
            )

        mock_executor.submit.assert_called_once()
        self.assertIn("completed", result)

        # Clean up
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

        def dummy_factory(msg):
            return lambda: expected

        with patch("muse.tools.orchestration._get_builtin_registry",
                    return_value={"research": dummy_factory}):
            result = spawn_subagent.invoke(
                {"message": "search", "agent_type": "research", "wait": True}
            )

        self.assertIn("completed", result)
        self.assertIn("1 accomplishment(s)", result)
        set_subagent_executor(None)

    def test_tool_returns_task_id_when_no_wait(self):
        from muse.tools.orchestration import set_subagent_executor, spawn_subagent

        mock_executor = MagicMock()
        mock_executor.submit.return_value = "task_789"
        set_subagent_executor(mock_executor)

        def dummy_factory(msg):
            return lambda: SubagentResult(status="completed")

        with patch("muse.tools.orchestration._get_builtin_registry",
                    return_value={"writing": dummy_factory}):
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
```

---

## Task 4: Create SubagentLimitMiddleware

**File:** `muse/middlewares/subagent_limit_middleware.py` (new file)

**Why:** LLMs sometimes emit more `spawn_subagent` tool calls in a single response
than the concurrency limit allows. Rather than relying on prompt instructions (which
are unreliable, per DeerFlow experience), this middleware hard-truncates excess tool
calls before they reach the executor.

**TDD steps:**

1. Write `tests/test_subagent_limit_middleware.py` with these tests:
   - `test_truncates_excess_spawn_calls` -- 5 spawn tool calls with limit=3 yields 3.
   - `test_preserves_non_spawn_tool_calls` -- other tool calls are never dropped.
   - `test_mixed_calls_truncate_only_spawn` -- 2 write_section + 4 spawn with limit=2
     yields 2 write_section + 2 spawn.
   - `test_under_limit_passes_all` -- 2 spawn with limit=3 yields 2 (no truncation).
   - `test_zero_spawn_passes_through` -- no spawn calls, nothing changes.
2. Run tests: all 5 fail.
3. Implement the middleware.
4. Run tests: all 5 pass.

**Implementation:**

```python
# muse/middlewares/subagent_limit_middleware.py
"""Middleware that truncates excess spawn_subagent tool calls."""

from __future__ import annotations

from typing import Any


_SPAWN_TOOL_NAME = "spawn_subagent"


class SubagentLimitMiddleware:
    """Hard-truncate spawn_subagent tool calls that exceed the concurrency limit.

    Position in middleware chain: BEFORE ClarificationMiddleware, AFTER
    SummarizationMiddleware. This ensures spawn limits are enforced before
    any tool call is actually executed.
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        self._max_concurrent = max_concurrent

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def filter_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return tool_calls with excess spawn_subagent calls removed.

        Non-spawn tool calls are always preserved. Spawn tool calls beyond
        max_concurrent are silently dropped.
        """
        result: list[dict[str, Any]] = []
        spawn_count = 0

        for tc in tool_calls:
            if tc.get("name") == _SPAWN_TOOL_NAME:
                if spawn_count < self._max_concurrent:
                    result.append(tc)
                    spawn_count += 1
                # else: silently drop
            else:
                result.append(tc)

        return result

    def dropped_count(self, tool_calls: list[dict[str, Any]]) -> int:
        """Return the number of spawn_subagent calls that would be dropped."""
        spawn_total = sum(1 for tc in tool_calls if tc.get("name") == _SPAWN_TOOL_NAME)
        return max(0, spawn_total - self._max_concurrent)
```

**Test file:**

```python
# tests/test_subagent_limit_middleware.py
"""Tests for SubagentLimitMiddleware."""

import unittest


class SubagentLimitMiddlewareTests(unittest.TestCase):
    def _make_middleware(self, max_concurrent=3):
        from muse.middlewares.subagent_limit_middleware import SubagentLimitMiddleware
        return SubagentLimitMiddleware(max_concurrent=max_concurrent)

    def _spawn_call(self, idx):
        return {"name": "spawn_subagent", "id": f"tc_{idx}", "args": {"message": f"task {idx}", "agent_type": "research"}}

    def _other_call(self, name, idx):
        return {"name": name, "id": f"tc_other_{idx}", "args": {}}

    def test_truncates_excess_spawn_calls(self):
        mw = self._make_middleware(max_concurrent=3)
        calls = [self._spawn_call(i) for i in range(5)]
        filtered = mw.filter_tool_calls(calls)
        spawn_filtered = [c for c in filtered if c["name"] == "spawn_subagent"]
        self.assertEqual(len(spawn_filtered), 3)

    def test_preserves_non_spawn_tool_calls(self):
        mw = self._make_middleware(max_concurrent=1)
        calls = [
            self._other_call("write_section", 0),
            self._other_call("self_review", 1),
            self._spawn_call(0),
            self._spawn_call(1),
        ]
        filtered = mw.filter_tool_calls(calls)
        non_spawn = [c for c in filtered if c["name"] != "spawn_subagent"]
        self.assertEqual(len(non_spawn), 2)

    def test_mixed_calls_truncate_only_spawn(self):
        mw = self._make_middleware(max_concurrent=2)
        calls = [
            self._other_call("write_section", 0),
            self._spawn_call(0),
            self._other_call("write_section", 1),
            self._spawn_call(1),
            self._spawn_call(2),
            self._spawn_call(3),
        ]
        filtered = mw.filter_tool_calls(calls)
        spawn_filtered = [c for c in filtered if c["name"] == "spawn_subagent"]
        other_filtered = [c for c in filtered if c["name"] != "spawn_subagent"]
        self.assertEqual(len(spawn_filtered), 2)
        self.assertEqual(len(other_filtered), 2)

    def test_under_limit_passes_all(self):
        mw = self._make_middleware(max_concurrent=3)
        calls = [self._spawn_call(0), self._spawn_call(1)]
        filtered = mw.filter_tool_calls(calls)
        self.assertEqual(len(filtered), 2)

    def test_zero_spawn_passes_through(self):
        mw = self._make_middleware(max_concurrent=3)
        calls = [self._other_call("read_file", 0), self._other_call("edit_file", 1)]
        filtered = mw.filter_tool_calls(calls)
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
```

---

## Task 5: Create built-in agent configs

**File:** `muse/agents/builtins.py` (new file)

**Why:** The three built-in sub-agent types (research, writing, bash) need concrete
factory functions that wire up the correct tool sets and system prompts. These
factories are referenced by `spawn_subagent` via the `BUILTIN_AGENT_FACTORIES`
registry dict.

**TDD steps:**

1. Write `tests/test_builtin_agents.py` with these tests:
   - `test_builtin_registry_has_three_types` -- `BUILTIN_AGENT_FACTORIES` has keys
     `research`, `writing`, `bash`.
   - `test_research_factory_returns_callable` -- calling `BUILTIN_AGENT_FACTORIES["research"]("find papers")`
     returns a callable.
   - `test_writing_factory_returns_callable` -- same for `writing`.
   - `test_bash_factory_returns_callable` -- same for `bash`.
   - `test_factory_callable_returns_subagent_result` -- execute the callable returned
     by the research factory (with a stub LLM), verify it returns `SubagentResult`.
2. Run tests: all 5 fail.
3. Implement the builtins.
4. Run tests: all 5 pass.

**Implementation:**

```python
# muse/agents/builtins.py
"""Built-in sub-agent type configurations and factories."""

from __future__ import annotations

from typing import Any, Callable

from muse.agents.result import SubagentResult


# ----- System prompts -----

_RESEARCH_SYSTEM = """\
You are a research sub-agent. Your task is to find and analyze academic literature
relevant to the given query. Use available search tools to find papers, read PDFs,
and synthesize findings. Return key findings and citation information.

Constraints:
- You CANNOT spawn further sub-agents.
- You CANNOT ask the human for clarification.
- Summarize findings concisely when done.
"""

_WRITING_SYSTEM = """\
You are a writing sub-agent. Your task is to draft, revise, or edit academic text
for a thesis section. Use available writing and review tools. Follow the style
guidelines provided in the task message.

Constraints:
- You CANNOT spawn further sub-agents.
- You CANNOT ask the human for clarification.
- Submit your output via the submit_result tool when done.
"""

_BASH_SYSTEM = """\
You are a command execution sub-agent. Your task is to run shell commands,
compile LaTeX, execute Python scripts, or perform file operations as described
in the task message.

Constraints:
- You CANNOT spawn further sub-agents.
- You CANNOT ask the human for clarification.
- Report results and any files created.
"""

# ----- Tool group names per agent type -----

AGENT_TOOL_PROFILES: dict[str, list[str]] = {
    "research": ["research", "file_read", "rag"],
    "writing": ["writing", "review_self", "file"],
    "bash": ["sandbox", "file"],
}

AGENT_MAX_TURNS: dict[str, int] = {
    "research": 15,
    "writing": 25,
    "bash": 15,
}

AGENT_SYSTEM_PROMPTS: dict[str, str] = {
    "research": _RESEARCH_SYSTEM,
    "writing": _WRITING_SYSTEM,
    "bash": _BASH_SYSTEM,
}

# Blocked tools for ALL sub-agent types (enforced at registration time).
BLOCKED_TOOLS: set[str] = {"spawn_subagent", "ask_clarification"}


def _make_stub_agent_fn(agent_type: str, message: str) -> Callable[[], SubagentResult]:
    """Create a stub agent function for use before Phase 1 ReAct agents exist.

    This is a placeholder that returns a minimal result. Once Phase 1 lands
    create_react_agent-based execution, this will be replaced with real
    agent invocation.
    """

    def run() -> SubagentResult:
        return SubagentResult(
            status="completed",
            accomplishments=[f"[stub] {agent_type} agent processed: {message[:100]}"],
            key_findings=[],
            files_created=[],
            issues=["Running in stub mode -- Phase 1 ReAct agents not yet implemented"],
            citations=[],
        )

    return run


def build_research_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for research sub-agent tasks."""
    return _make_stub_agent_fn("research", message)


def build_writing_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for writing sub-agent tasks."""
    return _make_stub_agent_fn("writing", message)


def build_bash_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for bash sub-agent tasks."""
    return _make_stub_agent_fn("bash", message)


# Registry used by spawn_subagent tool.
BUILTIN_AGENT_FACTORIES: dict[str, Callable[[str], Callable[[], SubagentResult]]] = {
    "research": build_research_agent,
    "writing": build_writing_agent,
    "bash": build_bash_agent,
}
```

**Test file:**

```python
# tests/test_builtin_agents.py
"""Tests for built-in sub-agent configurations."""

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
        self.assertTrue(any("research" in a for a in result.accomplishments))

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
```

---

## Task 6: Integration test -- Chapter Agent spawns research sub-agent

**File:** `tests/test_subagent_integration.py` (new file)

**Why:** End-to-end verification that the spawn mechanism works within the graph
context. A chapter-level agent (simulated) calls `spawn_subagent` to delegate
a literature search, the SubagentExecutor runs the sub-agent, the result is
returned to the parent agent, and the SubagentLimitMiddleware correctly caps
concurrent spawns.

**TDD steps:**

1. Write `tests/test_subagent_integration.py` with these tests:
   - `test_spawn_and_collect_result` -- a parent function calls `spawn_subagent`,
     gets back a `SubagentResult`, verifies it has `status="completed"`.
   - `test_limit_middleware_caps_spawns` -- emit 5 spawn tool calls, verify the
     middleware reduces them to 3.
   - `test_executor_runs_agent_to_completion` -- submit a stub agent via the
     executor, verify status transitions from `running` to `completed`.
   - `test_full_pipeline_spawn_collect_limit` -- combine executor + middleware +
     tool: set up executor with limit=2, emit 4 spawn tool calls, verify only 2
     run, and results are collected.
2. Run tests: all 4 fail.
3. Wire integration (connect executor, middleware, and tool).
4. Run tests: all 4 pass.

**Test file:**

```python
# tests/test_subagent_integration.py
"""Integration tests for sub-agent delegation pipeline."""

import time
import threading
import unittest
from unittest.mock import patch

from muse.agents.executor import SubagentExecutor
from muse.agents.result import SubagentResult
from muse.middlewares.subagent_limit_middleware import SubagentLimitMiddleware
from muse.tools.orchestration import set_subagent_executor, spawn_subagent


class SubagentIntegrationTests(unittest.TestCase):
    def setUp(self):
        """Clean up executor state before each test."""
        set_subagent_executor(None)

    def tearDown(self):
        set_subagent_executor(None)

    def test_spawn_and_collect_result(self):
        executor = SubagentExecutor(max_concurrent=2)
        set_subagent_executor(executor)

        def research_factory(msg):
            def run():
                return SubagentResult(
                    status="completed",
                    accomplishments=[f"researched: {msg}"],
                    key_findings=["finding1"],
                    files_created=[],
                    issues=[],
                    citations=[{"ref_id": "@test"}],
                )
            return run

        with patch("muse.tools.orchestration._get_builtin_registry",
                    return_value={"research": research_factory}):
            result_str = spawn_subagent.invoke({
                "message": "find papers on LLMs",
                "agent_type": "research",
                "wait": True,
            })

        self.assertIn("completed", result_str)
        self.assertIn("1 accomplishment(s)", result_str)
        executor.shutdown()

    def test_limit_middleware_caps_spawns(self):
        mw = SubagentLimitMiddleware(max_concurrent=3)
        calls = [
            {"name": "spawn_subagent", "id": f"tc_{i}", "args": {"message": f"task{i}", "agent_type": "research"}}
            for i in range(5)
        ]
        filtered = mw.filter_tool_calls(calls)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(mw.dropped_count(calls), 2)

    def test_executor_runs_agent_to_completion(self):
        executor = SubagentExecutor(max_concurrent=2)
        started = threading.Event()

        def slow_agent():
            started.set()
            time.sleep(0.2)
            return SubagentResult(status="completed", accomplishments=["done"])

        task_id = executor.submit(agent_fn=slow_agent)
        started.wait(timeout=3)

        # While running
        self.assertEqual(executor.get_status(task_id), "running")

        # After completion
        result = executor.get_result(task_id, timeout=5)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "completed")
        self.assertEqual(executor.get_status(task_id), "completed")
        executor.shutdown()

    def test_full_pipeline_spawn_collect_limit(self):
        executor = SubagentExecutor(max_concurrent=2)
        mw = SubagentLimitMiddleware(max_concurrent=2)
        set_subagent_executor(executor)

        collected_results = []

        def factory(msg):
            def run():
                return SubagentResult(status="completed", accomplishments=[msg])
            return run

        # Simulate 4 spawn tool calls, middleware should cap to 2
        raw_calls = [
            {"name": "spawn_subagent", "id": f"tc_{i}",
             "args": {"message": f"task_{i}", "agent_type": "research", "wait": True}}
            for i in range(4)
        ]
        filtered_calls = mw.filter_tool_calls(raw_calls)
        self.assertEqual(len(filtered_calls), 2)

        # Execute the filtered calls via the tool
        with patch("muse.tools.orchestration._get_builtin_registry",
                    return_value={"research": factory}):
            for call in filtered_calls:
                result_str = spawn_subagent.invoke(call["args"])
                collected_results.append(result_str)

        self.assertEqual(len(collected_results), 2)
        for r in collected_results:
            self.assertIn("completed", r)

        executor.shutdown()


if __name__ == "__main__":
    unittest.main()
```

---

## File inventory

| Action | Path |
|--------|------|
| Create | `muse/agents/__init__.py` |
| Create | `muse/agents/result.py` |
| Create | `muse/agents/executor.py` |
| Create | `muse/agents/builtins.py` |
| Extend | `muse/tools/orchestration.py` (add spawn_subagent; from Phase 2) |
| Create | `muse/middlewares/subagent_limit_middleware.py` |
| Create | `tests/test_subagent_result.py` |
| Create | `tests/test_subagent_executor.py` |
| Create | `tests/test_spawn_subagent_tool.py` |
| Create | `tests/test_subagent_limit_middleware.py` |
| Create | `tests/test_builtin_agents.py` |
| Create | `tests/test_subagent_integration.py` |

## Verification

After all 6 tasks are complete, run:

```bash
python -m pytest tests/test_subagent_result.py tests/test_subagent_executor.py tests/test_spawn_subagent_tool.py tests/test_subagent_limit_middleware.py tests/test_builtin_agents.py tests/test_subagent_integration.py -v
```

All tests must pass. Also verify no regressions:

```bash
python -m pytest tests/test_hitl_interrupt.py tests/test_structured_hitl_integration.py -v
```
