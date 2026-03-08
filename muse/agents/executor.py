"""SubagentExecutor: manages sub-agent lifecycle with bounded concurrency."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable

from muse.agents.result import SubagentResult


class SubagentExecutor:
    """Manage sub-agent execution with a bounded thread pool."""

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
            return sum(1 for entry in self._tasks.values() if entry.status == "running")

    def submit(
        self,
        *,
        agent_fn: Callable[[], SubagentResult],
        timeout: float | None = None,
        task_id: str | None = None,
    ) -> str:
        """Submit a sub-agent task and return the task id."""

        effective_timeout = timeout if timeout is not None else self._default_timeout
        task_id = task_id or f"subagent_{uuid.uuid4().hex[:8]}"
        entry = _TaskEntry(task_id=task_id, status="running", timeout=effective_timeout)
        with self._lock:
            self._tasks[task_id] = entry
        entry.future = self._pool.submit(self._run_task, task_id, agent_fn, effective_timeout)
        return task_id

    def _run_task(
        self,
        task_id: str,
        agent_fn: Callable[[], SubagentResult],
        timeout: float,
    ) -> SubagentResult:
        try:
            result = agent_fn()
        except TimeoutError:
            result = _timed_out_result(timeout)
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].status = "timed_out"
                    self._tasks[task_id].result = result
            return result
        except Exception as exc:  # noqa: BLE001
            result = SubagentResult(
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
                    self._tasks[task_id].result = result
            return result

        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is not None and entry.status == "running":
                entry.status = "completed"
                entry.result = result
        return result

    def get_status(self, task_id: str) -> str:
        """Return task status or ``unknown`` when absent."""

        with self._lock:
            entry = self._tasks.get(task_id)
            return entry.status if entry is not None else "unknown"

    def get_result(self, task_id: str, timeout: float | None = None) -> SubagentResult | None:
        """Wait for a task to complete and return its result."""

        with self._lock:
            entry = self._tasks.get(task_id)
        if entry is None:
            return None
        if entry.future is not None:
            try:
                entry.future.result(timeout=timeout if timeout is not None else self._default_timeout)
            except FutureTimeout:
                timed_out = _timed_out_result(timeout if timeout is not None else entry.timeout)
                with self._lock:
                    current = self._tasks.get(task_id)
                    if current is not None and current.status == "running":
                        current.status = "timed_out"
                        current.result = timed_out
                return timed_out
            except Exception:
                pass
        return entry.result

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool."""

        self._pool.shutdown(wait=wait)


def _timed_out_result(timeout: float) -> SubagentResult:
    return SubagentResult(
        status="timed_out",
        accomplishments=[],
        key_findings=[],
        files_created=[],
        issues=[f"Task timed out after {timeout}s"],
        citations=[],
    )


class _TaskEntry:
    __slots__ = ("task_id", "status", "result", "future", "timeout")

    def __init__(self, task_id: str, status: str, timeout: float) -> None:
        self.task_id = task_id
        self.status = status
        self.result: SubagentResult | None = None
        self.future: Future | None = None
        self.timeout = timeout
