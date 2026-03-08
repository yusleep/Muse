"""Middleware that converts ask_clarification tool calls into HITL interrupts."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from muse.tools.orchestration import (
    get_clarification_handler,
    set_clarification_handler,
)


_TOOL_NAME = "ask_clarification"


class ClarificationMiddleware:
    """Intercept structured clarification tool calls and fire LangGraph interrupts."""

    async def before_invoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Protocol no-op; runtime behavior is provided by ``wrap_node``."""

        del config
        return state

    async def after_invoke(
        self,
        state: dict[str, Any],
        result: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Protocol no-op; runtime behavior is provided by ``wrap_node``."""

        del state, config
        return result

    def wrap_node(self, node_fn):
        """Install a clarification handler for the duration of a node call."""

        def wrapped(*args, **kwargs):
            previous_handler = get_clarification_handler()

            def handler(
                *,
                question: str,
                clarification_type: str,
                context: str | None = None,
                options: list[dict[str, Any]] | None = None,
            ) -> Any:
                return self.fire_interrupt(
                    {
                        "name": _TOOL_NAME,
                        "id": "ask_clarification",
                        "args": {
                            "question": question,
                            "clarification_type": clarification_type,
                            "context": context,
                            "options": options,
                        },
                    }
                )

            set_clarification_handler(handler)
            try:
                return node_fn(*args, **kwargs)
            finally:
                set_clarification_handler(previous_handler)

        return wrapped

    def should_intercept(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the first matching clarification tool call, if present."""

        for tool_call in tool_calls:
            if tool_call.get("name") == _TOOL_NAME:
                return tool_call
        return None

    def build_interrupt_payload(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Build the structured HITL payload from a tool call."""

        args = tool_call.get("args", {})
        return {
            "question": args.get("question", ""),
            "clarification_type": args.get("clarification_type", "missing_info"),
            "context": args.get("context"),
            "options": args.get("options"),
            "tool_call_id": tool_call.get("id", ""),
            "source": _TOOL_NAME,
        }

    def fire_interrupt(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Trigger a LangGraph interrupt carrying the structured payload."""

        payload = self.build_interrupt_payload(tool_call)
        return interrupt(payload)

    def build_tool_message(self, *, tool_call_id: str, human_response: str) -> dict[str, Any]:
        """Convert a human response into a ToolMessage-compatible payload."""

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": human_response,
        }
