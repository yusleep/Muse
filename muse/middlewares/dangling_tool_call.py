"""Repair dangling tool calls in message histories emitted by ReAct nodes."""

from __future__ import annotations

from typing import Any


class DanglingToolCallMiddleware:
    """Detect assistant tool calls that never received tool responses."""

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return state

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        del state, config
        if not isinstance(result, dict):
            return result

        messages = result.get("messages")
        if not isinstance(messages, list):
            return result

        pending: dict[str, str] = {}
        answered: set[str] = set()
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = tool_call.get("id", "")
                    function = tool_call.get("function", {})
                    function_name = (
                        function.get("name", "unknown") if isinstance(function, dict) else "unknown"
                    )
                    if tool_call_id:
                        pending[tool_call_id] = function_name
            elif message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id", "")
                if tool_call_id:
                    answered.add(tool_call_id)

        dangling = {
            tool_call_id: function_name
            for tool_call_id, function_name in pending.items()
            if tool_call_id not in answered
        }
        if not dangling:
            return result

        patched_messages = list(messages)
        for tool_call_id, function_name in dangling.items():
            patched_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": (
                        f"Error: tool call '{function_name}' (id={tool_call_id}) was not "
                        "executed. The node exited before this tool call could be "
                        "processed. Please retry or choose an alternative approach."
                    ),
                }
            )

        patched_result = dict(result)
        patched_result["messages"] = patched_messages
        return patched_result
