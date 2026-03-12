"""LangChain chat model adapter backed by Muse's LLM client."""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from muse.services.providers import LLMClient


class MuseChatModel(BaseChatModel):
    """Wrap Muse's ``LLMClient`` with the LangChain ``BaseChatModel`` contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_client: LLMClient
    route: str = "default"
    temperature: float = 0.2
    max_tokens: int = 2500

    @property
    def _llm_type(self) -> str:
        return "muse-chat-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        system, user = _split_messages(messages)

        response = self.llm_client._chat_completion(
            system=system,
            user=user,
            route=self.route,
            temperature=self.temperature,
            response_format=kwargs.get("response_format"),
            max_tokens=self.max_tokens,
            tools=_coerce_tools(kwargs.get("tools")),
            tool_choice=_coerce_tool_choice(kwargs.get("tool_choice")),
        )

        message = _parse_response_to_ai_message(
            content=response.get("content", ""),
            raw=response.get("raw", {}),
        )
        generation = ChatGeneration(
            message=message,
            generation_info={"usage": response.get("usage", {})},
        )
        return ChatResult(generations=[generation])


def _split_messages(messages: list[BaseMessage]) -> tuple[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []

    for message in messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        if isinstance(message, SystemMessage):
            system_parts.append(content)
            continue
        if isinstance(message, HumanMessage):
            user_parts.append(content)
            continue
        role_label = getattr(message, "type", "assistant")
        user_parts.append(f"[{role_label}]: {content}")

    system = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."
    user = "\n\n".join(user_parts)
    return system, user


def _parse_response_to_ai_message(content: str, raw: dict[str, Any]) -> AIMessage:
    tool_calls = _extract_tool_calls(raw)
    if tool_calls:
        return AIMessage(content=content or "", tool_calls=tool_calls)
    return AIMessage(content=content or "")


def _extract_tool_calls(raw: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        raw_calls = message.get("tool_calls", [])
        if isinstance(raw_calls, list):
            for tool_call in raw_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "{}")
                calls.append(
                    {
                        "name": function.get("name", ""),
                        "args": _parse_tool_arguments(arguments),
                        "id": tool_call.get("id", ""),
                        "type": "tool_call",
                    }
                )

    if calls:
        return calls

    content_blocks = raw.get("content")
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(
                    {
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                        "id": block.get("id", ""),
                        "type": "tool_call",
                    }
                )

    if calls:
        return calls

    output_items = raw.get("output")
    if isinstance(output_items, list):
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "function_call":
                calls.append(
                    {
                        "name": item.get("name", ""),
                        "args": _parse_tool_arguments(item.get("arguments", "{}")),
                        "id": item.get("call_id", item.get("id", "")),
                        "type": "tool_call",
                    }
                )

    return calls


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_tools(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    tools = [item for item in value if isinstance(item, dict)]
    return tools or None


def _coerce_tool_choice(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
