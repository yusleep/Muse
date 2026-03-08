# Phase 0-A: Tool Abstraction Layer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bridge Muse's LLMClient with LangChain's BaseChatModel so ReAct agents can use tool-calling.

**Architecture:** Create MuseChatModel wrapping LLMClient, extend _build_request_payload to support tools parameter, create ToolRegistry for dynamic tool assembly.

**Tech Stack:** LangChain, LangGraph, Python 3.10

**Depends on:** Nothing (foundation)

---

## Task 1: Install LangChain dependencies

**Files:**
- Modify: `/home/planck/gradute/Muse/requirements.txt`

### Step 1.1 — Add langchain-core to requirements.txt

Add `langchain-core>=0.3.0` to `requirements.txt`. We only need the core
abstractions (BaseChatModel, BaseTool, AIMessage). We do NOT need
`langchain-openai` or `langchain-anthropic` since MuseChatModel wraps our own
LLMClient.

```bash
# Edit requirements.txt — append langchain-core
```

After editing, `requirements.txt` should read:

```
langgraph==1.0.10
langgraph-checkpoint-sqlite==3.0.3
typing_extensions==4.15.0
langchain-core>=0.3.0
```

### Step 1.2 — Install and verify

```bash
pip install -r requirements.txt
```

### Step 1.3 — Verify import works

```bash
python3 -c "from langchain_core.language_models.chat_models import BaseChatModel; print('OK')"
python3 -c "from langchain_core.tools import tool; print('OK')"
python3 -c "from langchain_core.messages import AIMessage, HumanMessage, SystemMessage; print('OK')"
```

Expected: all three print `OK`.

---

## Task 2: Create MuseChatModel adapter

**Files:**
- Create: `/home/planck/gradute/Muse/muse/models/__init__.py`
- Create: `/home/planck/gradute/Muse/muse/models/adapter.py`
- Create: `/home/planck/gradute/Muse/tests/test_muse_chat_model.py`

### Step 2.1 — Write failing test

Create `tests/test_muse_chat_model.py`:

```python
"""Tests for MuseChatModel adapter wrapping LLMClient."""

import unittest
from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class _StubHttp:
    """Minimal HttpClient stub for LLMClient."""

    def post_json(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        # Return OpenAI chat completions shape
        return {
            "choices": [
                {"message": {"content": "stub response"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def post_json_sse(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        return self.post_json(url, payload, headers)

    def get_json(self, url: str, headers: dict | None = None) -> dict:
        return {}


class MuseChatModelTests(unittest.TestCase):

    def _make_model(self, route: str = "default"):
        from muse.services.providers import LLMClient
        from muse.models.adapter import MuseChatModel

        llm_client = LLMClient(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="test-model",
            http=_StubHttp(),
        )
        return MuseChatModel(llm_client=llm_client, route=route)

    def test_is_base_chat_model(self):
        from langchain_core.language_models.chat_models import BaseChatModel
        from muse.models.adapter import MuseChatModel

        model = self._make_model()
        self.assertIsInstance(model, BaseChatModel)

    def test_invoke_returns_ai_message(self):
        model = self._make_model()
        result = model.invoke([HumanMessage(content="Hello")])
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "stub response")

    def test_system_message_extracted(self):
        """SystemMessage should be sent as the system parameter to LLMClient."""
        model = self._make_model()
        result = model.invoke([
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hi"),
        ])
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "stub response")

    def test_llm_type_property(self):
        model = self._make_model()
        self.assertEqual(model._llm_type, "muse-chat-model")

    def test_multiple_human_messages_concatenated(self):
        model = self._make_model()
        result = model.invoke([
            HumanMessage(content="First part."),
            HumanMessage(content="Second part."),
        ])
        self.assertIsInstance(result, AIMessage)

    def test_bind_tools_returns_runnable(self):
        from langchain_core.tools import tool

        @tool
        def dummy_tool(query: str) -> str:
            """Search for papers."""
            return "result"

        model = self._make_model()
        bound = model.bind_tools([dummy_tool])
        # bind_tools should return a new runnable, not mutate the original
        self.assertIsNotNone(bound)

    def test_tool_call_response_parsed(self):
        """When LLM returns a tool_calls response, AIMessage should contain tool_calls."""
        import json
        from muse.services.providers import LLMClient
        from muse.models.adapter import MuseChatModel

        class _ToolCallHttp:
            def post_json(self, url, payload, headers=None):
                return {
                    "choices": [{
                        "message": {
                            "content": None,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "academic_search",
                                    "arguments": json.dumps({"query": "graph neural networks"}),
                                },
                            }],
                        }
                    }],
                    "usage": {},
                }

            def post_json_sse(self, url, payload, headers=None):
                return self.post_json(url, payload, headers)

            def get_json(self, url, headers=None):
                return {}

        llm_client = LLMClient(
            api_key="k", base_url="http://localhost/v1",
            model="m", http=_ToolCallHttp(),
        )
        model = MuseChatModel(llm_client=llm_client, route="default")
        result = model.invoke([HumanMessage(content="search for papers")])
        self.assertIsInstance(result, AIMessage)
        self.assertTrue(len(result.tool_calls) > 0)
        self.assertEqual(result.tool_calls[0]["name"], "academic_search")


if __name__ == "__main__":
    unittest.main()
```

Run to verify failure (module does not exist yet):

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_muse_chat_model.py -x 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'muse.models'`

### Step 2.2 — Create muse/models/__init__.py

```python
"""LangChain model adapters for Muse."""
```

### Step 2.3 — Create muse/models/adapter.py

```python
"""MuseChatModel: wraps Muse's LLMClient as a LangChain BaseChatModel."""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from muse.services.providers import LLMClient


class MuseChatModel(BaseChatModel):
    """LangChain BaseChatModel backed by Muse's LLMClient.

    Converts LangChain message lists to the system+user string pair that
    LLMClient._chat_completion expects.  When tools are bound (via
    bind_tools), injects them into the request payload so the underlying
    provider can return tool_calls.
    """

    llm_client: Any  # LLMClient (Any to satisfy pydantic)
    route: str = "default"
    temperature: float = 0.2
    max_tokens: int = 2500

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "muse-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, user = _split_messages(messages)

        result = self.llm_client._chat_completion(
            system=system,
            user=user,
            route=self.route,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=kwargs.get("response_format"),
            **{k: v for k, v in kwargs.items() if k not in ("response_format",)},
        )

        raw = result.get("raw", {})
        ai_message = _parse_response_to_ai_message(result["content"], raw)

        generation = ChatGeneration(
            message=ai_message,
            generation_info={"usage": result.get("usage", {})},
        )
        return ChatResult(generations=[generation])


def _split_messages(messages: list[BaseMessage]) -> tuple[str, str]:
    """Extract system and user strings from a LangChain message list.

    - All SystemMessage contents are joined as the system prompt.
    - All HumanMessage (and other non-system) contents are joined as the user prompt.
    - AIMessage contents are prepended to user with an "Assistant:" prefix so the
      LLM sees prior conversation turns.
    """
    system_parts: list[str] = []
    user_parts: list[str] = []

    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, SystemMessage):
            system_parts.append(content)
        elif isinstance(msg, HumanMessage):
            user_parts.append(content)
        else:
            # AIMessage, ToolMessage, etc. — include in user context
            role_label = getattr(msg, "type", "assistant")
            user_parts.append(f"[{role_label}]: {content}")

    system = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."
    user = "\n\n".join(user_parts) if user_parts else ""
    return system, user


def _parse_response_to_ai_message(content: str, raw: dict[str, Any]) -> AIMessage:
    """Build an AIMessage, extracting tool_calls from the raw provider response."""
    tool_calls = _extract_tool_calls(raw)

    if tool_calls:
        return AIMessage(
            content=content or "",
            tool_calls=tool_calls,
        )
    return AIMessage(content=content)


def _extract_tool_calls(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull tool_calls from the raw API response (OpenAI or Anthropic shape)."""
    calls: list[dict[str, Any]] = []

    # OpenAI chat completions: choices[0].message.tool_calls
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        raw_calls = msg.get("tool_calls", [])
        for tc in raw_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append({
                "name": name,
                "args": args,
                "id": tc.get("id", ""),
                "type": "tool_call",
            })

    # Anthropic: content blocks with type=tool_use
    if not calls:
        content_blocks = raw.get("content", [])
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append({
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                        "id": block.get("id", ""),
                        "type": "tool_call",
                    })

    # OpenAI responses API: output items with type=function_call
    if not calls:
        output_items = raw.get("output", [])
        if isinstance(output_items, list):
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    args_str = item.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    calls.append({
                        "name": item.get("name", ""),
                        "args": args,
                        "id": item.get("call_id", item.get("id", "")),
                        "type": "tool_call",
                    })

    return calls
```

### Step 2.4 — Run tests to verify pass

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_muse_chat_model.py -v 2>&1 | tail -15
```

Expected output:

```
tests/test_muse_chat_model.py::MuseChatModelTests::test_is_base_chat_model PASSED
tests/test_muse_chat_model.py::MuseChatModelTests::test_invoke_returns_ai_message PASSED
tests/test_muse_chat_model.py::MuseChatModelTests::test_system_message_extracted PASSED
tests/test_muse_chat_model.py::MuseChatModelTests::test_llm_type_property PASSED
tests/test_muse_chat_model.py::MuseChatModelTests::test_multiple_human_messages_concatenated PASSED
tests/test_muse_chat_model.py::MuseChatModelTests::test_bind_tools_returns_runnable PASSED
tests/test_muse_chat_model.py::MuseChatModelTests::test_tool_call_response_parsed PASSED
```

### Step 2.5 — Commit

```bash
git add muse/models/__init__.py muse/models/adapter.py tests/test_muse_chat_model.py
git commit -m "feat: add MuseChatModel adapter wrapping LLMClient as BaseChatModel"
```

---

## Task 3: Extend _build_request_payload to support tools parameter

**Files:**
- Modify: `/home/planck/gradute/Muse/muse/services/providers.py`
- Create: `/home/planck/gradute/Muse/tests/test_payload_tools.py`

### Step 3.1 — Write failing test

Create `tests/test_payload_tools.py`:

```python
"""Tests for _build_request_payload with tools parameter."""

import unittest
from muse.services.providers import _build_request_payload, _ModelAttempt


def _make_attempt(api_style: str, endpoint: str = "http://localhost/v1/chat/completions") -> _ModelAttempt:
    return _ModelAttempt(
        route_name="default",
        model_id="test/model",
        provider_name="test",
        endpoint_url=endpoint,
        api_style=api_style,
        model_name="model",
        header_candidates=[{}],
        params={},
        requires_streaming=False,
    )


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "academic_search",
            "description": "Search for academic papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }
]


class PayloadToolsTests(unittest.TestCase):

    def test_openai_chat_completions_includes_tools(self):
        attempt = _make_attempt("chat_completions")
        payload = _build_request_payload(
            attempt=attempt, system="sys", user="usr",
            temperature=0.2, response_format=None, max_tokens=100,
            tools=SAMPLE_TOOLS,
        )
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"], SAMPLE_TOOLS)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_openai_chat_completions_no_tools_key_when_none(self):
        attempt = _make_attempt("chat_completions")
        payload = _build_request_payload(
            attempt=attempt, system="sys", user="usr",
            temperature=0.2, response_format=None, max_tokens=100,
        )
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_anthropic_includes_tools_in_anthropic_format(self):
        attempt = _make_attempt("anthropic", "http://localhost/v1/messages")
        payload = _build_request_payload(
            attempt=attempt, system="sys", user="usr",
            temperature=0.2, response_format=None, max_tokens=100,
            tools=SAMPLE_TOOLS,
        )
        self.assertIn("tools", payload)
        # Anthropic format: list of {name, description, input_schema}
        tool_entry = payload["tools"][0]
        self.assertIn("name", tool_entry)
        self.assertIn("input_schema", tool_entry)
        self.assertEqual(tool_entry["name"], "academic_search")

    def test_responses_includes_tools_as_function_tools(self):
        attempt = _make_attempt("responses", "http://localhost/v1/responses")
        payload = _build_request_payload(
            attempt=attempt, system="sys", user="usr",
            temperature=0.2, response_format=None, max_tokens=100,
            tools=SAMPLE_TOOLS,
        )
        self.assertIn("tools", payload)
        tool_entry = payload["tools"][0]
        self.assertEqual(tool_entry["type"], "function")
        self.assertIn("name", tool_entry)

    def test_codex_streaming_omits_tools(self):
        """Codex backend-api streaming does not support tools; they should be omitted."""
        attempt = _make_attempt(
            "responses",
            "https://chatgpt.com/backend-api/codex/responses",
        )
        attempt = _ModelAttempt(
            route_name="default",
            model_id="codex/o4-mini",
            provider_name="codex",
            endpoint_url="https://chatgpt.com/backend-api/codex/responses",
            api_style="responses",
            model_name="o4-mini",
            header_candidates=[{}],
            params={},
            requires_streaming=True,
        )
        payload = _build_request_payload(
            attempt=attempt, system="sys", user="usr",
            temperature=0.2, response_format=None, max_tokens=100,
            streaming=True,
            tools=SAMPLE_TOOLS,
        )
        # Codex streaming endpoint cannot handle tools — should be excluded
        self.assertNotIn("tools", payload)

    def test_tool_choice_can_be_overridden(self):
        attempt = _make_attempt("chat_completions")
        payload = _build_request_payload(
            attempt=attempt, system="sys", user="usr",
            temperature=0.2, response_format=None, max_tokens=100,
            tools=SAMPLE_TOOLS,
            tool_choice="required",
        )
        self.assertEqual(payload["tool_choice"], "required")


if __name__ == "__main__":
    unittest.main()
```

Run to verify failure:

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_payload_tools.py -x 2>&1 | tail -5
```

Expected: `TypeError: _build_request_payload() got an unexpected keyword argument 'tools'`

### Step 3.2 — Implement: extend _build_request_payload in providers.py

Modify the function signature and all three API style branches in
`/home/planck/gradute/Muse/muse/services/providers.py`.

Change the function signature from:

```python
def _build_request_payload(
    *,
    attempt: _ModelAttempt,
    system: str,
    user: str,
    temperature: float,
    response_format: dict[str, Any] | None,
    max_tokens: int,
    streaming: bool = False,
) -> dict[str, Any]:
```

to:

```python
def _build_request_payload(
    *,
    attempt: _ModelAttempt,
    system: str,
    user: str,
    temperature: float,
    response_format: dict[str, Any] | None,
    max_tokens: int,
    streaming: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
) -> dict[str, Any]:
```

In the **`responses`** branch, before `return payload`, add:

```python
        # Tools — only when NOT streaming through chatgpt.com/backend-api
        if tools and not streaming:
            payload["tools"] = [_to_responses_tool(t) for t in tools]
```

In the **`anthropic`** branch, before `return payload`, add:

```python
        if tools:
            payload["tools"] = [_to_anthropic_tool(t) for t in tools]
```

In the **`chat_completions`** (default) branch, before `return payload`, add:

```python
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
```

Add two helper functions before `_build_request_payload`:

```python
def _to_anthropic_tool(openai_tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-format tool schema to Anthropic tool_use format."""
    fn = openai_tool.get("function", openai_tool)
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _to_responses_tool(openai_tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-format tool schema to Responses API function tool."""
    fn = openai_tool.get("function", openai_tool)
    return {
        "type": "function",
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
    }
```

### Step 3.3 — Run tests

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_payload_tools.py -v 2>&1 | tail -15
```

Expected: all 6 tests pass.

### Step 3.4 — Verify existing tests still pass

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/ -x --ignore=tests/test_e2e.py 2>&1 | tail -10
```

Expected: no regressions.

### Step 3.5 — Commit

```bash
git add muse/services/providers.py tests/test_payload_tools.py
git commit -m "feat: extend _build_request_payload to support tools for all 3 API styles"
```

---

## Task 4: Create model factory

**Files:**
- Create: `/home/planck/gradute/Muse/muse/models/factory.py`
- Create: `/home/planck/gradute/Muse/tests/test_model_factory.py`

### Step 4.1 — Write failing test

Create `tests/test_model_factory.py`:

```python
"""Tests for the model factory."""

import unittest
from muse.config import Settings


def _make_settings() -> Settings:
    return Settings(
        llm_api_key="test-key",
        llm_base_url="http://localhost:11434/v1",
        llm_model="test-model",
        model_router_config={},
        runs_dir="runs",
        semantic_scholar_api_key=None,
        openalex_email=None,
        crossref_mailto=None,
        refs_dir=None,
        checkpoint_dir=None,
    )


class ModelFactoryTests(unittest.TestCase):

    def test_create_chat_model_returns_muse_chat_model(self):
        from muse.models.factory import create_chat_model
        from muse.models.adapter import MuseChatModel

        model = create_chat_model(_make_settings(), route="default")
        self.assertIsInstance(model, MuseChatModel)

    def test_create_chat_model_uses_specified_route(self):
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings(), route="writing")
        self.assertEqual(model.route, "writing")

    def test_create_chat_model_default_route(self):
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings())
        self.assertEqual(model.route, "default")

    def test_create_chat_model_custom_temperature(self):
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings(), route="default", temperature=0.7)
        self.assertAlmostEqual(model.temperature, 0.7)

    def test_create_chat_model_custom_max_tokens(self):
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings(), route="default", max_tokens=4000)
        self.assertEqual(model.max_tokens, 4000)

    def test_create_chat_model_with_router_config(self):
        from muse.models.factory import create_chat_model
        from muse.models.adapter import MuseChatModel

        settings = Settings(
            llm_api_key="key",
            llm_base_url="http://localhost/v1",
            llm_model="router/default",
            model_router_config={
                "providers": {
                    "local": {
                        "baseUrl": "http://localhost:11434/v1",
                        "auth": "default",
                        "models": {
                            "local/llama3": {"model": "llama3", "params": {}},
                        },
                    }
                },
                "models": {
                    "default": {"primary": "local/llama3", "fallbacks": []},
                    "writing": {"primary": "local/llama3", "fallbacks": []},
                },
            },
            runs_dir="runs",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
        )
        model = create_chat_model(settings, route="writing")
        self.assertIsInstance(model, MuseChatModel)
        self.assertEqual(model.route, "writing")


if __name__ == "__main__":
    unittest.main()
```

Run to verify failure:

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_model_factory.py -x 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'muse.models.factory'`

### Step 4.2 — Implement muse/models/factory.py

```python
"""Factory for creating LangChain-compatible chat models from Muse settings."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from muse.config import Settings
from muse.models.adapter import MuseChatModel
from muse.services.http import HttpClient
from muse.services.providers import LLMClient


def create_chat_model(
    settings: Settings,
    *,
    route: str = "default",
    temperature: float = 0.2,
    max_tokens: int = 2500,
    http_timeout: int = 120,
) -> BaseChatModel:
    """Build a MuseChatModel from Muse Settings.

    This is the single entry point for obtaining a LangChain-compatible
    chat model that uses Muse's multi-provider router under the hood.

    Args:
        settings: Muse runtime settings (API keys, model router config, etc.).
        route: Model router route name (e.g. "default", "writing", "reasoning").
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        http_timeout: HTTP request timeout in seconds.

    Returns:
        A BaseChatModel backed by Muse's LLMClient with full router support.
    """
    http = HttpClient(timeout_seconds=http_timeout)
    llm_client = LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        http=http,
        model_router_config=settings.model_router_config,
        env=dict(os.environ),
    )
    return MuseChatModel(
        llm_client=llm_client,
        route=route,
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

### Step 4.3 — Run tests

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_model_factory.py -v 2>&1 | tail -15
```

Expected: all 6 tests pass.

### Step 4.4 — Commit

```bash
git add muse/models/factory.py tests/test_model_factory.py
git commit -m "feat: add model factory create_chat_model(settings, route) -> BaseChatModel"
```

---

## Task 5: Create ToolRegistry

**Files:**
- Create: `/home/planck/gradute/Muse/muse/tools/__init__.py`
- Create: `/home/planck/gradute/Muse/muse/tools/registry.py`
- Create: `/home/planck/gradute/Muse/tests/test_tool_registry.py`

### Step 5.1 — Write failing test

Create `tests/test_tool_registry.py`:

```python
"""Tests for the ToolRegistry."""

import unittest
from langchain_core.tools import tool


@tool
def search_papers(query: str) -> str:
    """Search for academic papers by query."""
    return f"results for {query}"


@tool
def verify_doi(doi: str) -> bool:
    """Verify a DOI exists via CrossRef."""
    return True


@tool
def write_section(outline: str, references: str) -> str:
    """Write a thesis section from outline and references."""
    return "written section"


class ToolRegistryTests(unittest.TestCase):

    def test_register_and_get_by_group(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        tools = registry.get_tools(groups=["research"])
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "search_papers")

    def test_register_multiple_groups(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        registry.register(write_section, group="writing")

        research_tools = registry.get_tools(groups=["research"])
        self.assertEqual(len(research_tools), 1)

        review_tools = registry.get_tools(groups=["review"])
        self.assertEqual(len(review_tools), 1)

    def test_get_tools_multiple_groups(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        registry.register(write_section, group="writing")

        tools = registry.get_tools(groups=["research", "review"])
        names = {t.name for t in tools}
        self.assertEqual(names, {"search_papers", "verify_doi"})

    def test_get_tools_for_profile(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        registry.register(write_section, group="writing")

        registry.define_profile("chapter", groups=["research", "writing"])
        registry.define_profile("citation", groups=["review"])

        chapter_tools = registry.get_tools_for_profile("chapter")
        names = {t.name for t in chapter_tools}
        self.assertEqual(names, {"search_papers", "write_section"})

        citation_tools = registry.get_tools_for_profile("citation")
        self.assertEqual(len(citation_tools), 1)
        self.assertEqual(citation_tools[0].name, "verify_doi")

    def test_unknown_profile_returns_empty(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tools = registry.get_tools_for_profile("nonexistent")
        self.assertEqual(tools, [])

    def test_unknown_group_returns_empty(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tools = registry.get_tools(groups=["nonexistent"])
        self.assertEqual(tools, [])

    def test_list_groups(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        self.assertEqual(sorted(registry.list_groups()), ["research", "review"])

    def test_list_profiles(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.define_profile("chapter", groups=["research", "writing"])
        registry.define_profile("citation", groups=["review"])
        self.assertEqual(sorted(registry.list_profiles()), ["chapter", "citation"])

    def test_no_duplicate_tools_across_groups(self):
        """If same tool registered in two groups, profile returning both should deduplicate."""
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(search_papers, group="extra")
        registry.define_profile("all", groups=["research", "extra"])
        tools = registry.get_tools_for_profile("all")
        names = [t.name for t in tools]
        self.assertEqual(len(names), len(set(names)), "tools should be deduplicated")


if __name__ == "__main__":
    unittest.main()
```

Run to verify failure:

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_tool_registry.py -x 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'muse.tools'`

### Step 5.2 — Create muse/tools/__init__.py

```python
"""LangChain tool wrappers for Muse."""
```

### Step 5.3 — Implement muse/tools/registry.py

```python
"""ToolRegistry: dynamic tool assembly by group and profile."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool


class ToolRegistry:
    """Registry for organizing tools into groups and assembling profiles.

    Groups are collections of related tools (e.g. "research", "review", "writing").
    Profiles map to sub-graph agent roles and specify which groups of tools
    that agent may use.

    Example:
        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        registry.define_profile("chapter", groups=["research", "writing"])
        tools = registry.get_tools_for_profile("chapter")
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._groups: dict[str, list[str]] = {}
        self._profiles: dict[str, list[str]] = {}

    def register(self, tool: BaseTool, *, group: str) -> None:
        """Register a tool under a named group.

        The same tool can be registered under multiple groups.  Tools are
        keyed by their ``name`` attribute for deduplication.
        """
        self._tools[tool.name] = tool
        if group not in self._groups:
            self._groups[group] = []
        if tool.name not in self._groups[group]:
            self._groups[group].append(tool.name)

    def define_profile(self, profile: str, *, groups: list[str]) -> None:
        """Define a tool profile as a list of group names.

        Profiles correspond to sub-graph agents (e.g. "chapter", "citation",
        "composition").  When the agent is created it calls
        ``get_tools_for_profile`` to receive exactly the tools it may use.
        """
        self._profiles[profile] = list(groups)

    def get_tools(self, *, groups: list[str]) -> list[BaseTool]:
        """Return deduplicated list of tools from the specified groups."""
        seen: set[str] = set()
        result: list[BaseTool] = []
        for group in groups:
            for name in self._groups.get(group, []):
                if name not in seen:
                    seen.add(name)
                    tool = self._tools.get(name)
                    if tool is not None:
                        result.append(tool)
        return result

    def get_tools_for_profile(self, profile: str) -> list[BaseTool]:
        """Return deduplicated list of tools for a named profile."""
        groups = self._profiles.get(profile)
        if not groups:
            return []
        return self.get_tools(groups=groups)

    def list_groups(self) -> list[str]:
        """Return sorted list of registered group names."""
        return sorted(self._groups.keys())

    def list_profiles(self) -> list[str]:
        """Return sorted list of defined profile names."""
        return sorted(self._profiles.keys())
```

### Step 5.4 — Run tests

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_tool_registry.py -v 2>&1 | tail -15
```

Expected: all 9 tests pass.

### Step 5.5 — Commit

```bash
git add muse/tools/__init__.py muse/tools/registry.py tests/test_tool_registry.py
git commit -m "feat: add ToolRegistry with group-based tool assembly and profiles"
```

---

## Task 6: Create academic_search_tool

**Files:**
- Create: `/home/planck/gradute/Muse/muse/tools/academic_search.py`
- Create: `/home/planck/gradute/Muse/tests/test_tool_academic_search.py`

### Step 6.1 — Write failing test

Create `tests/test_tool_academic_search.py`:

```python
"""Tests for the academic_search LangChain tool."""

import unittest
from typing import Any


class _FakeSearchClient:
    """Stub matching AcademicSearchClient interface."""

    def __init__(self):
        self.last_query = None
        self.last_discipline = None

    def search_multi_source(
        self, topic: str, discipline: str, extra_queries: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        self.last_query = topic
        self.last_discipline = discipline
        return (
            [
                {
                    "ref_id": "@smith2024graph",
                    "title": "Graph Neural Networks Survey",
                    "authors": ["Alice Smith"],
                    "year": 2024,
                    "doi": "10.1000/gnn",
                    "venue": "NeurIPS",
                    "abstract": "A survey of GNN methods.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                }
            ],
            extra_queries or [topic],
        )


class AcademicSearchToolTests(unittest.TestCase):

    def test_tool_returns_string(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client)
        result = tool.invoke({"query": "graph neural networks"})
        self.assertIsInstance(result, str)
        self.assertIn("Graph Neural Networks Survey", result)

    def test_tool_has_correct_name(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client)
        self.assertEqual(tool.name, "academic_search")

    def test_tool_has_description(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client)
        self.assertTrue(len(tool.description) > 10)

    def test_tool_passes_query_to_client(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client)
        tool.invoke({"query": "transformer architectures"})
        self.assertEqual(client.last_query, "transformer architectures")

    def test_tool_with_discipline(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client, default_discipline="Computer Science")
        tool.invoke({"query": "attention mechanisms"})
        self.assertEqual(client.last_discipline, "Computer Science")

    def test_tool_handles_empty_results(self):
        from muse.tools.academic_search import make_academic_search_tool

        class _EmptySearch:
            def search_multi_source(self, topic, discipline, extra_queries=None):
                return ([], [topic])

        tool = make_academic_search_tool(_EmptySearch())
        result = tool.invoke({"query": "obscure topic"})
        self.assertIsInstance(result, str)
        self.assertIn("No papers found", result)

    def test_tool_is_langchain_base_tool(self):
        from langchain_core.tools import BaseTool
        from muse.tools.academic_search import make_academic_search_tool

        tool = make_academic_search_tool(_FakeSearchClient())
        self.assertIsInstance(tool, BaseTool)


if __name__ == "__main__":
    unittest.main()
```

Run to verify failure:

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_tool_academic_search.py -x 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'muse.tools.academic_search'`

### Step 6.2 — Implement muse/tools/academic_search.py

```python
"""LangChain tool wrapping Muse's AcademicSearchClient."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class AcademicSearchInput(BaseModel):
    """Input schema for academic_search tool."""
    query: str = Field(description="Search query for academic papers")


class AcademicSearchTool(BaseTool):
    """Search academic databases (Semantic Scholar, OpenAlex, arXiv).

    Wraps Muse's existing AcademicSearchClient.search_multi_source so that
    ReAct agents can discover and cite relevant literature.
    """

    name: str = "academic_search"
    description: str = (
        "Search Semantic Scholar, OpenAlex, and arXiv for academic papers. "
        "Input is a search query string. Returns a list of papers with "
        "title, authors, year, DOI, venue, and abstract."
    )
    args_schema: type[BaseModel] = AcademicSearchInput
    search_client: Any = None  # AcademicSearchClient
    default_discipline: str = ""

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        papers, queries = self.search_client.search_multi_source(
            topic=query,
            discipline=self.default_discipline,
        )
        if not papers:
            return "No papers found for the given query."

        lines: list[str] = [f"Found {len(papers)} paper(s):"]
        for i, paper in enumerate(papers[:20], 1):
            authors = ", ".join(paper.get("authors", [])[:3])
            year = paper.get("year", "n/a")
            title = paper.get("title", "Untitled")
            doi = paper.get("doi") or "no DOI"
            venue = paper.get("venue") or "unknown venue"
            abstract = (paper.get("abstract") or "")[:200]
            lines.append(
                f"\n{i}. [{paper.get('ref_id', '')}] {title}\n"
                f"   Authors: {authors}\n"
                f"   Year: {year} | Venue: {venue} | DOI: {doi}\n"
                f"   Abstract: {abstract}"
            )
        return "\n".join(lines)


def make_academic_search_tool(
    search_client: Any,
    default_discipline: str = "",
) -> BaseTool:
    """Factory: create an academic_search tool from an AcademicSearchClient instance."""
    return AcademicSearchTool(
        search_client=search_client,
        default_discipline=default_discipline,
    )
```

### Step 6.3 — Run tests

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_tool_academic_search.py -v 2>&1 | tail -15
```

Expected: all 7 tests pass.

### Step 6.4 — Commit

```bash
git add muse/tools/academic_search.py tests/test_tool_academic_search.py
git commit -m "feat: add academic_search LangChain tool wrapping AcademicSearchClient"
```

---

## Task 7: Create citation tools

**Files:**
- Create: `/home/planck/gradute/Muse/muse/tools/citation.py`
- Create: `/home/planck/gradute/Muse/tests/test_tool_citation.py`

### Step 7.1 — Write failing test

Create `tests/test_tool_citation.py`:

```python
"""Tests for citation verification LangChain tools."""

import unittest
from typing import Any


class _FakeMetadataClient:
    def __init__(self, doi_valid: bool = True, metadata_match: bool = True):
        self._doi_valid = doi_valid
        self._metadata_match = metadata_match

    def verify_doi(self, doi: str) -> bool:
        return self._doi_valid

    def crosscheck_metadata(self, ref: dict[str, Any]) -> bool:
        return self._metadata_match


class VerifyDoiToolTests(unittest.TestCase):

    def test_verify_doi_returns_string(self):
        from muse.tools.citation import make_verify_doi_tool

        tool = make_verify_doi_tool(_FakeMetadataClient(doi_valid=True))
        result = tool.invoke({"doi": "10.1000/test"})
        self.assertIsInstance(result, str)
        self.assertIn("valid", result.lower())

    def test_verify_doi_invalid(self):
        from muse.tools.citation import make_verify_doi_tool

        tool = make_verify_doi_tool(_FakeMetadataClient(doi_valid=False))
        result = tool.invoke({"doi": "10.9999/fake"})
        self.assertIn("invalid", result.lower())

    def test_verify_doi_tool_name(self):
        from muse.tools.citation import make_verify_doi_tool

        tool = make_verify_doi_tool(_FakeMetadataClient())
        self.assertEqual(tool.name, "verify_doi")

    def test_verify_doi_is_base_tool(self):
        from langchain_core.tools import BaseTool
        from muse.tools.citation import make_verify_doi_tool

        tool = make_verify_doi_tool(_FakeMetadataClient())
        self.assertIsInstance(tool, BaseTool)


class CrosscheckMetadataToolTests(unittest.TestCase):

    def test_crosscheck_returns_string(self):
        from muse.tools.citation import make_crosscheck_metadata_tool

        tool = make_crosscheck_metadata_tool(_FakeMetadataClient(metadata_match=True))
        result = tool.invoke({
            "title": "Graph Neural Networks",
            "authors": "Smith, Jones",
            "year": "2024",
            "doi": "10.1000/test",
        })
        self.assertIsInstance(result, str)
        self.assertIn("verified", result.lower())

    def test_crosscheck_mismatch(self):
        from muse.tools.citation import make_crosscheck_metadata_tool

        tool = make_crosscheck_metadata_tool(_FakeMetadataClient(metadata_match=False))
        result = tool.invoke({
            "title": "Fake Paper",
            "authors": "Nobody",
            "year": "2024",
            "doi": "",
        })
        self.assertIn("mismatch", result.lower())

    def test_crosscheck_tool_name(self):
        from muse.tools.citation import make_crosscheck_metadata_tool

        tool = make_crosscheck_metadata_tool(_FakeMetadataClient())
        self.assertEqual(tool.name, "crosscheck_metadata")

    def test_crosscheck_is_base_tool(self):
        from langchain_core.tools import BaseTool
        from muse.tools.citation import make_crosscheck_metadata_tool

        tool = make_crosscheck_metadata_tool(_FakeMetadataClient())
        self.assertIsInstance(tool, BaseTool)


if __name__ == "__main__":
    unittest.main()
```

Run to verify failure:

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_tool_citation.py -x 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'muse.tools.citation'`

### Step 7.2 — Implement muse/tools/citation.py

```python
"""LangChain tools wrapping Muse's CitationMetadataClient methods."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ── verify_doi ──────────────────────────────────────────────────────────

class VerifyDoiInput(BaseModel):
    """Input schema for verify_doi tool."""
    doi: str = Field(description="The DOI string to verify (e.g. '10.1038/nphys1170')")


class VerifyDoiTool(BaseTool):
    """Verify whether a DOI is valid and resolvable via CrossRef.

    Wraps CitationMetadataClient.verify_doi.
    """

    name: str = "verify_doi"
    description: str = (
        "Check whether a DOI is valid and exists in CrossRef. "
        "Input is a DOI string. Returns whether the DOI is valid or invalid."
    )
    args_schema: type[BaseModel] = VerifyDoiInput
    metadata_client: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, doi: str) -> str:
        is_valid = self.metadata_client.verify_doi(doi)
        if is_valid:
            return f"DOI '{doi}' is valid and exists in CrossRef."
        return f"DOI '{doi}' is invalid or not found in CrossRef."


def make_verify_doi_tool(metadata_client: Any) -> BaseTool:
    """Factory: create a verify_doi tool from a CitationMetadataClient."""
    return VerifyDoiTool(metadata_client=metadata_client)


# ── crosscheck_metadata ────────────────────────────────────────────────

class CrosscheckMetadataInput(BaseModel):
    """Input schema for crosscheck_metadata tool."""
    title: str = Field(description="Paper title")
    authors: str = Field(default="", description="Comma-separated author names")
    year: str = Field(default="", description="Publication year")
    doi: str = Field(default="", description="DOI if available")


class CrosscheckMetadataTool(BaseTool):
    """Cross-check a citation's metadata against CrossRef records.

    Wraps CitationMetadataClient.crosscheck_metadata.
    """

    name: str = "crosscheck_metadata"
    description: str = (
        "Verify a citation's metadata (title, authors, year, DOI) against "
        "CrossRef records. Returns whether the metadata is verified or has "
        "a mismatch."
    )
    args_schema: type[BaseModel] = CrosscheckMetadataInput
    metadata_client: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        title: str,
        authors: str = "",
        year: str = "",
        doi: str = "",
    ) -> str:
        ref = {
            "title": title,
            "authors": [a.strip() for a in authors.split(",") if a.strip()],
            "year": int(year) if year.isdigit() else None,
            "doi": doi or None,
        }
        is_match = self.metadata_client.crosscheck_metadata(ref)
        if is_match:
            return f"Citation metadata verified: '{title}' matches CrossRef records."
        return f"Citation metadata mismatch: '{title}' does not match CrossRef records."


def make_crosscheck_metadata_tool(metadata_client: Any) -> BaseTool:
    """Factory: create a crosscheck_metadata tool from a CitationMetadataClient."""
    return CrosscheckMetadataTool(metadata_client=metadata_client)
```

### Step 7.3 — Run tests

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_tool_citation.py -v 2>&1 | tail -15
```

Expected: all 8 tests pass.

### Step 7.4 — Commit

```bash
git add muse/tools/citation.py tests/test_tool_citation.py
git commit -m "feat: add verify_doi and crosscheck_metadata LangChain tools"
```

---

## Task 8: Integration test — MuseChatModel + tools end-to-end

**Files:**
- Create: `/home/planck/gradute/Muse/tests/test_integration_tools.py`

### Step 8.1 — Write and implement integration test

This test verifies the full stack: MuseChatModel receives tools via bind_tools,
the request payload includes them, and a simulated tool_calls response is parsed
back into AIMessage.tool_calls correctly. It also verifies ToolRegistry
assembles the right tools for a profile and they can be bound to the model.

Create `tests/test_integration_tools.py`:

```python
"""Integration tests: MuseChatModel + ToolRegistry + tools end-to-end."""

import json
import unittest
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class _ToolCallingHttp:
    """HTTP stub that captures payloads and returns tool_calls or text responses.

    First call: returns a tool_call for academic_search.
    Second call: returns a text response (simulating post-tool-call completion).
    """

    def __init__(self):
        self.call_count = 0
        self.captured_payloads: list[dict[str, Any]] = []

    def post_json(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        self.call_count += 1
        self.captured_payloads.append(payload)

        if self.call_count == 1:
            # First call: LLM decides to call a tool
            return {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "academic_search",
                                "arguments": json.dumps({"query": "graph neural networks"}),
                            },
                        }],
                    }
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }
        else:
            # Second call: LLM produces final text answer
            return {
                "choices": [{
                    "message": {"content": "Based on the search results, GNNs are..."}
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }

    def post_json_sse(self, url, payload, headers=None):
        return self.post_json(url, payload, headers)

    def get_json(self, url, headers=None):
        return {}


class _FakeSearchClient:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        return (
            [{
                "ref_id": "@smith2024gnn",
                "title": "Graph Neural Networks",
                "authors": ["Alice Smith"],
                "year": 2024,
                "doi": "10.1000/gnn",
                "venue": "NeurIPS",
                "abstract": "A survey.",
                "source": "semantic_scholar",
                "verified_metadata": True,
            }],
            [topic],
        )


class _FakeMetadataClient:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return True


class IntegrationToolsTests(unittest.TestCase):

    def test_registry_assembles_profile_and_binds_to_model(self):
        """Full path: registry -> profile -> bind_tools -> invoke."""
        from muse.models.adapter import MuseChatModel
        from muse.services.providers import LLMClient
        from muse.tools.academic_search import make_academic_search_tool
        from muse.tools.citation import make_verify_doi_tool, make_crosscheck_metadata_tool
        from muse.tools.registry import ToolRegistry

        http = _ToolCallingHttp()
        llm_client = LLMClient(
            api_key="key", base_url="http://localhost/v1",
            model="test", http=http,
        )
        model = MuseChatModel(llm_client=llm_client, route="default")

        # Build registry and profile
        registry = ToolRegistry()
        registry.register(
            make_academic_search_tool(_FakeSearchClient(), default_discipline="CS"),
            group="research",
        )
        registry.register(make_verify_doi_tool(_FakeMetadataClient()), group="review")
        registry.register(make_crosscheck_metadata_tool(_FakeMetadataClient()), group="review")
        registry.define_profile("chapter", groups=["research"])
        registry.define_profile("citation", groups=["review"])

        # Get tools for chapter profile and bind
        chapter_tools = registry.get_tools_for_profile("chapter")
        self.assertEqual(len(chapter_tools), 1)
        self.assertEqual(chapter_tools[0].name, "academic_search")

        bound_model = model.bind_tools(chapter_tools)
        self.assertIsNotNone(bound_model)

        # Invoke: first call should get tool_calls back
        result = bound_model.invoke([
            SystemMessage(content="You are a research assistant."),
            HumanMessage(content="Find papers about graph neural networks"),
        ])
        self.assertIsInstance(result, AIMessage)
        self.assertTrue(len(result.tool_calls) > 0)
        self.assertEqual(result.tool_calls[0]["name"], "academic_search")

    def test_tool_execution_and_follow_up(self):
        """Simulate: model calls tool -> tool executes -> result fed back -> model answers."""
        from muse.models.adapter import MuseChatModel
        from muse.services.providers import LLMClient
        from muse.tools.academic_search import make_academic_search_tool

        http = _ToolCallingHttp()
        llm_client = LLMClient(
            api_key="key", base_url="http://localhost/v1",
            model="test", http=http,
        )
        model = MuseChatModel(llm_client=llm_client, route="default")
        search_tool = make_academic_search_tool(_FakeSearchClient())

        # Step 1: Model produces tool call
        result1 = model.invoke([HumanMessage(content="Search for GNN papers")])
        self.assertTrue(len(result1.tool_calls) > 0)
        tc = result1.tool_calls[0]

        # Step 2: Execute the tool
        tool_output = search_tool.invoke(tc["args"])
        self.assertIn("Graph Neural Networks", tool_output)

        # Step 3: Feed tool result back and get final answer
        result2 = model.invoke([
            HumanMessage(content="Search for GNN papers"),
            result1,
            ToolMessage(content=tool_output, tool_call_id=tc["id"]),
        ])
        self.assertIsInstance(result2, AIMessage)
        self.assertIn("GNNs", result2.content)

    def test_payload_includes_tools_when_bound(self):
        """Verify the HTTP payload actually contains tools when bind_tools is used."""
        from muse.models.adapter import MuseChatModel
        from muse.services.providers import LLMClient
        from muse.tools.academic_search import make_academic_search_tool

        http = _ToolCallingHttp()
        llm_client = LLMClient(
            api_key="key", base_url="http://localhost/v1",
            model="test", http=http,
        )
        model = MuseChatModel(llm_client=llm_client, route="default")
        search_tool = make_academic_search_tool(_FakeSearchClient())

        bound = model.bind_tools([search_tool])
        bound.invoke([HumanMessage(content="Find papers")])

        # Check that the first captured payload had tools
        self.assertTrue(len(http.captured_payloads) > 0)
        payload = http.captured_payloads[0]
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"][0]["function"]["name"], "academic_search")

    def test_citation_profile_tools(self):
        """Verify citation profile gives verify_doi + crosscheck_metadata."""
        from muse.tools.citation import make_verify_doi_tool, make_crosscheck_metadata_tool
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(make_verify_doi_tool(_FakeMetadataClient()), group="review")
        registry.register(make_crosscheck_metadata_tool(_FakeMetadataClient()), group="review")
        registry.define_profile("citation", groups=["review"])

        tools = registry.get_tools_for_profile("citation")
        names = {t.name for t in tools}
        self.assertEqual(names, {"verify_doi", "crosscheck_metadata"})

        # Each tool should be independently invocable
        doi_result = tools[0].invoke({"doi": "10.1000/test"}) if tools[0].name == "verify_doi" else tools[1].invoke({"doi": "10.1000/test"})
        self.assertIsInstance(doi_result, str)


if __name__ == "__main__":
    unittest.main()
```

### Step 8.2 — Run integration tests

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/test_integration_tools.py -v 2>&1 | tail -15
```

Expected: all 4 tests pass.

### Step 8.3 — Run full test suite to verify no regressions

```bash
cd /home/planck/gradute/Muse && python3 -m pytest tests/ --ignore=tests/test_e2e.py -v 2>&1 | tail -20
```

Expected: all tests pass, no regressions.

### Step 8.4 — Commit

```bash
git add tests/test_integration_tools.py
git commit -m "test: add integration tests for MuseChatModel + ToolRegistry + tools"
```

---

## Summary of Files Created/Modified

### New files (10)

| File | Purpose |
|------|---------|
| `muse/models/__init__.py` | Package init |
| `muse/models/adapter.py` | MuseChatModel wrapping LLMClient as BaseChatModel |
| `muse/models/factory.py` | `create_chat_model(settings, route)` factory |
| `muse/tools/__init__.py` | Package init |
| `muse/tools/registry.py` | ToolRegistry with group/profile assembly |
| `muse/tools/academic_search.py` | academic_search tool wrapping AcademicSearchClient |
| `muse/tools/citation.py` | verify_doi + crosscheck_metadata tools |
| `tests/test_muse_chat_model.py` | Tests for MuseChatModel adapter |
| `tests/test_payload_tools.py` | Tests for _build_request_payload tools support |
| `tests/test_model_factory.py` | Tests for model factory |
| `tests/test_tool_registry.py` | Tests for ToolRegistry |
| `tests/test_tool_academic_search.py` | Tests for academic_search tool |
| `tests/test_tool_citation.py` | Tests for citation tools |
| `tests/test_integration_tools.py` | End-to-end integration tests |

### Modified files (2)

| File | Change |
|------|--------|
| `requirements.txt` | Add `langchain-core>=0.3.0` |
| `muse/services/providers.py` | Add `tools` + `tool_choice` params to `_build_request_payload`, add `_to_anthropic_tool` and `_to_responses_tool` helpers |

### What stays unchanged

- `LLMClient`, `_ModelRouter`, `HttpClient`, `post_json_sse` -- all preserved
- `AcademicSearchClient`, `CitationMetadataClient` -- wrapped but not modified
- `main_graph.py`, `state.py`, `runtime.py` -- untouched (Phase 1 changes those)
- All existing tests continue to pass
