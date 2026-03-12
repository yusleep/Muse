"""HTTP transport primitives shared by provider clients."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, timeout_seconds: int = 300) -> None:
        self.timeout_seconds = timeout_seconds

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        return self._json_request(req)

    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=merged_headers,
            method="POST",
        )
        return self._json_request(req)

    def post_json_sse(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=merged_headers,
            method="POST",
        )
        text_chunks: list[str] = []
        usage: dict[str, Any] = {}
        saw_chat_chunks = False
        finish_reason: str | None = None
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type", "")

                    # Responses API format (Codex)
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if isinstance(delta, str):
                            text_chunks.append(delta)
                    elif event_type == "response.output_text.done":
                        full_text = event.get("text")
                        if isinstance(full_text, str):
                            text_chunks = [full_text]
                    elif event_type == "response.completed":
                        response_data = event.get("response", {})
                        if isinstance(response_data, dict):
                            usage = response_data.get("usage", {})

                    # Standard OpenAI chat completions streaming format
                    elif event.get("object") == "chat.completion.chunk":
                        saw_chat_chunks = True
                        choices = event.get("choices")
                        if isinstance(choices, list) and choices:
                            choice0 = choices[0] if isinstance(choices[0], dict) else {}
                            delta = choice0.get("delta", {})
                            if isinstance(delta, dict):
                                content = delta.get("content")
                                if isinstance(content, str):
                                    text_chunks.append(content)
                                tool_calls = delta.get("tool_calls", [])
                                if isinstance(tool_calls, list):
                                    for tool_call in tool_calls:
                                        if not isinstance(tool_call, dict):
                                            continue
                                        index_value = tool_call.get("index")
                                        if isinstance(index_value, str) and index_value.isdigit():
                                            index_value = int(index_value)
                                        if not isinstance(index_value, int):
                                            continue

                                        current = tool_calls_by_index.setdefault(
                                            index_value,
                                            {
                                                "id": "",
                                                "type": "function",
                                                "function": {"name": "", "arguments": ""},
                                            },
                                        )
                                        call_id = tool_call.get("id")
                                        if isinstance(call_id, str) and call_id:
                                            current["id"] = call_id
                                        call_type = tool_call.get("type")
                                        if isinstance(call_type, str) and call_type:
                                            current["type"] = call_type
                                        function = tool_call.get("function", {})
                                        if isinstance(function, dict):
                                            name = function.get("name")
                                            if isinstance(name, str) and name:
                                                current["function"]["name"] = name
                                            arguments = function.get("arguments")
                                            if isinstance(arguments, str) and arguments:
                                                current["function"]["arguments"] += arguments

                            reason = choice0.get("finish_reason")
                            if isinstance(reason, str) and reason:
                                finish_reason = reason
                        chunk_usage = event.get("usage")
                        if isinstance(chunk_usage, dict) and chunk_usage:
                            usage = chunk_usage
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise ProviderError(f"HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Network error: {exc.reason}") from exc

        text = "".join(text_chunks).strip()
        if saw_chat_chunks:
            tool_calls_out = [call for _, call in sorted(tool_calls_by_index.items())]
            if not text and not tool_calls_out:
                raise ProviderError("SSE stream ended without yielding text content")

            choice: dict[str, Any] = {
                "message": {
                    "content": text if text else None,
                    "tool_calls": tool_calls_out,
                }
            }
            if finish_reason:
                choice["finish_reason"] = finish_reason
            return {
                "choices": [choice],
                "usage": usage,
                "output_text": text,
            }

        if not text:
            raise ProviderError("SSE stream ended without yielding text content")
        return {"output_text": text, "usage": usage}

    def _json_request(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise ProviderError(f"HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Network error: {exc.reason}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Invalid JSON response: {body[:300]}") from exc
