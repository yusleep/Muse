"""HTTP transport primitives shared by provider clients."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, timeout_seconds: int = 120) -> None:
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
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise ProviderError(f"HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Network error: {exc.reason}") from exc

        text = "".join(text_chunks).strip()
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
