# Claude API Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `"anthropic"` as a third `api_style` in the LLM provider layer so Muse can call Claude models (claude-opus-4-6, claude-sonnet-4-6, etc.) via the Anthropic Messages API.

**Architecture:** The existing `_ModelRouter` → `_ModelAttempt` → `_build_request_payload` pipeline already handles two API styles (`"responses"` for Codex/OpenAI Responses API, `"chat_completions"` for standard OpenAI). Adding `"anthropic"` follows the same pattern: a new branch in `_resolve_api_style`, a URL builder, a header mutator in `_build_attempt`, a payload builder in `_build_request_payload`, and a response extractor in `_extract_llm_message`. No new classes required — only targeted additions to existing functions.

**Tech Stack:** Anthropic Messages REST API (`https://api.anthropic.com/v1/messages`), existing `HttpClient.post_json`, Python 3.10+

---

## Anthropic API Quick Reference

**Request:**
```
POST https://api.anthropic.com/v1/messages
x-api-key: sk-ant-...
anthropic-version: 2023-06-01
content-type: application/json

{
  "model": "claude-opus-4-6",
  "max_tokens": 4000,
  "system": "You are a helpful assistant.",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**Response:**
```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Hello!"}],
  "model": "claude-opus-4-6",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 10, "output_tokens": 3}
}
```

Key differences from OpenAI:
- Auth: `x-api-key: {key}` (NOT `Authorization: Bearer {key}`)
- Required header: `anthropic-version: 2023-06-01`
- System prompt: top-level `"system"` field (not in messages array)
- Response: `content[0]["text"]` (not `choices[0].message.content`)
- No `response_format` parameter — JSON must be elicited via system prompt (already done by stages)

---

## Task 1: Add `"anthropic"` to `_resolve_api_style`

**Files:**
- Modify: `muse/providers.py:783-797`

**Step 1: Write the failing test**

Create `tests/test_anthropic_provider.py`:
```python
"""Tests for Anthropic api_style support in providers."""
import unittest
from muse.providers import LLMClient, ProviderError


def _make_anthropic_client(fake_http, model="claude-opus-4-6"):
    """Helper: LLMClient wired to an Anthropic provider config."""
    router_config = {
        "auth": {
            "profiles": {
                "anthro_key": {"apiKey": "sk-ant-test"}
            }
        },
        "providers": {
            "anthropic": {
                "baseUrl": "https://api.anthropic.com",
                "apiStyle": "anthropic",
                "auth": "anthro_key",
                "models": {
                    f"anthropic/{model}": {"model": model}
                },
            }
        },
        "models": {
            "default": {"primary": f"anthropic/{model}", "fallbacks": []},
            "outline": {"primary": f"anthropic/{model}", "fallbacks": []},
            "writing": {"primary": f"anthropic/{model}", "fallbacks": []},
            "review": {"primary": f"anthropic/{model}", "fallbacks": []},
            "polish": {"primary": f"anthropic/{model}", "fallbacks": []},
            "reasoning": {"primary": f"anthropic/{model}", "fallbacks": []},
        },
        "modelAliases": {},
    }
    client = LLMClient(
        api_key="unused",
        base_url="https://unused.local",
        model="unused",
        model_router_config=router_config,
    )
    client._http = fake_http
    return client


class _FakeHttp:
    def __init__(self, response):
        self.calls = []
        self._response = response

    def post_json(self, url, payload, headers=None):
        self.calls.append({"url": url, "payload": payload, "headers": headers or {}})
        return self._response

    def post_json_sse(self, url, payload, headers=None):
        return self.post_json(url, payload, headers=headers)


class TestAnthropicApiStyle(unittest.TestCase):
    def _anthropic_response(self, text="Test response."):
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

    def test_endpoint_is_v1_messages(self):
        """Requests must go to /v1/messages, not /v1/chat/completions."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=100)
        self.assertTrue(fake.calls[0]["url"].endswith("/v1/messages"))

    def test_auth_uses_x_api_key_not_bearer(self):
        """x-api-key header must be present; Authorization header must NOT be."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=100)
        hdrs = fake.calls[0]["headers"]
        self.assertIn("x-api-key", hdrs)
        self.assertEqual(hdrs["x-api-key"], "sk-ant-test")
        self.assertNotIn("Authorization", hdrs)

    def test_anthropic_version_header_present(self):
        """anthropic-version header must be set to 2023-06-01."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=100)
        hdrs = fake.calls[0]["headers"]
        self.assertIn("anthropic-version", hdrs)
        self.assertEqual(hdrs["anthropic-version"], "2023-06-01")

    def test_payload_has_system_as_top_level_key(self):
        """system prompt must be top-level 'system' field, not inside messages."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="Be a thesis writer.", user="hello", route="default", max_tokens=100)
        payload = fake.calls[0]["payload"]
        self.assertIn("system", payload)
        self.assertEqual(payload["system"], "Be a thesis writer.")
        # messages should only have user role
        for msg in payload.get("messages", []):
            self.assertNotEqual(msg.get("role"), "system")

    def test_payload_has_user_message(self):
        """messages array must have exactly one user message."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello world", route="default", max_tokens=100)
        payload = fake.calls[0]["payload"]
        msgs = payload.get("messages", [])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "hello world")

    def test_max_tokens_in_payload(self):
        """max_tokens must be passed, not max_output_tokens."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=1234)
        payload = fake.calls[0]["payload"]
        self.assertEqual(payload.get("max_tokens"), 1234)
        self.assertNotIn("max_output_tokens", payload)

    def test_response_text_extracted_from_content_array(self):
        """_extract_llm_message must handle Anthropic content array format."""
        fake = _FakeHttp(self._anthropic_response("Thesis output text."))
        client = _make_anthropic_client(fake)
        result = client.text(system="sys", user="hello", route="default", max_tokens=100)
        self.assertEqual(result, "Thesis output text.")

    def test_no_response_format_in_payload(self):
        """Anthropic does not accept response_format — must not be in payload."""
        fake = _FakeHttp(self._anthropic_response('{"chapters": []}'))
        client = _make_anthropic_client(fake)
        client.structured(system="Return JSON.", user="topic", route="outline", max_tokens=500)
        payload = fake.calls[0]["payload"]
        self.assertNotIn("response_format", payload)
```

**Step 2: Run test to verify it fails**

```bash
cd /home/planck/gradute/Muse
python3 -m pytest tests/test_anthropic_provider.py -v 2>&1 | head -40
```
Expected: All tests FAIL (ProviderError or wrong endpoint/headers/payload).

**Step 3: Add `"anthropic"` to `_resolve_api_style` (providers.py:783-797)**

After the `"chat_completions"` block and before the `codex_oauth` fallback:
```python
    if raw in {"anthropic", "claude", "anthropic_messages"}:
        return "anthropic"
```

**Step 4: Run test again — endpoint test will fail differently now (needs URL builder)**

```bash
python3 -m pytest tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_endpoint_is_v1_messages -v
```
Expected: FAIL with "goes to /v1/chat/completions" (wrong URL, not yet mapped).

**Step 5: Commit**

```bash
git add muse/providers.py tests/test_anthropic_provider.py
git commit -m "test: add failing tests for Anthropic api_style; add 'anthropic' to _resolve_api_style"
```

---

## Task 2: URL builder and endpoint routing for Anthropic

**Files:**
- Modify: `muse/providers.py:751-754` (`_to_provider_endpoint`)
- Add after line 763 (`_to_chat_completions_url`): `_to_anthropic_url`

**Step 1: Add `_to_anthropic_url` helper**

Insert after `_to_chat_completions_url` (line ~763):
```python
def _to_anthropic_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1/messages"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/messages"
    return normalized + "/v1/messages"
```

**Step 2: Update `_to_provider_endpoint` to route Anthropic**

Change (line 751-754):
```python
def _to_provider_endpoint(base_url: str, *, api_style: str, codex_oauth: bool) -> str:
    if api_style == "responses":
        return _to_responses_url(base_url, codex_oauth=codex_oauth)
    if api_style == "anthropic":
        return _to_anthropic_url(base_url)
    return _to_chat_completions_url(base_url)
```

**Step 3: Run endpoint test**

```bash
python3 -m pytest tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_endpoint_is_v1_messages -v
```
Expected: PASS.

**Step 4: Run all Anthropic tests**

```bash
python3 -m pytest tests/test_anthropic_provider.py -v
```
Expected: endpoint test PASS; auth/header tests still FAIL (headers not set yet).

**Step 5: Commit**

```bash
git add muse/providers.py
git commit -m "feat: add _to_anthropic_url and route anthropic api_style to /v1/messages"
```

---

## Task 3: Anthropic auth headers in `_build_attempt`

**Files:**
- Modify: `muse/providers.py:273-295` (header construction loop in `_build_attempt`)

**Context:** The current code sets `Authorization: Bearer {api_key}` for every profile. Anthropic needs `x-api-key: {key}` and `anthropic-version: 2023-06-01` instead.

**Step 1: Add Anthropic header mutation after the `codex_oauth` block**

After the `if codex_oauth:` block (line ~284-292), add:
```python
            if api_style == "anthropic":
                _pop_header_case_insensitive(headers, "Authorization")
                if api_key:
                    headers["x-api-key"] = api_key
                headers.setdefault("anthropic-version", "2023-06-01")
```

Note: `api_style` is resolved at line 271, before the loop — it's already available.

**Step 2: Run auth header tests**

```bash
python3 -m pytest tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_auth_uses_x_api_key_not_bearer tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_anthropic_version_header_present -v
```
Expected: Both PASS.

**Step 3: Run full Anthropic test suite**

```bash
python3 -m pytest tests/test_anthropic_provider.py -v
```
Expected: endpoint + auth tests PASS; payload/response tests still FAIL.

**Step 4: Commit**

```bash
git add muse/providers.py
git commit -m "feat: set x-api-key and anthropic-version headers for anthropic api_style"
```

---

## Task 4: Anthropic payload format in `_build_request_payload`

**Files:**
- Modify: `muse/providers.py:860-921` (`_build_request_payload`)

**Step 1: Add Anthropic branch before the `chat_completions` fallback**

After the `if attempt.api_style == "responses":` block (before line 906), insert:
```python
    if attempt.api_style == "anthropic":
        payload = {
            "model": attempt.model_name,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        payload.update(attempt.params)
        # Re-apply required fields so params cannot override them
        payload["model"] = attempt.model_name
        payload["max_tokens"] = max_tokens
        payload["system"] = system
        # Anthropic does not support response_format — JSON is elicited via system prompt
        return payload
```

**Step 2: Run payload tests**

```bash
python3 -m pytest tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_payload_has_system_as_top_level_key tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_payload_has_user_message tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_max_tokens_in_payload tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_no_response_format_in_payload -v
```
Expected: All PASS.

**Step 3: Run full Anthropic test suite**

```bash
python3 -m pytest tests/test_anthropic_provider.py -v
```
Expected: All except `test_response_text_extracted_from_content_array` pass.

**Step 4: Commit**

```bash
git add muse/providers.py
git commit -m "feat: add Anthropic payload format to _build_request_payload"
```

---

## Task 5: Anthropic response extraction in `_extract_llm_message`

**Files:**
- Modify: `muse/providers.py:924-950` (`_extract_llm_message`)

**Context:** Current `_extract_llm_message` handles:
1. `choices[0].message.content` (OpenAI chat_completions)
2. `output_text` (Responses API shortcut)
3. `output[]` array (Responses API full)
4. Nested `response` dict

Anthropic returns `{"content": [{"type": "text", "text": "..."}], ...}`. The `content` key at the top level must be handled before the existing checks to avoid conflict.

**Step 1: Add Anthropic response format extraction at the top of `_extract_llm_message`**

After the function signature (line 924), as the first check:
```python
    # Anthropic Messages API: {"content": [{"type": "text", "text": "..."}]}
    content_list = result.get("content")
    if isinstance(content_list, list) and content_list:
        for block in content_list:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    return text
```

Insert this before the `choices = result.get("choices")` line.

**Step 2: Run response extraction test**

```bash
python3 -m pytest tests/test_anthropic_provider.py::TestAnthropicApiStyle::test_response_text_extracted_from_content_array -v
```
Expected: PASS.

**Step 3: Run full Anthropic test suite**

```bash
python3 -m pytest tests/test_anthropic_provider.py -v
```
Expected: All 8 tests PASS.

**Step 4: Run full test suite to check for regressions**

```bash
python3 -m pytest tests/ -v 2>&1 | tail -20
```
Expected: All previously passing tests still pass; 8 new Anthropic tests pass.

**Step 5: Commit**

```bash
git add muse/providers.py
git commit -m "feat: extract Anthropic Messages API response from content array"
```

---

## Task 6: Create example config file for Anthropic provider

**Files:**
- Create: `model-router.anthropic.example.json`

**Step 1: Create the example config**

```json
{
  "auth": {
    "profiles": {
      "anthropic_api_key": {
        "apiKeyEnv": "ANTHROPIC_API_KEY"
      }
    }
  },
  "providers": {
    "anthropic": {
      "baseUrl": "https://api.anthropic.com",
      "apiStyle": "anthropic",
      "auth": "anthropic_api_key",
      "models": {
        "anthropic/opus": {
          "model": "claude-opus-4-6"
        },
        "anthropic/sonnet": {
          "model": "claude-sonnet-4-6"
        },
        "anthropic/haiku": {
          "model": "claude-haiku-4-5-20251001"
        }
      }
    }
  },
  "models": {
    "default": {
      "primary": "anthropic/opus",
      "fallbacks": ["anthropic/sonnet"]
    },
    "outline": {
      "primary": "anthropic/opus",
      "fallbacks": ["anthropic/sonnet"]
    },
    "writing": {
      "primary": "anthropic/opus",
      "fallbacks": ["anthropic/sonnet"]
    },
    "review": {
      "primary": "anthropic/sonnet",
      "fallbacks": ["anthropic/haiku"]
    },
    "reasoning": {
      "primary": "anthropic/opus",
      "fallbacks": ["anthropic/sonnet"]
    },
    "polish": {
      "primary": "anthropic/sonnet",
      "fallbacks": ["anthropic/haiku"]
    }
  },
  "modelAliases": {}
}
```

**Step 2: Verify the config loads correctly with a quick smoke test**

```bash
cd /home/planck/gradute/Muse
MUSE_MODEL_ROUTER_PATH=model-router.anthropic.example.json \
ANTHROPIC_API_KEY=sk-ant-test \
python3 -c "
from muse.config import load_settings
from muse.providers import LLMClient
s = load_settings()
print('refs_dir:', s.refs_dir)
c = LLMClient(s.llm_api_key, s.llm_base_url, s.llm_model, s.model_router_config)
attempts = c._router.resolve('writing')
a = attempts[0]
print('api_style:', a.api_style)
print('endpoint:', a.endpoint_url)
print('model:', a.model_name)
# Check headers
print('headers:', [list(h.keys()) for h in a.header_candidates])
"
```
Expected output:
```
api_style: anthropic
endpoint: https://api.anthropic.com/v1/messages
model: claude-opus-4-6
headers: [['x-api-key', 'anthropic-version']] (roughly)
```

**Step 3: Run full test suite one more time**

```bash
python3 -m pytest tests/ -v 2>&1 | tail -5
```
Expected: All tests pass.

**Step 4: Commit**

```bash
git add model-router.anthropic.example.json
git commit -m "feat: add model-router.anthropic.example.json for Claude API provider"
```

---

## Verification checklist

After all tasks complete, verify these key behaviors:

```bash
# 1. All tests pass (target: ~89 tests = 81 existing + 8 new)
python3 -m pytest tests/ -v

# 2. Anthropic api_style resolves correctly
python3 -c "
from muse.providers import _resolve_api_style
assert _resolve_api_style({'apiStyle': 'anthropic'}, codex_oauth=False) == 'anthropic'
assert _resolve_api_style({'apiStyle': 'claude'}, codex_oauth=False) == 'anthropic'
print('api_style resolution OK')
"

# 3. URL builder works
python3 -c "
from muse.providers import _to_provider_endpoint
url = _to_provider_endpoint('https://api.anthropic.com', api_style='anthropic', codex_oauth=False)
assert url == 'https://api.anthropic.com/v1/messages', url
print('URL builder OK:', url)
"

# 4. Payload format correct
python3 -c "
from muse.providers import _build_request_payload, _ModelAttempt
attempt = _ModelAttempt(
    route_name='writing', model_id='anthropic/opus', provider_name='anthropic',
    endpoint_url='https://api.anthropic.com/v1/messages',
    api_style='anthropic', model_name='claude-opus-4-6',
    header_candidates=[], params={}, requires_streaming=False,
)
p = _build_request_payload(attempt=attempt, system='sys', user='hello',
                           temperature=0.7, response_format=None, max_tokens=1000)
assert p['system'] == 'sys', p
assert p['messages'] == [{'role': 'user', 'content': 'hello'}], p
assert p['max_tokens'] == 1000, p
assert 'response_format' not in p, p
print('Payload format OK')
"

# 5. Live smoke test with real API key (optional)
# MUSE_MODEL_ROUTER_PATH=model-router.anthropic.example.json \
# ANTHROPIC_API_KEY=sk-ant-YOUR-KEY \
# python3 -m muse debug-llm
```

---

## Usage after implementation

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run Muse with Claude
MUSE_MODEL_ROUTER_PATH=model-router.anthropic.example.json \
python3 -m muse run \
  --topic "拜占庭容错分析" \
  --discipline "计算机科学" \
  --language zh \
  --auto-approve

# Or mix with fallback: use Claude for writing but Codex for reasoning
# (requires a merged config with both providers)
```
