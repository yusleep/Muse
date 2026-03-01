"""Real provider clients for LLM, literature search, and citation metadata."""

from __future__ import annotations

import json
import os
import re
import base64
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping


class ProviderError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, timeout_seconds: int = 45) -> None:
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
        """POST and consume a Server-Sent Events stream; returns a synthetic response dict."""
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


@dataclass(frozen=True)
class _ModelAttempt:
    route_name: str
    model_id: str
    provider_name: str
    endpoint_url: str
    api_style: str
    model_name: str
    header_candidates: list[dict[str, str]]
    params: dict[str, Any]
    requires_streaming: bool = False


@dataclass
class _ModelRouter:
    router_config: Mapping[str, Any]
    default_api_key: str
    default_base_url: str
    default_model: str
    env: Mapping[str, str]
    aliases: dict[str, str] = field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    routes: dict[str, dict[str, Any]] = field(default_factory=dict)
    auth_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    _token_cache: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cfg = dict(self.router_config or {})

        aliases = cfg.get("modelAliases", cfg.get("aliases", {}))
        self.aliases = aliases if isinstance(aliases, dict) else {}

        auth_root = cfg.get("auth", {})
        if isinstance(auth_root, dict) and isinstance(auth_root.get("profiles"), dict):
            self.auth_profiles = dict(auth_root.get("profiles", {}))
        elif isinstance(auth_root, dict):
            self.auth_profiles = dict(auth_root)
        else:
            self.auth_profiles = {}

        providers = cfg.get("providers", {})
        self.providers = providers if isinstance(providers, dict) else {}

        routes = cfg.get("models", {})
        self.routes = routes if isinstance(routes, dict) else {}

        # Legacy default route when router config is not provided.
        if not self.routes:
            self.routes = {
                "default": {
                    "primary": f"default/{self.default_model}",
                    "fallbacks": [],
                },
                "outline": {
                    "primary": f"default/{self.default_model}",
                    "fallbacks": [],
                },
                "writing": {
                    "primary": f"default/{self.default_model}",
                    "fallbacks": [],
                },
                "review": {
                    "primary": f"default/{self.default_model}",
                    "fallbacks": [],
                },
                "reasoning": {
                    "primary": f"default/{self.default_model}",
                    "fallbacks": [],
                },
                "polish": {
                    "primary": f"default/{self.default_model}",
                    "fallbacks": [],
                },
            }

        if "default" not in self.providers:
            self.providers["default"] = {
                "baseUrl": self.default_base_url,
                "auth": "default",
                "models": {
                    f"default/{self.default_model}": {
                        "model": self.default_model,
                        "params": {},
                    },
                    self.default_model: {
                        "model": self.default_model,
                        "params": {},
                    },
                },
            }

        if "default" not in self.auth_profiles:
            self.auth_profiles["default"] = {
                "apiKey": self.default_api_key,
            }

    def resolve(self, route_name: str) -> list[_ModelAttempt]:
        route = self.routes.get(route_name)
        if not isinstance(route, dict):
            route = self.routes.get("default", {})
        if not isinstance(route, dict):
            raise ProviderError("No route configuration available")

        primary = route.get("primary")
        if not isinstance(primary, str) or not primary.strip():
            raise ProviderError(f"Route '{route_name}' has no primary model")

        fallbacks = route.get("fallbacks", [])
        fallback_list = [m for m in fallbacks if isinstance(m, str) and m.strip()] if isinstance(fallbacks, list) else []

        ordered_ids = [primary.strip(), *fallback_list]
        deduped: list[str] = []
        seen: set[str] = set()
        for model_id in ordered_ids:
            alias_target = self.aliases.get(model_id, model_id)
            if alias_target not in seen:
                deduped.append(alias_target)
                seen.add(alias_target)

        attempts = [self._build_attempt(route_name=route_name, model_id=model_id) for model_id in deduped]
        return attempts

    def _build_attempt(self, route_name: str, model_id: str) -> _ModelAttempt:
        provider_name, short_model = _split_model_id(model_id)

        provider_cfg = self.providers.get(provider_name)
        if not isinstance(provider_cfg, dict):
            raise ProviderError(f"Unknown provider '{provider_name}' for model '{model_id}'")

        base_url = _get_text(provider_cfg, "baseUrl") or _get_text(provider_cfg, "base_url")
        if not base_url:
            raise ProviderError(f"Provider '{provider_name}' missing baseUrl")

        provider_models = provider_cfg.get("models", {})
        model_entry = None
        if isinstance(provider_models, dict):
            model_entry = provider_models.get(model_id)
            if model_entry is None:
                model_entry = provider_models.get(short_model)

        model_name = short_model
        params: dict[str, Any] = {}
        if isinstance(model_entry, dict):
            model_name = str(model_entry.get("model", short_model))
            if isinstance(model_entry.get("params"), dict):
                params = dict(model_entry["params"])

        provider_headers: dict[str, str] = {}
        provider_headers = provider_cfg.get("headers", {})
        if isinstance(provider_headers, dict):
            provider_headers = {str(k): str(v) for k, v in provider_headers.items()}
        else:
            provider_headers = {}

        profile_names = _normalize_auth_profile_names(provider_cfg.get("auth"))
        profile_configs: list[Mapping[str, Any]] = []
        for profile_name in profile_names:
            auth_profile = self.auth_profiles.get(profile_name, {})
            if not isinstance(auth_profile, dict):
                auth_profile = {}
            profile_configs.append(auth_profile)

        codex_oauth = bool(provider_cfg.get("codexOAuth")) or any(
            _is_codex_oauth_profile(profile) for profile in profile_configs
        )
        api_style = _resolve_api_style(provider_cfg, codex_oauth=codex_oauth)

        header_candidates: list[dict[str, str]] = []
        for auth_profile in profile_configs:
            headers = dict(provider_headers)
            auth_headers = auth_profile.get("headers", {})
            if isinstance(auth_headers, dict):
                headers.update({str(k): str(v) for k, v in auth_headers.items()})

            api_key = self._resolve_api_key(auth_profile)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            if codex_oauth:
                _pop_header_case_insensitive(headers, "x-api-key")
                account_id = _extract_chatgpt_account_id_from_jwt(api_key)
                if account_id:
                    headers.setdefault("chatgpt-account-id", account_id)
                headers.setdefault("OpenAI-Beta", "responses=experimental")
                headers.setdefault("originator", "codex_cli_rs")
                headers.setdefault("accept", "application/json")
            header_candidates.append(headers)

        if not header_candidates:
            header_candidates = [dict(provider_headers)]

        endpoint = _to_provider_endpoint(base_url, api_style=api_style, codex_oauth=codex_oauth)
        requires_streaming = "chatgpt.com/backend-api" in endpoint

        return _ModelAttempt(
            route_name=route_name,
            model_id=model_id,
            provider_name=provider_name,
            endpoint_url=endpoint,
            api_style=api_style,
            model_name=model_name,
            header_candidates=header_candidates,
            params=params,
            requires_streaming=requires_streaming,
        )

    def _resolve_api_key(self, auth_profile: Mapping[str, Any]) -> str:
        profile_type = str(auth_profile.get("type", "")).strip().lower()
        if profile_type == "oauth" or "oauthProvider" in auth_profile:
            oauth_token = self._resolve_oauth_access_token(auth_profile)
            if oauth_token:
                return oauth_token

        api_key = str(auth_profile.get("apiKey", "")).strip()
        if api_key:
            return api_key

        api_key_env = str(auth_profile.get("apiKeyEnv", "")).strip()
        if api_key_env:
            return str(self.env.get(api_key_env, "")).strip()

        return self.default_api_key

    def _resolve_oauth_access_token(self, auth_profile: Mapping[str, Any]) -> str:
        env_keys = [
            "oauthAccessTokenEnv",
            "accessTokenEnv",
            "tokenEnv",
            "apiKeyEnv",
        ]
        for env_key_name in env_keys:
            key = str(auth_profile.get(env_key_name, "")).strip()
            if key:
                token = str(self.env.get(key, "")).strip()
                if token:
                    return token

        provider = str(auth_profile.get("oauthProvider", "")).strip().lower()
        auth_file = str(auth_profile.get("authFile", "") or auth_profile.get("tokenFile", "")).strip()
        token_path = str(
            auth_profile.get("tokenPath", "")
            or auth_profile.get("accessTokenPath", "")
        ).strip()

        if not auth_file and provider in {"codex", "codex_plus", "codex-plus", "openai-codex"}:
            auth_file = "~/.codex/auth.json"
        if not token_path and provider in {"codex", "codex_plus", "codex-plus", "openai-codex"}:
            token_path = "tokens.access_token"

        if not auth_file:
            return ""
        if not token_path:
            token_path = "access_token"

        expanded = os.path.expanduser(auth_file)
        cache_key = f"{expanded}::{token_path}"
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        try:
            with open(expanded, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:  # noqa: BLE001
            return ""

        token = _extract_by_path(payload, token_path)
        if isinstance(token, str) and token.strip():
            token = token.strip()
            self._token_cache[cache_key] = token
            return token
        return ""


@dataclass
class LLMClient:
    api_key: str
    base_url: str
    model: str
    http: HttpClient
    model_router_config: Mapping[str, Any] | None = None
    env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        self.router = _ModelRouter(
            router_config=self.model_router_config or {},
            default_api_key=self.api_key,
            default_base_url=self.base_url,
            default_model=self.model,
            env=self.env or {},
        )

    def _chat_completion(
        self,
        *,
        system: str,
        user: str,
        route: str = "default",
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 2500,
    ) -> dict[str, Any]:
        attempts = self.router.resolve(route)
        failures: list[str] = []

        for attempt in attempts:
            payload = _build_request_payload(
                attempt=attempt,
                system=system,
                user=user,
                temperature=temperature,
                response_format=response_format,
                max_tokens=max_tokens,
                streaming=attempt.requires_streaming,
            )

            for profile_idx, headers in enumerate(attempt.header_candidates, start=1):
                try:
                    if attempt.requires_streaming:
                        result = self.http.post_json_sse(attempt.endpoint_url, payload, headers=headers)
                    else:
                        result = self.http.post_json(attempt.endpoint_url, payload, headers=headers)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{attempt.model_id}[profile#{profile_idx}]: {exc}")
                    continue

                try:
                    message = _extract_llm_message(result)
                except Exception:  # noqa: BLE001
                    failures.append(f"{attempt.model_id}[profile#{profile_idx}]: unexpected response shape")
                    continue

                return {
                    "content": message,
                    "usage": result.get("usage", {}),
                    "raw": result,
                    "model_id": attempt.model_id,
                    "provider": attempt.provider_name,
                }

        failure_text = " | ".join(failures) if failures else "no attempts were available"
        raise ProviderError(f"All model attempts failed for route '{route}': {failure_text}")

    def text(
        self,
        *,
        system: str,
        user: str,
        route: str = "default",
        temperature: float = 0.2,
        max_tokens: int = 2500,
    ) -> str:
        out = self._chat_completion(
            system=system,
            user=user,
            route=route,
            temperature=temperature,
            response_format=None,
            max_tokens=max_tokens,
        )
        return out["content"]

    def structured(self, *, system: str, user: str, route: str = "default", max_tokens: int = 2500) -> dict[str, Any]:
        out = self._chat_completion(
            system=system,
            user=user,
            route=route,
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        content = out["content"]
        parsed = _parse_json_relaxed(content)
        if not isinstance(parsed, dict):
            raise ProviderError("LLM structured response was not a JSON object")
        return parsed

    def entailment(self, *, premise: str, hypothesis: str, route: str = "reasoning") -> str:
        system = (
            "You are an NLI classifier. Output only JSON with key `label` where label is one of "
            "entailment, neutral, contradiction."
        )
        user = f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"
        out = self.structured(system=system, user=user, route=route, max_tokens=200)
        label = str(out.get("label", "neutral")).strip().lower()
        if label not in {"entailment", "neutral", "contradiction"}:
            return "neutral"
        return label

    def debug_probe(
        self,
        *,
        route: str = "default",
        system: str = "Reply with 'ok'.",
        user: str = "ok",
        max_tokens: int = 16,
    ) -> dict[str, Any]:
        """Run diagnostic LLM calls and report per-attempt failures safely."""

        attempts = self.router.resolve(route)
        diag_attempts: list[dict[str, Any]] = []

        for attempt in attempts:
            payload = _build_request_payload(
                attempt=attempt,
                system=system,
                user=user,
                temperature=0.0,
                response_format=None,
                max_tokens=max_tokens,
                streaming=attempt.requires_streaming,
            )

            model_attempt_ok = False
            for profile_idx, headers in enumerate(attempt.header_candidates, start=1):
                row = {
                    "model_id": attempt.model_id,
                    "provider": attempt.provider_name,
                    "endpoint": attempt.endpoint_url,
                    "api_style": attempt.api_style,
                    "profile_index": profile_idx,
                    "model": attempt.model_name,
                    "authorization_present": "Authorization" in headers,
                    "header_keys": sorted(headers.keys()),
                }
                try:
                    if attempt.requires_streaming:
                        result = self.http.post_json_sse(attempt.endpoint_url, payload, headers=headers)
                    else:
                        result = self.http.post_json(attempt.endpoint_url, payload, headers=headers)
                    message = _extract_llm_message(result)
                    row["ok"] = True
                    row["response_preview"] = str(message)[:200]
                    diag_attempts.append(row)
                    model_attempt_ok = True
                    break
                except Exception as exc:  # noqa: BLE001
                    row["ok"] = False
                    row["error"] = str(exc)
                    diag_attempts.append(row)

            if model_attempt_ok:
                return {
                    "success": True,
                    "route": route,
                    "attempts": diag_attempts,
                }

        return {
            "success": False,
            "route": route,
            "attempts": diag_attempts,
            "error": "all attempts failed",
        }


@dataclass
class AcademicSearchClient:
    http: HttpClient
    semantic_scholar_api_key: str | None = None
    openalex_email: str | None = None

    def search_multi_source(self, topic: str, discipline: str) -> tuple[list[dict[str, Any]], list[str]]:
        queries = [
            topic,
            f"{topic} {discipline}",
            f"{topic} methodology",
        ]

        papers: list[dict[str, Any]] = []
        for q in queries:
            for fetch, kwargs in [
                (self.search_semantic_scholar, {"query": q, "limit": 10}),
                (self.search_openalex, {"query": q, "limit": 10}),
                (self.search_arxiv, {"query": q, "limit": 8}),
            ]:
                try:
                    papers.extend(fetch(**kwargs))
                except Exception:  # noqa: BLE001
                    pass

        deduped = _dedupe_references(papers)
        return deduped[:80], queries

    def search_semantic_scholar(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": str(limit),
                "fields": "title,authors,year,doi,venue,abstract,url",
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        payload = self.http.get_json(url, headers=headers)
        records: list[dict[str, Any]] = []
        for item in payload.get("data", []):
            authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
            records.append(
                {
                    "ref_id": _reference_id(item.get("authors", []), item.get("year"), item.get("title")),
                    "title": item.get("title", ""),
                    "authors": authors,
                    "year": item.get("year"),
                    "doi": item.get("doi"),
                    "venue": item.get("venue"),
                    "abstract": item.get("abstract"),
                    "source": "semantic_scholar",
                    "verified_metadata": bool(item.get("title")),
                }
            )
        return records

    def search_openalex(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        mailto = f"&mailto={urllib.parse.quote(self.openalex_email)}" if self.openalex_email else ""
        url = (
            "https://api.openalex.org/works?"
            f"search={urllib.parse.quote(query)}&per-page={int(limit)}{mailto}"
        )
        payload = self.http.get_json(url)

        records: list[dict[str, Any]] = []
        for item in payload.get("results", []):
            authorships = item.get("authorships", [])
            authors = [
                auth.get("author", {}).get("display_name", "")
                for auth in authorships
                if auth.get("author", {}).get("display_name")
            ]
            doi = item.get("doi")
            if isinstance(doi, str):
                doi = doi.replace("https://doi.org/", "")

            records.append(
                {
                    "ref_id": _reference_id(authorships, item.get("publication_year"), item.get("title")),
                    "title": item.get("title", ""),
                    "authors": authors,
                    "year": item.get("publication_year"),
                    "doi": doi,
                    "venue": item.get("primary_location", {}).get("source", {}).get("display_name"),
                    "abstract": _openalex_abstract(item.get("abstract_inverted_index")),
                    "source": "openalex",
                    "verified_metadata": bool(item.get("title")),
                }
            )

        return records

    def search_arxiv(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query=all:{urllib.parse.quote(query)}&start=0&max_results={int(limit)}"
        )
        try:
            with urllib.request.urlopen(url, timeout=self.http.timeout_seconds) as resp:
                xml_text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"arXiv request failed: {exc}") from exc

        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, flags=re.DOTALL)
        records: list[dict[str, Any]] = []
        for entry in entries:
            title = _extract_xml_tag(entry, "title")
            summary = _extract_xml_tag(entry, "summary")
            published = _extract_xml_tag(entry, "published")
            year = int(published[:4]) if published and published[:4].isdigit() else None
            authors = re.findall(r"<name>(.*?)</name>", entry, flags=re.DOTALL)
            records.append(
                {
                    "ref_id": _reference_id(authors, year, title),
                    "title": title,
                    "authors": [a.strip() for a in authors if a.strip()],
                    "year": year,
                    "doi": None,
                    "venue": "arXiv",
                    "abstract": summary,
                    "source": "arxiv",
                    "verified_metadata": bool(title),
                }
            )
        return records


@dataclass
class CitationMetadataClient:
    http: HttpClient
    crossref_mailto: str | None = None

    def verify_doi(self, doi: str) -> bool:
        if not doi:
            return False
        encoded = urllib.parse.quote(doi)
        mailto = f"?mailto={urllib.parse.quote(self.crossref_mailto)}" if self.crossref_mailto else ""
        url = f"https://api.crossref.org/works/{encoded}{mailto}"
        try:
            payload = self.http.get_json(url)
        except ProviderError:
            return False
        return bool(payload.get("message", {}).get("DOI"))

    def crosscheck_metadata(self, ref: dict[str, Any]) -> bool:
        title = str(ref.get("title", "")).strip().lower()
        if not title:
            return False

        doi = str(ref.get("doi", "") or "").strip()
        if doi:
            return self.verify_doi(doi)

        # Title fallback using CrossRef works query.
        query = urllib.parse.quote(title)
        mailto = f"&mailto={urllib.parse.quote(self.crossref_mailto)}" if self.crossref_mailto else ""
        url = f"https://api.crossref.org/works?query.title={query}&rows=1{mailto}"
        try:
            payload = self.http.get_json(url)
        except ProviderError:
            return False

        items = payload.get("message", {}).get("items", [])
        if not items:
            return False
        remote_title = " ".join(items[0].get("title", [])) if items[0].get("title") else ""
        return title[:40] in remote_title.lower() or remote_title.lower()[:40] in title


def _get_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    return str(value).strip() if isinstance(value, str) else ""


def _split_model_id(model_id: str) -> tuple[str, str]:
    if "/" in model_id:
        provider, short = model_id.split("/", 1)
        return provider.strip(), short.strip()
    return "default", model_id.strip()


def _to_provider_endpoint(base_url: str, *, api_style: str, codex_oauth: bool) -> str:
    if api_style == "responses":
        return _to_responses_url(base_url, codex_oauth=codex_oauth)
    return _to_chat_completions_url(base_url)


def _to_chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/chat/completions"


def _to_responses_url(base_url: str, *, codex_oauth: bool) -> str:
    normalized = base_url.strip().rstrip("/")
    if codex_oauth and "api.openai.com" in normalized:
        return "https://chatgpt.com/backend-api/codex/responses"
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/responses"):
        if codex_oauth:
            return normalized[:-10] + "/codex/responses"
        return normalized
    if codex_oauth and normalized.endswith("/backend-api"):
        return normalized + "/codex/responses"
    if normalized.endswith("/v1"):
        return normalized + "/responses"
    return normalized + "/responses"


def _resolve_api_style(provider_cfg: Mapping[str, Any], *, codex_oauth: bool) -> str:
    raw = (
        _get_text(provider_cfg, "apiStyle")
        or _get_text(provider_cfg, "api_style")
        or _get_text(provider_cfg, "protocol")
    ).lower()
    if raw in {"responses", "response", "openai_responses"}:
        return "responses"
    if raw in {"codex_responses", "codex-oauth", "codex_oauth"}:
        return "responses"
    if raw in {"chat", "chat_completions", "chat-completions"}:
        return "chat_completions"
    if codex_oauth:
        return "responses"
    return "chat_completions"


def _is_codex_oauth_profile(auth_profile: Mapping[str, Any]) -> bool:
    profile_type = str(auth_profile.get("type", "")).strip().lower()
    provider = str(auth_profile.get("oauthProvider", "")).strip().lower()
    codex_names = {"codex", "codex_plus", "codex-plus", "openai-codex"}
    if provider in codex_names:
        return True
    if profile_type != "oauth":
        return False
    auth_file = str(auth_profile.get("authFile", "") or auth_profile.get("tokenFile", "")).strip().lower()
    return ".codex/" in auth_file or auth_file.endswith(".codex/auth.json")


def _extract_by_path(payload: Any, dot_path: str) -> Any:
    current: Any = payload
    for part in [p for p in dot_path.split(".") if p]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _normalize_auth_profile_names(raw_auth: Any) -> list[str]:
    if isinstance(raw_auth, str) and raw_auth.strip():
        return [raw_auth.strip()]
    if isinstance(raw_auth, list):
        out = [str(item).strip() for item in raw_auth if isinstance(item, str) and item.strip()]
        return out or ["default"]
    return ["default"]


def _extract_chatgpt_account_id_from_jwt(token: str) -> str:
    if token.count(".") != 2:
        return ""
    try:
        payload_b64 = token.split(".")[1]
        decoded = _base64url_decode(payload_b64)
        payload = json.loads(decoded)
    except Exception:  # noqa: BLE001
        return ""
    claim = payload.get("https://api.openai.com/auth")
    if isinstance(claim, dict):
        account_id = claim.get("chatgpt_account_id")
        if isinstance(account_id, str):
            return account_id.strip()
    return ""


def _base64url_decode(value: str) -> str:
    padded = value + "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")


def _pop_header_case_insensitive(headers: dict[str, str], key: str) -> None:
    lowered = key.lower()
    for existing in list(headers.keys()):
        if existing.lower() == lowered:
            headers.pop(existing, None)


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
    if attempt.api_style == "responses":
        payload: dict[str, Any] = {
            "model": attempt.model_name,
            "instructions": system,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user}],
                },
            ],
            "stream": streaming,
        }
        # The chatgpt.com/backend-api/codex endpoint only accepts a minimal set of
        # parameters; temperature and max_output_tokens are rejected with HTTP 400.
        if not streaming:
            payload["temperature"] = temperature
            payload["max_output_tokens"] = max_tokens
        if "chatgpt.com/backend-api" in attempt.endpoint_url:
            payload["store"] = False
        payload.update(attempt.params)
        payload["model"] = attempt.model_name
        payload["stream"] = streaming
        if not streaming:
            payload["temperature"] = temperature
            payload["max_output_tokens"] = max_tokens
        # chatgpt.com/backend-api streaming does not support text.format constraints;
        # the stages already request JSON in their system prompts so _parse_json_relaxed
        # can extract the output without explicit format enforcement.
        if not streaming and response_format is not None and response_format.get("type") == "json_object":
            text_cfg = payload.get("text")
            if not isinstance(text_cfg, dict):
                text_cfg = {}
            text_cfg["format"] = {"type": "json_object"}
            payload["text"] = text_cfg
        return payload

    payload = {
        "model": attempt.model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(attempt.params)
    payload["temperature"] = temperature
    payload["max_tokens"] = max_tokens
    payload["model"] = attempt.model_name
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


def _extract_llm_message(result: Mapping[str, Any]) -> str:
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content

    direct = result.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    output = result.get("output")
    from_output = _extract_text_from_responses_output(output)
    if from_output:
        return from_output

    nested = result.get("response")
    if isinstance(nested, dict):
        nested_text = _extract_llm_message(nested)
        if nested_text:
            return nested_text

    raise ProviderError("unexpected response shape")


def _extract_text_from_responses_output(output: Any) -> str:
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
                    continue
                value = part.get("value")
                if isinstance(value, str) and value.strip():
                    chunks.append(value)
    return "\n".join(chunks).strip()


def _reference_id(authors: Any, year: Any, title: Any) -> str:
    year_str = str(year) if year else "noyear"
    if isinstance(authors, list) and authors:
        first = authors[0]
        if isinstance(first, dict):
            name = first.get("name") or first.get("author", {}).get("display_name") or "anon"
        else:
            name = str(first)
    else:
        name = "anon"

    surname = re.sub(r"[^a-zA-Z0-9]", "", name.split()[-1].lower()) or "anon"
    title_token = re.sub(r"[^a-zA-Z0-9]", "", str(title).split(" ")[0].lower()) or "paper"
    return f"@{surname}{year_str}{title_token}"


def _openalex_abstract(inverted: Any) -> str | None:
    if not isinstance(inverted, dict) or not inverted:
        return None

    max_position = 0
    for positions in inverted.values():
        if isinstance(positions, list):
            max_position = max(max_position, max(positions, default=0))
    words = [""] * (max_position + 1)
    for token, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = token
    return " ".join(w for w in words if w)


def _extract_xml_tag(xml_fragment: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", xml_fragment, flags=re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _fix_invalid_escapes(s: str) -> str:
    """Fix invalid JSON escape sequences by properly escaping lone backslashes.

    Handles cases like LaTeX \\alpha, \\unicode (non-4-hex \\u), etc.
    Correctly skips already-valid escapes (\\\\, \\n, \\uXXXX, etc.).
    """
    result: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            result.append(c)
            i += 1
            continue
        if i + 1 >= len(s):
            # Trailing lone backslash
            result.append("\\\\")
            i += 1
            continue
        nc = s[i + 1]
        if nc in '"\\\/bfnrt':
            # Valid simple escape – keep as-is
            result.append(c)
            result.append(nc)
            i += 2
        elif nc == "u":
            hex_digits = s[i + 2 : i + 6]
            if len(hex_digits) == 4 and all(h in "0123456789abcdefABCDEF" for h in hex_digits):
                # Valid \uXXXX
                result.append(s[i : i + 6])
                i += 6
            else:
                # Invalid \u (e.g. \unicode) – escape the backslash only
                result.append("\\\\")
                i += 1
        else:
            # Invalid escape (e.g. \p, \alpha) – escape the backslash only
            result.append("\\\\")
            i += 1
    return "".join(result)


def _parse_json_relaxed(raw: str) -> Any:
    def _try_loads(s: str) -> Any:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            fixed = _fix_invalid_escapes(s)
            if fixed != s:
                return json.loads(fixed)
            raise

    try:
        return _try_loads(raw)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        try:
            return _try_loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Try bracket extraction fallback.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return _try_loads(raw[start : end + 1])

    raise ProviderError("Could not parse JSON from LLM response")


def _dedupe_references(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for p in papers:
        doi = str(p.get("doi") or "").strip().lower()
        title = str(p.get("title") or "").strip().lower()
        key = doi or title
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
