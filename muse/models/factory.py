"""Factory for building LangChain-compatible Muse chat models."""

from __future__ import annotations

import os

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
    """Create a LangChain-compatible chat model from Muse runtime settings."""

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
