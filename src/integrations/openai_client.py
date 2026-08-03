"""OpenAI provider client (text + image generation)."""

from __future__ import annotations

import base64
import uuid
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from src.core.config import get_settings
from src.integrations.observability import record_cost, traced_tool
from src.integrations.performance import rate_limited
from src.integrations.resilience import ProviderError, with_retry

_IMAGE_STORAGE_DIR = Path(__file__).resolve().parents[2] / "src" / "web_app" / "static" / "generated"

_RETRYABLE_ERRORS = (APIConnectionError, APIError, RateLimitError)


def get_chat_model(temperature: float = 0.7, tools: list | None = None) -> ChatOpenAI:
    """Returns a configured ChatOpenAI instance using the resolved model/API key."""
    settings = get_settings()
    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )
    return model.bind_tools(tools) if tools else model


@with_retry(retry_on=(ProviderError,))
@rate_limited()
def invoke_chat(
    messages: list[BaseMessage],
    temperature: float = 0.7,
    tools: list | None = None,
):
    """Single chat-completion call. Raises ProviderError on transient failures so
    with_retry can retry it."""
    try:
        model = get_chat_model(temperature=temperature, tools=tools)
        return model.invoke(messages)
    except _RETRYABLE_ERRORS as exc:
        raise ProviderError(str(exc)) from exc


def generate_text(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Convenience helper for single-turn, tool-free text generation."""
    response = invoke_chat(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        temperature=temperature,
    )
    return response.content


@lru_cache
def _get_image_client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


@traced_tool("openai_generate_image")
@with_retry(retry_on=(ProviderError,))
@rate_limited()
def generate_image(prompt: str, size: str = "1024x1024", quality: str = "auto") -> dict:
    """Generates an image via OpenAI's gpt-image-1-mini — the cheapest current
    image model (dall-e-3/dall-e-2 are discontinued; "model does not exist" for
    this account). Returns {"url": ...} if the API gives a hosted URL, or
    {"path": ...} to a locally-saved file if it returns base64 data instead
    (gpt-image-1 family typically returns b64_json, not a url, unlike the
    discontinued DALL-E endpoints) — callers should check whichever key is
    non-None. Raises ProviderError on transient failures so with_retry can retry
    it."""
    try:
        response = _get_image_client().images.generate(
            model="gpt-image-1-mini",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
    except _RETRYABLE_ERRORS as exc:
        raise ProviderError(str(exc)) from exc

    record_cost("openai_generate_image", size=size, quality=quality)

    item = response.data[0]
    if getattr(item, "url", None):
        return {"url": item.url, "path": None}

    _IMAGE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _IMAGE_STORAGE_DIR / f"{uuid.uuid4().hex}.png"
    filepath.write_bytes(base64.b64decode(item.b64_json))
    return {"url": None, "path": str(filepath)}
