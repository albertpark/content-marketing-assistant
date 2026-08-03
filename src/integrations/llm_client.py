"""Provider-agnostic LLM dispatch and fallback-chain ordering (OpenAI / Anthropic / Gemini)."""

from __future__ import annotations

from src.core.config import Settings, get_settings
from src.integrations import anthropic_client, gemini_client, openai_client

PROVIDERS = ("openai", "anthropic", "gemini")

_CLIENTS = {
    "openai": openai_client.get_chat_model,
    "anthropic": anthropic_client.get_chat_model,
    "gemini": gemini_client.get_chat_model,
}


def get_chat_model(provider: str, temperature: float = 0.7, tools: list | None = None):
    """Constructs a chat model for the named provider ("openai" | "anthropic" | "gemini")."""
    try:
        client_fn = _CLIENTS[provider]
    except KeyError:
        raise ValueError(f"Unknown LLM provider: {provider!r}") from None
    return client_fn(temperature=temperature, tools=tools)


def _api_key_for(provider: str, settings: Settings) -> str:
    return {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.google_api_key,
    }[provider]


def fallback_chain(primary: str | None = None) -> list[str]:
    """Ordered provider chain: primary -> the configured llm_fallback_provider ->
    any remaining provider, deduplicated, skipping any provider whose API key is
    empty. Raises ValueError if the resulting chain is empty (no provider has a
    usable key)."""
    settings = get_settings()
    primary = primary or settings.llm_primary_provider

    ordered = [primary, settings.llm_fallback_provider, *PROVIDERS]
    seen: set[str] = set()
    chain: list[str] = []
    for provider in ordered:
        if provider in seen or provider not in PROVIDERS:
            continue
        seen.add(provider)
        if _api_key_for(provider, settings):
            chain.append(provider)

    if not chain:
        raise ValueError("No LLM provider has a configured API key")
    return chain
