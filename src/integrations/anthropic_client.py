"""Anthropic provider client (text generation) — alternative to OpenAI."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src.core.config import get_settings


def get_chat_model(temperature: float = 0.7, tools: list | None = None) -> ChatAnthropic:
    """Returns a configured ChatAnthropic instance using the resolved model/API key."""
    settings = get_settings()
    model = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=temperature,
    )
    return model.bind_tools(tools) if tools else model
