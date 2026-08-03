"""Google Gemini provider client (text generation) — alternative to OpenAI."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import get_settings


def get_chat_model(temperature: float = 0.7, tools: list | None = None) -> ChatGoogleGenerativeAI:
    """Returns a configured ChatGoogleGenerativeAI instance using the resolved model/API key."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
        temperature=temperature,
    )
    return model.bind_tools(tools) if tools else model
