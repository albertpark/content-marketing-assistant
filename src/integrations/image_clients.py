"""Image generation provider clients and the fallback chain (gpt-image-1-mini + fallback)."""

from __future__ import annotations

from dataclasses import dataclass

from src.integrations import openai_client
from src.integrations.resilience import AllRetriesExhaustedError


@dataclass
class ImageAsset:
    url: str | None
    path: str | None
    prompt: str
    provider_used: str
    alt_text: str


def generate_image_with_fallback(prompt: str, alt_text: str = "") -> ImageAsset:
    """Tries the primary image provider (OpenAI gpt-image-1-mini, retry-wrapped). Falls
    back to a placeholder asset if it's exhausted its retries — no secondary image
    provider is configured yet (config/*.yaml's image.fallback_provider is null),
    so this degrades to a placeholder rather than raising, consistent with every
    other fallback chain in this codebase (never crash the graph over a provider
    outage)."""
    if not prompt:
        return ImageAsset(url=None, path=None, prompt=prompt, provider_used="stub", alt_text=alt_text)

    try:
        result = openai_client.generate_image(prompt)
    except AllRetriesExhaustedError:
        return ImageAsset(
            url=None,
            path=None,
            prompt=prompt,
            provider_used="stub",
            alt_text=alt_text or prompt[:120],
        )

    return ImageAsset(
        url=result.get("url"),
        path=result.get("path"),
        prompt=prompt,
        provider_used="openai",
        alt_text=alt_text or prompt[:120],
    )
