from src.integrations import image_clients
from src.integrations.resilience import AllRetriesExhaustedError


def test_generate_image_with_fallback_success_url(monkeypatch):
    monkeypatch.setattr(
        image_clients.openai_client,
        "generate_image",
        lambda prompt, **k: {"url": "https://example.com/img.png", "path": None},
    )

    asset = image_clients.generate_image_with_fallback("a cat riding a bike", alt_text="Cat on a bike")

    assert asset.url == "https://example.com/img.png"
    assert asset.path is None
    assert asset.provider_used == "openai"
    assert asset.alt_text == "Cat on a bike"


def test_generate_image_with_fallback_success_local_path(monkeypatch):
    # gpt-image-1-mini typically returns base64 data (saved locally), not a hosted URL.
    monkeypatch.setattr(
        image_clients.openai_client,
        "generate_image",
        lambda prompt, **k: {"url": None, "path": "/tmp/generated/abc123.png"},
    )

    asset = image_clients.generate_image_with_fallback("a cat riding a bike")

    assert asset.url is None
    assert asset.path == "/tmp/generated/abc123.png"
    assert asset.provider_used == "openai"


def test_generate_image_with_fallback_degrades_to_placeholder_on_failure(monkeypatch):
    def _always_fails(prompt, **k):
        raise AllRetriesExhaustedError("image provider unavailable")

    monkeypatch.setattr(image_clients.openai_client, "generate_image", _always_fails)

    asset = image_clients.generate_image_with_fallback("a cat riding a bike")

    assert asset.url is None
    assert asset.path is None
    assert asset.provider_used == "stub"


def test_generate_image_with_fallback_empty_prompt_skips_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        image_clients.openai_client,
        "generate_image",
        lambda prompt, **k: calls.append(prompt) or {"url": "https://example.com/img.png", "path": None},
    )

    asset = image_clients.generate_image_with_fallback("")

    assert calls == []
    assert asset.provider_used == "stub"
    assert asset.url is None
