import pytest

from src.core import config as config_module
from src.integrations import llm_client

_ENV_VARS_TO_CLEAR = (
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "PERPLEXITY_API_KEY",
    "SESSION_STORE_URL",
    "SESSION_STORE_BACKEND",
    "SESSION_STORE_PATH",
)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-serp-key")
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


def test_fallback_chain_defaults_to_primary_then_configured_fallback(monkeypatch):
    # development.yaml: llm.primary_provider=openai, llm.fallback_provider=anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    chain = llm_client.fallback_chain()

    assert chain == ["openai", "anthropic", "gemini"]


def test_fallback_chain_puts_explicit_primary_first(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    chain = llm_client.fallback_chain("gemini")

    assert chain == ["gemini", "anthropic", "openai"]


def test_fallback_chain_dedupes_when_primary_equals_configured_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    chain = llm_client.fallback_chain("anthropic")  # anthropic is also the configured fallback

    assert chain.count("anthropic") == 1
    assert chain == ["anthropic", "openai", "gemini"]


def test_fallback_chain_skips_providers_without_api_keys():
    # ANTHROPIC_API_KEY/GOOGLE_API_KEY cleared by the fixture -> only openai has a key
    chain = llm_client.fallback_chain()

    assert chain == ["openai"]


def test_fallback_chain_raises_when_no_provider_has_a_key(monkeypatch):
    # openai_api_key is a required Settings field (ConfigError if empty), so a
    # real Settings object can never have every provider key empty. Exercise
    # fallback_chain()'s defensive empty-chain branch directly via a stand-in
    # settings object instead of going through the real config loader.
    class _EmptyKeysSettings:
        llm_primary_provider = "openai"
        llm_fallback_provider = "anthropic"
        openai_api_key = ""
        anthropic_api_key = ""
        google_api_key = ""

    monkeypatch.setattr(llm_client, "get_settings", lambda: _EmptyKeysSettings())

    with pytest.raises(ValueError):
        llm_client.fallback_chain()


def test_get_chat_model_raises_on_unknown_provider():
    with pytest.raises(ValueError):
        llm_client.get_chat_model("not-a-real-provider")
