from types import SimpleNamespace

import pytest

from src.agents import base_agent
from src.agents.base_agent import BaseAgent
from src.core import config as config_module
from src.integrations import llm_client
from src.integrations.resilience import AllRetriesExhaustedError

_ALWAYS_ALLOWS = SimpleNamespace(try_acquire=lambda: True)
_ALWAYS_REJECTS = SimpleNamespace(try_acquire=lambda: False)

_ENV_VARS_TO_CLEAR = ("SESSION_STORE_URL", "SESSION_STORE_BACKEND", "SESSION_STORE_PATH")


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-serp-key")
    # Fast, non-sleeping retries: one attempt per provider before moving on.
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setenv("BACKOFF_SECONDS", "0")
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


class _FakeAIMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []


class _AlwaysFailsModel:
    async def ainvoke(self, conversation):
        raise RuntimeError("simulated provider failure")


class _SucceedsModel:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, conversation):
        return _FakeAIMessage(self._content)


def _patch_providers(monkeypatch, behavior: dict):
    """behavior: {provider_name: "fail" | "succeed"}"""

    def fake_get_chat_model(provider, temperature=0.7, tools=None):
        if behavior.get(provider) == "succeed":
            return _SucceedsModel(f"response from {provider}")
        return _AlwaysFailsModel()

    monkeypatch.setattr(llm_client, "get_chat_model", fake_get_chat_model)


@pytest.mark.asyncio
async def test_falls_back_to_next_provider_after_primary_exhausts_retries(monkeypatch):
    _patch_providers(monkeypatch, {"openai": "fail", "anthropic": "succeed"})

    agent = BaseAgent(agent_name="test_agent", system_prompt="test", provider="openai")
    response = await agent.invoke([])

    assert response.content == "response from anthropic"


@pytest.mark.asyncio
async def test_raises_when_every_provider_in_chain_is_exhausted(monkeypatch):
    _patch_providers(monkeypatch, {"openai": "fail", "anthropic": "fail", "gemini": "fail"})

    agent = BaseAgent(agent_name="test_agent", system_prompt="test", provider="openai")

    with pytest.raises(AllRetriesExhaustedError):
        await agent.invoke([])


@pytest.mark.asyncio
async def test_primary_succeeding_never_touches_fallback(monkeypatch):
    calls = []

    def fake_get_chat_model(provider, temperature=0.7, tools=None):
        calls.append(provider)
        return _SucceedsModel(f"response from {provider}")

    monkeypatch.setattr(llm_client, "get_chat_model", fake_get_chat_model)

    agent = BaseAgent(agent_name="test_agent", system_prompt="test", provider="openai")
    response = await agent.invoke([])

    assert response.content == "response from openai"
    assert calls == ["openai"]


@pytest.mark.asyncio
async def test_admission_rejected_serves_cached_response_without_calling_provider(monkeypatch):
    calls = []

    def fake_get_chat_model(provider, temperature=0.7, tools=None):
        calls.append(provider)
        return _SucceedsModel("should never be reached")

    monkeypatch.setattr(llm_client, "get_chat_model", fake_get_chat_model)
    base_agent._RESPONSE_CACHE[base_agent._cache_key("test_agent", [])] = "cached response"

    agent = BaseAgent(
        agent_name="test_agent", system_prompt="test", provider="openai", admission_limiter=_ALWAYS_REJECTS
    )
    response = await agent.invoke([])

    assert response.content == "cached response"
    assert calls == []  # rejected at admission — no provider was ever attempted


@pytest.mark.asyncio
async def test_admission_rejected_with_no_cache_serves_static_fallback(monkeypatch):
    def fake_get_chat_model(provider, temperature=0.7, tools=None):
        return _SucceedsModel("should never be reached")

    monkeypatch.setattr(llm_client, "get_chat_model", fake_get_chat_model)

    agent = BaseAgent(
        agent_name="test_agent",
        system_prompt="test",
        provider="openai",
        admission_limiter=_ALWAYS_REJECTS,
        static_fallback="static fallback text",
    )
    response = await agent.invoke([])

    assert response.content == "static fallback text"


@pytest.mark.asyncio
async def test_all_providers_exhausted_falls_back_to_static_fallback(monkeypatch):
    _patch_providers(monkeypatch, {"openai": "fail", "anthropic": "fail", "gemini": "fail"})

    agent = BaseAgent(
        agent_name="test_agent",
        system_prompt="test",
        provider="openai",
        admission_limiter=_ALWAYS_ALLOWS,
        static_fallback="static fallback text",
    )
    response = await agent.invoke([])

    assert response.content == "static fallback text"


@pytest.mark.asyncio
async def test_raises_when_admission_rejected_and_no_fallback_available(monkeypatch):
    def fake_get_chat_model(provider, temperature=0.7, tools=None):
        return _SucceedsModel("should never be reached")

    monkeypatch.setattr(llm_client, "get_chat_model", fake_get_chat_model)

    agent = BaseAgent(
        agent_name="test_agent", system_prompt="test", provider="openai", admission_limiter=_ALWAYS_REJECTS
    )

    with pytest.raises(AllRetriesExhaustedError):
        await agent.invoke([])
