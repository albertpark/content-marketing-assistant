from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from src.agents.research_agent import research_agent_node

_TEST_CAP = 3


def _patch_call_once(monkeypatch, response: AIMessage):
    async def fake_call_once(self, conversation):
        return response

    monkeypatch.setattr("src.agents.research_agent.BaseAgent.call_once", fake_call_once)


def _patch_settings(monkeypatch, cap: int = _TEST_CAP):
    # Decouples this test's cap from config/services.yaml's real default, and from
    # whatever RESEARCH_TOOL_ITERATIONS_CAP happens to be set in the environment —
    # research_agent_node resolves the cap via get_settings(), same as
    # quality_pipeline_node resolves revision_loop_cap.
    monkeypatch.setattr(
        "src.agents.research_agent.get_settings",
        lambda: SimpleNamespace(research_tool_iterations_cap=cap),
    )


@pytest.mark.asyncio
async def test_fresh_start_anchors_iterations_to_zero(monkeypatch):
    response = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}])
    _patch_call_once(monkeypatch, response)
    _patch_settings(monkeypatch)

    state = {"user_query": "topic", "research_messages": [], "llm_provider": "openai"}
    update = await research_agent_node(state)

    assert update["research_tool_iterations"] == 0
    # Still under the cap, so the loop isn't finalized yet.
    assert update["research_tool_iterations_capped"] is False
    assert "last_agent_used" not in update


@pytest.mark.asyncio
async def test_continues_without_finalizing_when_under_cap(monkeypatch):
    response = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}])
    _patch_call_once(monkeypatch, response)
    _patch_settings(monkeypatch)

    state = {
        "user_query": "topic",
        "research_messages": [AIMessage(content="prior turn")],
        "research_tool_iterations": _TEST_CAP - 1,
        "llm_provider": "openai",
    }
    update = await research_agent_node(state)

    assert update["research_tool_iterations"] == _TEST_CAP - 1
    assert update["research_tool_iterations_capped"] is False
    assert "last_agent_used" not in update
    assert "research_provider_used" not in update


@pytest.mark.asyncio
async def test_finalizes_when_cap_reached_even_with_pending_tool_calls(monkeypatch):
    # Regression guard: without this, the graph loop never terminates once the
    # model keeps requesting tools past the configured cap.
    response = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}])
    _patch_call_once(monkeypatch, response)
    _patch_settings(monkeypatch)

    state = {
        "user_query": "topic",
        "research_messages": [AIMessage(content="prior turn")],
        "research_findings": [{"source": "serpapi", "title": "t", "url": "u", "snippet": "s"}],
        "research_tool_iterations": _TEST_CAP,
        "llm_provider": "openai",
    }
    update = await research_agent_node(state)

    assert update["research_tool_iterations_capped"] is True
    assert update["last_agent_used"] == "research_agent"
    assert update["research_provider_used"] == "serpapi"


@pytest.mark.asyncio
async def test_respects_a_lower_configured_cap(monkeypatch):
    # Proves the cap is actually read from settings, not hardcoded — the point of
    # this fix (PR review: "Max research tool should be configurable").
    response = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}])
    _patch_call_once(monkeypatch, response)
    _patch_settings(monkeypatch, cap=1)

    state = {
        "user_query": "topic",
        "research_messages": [AIMessage(content="prior turn")],
        "research_tool_iterations": 1,
        "llm_provider": "openai",
    }
    update = await research_agent_node(state)

    assert update["research_tool_iterations_capped"] is True
    assert update["last_agent_used"] == "research_agent"


@pytest.mark.asyncio
async def test_finalizes_normally_when_model_stops_requesting_tools(monkeypatch):
    response = AIMessage(content="Here's what I found.")
    _patch_call_once(monkeypatch, response)
    _patch_settings(monkeypatch)

    state = {
        "user_query": "topic",
        "research_messages": [AIMessage(content="prior turn")],
        "research_findings": [{"source": "perplexity", "title": "t", "url": "u", "snippet": "s"}],
        "research_tool_iterations": 1,
        "llm_provider": "openai",
    }
    update = await research_agent_node(state)

    assert update["last_agent_used"] == "research_agent"
    assert update["research_provider_used"] == "perplexity"
    assert update["research_tool_iterations"] == 1
    assert update["research_tool_iterations_capped"] is False
