import pytest

from src.agents.query_handler import (
    OrchestratorAgent,
    OrchestratorDecision,
    _normalize_targets,
    orchestrator_node,
)


def test_normalize_targets_collapses_exclusive_targets():
    # research/blog/package can't run alongside anything else — first match wins.
    assert _normalize_targets(["research", "linkedin"]) == ["research"]
    assert _normalize_targets(["blog"]) == ["blog"]
    assert _normalize_targets(["package", "image"]) == ["package"]


def test_normalize_targets_keeps_parallel_linkedin_and_image():
    assert set(_normalize_targets(["linkedin", "image"])) == {"linkedin", "image"}
    assert _normalize_targets(["linkedin"]) == ["linkedin"]


def test_normalize_targets_falls_back_to_research_when_nothing_recognized():
    assert _normalize_targets([]) == ["research"]


@pytest.mark.asyncio
async def test_orchestrator_node_sends_to_multiple_targets(monkeypatch):
    async def fake_decide(self, state):
        return OrchestratorDecision(intent="refinement", targets=["linkedin", "image"])

    monkeypatch.setattr(OrchestratorAgent, "decide", fake_decide)

    state = {
        "user_query": "refresh the linkedin post and image",
        "blog_post": {"title": "Existing Post"},
        "llm_provider": "openai",
    }
    command = await orchestrator_node(state)

    assert {send.node for send in command.goto} == {"linkedin_writer", "image_generator"}
    assert command.update["route"] == ["linkedin", "image"]
    assert command.update["intent"] == "refinement"
    assert command.update["last_agent_used"] == "orchestrator"


@pytest.mark.asyncio
async def test_orchestrator_node_dispatches_single_target(monkeypatch):
    async def fake_decide(self, state):
        return OrchestratorDecision(intent="refinement", targets=["blog"])

    monkeypatch.setattr(OrchestratorAgent, "decide", fake_decide)

    state = {"user_query": "rewrite the blog", "blog_post": {"title": "Existing Post"}, "llm_provider": "openai"}
    command = await orchestrator_node(state)

    assert len(command.goto) == 1
    assert command.goto[0].node == "blog_writer"


@pytest.mark.asyncio
async def test_orchestrator_node_resets_research_messages_only_when_dispatching_research(monkeypatch):
    async def fake_decide(self, state):
        return OrchestratorDecision(intent="new_content", targets=["research"])

    monkeypatch.setattr(OrchestratorAgent, "decide", fake_decide)

    # A stale, non-empty buffer left over from a prior completed research run.
    state = {"user_query": "write about x", "research_messages": ["stale"], "llm_provider": "openai"}
    command = await orchestrator_node(state)

    assert command.goto[0].node == "research_agent"
    assert command.goto[0].arg["research_messages"] == []


@pytest.mark.asyncio
async def test_orchestrator_node_leaves_research_messages_untouched_for_other_targets(monkeypatch):
    async def fake_decide(self, state):
        return OrchestratorDecision(intent="refinement", targets=["image"])

    monkeypatch.setattr(OrchestratorAgent, "decide", fake_decide)

    state = {
        "user_query": "regenerate the image",
        "blog_post": {"title": "Existing Post"},
        "research_messages": ["finished conversation from an earlier run"],
        "llm_provider": "openai",
    }
    command = await orchestrator_node(state)

    assert command.goto[0].arg["research_messages"] == ["finished conversation from an earlier run"]
