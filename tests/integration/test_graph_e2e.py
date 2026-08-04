import json

import pytest
from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.agents.query_handler import OrchestratorDecision
from src.core import config as config_module
from src.workflow.langgraph_workflow import build_graph
from src.workflow.state_management import initial_state


class _FakeAIMessage(AIMessage):
    # A real AIMessage subclass (not a bare stand-in) — research_agent_node's
    # response now lands in the checkpointed research_messages field, which the
    # checkpointer serializes after every step, so it must be a real, serializable
    # LangChain message, not just something that quacks like one.
    def __init__(self, content, tool_calls=None):
        super().__init__(content=content, tool_calls=tool_calls or [])


class _FakeStructuredModel:
    """Stands in for llm.with_structured_output(schema): returns the canned
    Pydantic instance directly, same as a real structured-output call would."""

    def __init__(self, decision):
        self._decision = decision

    async def ainvoke(self, _messages):
        return self._decision

    def invoke(self, _messages):
        return self._decision


class _FakeChatModel:
    """Stands in for ChatOpenAI: same bind_tools/with_structured_output/(a)invoke
    surface, canned content. `content` is a plain string for every agent except the
    orchestrator, which only ever calls with_structured_output — there `content` is
    an OrchestratorDecision instance (see ORCHESTRATOR_DECISION below)."""

    def __init__(self, content):
        self._content = content

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):
        return _FakeStructuredModel(self._content)

    async def ainvoke(self, _messages):
        return _FakeAIMessage(self._content)

    def invoke(self, _messages):
        return _FakeAIMessage(self._content)


ORCHESTRATOR_DECISION = OrchestratorDecision(intent="new_content", targets=["research"])
RESEARCH_RESPONSE = "Research summary: this topic has strong market interest."
STRATEGIST_RESPONSE = json.dumps(
    {
        "angle": "beginner guide",
        "outline": ["intro", "body", "conclusion"],
        "key_points": ["point one", "point two"],
        "target_keywords": ["test topic"],
    }
)
BLOG_RESPONSE_GOOD = json.dumps(
    {
        "title": "Test Topic: A Complete Guide",
        "body_markdown": "# Intro\n\n" + ("word " * 60),
        "meta_description": "A short meta description for the test blog post.",
        "headers": ["Intro"],
    }
)
BLOG_RESPONSE_BAD = json.dumps(
    {"title": "", "body_markdown": "too short", "meta_description": "", "headers": []}
)
LINKEDIN_RESPONSE = json.dumps(
    {"text": "Big news! Check out our new guide.\n\nRead it here: /blog/test-topic", "hashtags": ["#AI"]}
)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-serp-key")
    # development.yaml's default backend is postgres (Supabase) — never let
    # tests touch the real database. build_graph() with no explicit
    # checkpointer falls back to get_checkpointer(get_settings()), which only
    # supports "memory", so forcing that here keeps these tests fully local.
    monkeypatch.setenv("SESSION_STORE_BACKEND", "memory")
    monkeypatch.delenv("REVISION_LOOP_CAP", raising=False)
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


def _patch_agent_llms(monkeypatch, blog_response: str):
    # BaseAgent now dispatches LLM construction through a provider fallback
    # chain (src.integrations.llm_client) rather than each agent importing
    # openai_client.get_chat_model directly, so the single seam to patch is
    # BaseAgent._llm_for — keyed by agent_name, the real identity signal
    # (unlike temperature, which only incidentally differs per agent today).
    responses = {
        "orchestrator": ORCHESTRATOR_DECISION,
        "research_agent": RESEARCH_RESPONSE,
        "content_strategist": STRATEGIST_RESPONSE,
        "blog_writer": blog_response,
        "linkedin_writer": LINKEDIN_RESPONSE,
    }
    monkeypatch.setattr(
        BaseAgent, "_llm_for", lambda self, provider: _FakeChatModel(responses[self.agent_name])
    )
    # Never hit the real image-generation API (or its real retry/backoff delays) in tests.
    monkeypatch.setattr(
        "src.integrations.image_clients.openai_client.generate_image",
        lambda prompt, **k: {"url": "https://example.com/fake-image.png", "path": None},
    )


@pytest.mark.asyncio
async def test_graph_e2e_passes_quality_gate(monkeypatch):
    _patch_agent_llms(monkeypatch, BLOG_RESPONSE_GOOD)

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-thread-pass"}}
    state = initial_state("test-thread-pass", "Write about test topic")

    result = await graph.ainvoke(state, config=config)

    assert result["blog_post"]["title"] == "Test Topic: A Complete Guide"
    assert result["content_package"]["blog"]["title"] == "Test Topic: A Complete Guide"
    assert result["content_package"]["linkedin"]["text"]
    assert result["quality_report"]["passed"] is True
    assert result["revision_count"] == 0


@pytest.mark.asyncio
async def test_graph_e2e_cap_reached_preserves_draft_and_requires_review(monkeypatch):
    _patch_agent_llms(monkeypatch, BLOG_RESPONSE_BAD)

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-thread-cap"}}
    state = initial_state("test-thread-cap", "Write about test topic")

    result = await graph.ainvoke(state, config=config)

    # revision_loop_cap defaults to 1: exactly one revise pass, then hard stop.
    assert result["revision_count"] == 1
    assert result["quality_report"]["passed"] is False
    assert result["quality_report"]["capped"] is True
    assert result["quality_report"]["requires_human_review"] is True
    assert result["human_approved_override"] is False

    # The last (failing) draft must survive for human review, not be wiped.
    assert result["blog_post"] is not None
    assert result["content_package"] is not None
