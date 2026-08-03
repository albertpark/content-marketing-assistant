"""Research Agent: multi-source web research via the Search tools (SERP + Perplexity)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.integrations.perplexity_client import search_perplexity
from src.integrations.resilience import AllRetriesExhaustedError
from src.integrations.serp_client import search_serp

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_SYSTEM_PROMPT = """You are the Research Agent for ContentAlchemy. Use the \
web_search tool to gather comprehensive, accurate information on the user's topic \
from multiple angles before answering. Once you have enough coverage, respond with \
a concise research summary (a few paragraphs) synthesizing what you found, citing \
sources by URL."""


def _make_web_search_tool(collected: list[dict]):
    """Builds a web_search tool bound to a per-run `collected` accumulator (via
    closure) instead of module-level state, so concurrent/repeated agent runs never
    share results with each other."""

    @tool
    def web_search(query: str) -> str:
        """Search the web for a query. Tries SERP API first, falling back to
        Perplexity Sonar if SERP is unavailable. Returns a text summary of results
        for the model to read."""
        try:
            results = search_serp(query)
            provider = "serpapi"
        except AllRetriesExhaustedError:
            try:
                results = search_perplexity(query)
                provider = "perplexity"
            except AllRetriesExhaustedError:
                return "Search failed: both SERP API and Perplexity are unavailable right now."

        collected.extend(results)
        lines = [f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results]
        return f"Results via {provider}:\n" + "\n".join(lines)

    return web_search


class ResearchAgent(BaseAgent):
    def __init__(self, provider: str | None = None, debug: bool = False):
        self._collected: list[dict] = []
        super().__init__(
            agent_name="research_agent",
            provider=provider,
            temperature=0.3,
            system_prompt=_SYSTEM_PROMPT,
            tools=[_make_web_search_tool(self._collected)],
            max_tool_iterations=3,
            debug=debug,
        )

    async def run(self, state: "AgentState") -> dict:
        self._collected.clear()
        await self.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=state["user_query"])]
        )
        provider_used = self._collected[0]["source"] if self._collected else None
        return {
            "research_findings": list(self._collected),
            "research_provider_used": provider_used,
            "last_agent_used": "research_agent",
        }


async def research_agent_node(state: "AgentState") -> dict:
    return await ResearchAgent(provider=state.get("llm_provider")).run(state)
