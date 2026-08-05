"""Research Agent: multi-source web research via a graph-level tool loop.

Unlike the other agents (which have no tools and so never actually loop), this is the
one agent that calls a tool — web_search — so it's the one place a graph-level
agent_node <-> tools_node conditional-edge loop (see should_continue_research in
src/core/router.py) actually shows a benefit over BaseAgent.invoke()'s in-process
loop: every search becomes its own visible graph step, streamed live to the UI
(see streamlit_app.py's _NODE_INFO / st.status progress panel) instead of being
hidden inside one opaque "research_agent" node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.agents.prompts import RESEARCH_AGENT_PROMPT
from src.core.config import get_settings
from src.integrations.perplexity_client import search_perplexity
from src.integrations.resilience import AllRetriesExhaustedError
from src.integrations.serp_client import search_serp

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState


@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, list[dict]]:
    """Search the web for a query. Tries SERP API first, falling back to
    Perplexity Sonar if SERP is unavailable. Returns a text summary for the
    model plus the structured results as the tool's artifact."""
    try:
        results = search_serp(query)
        provider = "serpapi"
    except AllRetriesExhaustedError:
        try:
            results = search_perplexity(query)
            provider = "perplexity"
        except AllRetriesExhaustedError:
            return "Search failed: both SERP API and Perplexity are unavailable right now.", []

    lines = [f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results]
    return f"Results via {provider}:\n" + "\n".join(lines), results


_TOOLS = [web_search]
_TOOL_MAP = {t.name: t for t in _TOOLS}

# Plain text, not JSON: unlike blog_writer/linkedin_writer/content_strategist,
# research_agent's raw response.content is never JSON-parsed downstream —
# content_strategist reads research_findings (populated by the tool call
# itself), not this message's text.
_STATIC_FALLBACK = (
    "Research is temporarily unavailable due to high demand. Proceeding with "
    "general knowledge only — results may be less current than usual."
)


def _new_agent(provider: str | None) -> BaseAgent:
    return BaseAgent(
        agent_name="research_agent",
        provider=provider,
        temperature=0.3,
        system_prompt=RESEARCH_AGENT_PROMPT,
        tools=_TOOLS,
        debug=False,
        static_fallback=_STATIC_FALLBACK,
    )


async def research_agent_node(state: "AgentState") -> dict:
    """Calls the LLM once. On first invocation (state["research_messages"] empty —
    reset by the orchestrator's Send() payload for every fresh research run, see
    query_handler.py) builds the initial prompt and a clean research_findings/
    research_tool_iterations baseline; on subsequent invocations (after
    research_tools_node ran) continues from the accumulated buffer.

    research_findings and research_tool_iterations are always returned explicitly
    (not just on finalize): both are read by research_tools_node, which is reached
    via a normal conditional edge and so only ever sees the persisted/merged state,
    never the orchestrator's Send() payload — this is what actually anchors a clean
    baseline for research_tools_node to accumulate/increment onto.

    Finalizes (sets research_provider_used/last_agent_used) either when the model
    stops requesting tools, or when research_tool_iterations has already hit
    settings.research_tool_iterations_cap completed round-trips. Also writes
    research_tool_iterations_capped, which should_continue_research (src/core/
    router.py) reads to decide whether the loop continues — the cap value itself
    is resolved here, not in the router, so router.py's conditional-edge functions
    stay pure state reads (see route_after_quality / quality_pipeline_node for the
    same node-owns-settings split)."""
    agent = _new_agent(state.get("llm_provider"))

    existing = state.get("research_messages") or []
    is_fresh_start = not existing
    if is_fresh_start:
        all_msgs = [
            SystemMessage(content=RESEARCH_AGENT_PROMPT),
            HumanMessage(content=state["user_query"]),
        ]
        findings_so_far: list = []
        iterations_so_far = 0
    else:
        all_msgs = list(existing)
        findings_so_far = state.get("research_findings") or []
        iterations_so_far = state.get("research_tool_iterations", 0)

    response = await agent.call_once(all_msgs)
    all_msgs = [*all_msgs, response]

    has_pending_tool_calls = bool(getattr(response, "tool_calls", None))
    capped = iterations_so_far >= get_settings().research_tool_iterations_cap

    update: dict = {
        "research_messages": all_msgs,
        "research_findings": findings_so_far,
        "research_tool_iterations": iterations_so_far,
        "research_tool_iterations_capped": capped,
    }
    if not has_pending_tool_calls or capped:
        update["research_provider_used"] = findings_so_far[0]["source"] if findings_so_far else None
        update["last_agent_used"] = "research_agent"
    return update


async def research_tools_node(state: "AgentState") -> dict:
    """Executes the tool_calls on the last message in research_messages, appending
    each ToolMessage to the buffer and each tool's structured artifact to
    research_findings. Increments research_tool_iterations by one completed
    round-trip — read back by research_agent_node/should_continue_research to
    enforce settings.research_tool_iterations_cap."""
    all_msgs = list(state["research_messages"])
    last_msg = all_msgs[-1]

    findings = list(state.get("research_findings") or [])
    for tool_call in last_msg.tool_calls:
        tool_fn = _TOOL_MAP.get(tool_call["name"])
        if tool_fn is None:
            all_msgs.append(
                ToolMessage(content=f"Unknown tool: {tool_call['name']!r}", tool_call_id=tool_call["id"])
            )
            continue
        result_msg = tool_fn.invoke(tool_call)
        all_msgs.append(result_msg)
        if result_msg.artifact:
            findings.extend(result_msg.artifact)

    iterations = state.get("research_tool_iterations", 0) + 1
    return {"research_messages": all_msgs, "research_findings": findings, "research_tool_iterations": iterations}
