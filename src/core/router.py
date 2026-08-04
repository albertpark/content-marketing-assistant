"""Routing helpers used by the Orchestrator to dispatch requests to agents.

These are LangGraph conditional-edge functions: pure reads of the state produced
by the node that just ran. They decide where to go next — they must not mutate
state (LangGraph conditional-edge callables can't return state updates, only a
routing key). Any state mutation (revision_count, quality_report flags, etc.)
happens inside the node itself (see src/utils/quality_validation.py).
"""

from __future__ import annotations

# A real (non-TYPE_CHECKING) import: LangGraph's add_conditional_edges resolves
# these functions' type hints via get_type_hints() at graph-build time to infer
# each branch's input schema, which fails on an unresolvable forward reference if
# AgentState is only available under TYPE_CHECKING.
from src.workflow.state_management import AgentState

# Note: routing out of the Orchestrator itself is no longer a conditional edge here —
# orchestrator_node (src/agents/query_handler.py) dispatches directly via
# Command(goto=[Send(...)]), since it needs to fan out to a dynamic, possibly
# multi-target, set of agents. What remains here are the two routing decisions that
# genuinely are pure state reads gating a fixed edge.

# Caps the research_agent <-> research_tools_node graph-level tool loop (see
# should_continue_research and research_agent_node in src/agents/research_agent.py).
# This is what hld.md/README document as the Research Agent's "tool loop (bidirectional,
# capped retries)" — mirrors BaseAgent.invoke()'s historical max_tool_iterations=3
# default, now enforced at the graph level since this loop no longer runs through
# BaseAgent.invoke() (it uses call_once() instead, which has no cap of its own).
MAX_RESEARCH_TOOL_ITERATIONS = 3


def should_continue_research(state: AgentState) -> str:
    """Reads the last message in research_messages (see research_agent_node /
    research_tools_node in src/agents/research_agent.py) and decides whether the
    tool loop needs another turn. Capped at MAX_RESEARCH_TOOL_ITERATIONS completed
    research_tools_node round-trips — research_agent_node mirrors this same
    condition to finalize its output once the cap is hit, so the two never
    disagree about whether the loop is over."""
    research_messages = state.get("research_messages") or []
    has_pending_tool_calls = bool(
        research_messages and getattr(research_messages[-1], "tool_calls", None)
    )
    iterations = state.get("research_tool_iterations", 0)
    if has_pending_tool_calls and iterations < MAX_RESEARCH_TOOL_ITERATIONS:
        return "research_tools_node"
    return "content_strategist"


def route_after_quality(state: AgentState) -> str:
    """Reads the quality_report the Quality & Enhancement Pipeline node already
    computed and decides the next step:

    - passed -> "pass" (-> END)
    - failed, not capped -> "revise" (-> blog_writer, revision_count already
      incremented by the node)
    - failed and capped -> "cap_reached" (-> END). The node guarantees the last
      draft (blog_post/linkedin_post/image_assets/content_package) is left
      untouched and quality_report.requires_human_review is True, so the UI blocks
      export until a human explicitly approves it.
    """
    report = state.get("quality_report") or {}
    if report.get("passed"):
        return "pass"
    if report.get("capped"):
        return "cap_reached"
    return "revise"
