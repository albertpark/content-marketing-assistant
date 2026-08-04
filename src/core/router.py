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

def should_continue_research(state: AgentState) -> str:
    """Reads the last message in research_messages (see research_agent_node /
    research_tools_node in src/agents/research_agent.py) and decides whether the
    tool loop needs another turn.

    The cap itself — settings.research_tool_iterations_cap (config/services.yaml,
    override via RESEARCH_TOOL_ITERATIONS_CAP) — is resolved and compared by
    research_agent_node, not here, matching this module's rule that node functions
    own settings/config reads and conditional-edge functions stay pure state reads
    (see route_after_quality / quality_pipeline_node for the same split). This is
    what hld.md/README document as the Research Agent's "tool loop (bidirectional,
    capped retries)", now enforced at the graph level since this loop runs via
    call_once() rather than BaseAgent.invoke() (which had its own, uncapped-here,
    max_tool_iterations)."""
    research_messages = state.get("research_messages") or []
    has_pending_tool_calls = bool(
        research_messages and getattr(research_messages[-1], "tool_calls", None)
    )
    if has_pending_tool_calls and not state.get("research_tool_iterations_capped", False):
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
