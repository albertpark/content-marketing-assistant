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

_ORCHESTRATOR_ROUTES = {"research", "strategy", "blog", "linkedin", "image", "package", "end"}


def route_after_orchestrator(state: AgentState) -> str:
    """Reads the route the Orchestrator node already decided (state["route"]) and
    returns one of the StateGraph's conditional-edge keys. Defaults to "research"
    (the standard entry point for a brand-new content request) if the Orchestrator
    didn't set a recognized route."""
    route = state.get("route")
    return route if route in _ORCHESTRATOR_ROUTES else "research"


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
