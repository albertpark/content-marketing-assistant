"""LangGraph StateGraph assembly: nodes, fixed/conditional edges, and the revision loop."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.blog_writer import blog_writer_node
from src.agents.content_strategist import content_strategist_node
from src.agents.image_generator import image_generator_node
from src.agents.linkedin_writer import linkedin_writer_node
from src.agents.query_handler import orchestrator_node
from src.agents.research_agent import research_agent_node, research_tools_node
from src.core.config import get_settings
from src.core.router import route_after_quality, should_continue_research
from src.utils.quality_validation import quality_pipeline_node
from src.workflow.state_management import AgentState, get_checkpointer
from src.workflow.synthesizer import synthesizer_node


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Assembles and compiles the full ContentAlchemy pipeline graph, matching
    docs/hld.md's data flow: Orchestrator -> Research -> Strategist -> {Blog Writer,
    Image Generator} (parallel) -> Blog Writer -> LinkedIn Writer -> Synthesizer
    (joined with Image Generator) -> Quality Pipeline, with a conditional revision
    loop back to Blog Writer.

    checkpointer: pass an explicit one for backends that require a
    freshly-opened, per-call checkpointer (sqlite/postgres — see
    state_management.open_checkpointer()). When omitted, falls back to
    get_checkpointer(get_settings()), which only supports the "memory"
    backend — the right default for existing callers/tests."""
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("research_tools_node", research_tools_node)
    workflow.add_node("content_strategist", content_strategist_node)
    workflow.add_node("blog_writer", blog_writer_node)
    workflow.add_node("linkedin_writer", linkedin_writer_node)
    workflow.add_node("image_generator", image_generator_node)
    # defer=True: synthesizer's two incoming branches (linkedin_writer, via
    # blog_writer — 2 hops from content_strategist; image_generator — 1 hop) are no
    # longer the same length now that image_generator fans out directly from
    # content_strategist. LangGraph's default fan-in fires as soon as any incoming
    # edge delivers a value, not once all of them have — that only happens to look
    # like a wait-for-all join when every branch is the same length. defer=True is
    # LangGraph's built-in fix: it holds this node's execution until nothing else
    # is left pending in the run, which is what actually guarantees both branches
    # (and, transitively, blog_writer) have completed first.
    workflow.add_node("synthesizer", synthesizer_node, defer=True)
    workflow.add_node("quality_pipeline", quality_pipeline_node)

    workflow.add_edge(START, "orchestrator")
    # orchestrator_node dispatches to one or more downstream nodes itself, via
    # Command(goto=[Send(...)]) — see src/agents/query_handler.py. No outgoing
    # conditional-edge map is needed here.

    # research_agent <-> research_tools_node: graph-level tool loop (see
    # should_continue_research), replacing the in-process loop every other agent
    # still uses via BaseAgent.invoke().
    workflow.add_conditional_edges(
        "research_agent",
        should_continue_research,
        {"research_tools_node": "research_tools_node", "content_strategist": "content_strategist"},
    )
    workflow.add_edge("research_tools_node", "research_agent")

    # Fan-out: both edges originate from content_strategist, so blog_writer and
    # image_generator run concurrently — image_generator no longer waits on the
    # finished blog, it works from content_brief.image_brief instead (see
    # src/agents/image_generator.py).
    workflow.add_edge("content_strategist", "blog_writer")
    workflow.add_edge("content_strategist", "image_generator")

    workflow.add_edge("blog_writer", "linkedin_writer")

    # Fan-in: synthesizer has incoming edges from both linkedin_writer and
    # image_generator. Since those two branches are different lengths (image_generator
    # is 1 hop from content_strategist; linkedin_writer is 2, via blog_writer),
    # synthesizer needs defer=True (set on add_node above) to actually wait for both —
    # a plain fixed-edge join only reliably waits for all predecessors when every
    # branch is the same length; otherwise it fires as soon as the first one arrives.
    # Safe once deferred, because the two branches write disjoint state keys
    # (linkedin_post vs. image_assets) — see AgentState's docstring note on reducers.
    workflow.add_edge("linkedin_writer", "synthesizer")
    workflow.add_edge("image_generator", "synthesizer")

    workflow.add_edge("synthesizer", "quality_pipeline")
    workflow.add_conditional_edges(
        "quality_pipeline",
        route_after_quality,
        {"revise": "blog_writer", "pass": END, "cap_reached": END},
    )

    if checkpointer is None:
        checkpointer = get_checkpointer(get_settings())
    return workflow.compile(checkpointer=checkpointer)
