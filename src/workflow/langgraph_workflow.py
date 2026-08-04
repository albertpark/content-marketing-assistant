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
    docs/hld.md's data flow: Orchestrator -> Research -> Strategist -> Blog Writer
    -> {LinkedIn Writer, Image Generator} -> Synthesizer -> Quality Pipeline, with
    a conditional revision loop back to Blog Writer.

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
    workflow.add_node("synthesizer", synthesizer_node)
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

    workflow.add_edge("content_strategist", "blog_writer")

    # Fan-out: both edges originate from blog_writer, so LangGraph's Pregel
    # scheduler runs linkedin_writer and image_generator concurrently in the same
    # superstep.
    workflow.add_edge("blog_writer", "linkedin_writer")
    workflow.add_edge("blog_writer", "image_generator")

    # Fan-in: synthesizer has incoming edges from both and only runs once both have
    # completed. Safe because they write disjoint state keys (linkedin_post vs.
    # image_assets) — see AgentState's docstring note on reducers.
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
