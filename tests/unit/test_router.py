from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.router import MAX_RESEARCH_TOOL_ITERATIONS, route_after_quality, should_continue_research


def _tool_call_messages():
    return [
        HumanMessage(content="research this"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}]),
    ]


def test_should_continue_research_routes_to_tools_when_tool_calls_present():
    state = {"research_messages": _tool_call_messages()}
    assert should_continue_research(state) == "research_tools_node"


def test_should_continue_research_routes_to_tools_when_under_cap():
    state = {
        "research_messages": _tool_call_messages(),
        "research_tool_iterations": MAX_RESEARCH_TOOL_ITERATIONS - 1,
    }
    assert should_continue_research(state) == "research_tools_node"


def test_should_continue_research_stops_at_cap_even_with_pending_tool_calls():
    # Regression guard: an uncapped graph-level loop would keep calling
    # research_tools_node forever if the model never stops requesting tools.
    state = {
        "research_messages": _tool_call_messages(),
        "research_tool_iterations": MAX_RESEARCH_TOOL_ITERATIONS,
    }
    assert should_continue_research(state) == "content_strategist"


def test_should_continue_research_routes_to_strategist_when_no_tool_calls():
    state = {
        "research_messages": [
            HumanMessage(content="research this"),
            ToolMessage(content="results", tool_call_id="1"),
            AIMessage(content="Here's what I found."),
        ]
    }
    assert should_continue_research(state) == "content_strategist"


def test_should_continue_research_handles_empty_buffer():
    assert should_continue_research({}) == "content_strategist"
    assert should_continue_research({"research_messages": []}) == "content_strategist"


def test_route_after_quality_pass():
    assert route_after_quality({"quality_report": {"passed": True}}) == "pass"


def test_route_after_quality_revise_when_not_capped():
    state = {"quality_report": {"passed": False, "capped": False}}
    assert route_after_quality(state) == "revise"


def test_route_after_quality_cap_reached_and_state_untouched():
    original_blog = {"title": "Draft"}
    state = {
        "quality_report": {"passed": False, "capped": True, "requires_human_review": True},
        "blog_post": original_blog,
        "revision_count": 1,
    }
    assert route_after_quality(state) == "cap_reached"
    # route_after_quality is a pure read (LangGraph conditional-edge functions
    # can't mutate state) — confirm it didn't touch the draft.
    assert state["blog_post"] is original_blog
