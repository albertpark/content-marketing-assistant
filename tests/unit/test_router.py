from src.core.router import route_after_orchestrator, route_after_quality


def test_route_after_orchestrator_known_route():
    assert route_after_orchestrator({"route": "blog"}) == "blog"


def test_route_after_orchestrator_unknown_defaults_to_research():
    assert route_after_orchestrator({"route": "nonsense"}) == "research"
    assert route_after_orchestrator({}) == "research"


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
