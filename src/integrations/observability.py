"""Single point of contact for LangSmith instrumentation.

Every outbound provider call that isn't already a LangChain object (and thus
already auto-traced/auto-priced by LangSmith) should be wrapped with
@traced_tool instead of importing `langsmith` directly, so there's exactly one
place that knows how ContentAlchemy talks to its observability backend.

Cost note: LangSmith computes $ cost automatically for LangChain chat-model
runs from token usage against its own maintained pricing tables — nothing to
do there. It has no way to price a raw REST call (image generation, SerpApi,
Perplexity), so those get a manual `cost_usd` attached via record_cost(),
using rates you supply in _MANUAL_PRICING below. Verify those rates against
your actual provider billing pages before trusting the numbers — they are
not fetched from anywhere authoritative.
"""

from __future__ import annotations

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# Fill in with rates from your provider billing pages. Left at 0.0 (no cost
# attached) until you do — a wrong guess here is worse than an honest gap.
_MANUAL_PRICING: dict[str, float] = {
    "openai_generate_image": 0.0,  # $ per image, e.g. gpt-image-1-mini at your chosen size/quality
    "serpapi_search": 0.0,  # $ per search call, per your SerpApi plan
    "perplexity_search": 0.0,  # $ per Sonar request, per your Perplexity plan
}


def traced_tool(name: str):
    """Decorator for a non-LangChain outbound call: traces it as a LangSmith
    "tool" run under whatever parent run is active (LangGraph node, etc.)."""
    return traceable(run_type="tool", name=name)


def record_cost(name: str, usd: float | None = None, **extra_metadata) -> None:
    """Attaches a manual $ figure to the currently-active traced run's metadata
    (visible in the LangSmith UI under that run's Metadata tab). Silently does
    nothing if tracing is off or called outside a traced call — this is a
    best-effort annotation, never a control-flow dependency.

    Pass `usd` explicitly to override _MANUAL_PRICING (e.g. a per-call price
    that depends on image size); otherwise falls back to the static rate for
    `name`, if one has been filled in.
    """
    cost = usd if usd is not None else _MANUAL_PRICING.get(name)
    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    metadata = {**extra_metadata}
    if cost is not None and cost > 0:
        metadata["cost_usd"] = cost
    elif cost is None or cost == 0:
        metadata["cost_usd_note"] = f"no rate configured for {name!r} in observability._MANUAL_PRICING"
    if metadata:
        run_tree.add_metadata(metadata)
