"""Runs the ContentAlchemy pipeline against the eval dataset and scores each
result in LangSmith (structural gates + LLM-judged relevance). Requires
LANGSMITH_TRACING=true and LANGSMITH_API_KEY set — see .env.

Usage:
    uv run python scripts/run_evaluation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Running this file directly (`python scripts/run_evaluation.py`) puts
# scripts/ on sys.path, not the project root — `import src...` needs the
# latter, so add it explicitly rather than requiring `-m` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langsmith.evaluation import aevaluate  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.evaluation.dataset import sync_dataset  # noqa: E402
from src.evaluation.evaluators import (  # noqa: E402
    llm_judge_completeness,
    llm_judge_hallucination,
    llm_judge_relevance,
    structural_gates,
)
from src.workflow.langgraph_workflow import build_graph  # noqa: E402
from src.workflow.state_management import initial_state, new_session_id, open_checkpointer  # noqa: E402


async def _run_pipeline(user_query: str) -> dict:
    settings = get_settings()
    session_id = new_session_id()
    state = initial_state(session_id, user_query)
    config = {"configurable": {"thread_id": session_id}}

    if settings.session_store_backend in ("sqlite", "postgres"):
        async with open_checkpointer(settings) as saver:
            graph = build_graph(checkpointer=saver)
            final_state = await graph.ainvoke(state, config=config)
    else:
        graph = build_graph()
        final_state = await graph.ainvoke(state, config=config)

    return {
        "content_package": final_state.get("content_package"),
        "blog_post": final_state.get("blog_post"),
        "quality_report": final_state.get("quality_report"),
        "research_findings": final_state.get("research_findings"),
    }


async def _target(inputs: dict) -> dict:
    return await _run_pipeline(inputs["user_query"])


async def main() -> None:
    get_settings()  # loads .env (LANGCHAIN_API_KEY etc.) before any LangSmith client is built
    dataset_name = sync_dataset()
    results = await aevaluate(
        _target,
        data=dataset_name,
        evaluators=[structural_gates, llm_judge_relevance, llm_judge_hallucination, llm_judge_completeness],
        experiment_prefix="contentalchemy",
        description=(
            "End-to-end ContentAlchemy pipeline eval: deterministic structural "
            "gates + LLM-judged relevance, hallucination/groundedness, and "
            "completeness against the original request."
        ),
        max_concurrency=2,
    )
    print(f"\nView results: {results.experiment_url if hasattr(results, 'experiment_url') else results}")


if __name__ == "__main__":
    # SelectorEventLoop, not asyncio.run()'s Windows default ProactorEventLoop —
    # required by psycopg's async mode (the postgres backend). See
    # streamlit_app.py's _run_async for the same constraint.
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
