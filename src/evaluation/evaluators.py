"""Row-level evaluators for LangSmith experiments run against the
ContentAlchemy graph. Each takes (run, example) and returns a LangSmith
feedback dict: {"key": ..., "score": ..., "comment": ...}."""

from __future__ import annotations

from langsmith.schemas import Example, Run

from src.integrations.openai_client import generate_text
from src.utils.quality_validation import run_gates


def structural_gates(run: Run, example: Example) -> dict:
    """Reuses the exact same deterministic quality gates the production
    quality_pipeline_node runs (src.utils.quality_validation.run_gates), so
    this score reflects precisely what would block a real draft from
    shipping — not a separate, parallel notion of "quality"."""
    content_package = (run.outputs or {}).get("content_package") or {}
    result = run_gates(content_package)
    return {
        "key": "structural_gates",
        "score": 1.0 if result["passed"] else 0.0,
        "comment": "; ".join(result["issues"]) or "all structural gates passed",
    }


_JUDGE_PROMPT = """You are grading a blog post written in response to a content \
request. Score how well the post addresses the request on a 1-5 scale (5 = \
excellent, thorough, on-topic; 1 = off-topic or unusable). Respond with ONLY \
the digit, nothing else.

Request: {query}

Blog post:
{body}
"""


def llm_judge_relevance(run: Run, example: Example) -> dict:
    """LLM-as-judge: does the generated blog post actually address the
    original request? A coarse, subjective signal meant to complement the
    deterministic structural_gates above, not replace it — this model can be
    wrong, and its score is only as good as the judge prompt."""
    query = (example.inputs or {}).get("user_query", "")
    blog_post = (run.outputs or {}).get("blog_post") or {}
    body = blog_post.get("body_markdown", "")
    if not body:
        return {"key": "llm_judge_relevance", "score": 0.0, "comment": "no blog post body produced"}

    raw = generate_text(
        "You are a strict, terse content-quality grader.",
        _JUDGE_PROMPT.format(query=query, body=body[:6000]),
        temperature=0,
    )
    digits = [c for c in raw if c.isdigit()]
    score = int(digits[0]) if digits else 0
    return {"key": "llm_judge_relevance", "score": score / 5, "comment": raw.strip()}


_HALLUCINATION_PROMPT = """You are fact-checking a blog post against the research \
findings it was supposed to be grounded in. Score whether the post's factual \
claims (statistics, quotes, sources, specific details) are actually supported \
by the research, on a 1-5 scale:

5 = every claim is grounded in the research (or is clearly general knowledge, not a fabricated specific)
4 = minor embellishment but the core facts check out
3 = some claims aren't traceable to the research
2 = significant fabrication of facts, statistics, or sources
1 = mostly fabricated content

Respond with ONLY the digit, nothing else.

Research findings:
{research_summary}

Blog post:
{body}
"""


def llm_judge_hallucination(run: Run, example: Example) -> dict:
    """LLM-as-judge: is the post grounded in research_findings, or fabricated?
    Requires research_findings in the run's outputs (see run_evaluation.py)."""
    blog_post = (run.outputs or {}).get("blog_post") or {}
    body = blog_post.get("body_markdown", "")
    if not body:
        return {"key": "llm_judge_hallucination", "score": 0.0, "comment": "no blog post body produced"}

    findings = (run.outputs or {}).get("research_findings") or []
    research_summary = (
        "\n".join(f"- {f.get('title', '')}: {f.get('snippet', '')}" for f in findings)
        or "No research findings were available for this run."
    )

    raw = generate_text(
        "You are a strict, terse fact-checking grader.",
        _HALLUCINATION_PROMPT.format(research_summary=research_summary[:4000], body=body[:6000]),
        temperature=0,
    )
    digits = [c for c in raw if c.isdigit()]
    score = int(digits[0]) if digits else 0
    return {"key": "llm_judge_hallucination", "score": score / 5, "comment": raw.strip()}


_COMPLETENESS_PROMPT = """Score how completely this blog post addresses every aspect \
of the original content request, on a 1-5 scale:

5 = addresses every aspect of the request with specific, actionable detail
4 = covers most aspects, one minor gap
3 = covers roughly half of what was asked
2 = major gaps — only partially addresses the request
1 = does not address the request at all

Respond with ONLY the digit, nothing else.

Request: {query}

Blog post:
{body}
"""


def llm_judge_completeness(run: Run, example: Example) -> dict:
    """LLM-as-judge: does the post cover everything the request asked for?"""
    query = (example.inputs or {}).get("user_query", "")
    blog_post = (run.outputs or {}).get("blog_post") or {}
    body = blog_post.get("body_markdown", "")
    if not body:
        return {"key": "llm_judge_completeness", "score": 0.0, "comment": "no blog post body produced"}

    raw = generate_text(
        "You are a strict, terse content-quality grader.",
        _COMPLETENESS_PROMPT.format(query=query, body=body[:6000]),
        temperature=0,
    )
    digits = [c for c in raw if c.isdigit()]
    score = int(digits[0]) if digits else 0
    return {"key": "llm_judge_completeness", "score": score / 5, "comment": raw.strip()}
