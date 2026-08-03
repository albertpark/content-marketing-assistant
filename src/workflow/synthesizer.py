"""Synthesizer: deterministic assembly of the final cross-linked content package.

No LLM call — hld.md's "reconciled," "links checked, formats matched up" language
describes structural work (blog/LinkedIn link consistency, citation carry-through,
non-empty required fields), not tone/quality judgment, which is the Quality &
Enhancement Pipeline's job. Keeping this a plain function also makes it trivially
unit-testable without mocking an LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState


def synthesize(state: "AgentState") -> dict:
    """Builds the cross-linked content_package from blog_post + linkedin_post +
    image_assets + research_findings."""
    blog_post = state.get("blog_post") or {}
    linkedin_post = state.get("linkedin_post") or {}
    image_assets = state.get("image_assets") or []
    research_findings = state.get("research_findings") or []

    citations = [
        {"title": f["title"], "url": f["url"]} for f in research_findings if f.get("url")
    ]

    package = {
        "blog": {
            "title": blog_post.get("title", ""),
            "slug": blog_post.get("slug", ""),
            "body_markdown": blog_post.get("body_markdown", ""),
            "meta_description": blog_post.get("meta_description", ""),
        },
        "linkedin": {
            "text": linkedin_post.get("text", ""),
            "hashtags": linkedin_post.get("hashtags", []),
        },
        "images": image_assets,
        "citations": citations,
    }
    return {"content_package": package}


async def synthesizer_node(state: "AgentState") -> dict:
    return synthesize(state)
