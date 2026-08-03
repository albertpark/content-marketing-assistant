"""LinkedIn Writer: produces a short-form post with a hook and a link back to the blog."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState


def _build_linkedin_post(blog_post: dict) -> dict:
    title = blog_post.get("title") or "our latest post"
    slug = blog_post.get("slug", "")
    link = f"/blog/{slug}" if slug else ""
    text = f"New post: {title}\n\nRead the full breakdown here: {link}".strip()
    hashtags = ["#ContentMarketing", "#AI"]
    return {
        "text": text,
        "hashtags": hashtags,
        "char_count": len(text),
    }


async def linkedin_writer_node(state: "AgentState") -> dict:
    """Milestone 1 stub: a deterministic template off the finished blog post, no
    LLM call yet. Milestone 2 replaces this with a real writer agent that produces
    genuine hook/engagement copy — see docs/hld.md's LinkedIn Writer entry."""
    # Note: does NOT set last_agent_used — this node runs concurrently with
    # image_generator (both fan out from blog_writer in the same superstep), and
    # last_agent_used uses LangGraph's default last-value channel, which rejects
    # two writes in the same step (InvalidUpdateError). "Last agent used" is
    # ambiguous for a parallel step anyway.
    blog_post = state.get("blog_post") or {}
    return {"linkedin_post": _build_linkedin_post(blog_post)}
