"""Image Generator: produces a visual derived from the finished blog post."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from src.integrations.image_clients import generate_image_with_fallback

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState


async def image_generator_node(state: "AgentState") -> dict:
    """Generates a real DALL-E 3 image from the finished blog post via the
    fallback chain in src.integrations.image_clients (degrades to a placeholder
    ImageAsset if generation fails after retries)."""
    blog_post = state.get("blog_post") or {}
    title = blog_post.get("title", "")
    prompt = f"A marketing header image for a blog post titled: {title}" if title else ""

    # Note: does NOT set last_agent_used — see linkedin_writer.py's node for why
    # (this node runs concurrently with it in the same superstep).
    asset = generate_image_with_fallback(prompt, alt_text=title)
    return {"image_assets": [asdict(asset)]}
