"""Image Generator: produces a visual guided by the Content Strategist's image brief;
runs as an agent so image style stays consistent. Fans out from Content Strategist in
parallel with Blog Writer (see langgraph_workflow.py), so it does not depend on the
finished blog — only on content_brief.image_brief. When state already has a blog_post
(e.g. an "regenerate only the image" refinement turn on existing content), its title
is passed along too as supporting context, but it's never required."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.agents.prompts import IMAGE_GENERATOR_PROMPT
from src.integrations.image_clients import generate_image_with_fallback

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_image_prompt(raw: str, fallback: str) -> str:
    match = _JSON_PATTERN.search(raw or "")
    if match:
        try:
            prompt = json.loads(match.group()).get("image_prompt", "")
            if prompt:
                return prompt
        except json.JSONDecodeError:
            pass
    return f"A marketing header image for: {fallback}" if fallback else ""


class ImageGeneratorAgent(BaseAgent):
    def __init__(self, provider: str | None = None, debug: bool = False):
        super().__init__(
            agent_name="image_generator",
            provider=provider,
            temperature=0.6,
            system_prompt=IMAGE_GENERATOR_PROMPT,
            debug=debug,
            # _parse_image_prompt already synthesizes a templated prompt from empty content.
            static_fallback="",
        )

    async def run(self, state: "AgentState") -> dict:
        brief = state.get("content_brief") or {}
        image_brief = brief.get("image_brief", "")
        # blog_post is NOT guaranteed here — this node fans out from
        # content_strategist in parallel with blog_writer, so on a fresh run the
        # blog isn't written yet. It's only present on a "regenerate only the
        # image" refinement turn (existing content, single-target Send() — see
        # query_handler.py), where it's still useful supporting context.
        title = (state.get("blog_post") or {}).get("title", "")

        context = f"Image brief: {image_brief}"
        if title:
            context += f"\nBlog title (existing, for supporting context only): {title}"

        response = await self.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=context)]
        )
        image_prompt = _parse_image_prompt(response.content, image_brief or title)

        # Note: does NOT set last_agent_used — this node runs concurrently with
        # blog_writer (both fan out from content_strategist), which already writes
        # it in the same step; last_agent_used's last-value channel rejects two
        # writes in one step (InvalidUpdateError).
        asset = generate_image_with_fallback(image_prompt, alt_text=title)
        return {"image_assets": [asdict(asset)]}


async def image_generator_node(state: "AgentState") -> dict:
    return await ImageGeneratorAgent(provider=state.get("llm_provider")).run(state)
