"""Image Generator: produces a visual guided by the Content Strategist's image brief
and the finished blog post; runs as an agent so image style stays consistent."""

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


def _parse_image_prompt(raw: str, fallback_title: str) -> str:
    match = _JSON_PATTERN.search(raw or "")
    if match:
        try:
            prompt = json.loads(match.group()).get("image_prompt", "")
            if prompt:
                return prompt
        except json.JSONDecodeError:
            pass
    return f"A marketing header image for a blog post titled: {fallback_title}" if fallback_title else ""


class ImageGeneratorAgent(BaseAgent):
    def __init__(self, provider: str | None = None, debug: bool = False):
        super().__init__(
            agent_name="image_generator",
            provider=provider,
            temperature=0.6,
            system_prompt=IMAGE_GENERATOR_PROMPT,
            debug=debug,
        )

    async def run(self, state: "AgentState") -> dict:
        brief = state.get("content_brief") or {}
        blog_post = state.get("blog_post") or {}
        title = blog_post.get("title", "")
        context = f"Image brief: {brief.get('image_brief', '')}\nBlog title: {title}"

        response = await self.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=context)]
        )
        image_prompt = _parse_image_prompt(response.content, title)

        # Note: does NOT set last_agent_used — see linkedin_writer.py's node for why
        # (this node runs concurrently with it in the same superstep).
        asset = generate_image_with_fallback(image_prompt, alt_text=title)
        return {"image_assets": [asdict(asset)]}


async def image_generator_node(state: "AgentState") -> dict:
    return await ImageGeneratorAgent(provider=state.get("llm_provider")).run(state)
