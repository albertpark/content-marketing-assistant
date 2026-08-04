"""LinkedIn Writer: produces a short-form post with a hook and a link back to the blog."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.agents.prompts import LINKEDIN_WRITER_PROMPT

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_linkedin_post(raw: str) -> dict:
    match = _JSON_PATTERN.search(raw or "")
    parsed = {}
    if match:
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            parsed = {}

    text = parsed.get("text", raw or "")
    return {
        "text": text,
        "hashtags": parsed.get("hashtags", []),
        "char_count": len(text),
    }


class LinkedInWriterAgent(BaseAgent):
    def __init__(self, provider: str | None = None, debug: bool = False):
        super().__init__(
            agent_name="linkedin_writer",
            provider=provider,
            temperature=0.7,
            system_prompt=LINKEDIN_WRITER_PROMPT,
            debug=debug,
        )

    async def run(self, state: "AgentState") -> dict:
        blog_post = state.get("blog_post") or {}
        title = blog_post.get("title") or "our latest post"
        slug = blog_post.get("slug", "")
        link = f"/blog/{slug}" if slug else ""
        context = (
            f"Blog title: {title}\n"
            f"Blog link: {link}\n"
            f"Blog summary: {blog_post.get('meta_description', '')}"
        )

        response = await self.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=context)]
        )
        # Note: does NOT set last_agent_used — this node runs concurrently with
        # image_generator (both fan out from blog_writer in the same superstep), and
        # last_agent_used uses LangGraph's default last-value channel, which rejects
        # two writes in the same step (InvalidUpdateError). "Last agent used" is
        # ambiguous for a parallel step anyway.
        return {"linkedin_post": _parse_linkedin_post(response.content)}


async def linkedin_writer_node(state: "AgentState") -> dict:
    return await LinkedInWriterAgent(provider=state.get("llm_provider")).run(state)
