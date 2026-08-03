"""LinkedIn Writer: produces a short-form post with a hook and a link back to the blog."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_SYSTEM_PROMPT = """You are the LinkedIn Writer for ContentAlchemy. Given a \
finished blog post, write a short-form LinkedIn post that hooks readers \
with a compelling narrative and links back to the blog. Respond with ONLY a \
JSON object, no other text:
{"text": "...", "hashtags": ["...", "..."]}

- text: an attention-grabbing hook as the first line, 3-4 short paragraphs \
  total, ending with a link back to the blog post (use the provided link \
  exactly, verbatim)
- hashtags: 2-4 relevant hashtags, each starting with #
- Keep the whole post under 1300 characters (LinkedIn's practical sweet spot)."""

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
            system_prompt=_SYSTEM_PROMPT,
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
