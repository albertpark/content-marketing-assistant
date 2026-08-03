"""Blog Writer: produces the full SEO-optimized blog post; runs first in the pipeline."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_SYSTEM_PROMPT = """You are the Blog Writer for ContentAlchemy. Given a content \
brief, write a full SEO-optimized blog post. Respond with ONLY a JSON object, no \
other text:
{"title": "...", "body_markdown": "...", "meta_description": "...", \
"headers": ["...", "..."]}

- meta_description: 150-160 characters
- body_markdown: the full post body in Markdown, using the headers above as H2 \
  sections, naturally incorporating the brief's target_keywords
- Aim for at least 400 words in body_markdown."""

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _parse_blog_post(raw: str) -> dict:
    match = _JSON_PATTERN.search(raw or "")
    parsed = {}
    if match:
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            parsed = {}

    title = parsed.get("title", "")
    body_markdown = parsed.get("body_markdown", raw or "")
    return {
        "title": title,
        "slug": _slugify(title) if title else "",
        "body_markdown": body_markdown,
        "meta_description": parsed.get("meta_description", ""),
        "headers": parsed.get("headers", []),
        "word_count": len(body_markdown.split()),
    }


class BlogWriterAgent(BaseAgent):
    def __init__(self, provider: str | None = None, debug: bool = False):
        super().__init__(
            agent_name="blog_writer",
            provider=provider,
            temperature=0.6,
            system_prompt=_SYSTEM_PROMPT,
            debug=debug,
        )

    async def run(self, state: "AgentState") -> dict:
        brief = state.get("content_brief") or {}
        context = (
            f"Angle: {brief.get('angle', '')}\n"
            f"Outline: {brief.get('outline', [])}\n"
            f"Key points: {brief.get('key_points', [])}\n"
            f"Target keywords: {brief.get('target_keywords', [])}"
        )
        feedback = state.get("revision_feedback")
        if feedback:
            context += (
                "\n\nThe previous draft failed quality review. Fix these specific "
                f"issues before writing the new draft:\n{feedback}"
            )

        response = await self.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=context)]
        )
        return {
            "blog_post": _parse_blog_post(response.content),
            "last_agent_used": "blog_writer",
        }


async def blog_writer_node(state: "AgentState") -> dict:
    return await BlogWriterAgent(provider=state.get("llm_provider")).run(state)
