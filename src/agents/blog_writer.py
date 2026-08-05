"""Blog Writer: produces the full SEO-optimized blog post; runs first in the pipeline."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.agents.prompts import BLOG_WRITER_PROMPT

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

# Served (see BaseAgent._ainvoke_llm) when the provider fallback chain is
# exhausted and there's no cached draft to fall back to. Deliberately still
# JSON matching _parse_blog_post's schema, so a degraded run still produces a
# valid (if generic) blog_post rather than an empty one — and its short word
# count will legitimately fail the quality gates, correctly routing it to
# human review instead of shipping silently.
_STATIC_FALLBACK = json.dumps(
    {
        "title": "Your post is on its way",
        "body_markdown": (
            "# We're experiencing high demand right now\n\n"
            "This is a placeholder draft — our writing service is temporarily "
            "degraded. Please retry in a few minutes for the full post."
        ),
        "meta_description": "Placeholder draft — full content will be available shortly.",
        "headers": ["We're experiencing high demand right now"],
    }
)


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
            system_prompt=BLOG_WRITER_PROMPT,
            debug=debug,
            static_fallback=_STATIC_FALLBACK,
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
