"""Orchestrator: classifies user intent and dispatches to the appropriate agent."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_SYSTEM_PROMPT = """You are the Orchestrator for ContentAlchemy, a content-marketing \
assistant. Given the user's latest message and whether a blog/LinkedIn post/image \
already exist for this session, decide:

- intent: "new_content" (start a fresh research-to-content run) or "refinement" \
  (the user wants to change something that already exists)
- route: exactly one of "research", "blog", "linkedin", "image", "package"
  - "research": start a brand-new content run from scratch
  - "blog": regenerate the blog post (and downstream LinkedIn post/image) from the \
    existing brief/research
  - "linkedin": regenerate only the LinkedIn post from the existing blog
  - "image": regenerate only the image from the existing blog
  - "package": just re-assemble the existing blog/LinkedIn/image into a package

Respond with ONLY a JSON object, no other text: {"intent": "...", "route": "..."}"""

_VALID_ROUTES = {"research", "blog", "linkedin", "image", "package"}
_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_decision(raw: str) -> dict:
    match = _JSON_PATTERN.search(raw or "")
    if not match:
        return {"intent": "new_content", "route": "research"}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {"intent": "new_content", "route": "research"}
    route = parsed.get("route")
    if route not in _VALID_ROUTES:
        route = "research"
    return {"intent": parsed.get("intent", "new_content"), "route": route}


class OrchestratorAgent(BaseAgent):
    def __init__(self, provider: str | None = None, debug: bool = False):
        super().__init__(
            agent_name="orchestrator",
            provider=provider,
            temperature=0,
            system_prompt=_SYSTEM_PROMPT,
            debug=debug,
        )

    async def run(self, state: "AgentState") -> dict:
        has_blog = state.get("blog_post") is not None
        has_linkedin = state.get("linkedin_post") is not None
        has_image = bool(state.get("image_assets"))

        context = (
            f"User message: {state['user_query']}\n"
            f"Existing blog post: {has_blog}\n"
            f"Existing LinkedIn post: {has_linkedin}\n"
            f"Existing image: {has_image}"
        )
        response = await self.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=context)]
        )
        decision = _parse_decision(response.content)

        # Safety net: never route straight to a refinement target for content that
        # doesn't exist yet, regardless of what the model decided.
        if not has_blog:
            decision["route"] = "research"

        return {
            "intent": decision["intent"],
            "route": decision["route"],
            "last_agent_used": "orchestrator",
        }


async def orchestrator_node(state: "AgentState") -> dict:
    return await OrchestratorAgent(provider=state.get("llm_provider")).run(state)
