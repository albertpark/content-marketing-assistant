"""Content Strategist: formats research findings into a structured content brief."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.agents.prompts import CONTENT_STRATEGIST_PROMPT

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_brief(raw: str) -> dict:
    match = _JSON_PATTERN.search(raw or "")
    if not match:
        return {"angle": raw or "", "outline": [], "key_points": [], "target_keywords": []}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {"angle": raw or "", "outline": [], "key_points": [], "target_keywords": []}
    return {
        "angle": parsed.get("angle", ""),
        "outline": parsed.get("outline", []),
        "key_points": parsed.get("key_points", []),
        "target_keywords": parsed.get("target_keywords", []),
    }


class ContentStrategistAgent(BaseAgent):
    def __init__(self, provider: str | None = None, debug: bool = False):
        super().__init__(
            agent_name="content_strategist",
            provider=provider,
            temperature=0.4,
            system_prompt=CONTENT_STRATEGIST_PROMPT,
            debug=debug,
        )

    async def run(self, state: "AgentState") -> dict:
        findings = state.get("research_findings", [])
        findings_text = (
            "\n".join(f"- {f['title']} ({f['url']}): {f['snippet']}" for f in findings)
            or "No research findings available — use general knowledge."
        )
        context = f"Topic: {state['user_query']}\n\nResearch findings:\n{findings_text}"

        response = await self.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=context)]
        )
        return {
            "content_brief": _parse_brief(response.content),
            "last_agent_used": "content_strategist",
        }


async def content_strategist_node(state: "AgentState") -> dict:
    return await ContentStrategistAgent(provider=state.get("llm_provider")).run(state)
