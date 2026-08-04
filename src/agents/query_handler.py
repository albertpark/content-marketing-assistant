"""Orchestrator: classifies user intent and dispatches to agent(s) via LangGraph Send()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.agents.prompts import ORCHESTRATOR_PROMPT

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

_NODE_MAP = {
    "research": "research_agent",
    "blog": "blog_writer",
    "linkedin": "linkedin_writer",
    "image": "image_generator",
    "package": "synthesizer",
}

# Checked in this priority order: any of these three can't run alongside anything
# else in the pipeline (research/blog/package each need the full downstream chain,
# or none of it, to themselves). Only linkedin+image are genuinely parallel-compatible
# — both only depend on an existing blog_post.
_EXCLUSIVE_TARGETS = ("research", "blog", "package")


class OrchestratorDecision(BaseModel):
    """The orchestrator's routing decision."""

    intent: Literal["new_content", "refinement"]
    targets: list[Literal["research", "blog", "linkedin", "image", "package"]] = Field(
        description="One or more of research/blog/linkedin/image/package. "
        "linkedin and image may be combined in one turn; every other target must stand alone.",
        min_length=1,
    )


def _normalize_targets(targets: list[str]) -> list[str]:
    for exclusive in _EXCLUSIVE_TARGETS:
        if exclusive in targets:
            return [exclusive]
    parallel = [t for t in targets if t in ("linkedin", "image")]
    return parallel or ["research"]


class OrchestratorAgent(BaseAgent):
    def __init__(self, provider: str | None = None, debug: bool = False):
        super().__init__(
            agent_name="orchestrator",
            provider=provider,
            temperature=0,
            system_prompt=ORCHESTRATOR_PROMPT,
            debug=debug,
        )

    async def decide(self, state: "AgentState") -> OrchestratorDecision:
        has_blog = state.get("blog_post") is not None
        has_linkedin = state.get("linkedin_post") is not None
        has_image = bool(state.get("image_assets"))

        context = (
            f"User message: {state['user_query']}\n"
            f"Existing blog post: {has_blog}\n"
            f"Existing LinkedIn post: {has_linkedin}\n"
            f"Existing image: {has_image}"
        )
        raw = await self.invoke_structured(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=context)],
            OrchestratorDecision,
        )
        targets = _normalize_targets(raw.targets)

        # Safety net: never route straight to a refinement target for content that
        # doesn't exist yet, regardless of what the model decided.
        if not has_blog:
            targets = ["research"]

        return OrchestratorDecision(intent=raw.intent, targets=targets)


async def orchestrator_node(
    state: "AgentState",
) -> Command[Literal["research_agent", "blog_writer", "linkedin_writer", "image_generator", "synthesizer"]]:
    decision = await OrchestratorAgent(provider=state.get("llm_provider")).decide(state)

    dispatch_state = dict(state)
    if "research" in decision.targets:
        # Reset the tool-loop buffer before Send()ing into a fresh research run — see
        # AgentState.research_messages's docstring for why this matters.
        dispatch_state["research_messages"] = []

    sends = [Send(_NODE_MAP[target], dispatch_state) for target in decision.targets]
    return Command(
        goto=sends,
        update={"intent": decision.intent, "route": decision.targets, "last_agent_used": "orchestrator"},
    )
