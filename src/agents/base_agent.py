"""BaseAgent: shared LLM-invocation and tool-loop machinery for every ContentAlchemy agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from src.integrations import llm_client
from src.integrations.resilience import AllRetriesExhaustedError, ProviderError, with_retry

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

logger = logging.getLogger(__name__)

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


@with_retry(retry_on=(ProviderError,))
async def _invoke_one(llm: Any, conversation: list[BaseMessage]) -> AIMessage:
    """Single provider attempt, retried per with_retry's policy. Catches broadly
    (Exception) rather than importing each provider SDK's own exception classes:
    OpenAI's, Anthropic's, and Google's exception hierarchies are three
    genuinely unrelated class trees (verified — Anthropic's share *names* with
    OpenAI's but not ancestry; Google's come from google.api_core.exceptions),
    so enumerating all three here would force this shared module to depend on
    every provider SDK directly. Tradeoff: a terminal error (e.g. a bad key)
    still burns the retry budget before BaseAgent moves to the next provider in
    the chain, since nothing here distinguishes retryable from terminal under
    the broad catch — acceptable for this project's scope."""
    try:
        if hasattr(llm, "ainvoke"):
            return await llm.ainvoke(conversation)
        return llm.invoke(conversation)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(str(exc)) from exc


@with_retry(retry_on=(ProviderError,))
async def _invoke_one_structured(llm: Any, conversation: list[BaseMessage], schema: type[_SchemaT]) -> _SchemaT:
    """Structured-output counterpart to _invoke_one: same broad-catch rationale
    applies (see _invoke_one's docstring)."""
    try:
        structured_llm = llm.with_structured_output(schema)
        if hasattr(structured_llm, "ainvoke"):
            return await structured_llm.ainvoke(conversation)
        return structured_llm.invoke(conversation)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(str(exc)) from exc


class BaseAgent:
    """Shared tool-loop machinery for LangGraph node agents.

    Subclasses implement run(state) -> dict (a LangGraph partial-state update) and
    call self.invoke(...) internally to get a final LLM response, executing any
    tool calls the model requests along the way. This is what implements hld.md's
    "tool loop (bidirectional, capped retries)" semantics without a separate MCP
    server process — the cap is max_tool_iterations.

    LLM invocation walks a provider fallback chain (see
    src.integrations.llm_client.fallback_chain): each provider is retried per
    with_retry's policy, and if its retry budget is exhausted, the next provider
    in the chain is tried before giving up — this is the milestone-2 "fallback
    mechanisms and error handling for API failures" requirement, centralized
    here rather than duplicated per agent.
    """

    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        provider: str | None = None,
        temperature: float = 0.7,
        tools: list[BaseTool] | None = None,
        max_tool_iterations: int = 3,
        debug: bool = False,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.tools = {tool.name: tool for tool in (tools or [])}
        self._tools_list = tools
        self.max_tool_iterations = max_tool_iterations
        self.debug = debug
        self._provider_chain = llm_client.fallback_chain(provider)
        self._llm_cache: dict[str, Any] = {}

    def _llm_for(self, provider: str) -> Any:
        if provider not in self._llm_cache:
            self._llm_cache[provider] = llm_client.get_chat_model(
                provider, temperature=self.temperature, tools=self._tools_list
            )
        return self._llm_cache[provider]

    async def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Runs the tool-call loop: invoke the LLM, execute any tool_calls it
        requests, append the results, and repeat until it stops requesting tools or
        max_tool_iterations is reached. Returns the final AIMessage (which may still
        carry unresolved tool_calls if the budget was exhausted)."""
        conversation = list(messages)
        response: AIMessage | None = None
        for iteration in range(self.max_tool_iterations):
            response = await self._ainvoke_llm(conversation)
            if self.debug:
                logger.debug("[%s] iteration %d: %r", self.agent_name, iteration, response)
            if not getattr(response, "tool_calls", None):
                return response
            conversation.append(response)
            for tool_call in response.tool_calls:
                conversation.append(await self._execute_tool(tool_call))
        return response

    async def call_once(self, conversation: list[BaseMessage]) -> AIMessage:
        """A single provider-fallback LLM call with no tool loop — for agents whose
        tool loop is implemented at the graph level (agent_node ⇄ tools_node
        conditional edges) rather than in-process via invoke()."""
        return await self._ainvoke_llm(conversation)

    async def invoke_structured(self, messages: list[BaseMessage], schema: type[_SchemaT]) -> _SchemaT:
        """Same provider-fallback chain as invoke()/call_once(), but binds
        with_structured_output(schema) on each provider's model and returns a
        validated instance of schema directly instead of an AIMessage."""
        return await self._with_fallback_chain(
            lambda llm: _invoke_one_structured(llm, messages, schema)
        )

    async def _ainvoke_llm(self, conversation: list[BaseMessage]) -> AIMessage:
        return await self._with_fallback_chain(lambda llm: _invoke_one(llm, conversation))

    async def _with_fallback_chain(self, call_fn: Callable[[Any], Awaitable[Any]]) -> Any:
        last_exc: Exception | None = None
        for provider in self._provider_chain:
            try:
                return await call_fn(self._llm_for(provider))
            except AllRetriesExhaustedError as exc:
                last_exc = exc
                logger.warning(
                    "[%s] provider %r exhausted retries; falling back", self.agent_name, provider
                )
        raise AllRetriesExhaustedError(
            f"{self.agent_name}: all providers in fallback chain exhausted: {self._provider_chain}"
        ) from last_exc

    async def _execute_tool(self, tool_call: dict) -> ToolMessage:
        tool = self.tools.get(tool_call["name"])
        if tool is None:
            content = f"Unknown tool: {tool_call['name']!r}"
        else:
            try:
                result = tool.invoke(tool_call["args"])
                content = str(result)
            except Exception as exc:  # a failing tool must not crash the agent loop
                content = f"Tool {tool_call['name']!r} failed: {exc}"
        return ToolMessage(content=content, tool_call_id=tool_call["id"])

    async def run(self, state: "AgentState") -> dict:
        """Subclasses must override: read what they need from state, call
        self.invoke(...), and return a partial-state update dict."""
        raise NotImplementedError(f"{type(self).__name__} must implement run()")
