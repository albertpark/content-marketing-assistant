"""BaseAgent: shared LLM-invocation and tool-loop machinery for every ContentAlchemy agent."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from src.integrations import llm_client, performance
from src.integrations.resilience import AllRetriesExhaustedError, ProviderError, with_retry

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState

logger = logging.getLogger(__name__)

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

# Shared across every BaseAgent instance for a given agent_name — a fresh
# BaseAgent is constructed on every graph-node call, so a degraded-response
# cache (or the admission limiter it backstops, see performance.get_admission_limiter)
# only works if it lives at module scope rather than on self. Not TTL'd: mirrors
# the notebook's response_cache exactly (last-successful-wins, no expiry) since a
# stale cached draft is still strictly better than the static tier below it.
_RESPONSE_CACHE: dict[str, str] = {}


def _cache_key(agent_name: str, conversation: list[BaseMessage]) -> str:
    # Includes each message's type, not just content, so e.g. a HumanMessage and
    # a SystemMessage with identical text don't collide on the same digest.
    text = "\n".join(f"{type(m).__name__}:{getattr(m, 'content', '')}" for m in conversation)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return f"{agent_name}:{digest}"


class _FallbackAIMessage(AIMessage):
    """A real AIMessage subclass (not a duck-typed stand-in) for degraded
    (cached/static) responses — research_agent_node appends whatever this
    returns into state["research_messages"], which the sqlite/postgres
    checkpointers serialize on every step (see test_graph_e2e.py's own
    _FakeAIMessage for the same requirement)."""

    def __init__(self, content: str):
        super().__init__(content=content, tool_calls=[])


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

    Text-producing calls (invoke()/call_once(), via _ainvoke_llm — NOT
    invoke_structured(), see below) sit behind a further, non-blocking
    admission gate (src.integrations.performance.TokenBucketLimiter) and a
    3-tier degradation ladder: (1) admission-gated provider-fallback chain
    above, (2) the last cached response to an identical conversation, (3) the
    subclass's static_fallback string. This collapses the notebook's 4-tier
    "primary model -> smaller model -> cache -> static" design into 3 tiers,
    since provider-fallback here already plays the "swap to a different model"
    role the notebook's Tier 2 does — there's no separate cheaper model to add.
    invoke_structured() deliberately stays on the plain provider chain with no
    degradation: it's only used for the orchestrator's routing decision, and
    there's no generically-safe fallback schema instance for an arbitrary
    Pydantic model.
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
        static_fallback: str | None = None,
        admission_limiter: performance.TokenBucketLimiter | None = None,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.tools = {tool.name: tool for tool in (tools or [])}
        self._tools_list = tools
        self.max_tool_iterations = max_tool_iterations
        self.debug = debug
        self.static_fallback = static_fallback
        self._provider_chain = llm_client.fallback_chain(provider)
        self._llm_cache: dict[str, Any] = {}
        # Resolved here (not as a `= performance.get_admission_limiter()` default
        # expression) so each call sees whatever the current singleton is —
        # a default expression would capture one instance at class-definition
        # time and never observe a later get_admission_limiter.cache_clear().
        self._admission_limiter = admission_limiter or performance.get_admission_limiter()

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
        """Tier 1: full provider-fallback chain, attempted only if the shared
        admission limiter grants a token (reject-not-queue: see
        TokenBucketLimiter — a rejection degrades immediately rather than
        adding latency). Falls through to Tier 2 (cached response to an
        identical conversation) and Tier 3 (self.static_fallback) on either an
        admission rejection or AllRetriesExhaustedError, re-raising only if
        neither tier below has anything to offer. See the class docstring for
        why this ladder has 3 tiers, not the notebook's 4."""
        cache_key = _cache_key(self.agent_name, conversation)

        last_exc: AllRetriesExhaustedError | None = None
        if self._admission_limiter.try_acquire():
            try:
                response = await self._with_fallback_chain(lambda llm: _invoke_one(llm, conversation))
                _RESPONSE_CACHE[cache_key] = response.content
                return response
            except AllRetriesExhaustedError as exc:
                last_exc = exc
                logger.warning("[%s] all providers exhausted; degrading", self.agent_name)
        else:
            logger.warning("[%s] admission rejected by rate limiter; degrading without a call", self.agent_name)

        cached_content = _RESPONSE_CACHE.get(cache_key)
        if cached_content is not None:
            logger.warning("[%s] serving cached response (degraded)", self.agent_name)
            return _FallbackAIMessage(cached_content)

        if self.static_fallback is not None:
            logger.warning("[%s] serving static fallback response (degraded)", self.agent_name)
            return _FallbackAIMessage(self.static_fallback)

        raise last_exc or AllRetriesExhaustedError(
            f"{self.agent_name}: rate-limited with no cached or static fallback available"
        )

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
