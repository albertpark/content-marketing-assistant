"""Session Store integration: LangGraph checkpointing and conversation state."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, AsyncIterator, TypedDict

import psycopg
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from src.core.config import Settings


class SearchResult(TypedDict):
    source: str  # "serpapi" | "perplexity"
    title: str
    url: str
    snippet: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_query: str

    # orchestrator / routing
    intent: str | None  # "new_content" | "refinement" | "research_only"
    route: str | None
    last_agent_used: str | None
    llm_provider: str | None  # "openai" | "anthropic" | "gemini"; locked for the session

    # research
    research_findings: list[SearchResult]
    research_provider_used: str | None  # "serpapi" | "perplexity" | None

    # strategy
    content_brief: dict | None  # {angle, outline, key_points, target_keywords}

    # drafts
    blog_post: dict | None  # {title, body_markdown, meta_description, headers, word_count}
    linkedin_post: dict | None  # {text, hashtags, char_count}
    image_assets: list[dict]  # [{url, path, prompt, provider_used, alt_text}]

    # synthesis
    content_package: dict | None

    # quality
    quality_report: dict | None  # {passed, capped, requires_human_review, gates, issues}
    revision_count: int
    revision_feedback: str | None
    human_approved_override: bool

    errors: list[str]


def new_session_id() -> str:
    return str(uuid.uuid4())


def initial_state(session_id: str, user_query: str, llm_provider: str | None = None) -> AgentState:
    """Full starting state for the FIRST turn of a brand-new session (no checkpoint
    exists yet for this thread_id). Do not reuse this for later turns — see
    new_turn_input()."""
    return AgentState(
        messages=[HumanMessage(content=user_query)],
        session_id=session_id,
        user_query=user_query,
        intent=None,
        route=None,
        last_agent_used=None,
        llm_provider=llm_provider,
        research_findings=[],
        research_provider_used=None,
        content_brief=None,
        blog_post=None,
        linkedin_post=None,
        image_assets=[],
        content_package=None,
        quality_report=None,
        revision_count=0,
        revision_feedback=None,
        human_approved_override=False,
        errors=[],
    )


def new_turn_input(user_query: str) -> dict:
    """Partial-state update for turn 2+ of an EXISTING session. Only messages/
    user_query are set — every other field is intentionally omitted so the
    checkpointer keeps the prior turn's values (blog_post, revision_count, etc.)
    instead of resetting them. This is what makes refinement turns ("make the
    LinkedIn post punchier") continue from existing content instead of restarting.
    llm_provider is deliberately one of the omitted fields too — the provider is
    locked for the life of a session (see docs/hld.md); start a new session to
    switch providers rather than changing it mid-refinement."""
    return {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
    }


def get_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    """Returns a checkpointer for backends that are safe to cache and reuse
    across many operations within one process. Today that's only "memory" —
    InMemorySaver has no OS-level connection or event-loop binding. "sqlite"
    and "postgres" are async-native and must NOT be cached this way: verified
    empirically that both aiosqlite and psycopg's async connection bind to the
    event loop that created them, and reusing one across a second, separate
    asyncio.run() call fails ("threads can only be started once" / "the
    connection is closed"). Use open_checkpointer() for those instead."""
    if settings.session_store_backend == "memory":
        return InMemorySaver()
    if settings.session_store_backend in ("sqlite", "postgres"):
        raise ValueError(
            f"{settings.session_store_backend!r} requires the per-call async path — "
            "use `async with open_checkpointer(settings) as saver:` instead of get_checkpointer()."
        )
    raise ValueError(f"Unknown session_store_backend: {settings.session_store_backend!r}")


def open_checkpointer(settings: Settings):
    """Returns an unentered async context manager yielding a fresh
    BaseCheckpointSaver, scoped to a single asyncio.run() call. Callers must
    `async with` this FRESH for every distinct graph operation (ainvoke /
    aget_state / aupdate_state) — see get_checkpointer()'s docstring for why."""
    backend = settings.session_store_backend
    if backend == "sqlite":
        return AsyncSqliteSaver.from_conn_string(settings.session_store_path)
    if backend == "postgres":
        return _open_postgres_checkpointer(settings.session_store_url)
    raise ValueError(f"{backend!r} does not use the per-call async checkpointer path")


@asynccontextmanager
async def _open_postgres_checkpointer(conn_string: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Constructs AsyncPostgresSaver manually rather than via its own
    from_conn_string() (which hardcodes prepare_threshold=0 — always use
    server-side prepared statements). That default is broken under
    connection-pooled Postgres in transaction mode (e.g. Supabase's pooler):
    transaction pooling can hand different transactions on the same client
    connection to different backend server processes, and a prepared
    statement created on one doesn't exist — or collides by name — on
    another. Verified empirically against a real Supabase transaction
    pooler: prepare_threshold=0 raises psycopg.errors.DuplicatePreparedStatement;
    prepare_threshold=None (never prepare) works correctly."""
    conn = await AsyncConnection.connect(
        conn_string, autocommit=True, prepare_threshold=None, row_factory=dict_row
    )
    async with conn:
        saver = AsyncPostgresSaver(conn=conn)
        await saver.setup()
        yield saver


class SessionRegistry:
    """Lightweight session list/title registry (session_id, title, timestamps,
    turn_count), stored alongside the checkpointer's own data (sqlite file or
    the same Postgres database) but through a plain, non-async connection —
    this CRUD is simple enough not to need the checkpointer's fresh-per-call
    async handling, and a cached sync connection is safe to reuse across
    Streamlit reruns for both backends (only the async drivers used by the
    checkpointer are event-loop-bound, not plain sqlite3/psycopg connections).

    Note: even though this connection isn't asyncio-native, prepare_threshold
    still needs to be disabled for the postgres backend — the pooler's
    DuplicatePreparedStatement issue (see _open_postgres_checkpointer) is
    caused by transaction-mode pooling itself, not by asyncio/event-loop
    binding, so it affects sync connections just as much."""

    def __init__(self, conn: sqlite3.Connection | psycopg.Connection, backend: str):
        self._conn = conn
        self._is_postgres = backend == "postgres"
        self._ensure_table()

    def _sql(self, query: str) -> str:
        return query.replace("?", "%s") if self._is_postgres else query

    def _commit_if_needed(self) -> None:
        # The postgres connection is opened with autocommit=True; sqlite3 needs an explicit commit.
        if not self._is_postgres:
            self._conn.commit()

    def _ensure_table(self) -> None:
        self._conn.execute(
            self._sql(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    turn_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )
        )
        self._commit_if_needed()

    def record_start(self, session_id: str, title: str) -> None:
        """Call once, on the FIRST turn of a session (mirrors initial_state())."""
        now = _utcnow()
        self._conn.execute(
            self._sql(
                "INSERT INTO sessions (session_id, title, created_at, updated_at, turn_count) "
                "VALUES (?, ?, ?, ?, 1) ON CONFLICT (session_id) DO NOTHING"
            ),
            (session_id, title, now, now),
        )
        self._commit_if_needed()

    def record_turn(self, session_id: str) -> None:
        """Call on every subsequent turn (mirrors new_turn_input())."""
        self._conn.execute(
            self._sql("UPDATE sessions SET updated_at = ?, turn_count = turn_count + 1 WHERE session_id = ?"),
            (_utcnow(), session_id),
        )
        self._commit_if_needed()

    def list_sessions(self) -> list[dict]:
        """Most-recently-updated first, for the sidebar."""
        cur = self._conn.execute(
            self._sql(
                "SELECT session_id, title, created_at, updated_at, turn_count "
                "FROM sessions ORDER BY updated_at DESC"
            )
        )
        cols = ("session_id", "title", "created_at", "updated_at", "turn_count")
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_registry_connection(settings: Settings) -> SessionRegistry:
    """Opens the (cacheable) connection backing the sessions registry. Only
    "sqlite" and "postgres" persist across restarts; "memory" has no durable
    registry (see streamlit_app.py, which hides the sidebar's session list for
    that backend rather than showing a registry that would outlive the actual
    checkpointed data it points to)."""
    backend = settings.session_store_backend
    if backend == "sqlite":
        conn = sqlite3.connect(settings.session_store_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return SessionRegistry(conn, backend)
    if backend == "postgres":
        conn = psycopg.connect(settings.session_store_url, autocommit=True, prepare_threshold=None)
        return SessionRegistry(conn, backend)
    raise ValueError(f"Session registry is not available for backend {backend!r}")


def derive_session_title(user_query: str, max_len: int = 60) -> str:
    """First-user-query-derived label, truncated at a word boundary — NOT
    LLM-summarized, to avoid a paid call just for a sidebar label."""
    text = " ".join(user_query.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def _utcnow() -> str:
    # Isolated as its own function so tests can monkeypatch it for deterministic
    # updated_at/ordering assertions (matches this repo's monkeypatch-heavy style).
    return datetime.now(timezone.utc).isoformat()
