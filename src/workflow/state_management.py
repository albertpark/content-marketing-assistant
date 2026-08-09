"""Session Store integration: LangGraph checkpointing and conversation state."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
    route: list[str] | None  # the target node(s) the orchestrator dispatched to via Send()
    last_agent_used: str | None
    llm_provider: str | None  # "openai" | "anthropic" | "gemini"; locked for the session

    # research
    # Graph-level tool-loop buffer for research_agent <-> research_tools_node (see
    # src/agents/research_agent.py). Deliberately NOT an add_messages-reducer channel
    # (unlike `messages`): each write is a full replacement of the whole list, so a
    # fresh research run (reset to [] by the orchestrator's Send() payload — see
    # query_handler.py) never has old turns' tool-loop history bleed back in. An
    # add_messages reducer would instead merge onto whatever the checkpointer already
    # had for this key, defeating that reset.
    research_messages: list[BaseMessage]
    research_findings: list[SearchResult]
    # Completed research_tools_node round-trips for the current research run, anchored
    # to 0 on every fresh start by research_agent_node (same anchoring rationale as
    # research_findings above).
    research_tool_iterations: int
    # Whether research_tool_iterations has hit settings.research_tool_iterations_cap
    # (config/services.yaml, override via RESEARCH_TOOL_ITERATIONS_CAP) — computed by
    # research_agent_node, read by should_continue_research to end the loop even if
    # the model still wants to call a tool. See research_agent_node's docstring.
    research_tool_iterations_capped: bool
    research_provider_used: str | None  # "serpapi" | "perplexity" | None

    # strategy
    content_brief: dict | None  # {angle, outline, key_points, target_keywords, image_brief}

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
        research_messages=[],
        research_findings=[],
        research_tool_iterations=0,
        research_tool_iterations_capped=False,
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

    _SESSION_COLS = ("session_id", "title", "created_at", "updated_at", "turn_count", "archived", "deleted_at")

    def __init__(self, conn: sqlite3.Connection | psycopg.Connection, backend: str, conn_string: str | None = None):
        self._conn = conn
        self._is_postgres = backend == "postgres"
        self._conn_string = conn_string
        self._ensure_table()

    def _sql(self, query: str) -> str:
        return query.replace("?", "%s") if self._is_postgres else query

    def _execute(self, query: str, params: tuple = ()):
        """Runs a query, transparently reconnecting once on a dead connection.
        get_registry_connection()'s postgres connection is cached for the whole
        Streamlit process lifetime (see streamlit_app.py's @st.cache_resource
        get_registry_conn) — Supabase's pooler silently drops idle connections,
        which otherwise permanently breaks the sidebar until the process is
        restarted."""
        try:
            return self._conn.execute(query, params)
        except psycopg.OperationalError:
            if not self._is_postgres:
                raise
            self._conn = psycopg.connect(self._conn_string, autocommit=True, prepare_threshold=None)
            return self._conn.execute(query, params)

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
                    turn_count INTEGER NOT NULL DEFAULT 1,
                    archived   INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT
                )
                """
            )
        )
        # Migrates sessions tables created before the archived/deleted_at columns
        # existed. Neither sqlite's ALTER TABLE ADD COLUMN nor (for simplicity,
        # treating both backends the same) this codepath's postgres usage rely
        # on "IF NOT EXISTS" — sqlite's ALTER TABLE grammar doesn't support that
        # clause on ADD COLUMN (verified: raises "near EXISTS: syntax error")
        # — so this just attempts each add and swallows the "already exists"
        # error every subsequent call hits once the column is there.
        for ddl in (
            "ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE sessions ADD COLUMN deleted_at TEXT",
        ):
            try:
                self._conn.execute(self._sql(ddl))
                self._commit_if_needed()
            except (sqlite3.OperationalError, psycopg.errors.DuplicateColumn):
                pass  # column already exists from a prior _ensure_table() call
        self._conn.execute(
            self._sql(
                """
                CREATE TABLE IF NOT EXISTS turn_routes (
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    routes     TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn_index)
                )
                """
            )
        )
        self._commit_if_needed()

    def record_start(self, session_id: str, title: str) -> None:
        """Call once, on the FIRST turn of a session (mirrors initial_state())."""
        now = _utcnow()
        self._execute(
            self._sql(
                "INSERT INTO sessions (session_id, title, created_at, updated_at, turn_count) "
                "VALUES (?, ?, ?, ?, 1) ON CONFLICT (session_id) DO NOTHING"
            ),
            (session_id, title, now, now),
        )
        self._commit_if_needed()

    def record_turn(self, session_id: str) -> None:
        """Call on every subsequent turn (mirrors new_turn_input())."""
        self._execute(
            self._sql("UPDATE sessions SET updated_at = ?, turn_count = turn_count + 1 WHERE session_id = ?"),
            (_utcnow(), session_id),
        )
        self._commit_if_needed()

    def list_sessions(self, include_archived: bool = False) -> list[dict]:
        """Most-recently-updated first, for the sidebar. Always excludes
        soft-deleted (trashed) sessions. By default also excludes archived
        ones; pass include_archived=True to list every non-trashed session
        (each row's "archived" field distinguishes them)."""
        where = "deleted_at IS NULL" if include_archived else "archived = 0 AND deleted_at IS NULL"
        cur = self._execute(
            self._sql(f"SELECT {', '.join(self._SESSION_COLS)} FROM sessions WHERE {where} ORDER BY updated_at DESC")
        )
        return [dict(zip(self._SESSION_COLS, row)) for row in cur.fetchall()]

    def list_trashed_sessions(self) -> list[dict]:
        """Soft-deleted sessions, most-recently-deleted first — for the
        sidebar's Trash view (restore, or delete forever before the retention
        worker would purge them anyway)."""
        cur = self._execute(
            self._sql(
                f"SELECT {', '.join(self._SESSION_COLS)} FROM sessions "
                "WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC"
            )
        )
        return [dict(zip(self._SESSION_COLS, row)) for row in cur.fetchall()]

    def list_expired_trashed_sessions(self, retention_days: int) -> list[dict]:
        """Soft-deleted sessions whose retention window has elapsed — for the
        periodic purge worker (see scripts/purge_deleted_sessions.py)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        cur = self._execute(
            self._sql(
                f"SELECT {', '.join(self._SESSION_COLS)} FROM sessions "
                "WHERE deleted_at IS NOT NULL AND deleted_at <= ? ORDER BY deleted_at"
            ),
            (cutoff,),
        )
        return [dict(zip(self._SESSION_COLS, row)) for row in cur.fetchall()]

    def set_archived(self, session_id: str, archived: bool) -> None:
        self._execute(
            self._sql("UPDATE sessions SET archived = ? WHERE session_id = ?"),
            (1 if archived else 0, session_id),
        )
        self._commit_if_needed()

    def rename_session(self, session_id: str, title: str) -> None:
        """Updates a session's display title after creation (record_start()
        only sets it once, immutably). Re-normalizes via
        derive_session_title() — same rule as the initial title — and
        no-ops if the normalized result is empty, so a title can never be
        blanked out. Does NOT touch updated_at: a rename is a metadata edit,
        not conversation activity, and must not reorder the sidebar's
        most-recently-updated list (mirrors set_archived, which also leaves
        updated_at untouched)."""
        normalized = derive_session_title(title)
        if not normalized:
            return
        self._execute(
            self._sql("UPDATE sessions SET title = ? WHERE session_id = ?"),
            (normalized, session_id),
        )
        self._commit_if_needed()

    def soft_delete_session(self, session_id: str) -> None:
        """Marks a session deleted without touching its data — it drops out of
        the sidebar's active/archived lists but stays fully intact (registry
        row + checkpointer thread) until restore_session() or the retention
        worker's hard_delete_session() call."""
        self._execute(
            self._sql("UPDATE sessions SET deleted_at = ? WHERE session_id = ?"),
            (_utcnow(), session_id),
        )
        self._commit_if_needed()

    def restore_session(self, session_id: str) -> None:
        """Undoes soft_delete_session(); the session reappears in the active
        list, or the archived one if it was archived before being deleted."""
        self._execute(
            self._sql("UPDATE sessions SET deleted_at = NULL WHERE session_id = ?"),
            (session_id,),
        )
        self._commit_if_needed()

    def hard_delete_session(self, session_id: str) -> None:
        """Permanently removes the registry row (title/timestamps) and its
        routing trace. Does NOT touch the checkpointer's own thread data —
        callers that also want the underlying conversation gone must delete
        that separately (see streamlit_app.py's _delete_session_thread, or
        scripts/purge_deleted_sessions.py for the retention worker)."""
        self._execute(self._sql("DELETE FROM turn_routes WHERE session_id = ?"), (session_id,))
        self._execute(self._sql("DELETE FROM sessions WHERE session_id = ?"), (session_id,))
        self._commit_if_needed()

    def record_routes(self, session_id: str, turn_index: int, routes: list[str]) -> None:
        """Persists one turn's agent-routing trace (the lines shown in the
        "Agent routing" expander) so it survives switching sessions and
        reloads — the graph checkpoint itself has no field for this, it's
        UI-only telemetry derived from streaming debug events."""
        self._execute(
            self._sql(
                "INSERT INTO turn_routes (session_id, turn_index, routes) VALUES (?, ?, ?) "
                "ON CONFLICT (session_id, turn_index) DO UPDATE SET routes = EXCLUDED.routes"
            ),
            (session_id, turn_index, json.dumps(routes)),
        )
        self._commit_if_needed()

    def get_routes(self, session_id: str) -> dict[int, list[str]]:
        """turn_index -> routing lines, for every turn of this session that has one."""
        cur = self._execute(
            self._sql("SELECT turn_index, routes FROM turn_routes WHERE session_id = ?"),
            (session_id,),
        )
        return {row[0]: json.loads(row[1]) for row in cur.fetchall()}


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
        return SessionRegistry(conn, backend, conn_string=settings.session_store_url)
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
