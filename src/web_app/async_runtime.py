"""Persistent background event loop for the Streamlit process.

Streamlit reruns the whole script on every interaction with no event loop of
its own, and psycopg's/aiosqlite's async connections bind to the loop that
created them (see state_management.get_checkpointer's docstring). The
previous approach — opening a fresh connection and rerunning checkpointer
setup() inside a fresh asyncio.run() on every click — measured ~650-700ms of
pure overhead per interaction against Supabase (connection/TLS handshake +
the two setup() round trips), on top of a slower first query.

This module runs ONE event loop on a dedicated daemon thread for the app's
lifetime, so the checkpointer connection can be opened and set up exactly
once (via build_persistent_graph, called from streamlit_app.get_persistent_graph)
and reused across every rerun and every browser session in this process —
mirroring how state_management.get_registry_connection already caches a
single sync connection app-wide. psycopg's async connection serializes
overlapping operations internally, so sharing it across concurrently
rerunning sessions on this one loop is safe.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import TypeVar

from src.core.config import Settings
from src.workflow.langgraph_workflow import build_graph
from src.workflow.state_management import open_checkpointer

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()
# Which background loop (by id) each cache_resource-held graph's checkpointer
# was opened against. Streamlit's file watcher can reload this module (e.g.
# after editing state_management.py, which it imports) independently of the
# @st.cache_resource-cached graph in streamlit_app.py — that reset _loop to a
# fresh loop/thread while the cached graph's psycopg async connection (and its
# internal asyncio.Lock) stayed bound to the old, now-orphaned one, raising
# "bound to a different event loop" on the next call. graph_matches_current_loop
# lets st.cache_resource(validate=...) detect that and rebuild instead of crashing.
_graph_loop_ids: dict[int, int] = {}
# Holds the entered open_checkpointer() context manager alive for the process
# lifetime. open_checkpointer() is an @asynccontextmanager generator; entering
# it via __aenter__() without keeping a reference to the CM object itself lets
# it get garbage-collected, which resumes the suspended generator past its
# yield and runs its cleanup (closing the connection) — verified empirically
# (the connection closed itself within milliseconds of the first successful
# call). Assigning it here is what keeps it alive.
_checkpointer_cm = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None:
            loop = asyncio.SelectorEventLoop()
            threading.Thread(target=loop.run_forever, name="checkpointer-loop", daemon=True).start()
            _loop = loop
    return _loop


def current_loop_id() -> int:
    return id(_get_loop())


def register_graph_loop(graph: object) -> None:
    """Records which loop `graph`'s checkpointer was opened against. Call once,
    right after building it."""
    _graph_loop_ids[id(graph)] = current_loop_id()


def graph_matches_current_loop(graph: object) -> bool:
    """False once the background loop has been replaced (e.g. this module got
    hot-reloaded) since `graph` was built — signals st.cache_resource to
    rebuild rather than reuse a graph whose checkpointer lock can never be
    acquired again."""
    return _graph_loop_ids.get(id(graph)) == current_loop_id()


def run(coro: Coroutine[None, None, T]) -> T:
    """Schedules coro on the background loop and blocks the calling thread for
    the result. Must be called from a thread other than the background loop's
    own thread — e.g. never from inside a coroutine that reached here via
    `run` itself — otherwise the loop thread deadlocks waiting on itself."""
    future = asyncio.run_coroutine_threadsafe(coro, _get_loop())
    return future.result()


async def build_persistent_graph(settings: Settings):
    """Opens the checkpointer connection once and never closes it — entering
    the context manager without a matching __aexit__ is deliberate here, since
    the daemon thread and its connection are meant to live for the process's
    lifetime; process exit tears down the socket regardless."""
    global _checkpointer_cm
    _checkpointer_cm = open_checkpointer(settings)
    saver = await _checkpointer_cm.__aenter__()
    return build_graph(checkpointer=saver)
