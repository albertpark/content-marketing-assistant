"""Streamlit entrypoint: chat interface, content dashboard, research panel, export tools."""

from __future__ import annotations

import html
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from src.core.config import get_settings
from src.utils.export_tools import blog_export_filename, blog_to_markdown, linkedin_post_to_text
from src.web_app import async_runtime
from src.workflow.langgraph_workflow import build_graph
from src.workflow.state_management import (
    derive_session_title,
    get_registry_connection,
    initial_state,
    new_session_id,
    new_turn_input,
)

# -----------------------------------------------------------------------------
# Set things up.

st.set_page_config(page_title="ContentAlchemy", page_icon="📝", layout="wide")

# Hides just the "Record a screencast" from main menu
st.markdown(
    '<style>[data-testid="stMainMenuItem-recordScreencast"] { display: none; }</style>',
    unsafe_allow_html=True,
)


def _run_async(coro):
    # Dispatches onto async_runtime's persistent background-thread loop rather
    # than asyncio.run()'ing a fresh one per call. Required for get_persistent_graph's
    # checkpointer connection to survive across reruns (see async_runtime.py);
    # also just avoids spinning up a new event loop on every interaction.
    return async_runtime.run(coro)


@st.cache_resource
def get_graph():
    # Only used for the "memory" backend: InMemorySaver has no event-loop/OS
    # binding, so it's safe to build once and reuse across reruns — which is
    # required, since a fresh InMemorySaver would be an empty store.
    return build_graph()


@st.cache_resource(validate=async_runtime.graph_matches_current_loop)
def get_persistent_graph():
    """"sqlite"/"postgres" backends: builds the graph once against a
    checkpointer connection opened on async_runtime's background loop, then
    reuses both for the app's lifetime. Must be called from main() before any
    _run_async() dispatch touches it — st.cache_resource's factory runs
    synchronously on the calling thread, and async_runtime.run() would
    deadlock if the factory is entered from a coroutine already running on
    the background loop it depends on.

    `validate` rebuilds this if async_runtime's background loop was ever
    replaced after this graph was built (see _graph_loop_ids' docstring) —
    otherwise a stale graph's checkpointer lock raises "bound to a different
    event loop" the next time it's used."""
    graph = async_runtime.run(async_runtime.build_persistent_graph(get_settings()))
    async_runtime.register_graph_loop(graph)
    return graph


@st.cache_resource
def get_registry_conn():
    return get_registry_connection(get_settings())


async def _with_graph(op):
    """Runs op(graph) against a correctly-scoped graph + checkpointer for the
    configured backend. op: an async-returning callable taking the graph."""
    settings = get_settings()
    if settings.session_store_backend in ("sqlite", "postgres"):
        return await op(get_persistent_graph())
    return await op(get_graph())


# Static per-node description of what each graph node actually does, shown in the
# expanded progress panel. "LLM" gets the live-selected provider spliced in by
# _node_progress_line; nodes with no LLM call (templates/deterministic gates) say
# so explicitly rather than leaving the user guessing why no model shows up.
_NODE_INFO: dict[str, dict[str, str]] = {
    "orchestrator": {"label": "Orchestrator", "detail": "LLM — routes the request"},
    "research_agent": {
        "label": "Research Agent",
        "detail": "LLM — decides what to search next",
    },
    "research_tools_node": {
        "label": "Research Tools",
        "detail": "Executes web_search (SerpAPI → Perplexity fallback), no LLM call",
    },
    "content_strategist": {"label": "Content Strategist", "detail": "LLM — builds the content brief"},
    "blog_writer": {"label": "Blog Writer", "detail": "LLM — drafts the blog post"},
    "linkedin_writer": {"label": "LinkedIn Writer", "detail": "LLM — writes a short-form post"},
    "image_generator": {"label": "Image Generator", "detail": "OpenAI image generation (gpt-image-1-mini)"},
    "synthesizer": {"label": "Synthesizer", "detail": "Packages the content bundle, no LLM call"},
    "quality_pipeline": {"label": "Quality Pipeline", "detail": "Deterministic quality gates, no LLM call"},
}


def _node_progress_line(node_name: str, provider_label: str) -> tuple[str, str]:
    info = _NODE_INFO.get(node_name, {"label": node_name, "detail": ""})
    detail = info["detail"]
    if detail.startswith("LLM"):
        detail = detail.replace("LLM", f"LLM ({provider_label})", 1)
    return info["label"], detail


# blog_writer and linkedin_writer both respond with a single JSON blob (see
# their _SYSTEM_PROMPTs), not plain prose, so raw token chunks can't just be
# appended and displayed — that would show a growing, escaped, incomplete
# JSON string. This does a best-effort incremental decode of one string
# field's value out of a still-streaming JSON blob, for live display only;
# the authoritative parse is each agent's own _parse_*_post once the node
# actually finishes.
def _extract_streaming_field(accumulated_raw: str, field_name: str) -> str:
    marker = f'"{field_name}":'
    idx = accumulated_raw.find(marker)
    if idx == -1:
        return ""
    rest = accumulated_raw[idx + len(marker) :].lstrip()
    if not rest.startswith('"'):
        return ""
    rest = rest[1:]
    out = []
    i = 0
    while i < len(rest):
        ch = rest[i]
        if ch == "\\":
            if i + 1 >= len(rest):
                break  # incomplete escape sequence — wait for the next chunk
            out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(rest[i + 1], rest[i + 1]))
            i += 2
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1
    return "".join(out)


async def _run_turn(
    input_state: dict,
    thread_id: str,
    status=None,
    provider_label: str = "",
    blog_placeholder=None,
    linkedin_placeholder=None,
    ctx=None,
    route_log: list[str] | None = None,
) -> None:
    """Runs the graph. When `status` (an st.status container) is given, streams
    per-node start/finish events into it live. When blog_placeholder /
    linkedin_placeholder (st.empty() containers) are given, also streams the
    blog post body / LinkedIn post text into them token-by-token as they're
    generated, instead of only appearing once the whole run finishes.

    `route_log`: if given, every line written to `status` is also appended
    here, so the routing trace can be replayed (e.g. in an expander) after
    this turn's live `st.status` widget is gone post-rerun.

    `ctx`: caller's ScriptRunContext (get_script_run_ctx()). This coroutine
    runs on async_runtime's shared background thread, not the Streamlit
    script thread, so st.* calls need a context re-attached — and since that
    thread is shared across concurrent sessions, it's re-attached after every
    await, not just once, in case another session's coroutine ran in between."""
    config = {"configurable": {"thread_id": thread_id}}
    streaming = status is not None or blog_placeholder is not None or linkedin_placeholder is not None

    async def op(graph):
        if not streaming:
            return await graph.ainvoke(input_state, config=config)

        raw_by_node: dict[str, str] = {}
        # Rendering on every token chunk (potentially hundreds/sec for a full
        # post) sends that many full markdown re-renders to the browser and
        # visibly janks the tab — throttle the live preview to a fixed cadence
        # instead. The final flush after the loop guarantees the last chunk
        # (which the throttle would otherwise drop) still gets shown; the
        # authoritative content comes from the post-rerun dashboard regardless.
        last_rendered_at: dict[str, float] = {}

        def _due(node_name: str) -> bool:
            now = time.monotonic()
            if now - last_rendered_at.get(node_name, 0.0) < 0.15:
                return False
            last_rendered_at[node_name] = now
            return True

        async for mode, payload in graph.astream(
            input_state, config=config, stream_mode=["debug", "messages"]
        ):
            if ctx is not None:
                add_script_run_ctx(threading.current_thread(), ctx)
            if mode == "debug":
                if status is None:
                    continue
                node_name = payload.get("payload", {}).get("name")
                if not node_name:
                    continue
                label, detail = _node_progress_line(node_name, provider_label)
                if payload["type"] == "task":
                    status.update(label=f"Working — {label}...")
                    line = f"▶ {label} started" + (f" — {detail}" if detail else "")
                    status.write(line)
                    if route_log is not None:
                        route_log.append(line)
                elif payload["type"] == "task_result":
                    line = f"✓ {label} done"
                    status.write(line)
                    if route_log is not None:
                        route_log.append(line)
            elif mode == "messages":
                msg_chunk, metadata = payload
                node_name = metadata.get("langgraph_node")
                if node_name not in ("blog_writer", "linkedin_writer"):
                    continue
                raw_by_node[node_name] = raw_by_node.get(node_name, "") + (msg_chunk.content or "")
                if node_name == "blog_writer" and blog_placeholder is not None and _due(node_name):
                    body = _extract_streaming_field(raw_by_node[node_name], "body_markdown")
                    if body:
                        blog_placeholder.markdown(body)
                elif node_name == "linkedin_writer" and linkedin_placeholder is not None and _due(node_name):
                    text = _extract_streaming_field(raw_by_node[node_name], "text")
                    if text:
                        linkedin_placeholder.text(text)

        if blog_placeholder is not None and "blog_writer" in raw_by_node:
            body = _extract_streaming_field(raw_by_node["blog_writer"], "body_markdown")
            if body:
                blog_placeholder.markdown(body)
        if linkedin_placeholder is not None and "linkedin_writer" in raw_by_node:
            text = _extract_streaming_field(raw_by_node["linkedin_writer"], "text")
            if text:
                linkedin_placeholder.text(text)
        return None

    await _with_graph(op)


async def _read_state(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await _with_graph(lambda graph: graph.aget_state(config))
    return snapshot.values or {}


async def _approve_draft(thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    await _with_graph(lambda graph: graph.aupdate_state(config, {"human_approved_override": True}))


async def _delete_session_thread(thread_id: str) -> None:
    # Removes the checkpointer's own conversation data for this session, on
    # top of the registry row (purged separately by the caller via
    # registry.hard_delete_session). Only called for a *permanent* delete
    # ("Delete forever" from Trash) — the regular sidebar delete button is a
    # soft delete (registry.soft_delete_session) that leaves this untouched
    # so the conversation is still there if the session gets restored.
    await _with_graph(lambda graph: graph.checkpointer.adelete_thread(thread_id))


def _chat_log_from_state(state: dict, routes_by_turn: dict[int, list[str]] | None = None) -> list[dict]:
    """Reconstructs the visible transcript when switching to a resumed session
    — st.session_state.chat_log is local to this browser tab and won't have
    that session's history otherwise.

    No graph node ever appends a non-HumanMessage to `messages` (agent output
    lives in dedicated state keys like blog_post/linkedin_post instead), so
    every entry here is actually a HumanMessage — a synthetic assistant reply
    is added after each one to mirror the live turn's "Done — see the results
    below." bubble, with that turn's persisted routing trace (if any) attached.
    """
    routes_by_turn = routes_by_turn or {}
    log = []
    turn_index = 0
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            turn_index += 1
            log.append({"role": "user", "content": message.content})
            log.append(
                {
                    "role": "assistant",
                    "content": "Done — see the results below.",
                    "routes": routes_by_turn.get(turn_index, []),
                }
            )
        else:
            log.append({"role": "assistant", "content": "Done — see the results below."})
    return log


def _provider_labels(settings) -> dict[str, str]:
    # Named after the actually-configured model (settings.*_model, which env
    # vars like OPENAI_MODEL override) rather than a hardcoded display name,
    # so this can't silently drift from what's really being called.
    return {
        "openai": f"OpenAI ({settings.openai_model})",
        "anthropic": f"Claude ({settings.anthropic_model})",
        "gemini": f"Gemini ({settings.google_model})",
    }


def _render_provider_selector(settings) -> str:
    labels = _provider_labels(settings)
    with st.sidebar:
        st.header("Model")
        options = list(labels)
        default_index = options.index(settings.llm_primary_provider) if settings.llm_primary_provider in options else 0
        selected = st.selectbox(
            "LLM provider",
            options=options,
            index=default_index,
            format_func=lambda key: labels[key],
            key="llm_provider_choice",
        )
        st.caption("Provider is fixed for a session — start a new session to switch.")
        st.divider()
    return selected


def _format_updated_at(iso_timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return iso_timestamp
    return f"{dt.strftime('%b')} {dt.day}, {dt.year}, {dt.strftime('%I:%M %p').lstrip('0')}"


def _switch_to_new_session() -> None:
    st.session_state.session_id = new_session_id()
    st.session_state.chat_log = []


def _open_session(registry, session_id: str) -> None:
    st.session_state.session_id = session_id
    resumed_state = _run_async(_read_state(session_id))
    routes_by_turn = registry.get_routes(session_id)
    st.session_state.chat_log = _chat_log_from_state(resumed_state, routes_by_turn)


def _render_active_session_row(registry, session: dict) -> None:
    """One row of the main (non-trashed) session list: title button, then
    archive/unarchive, then delete (soft — moves it to Trash) on the far
    right, per the sidebar's icon layout."""
    session_id = session["session_id"]
    is_current = session_id == st.session_state.session_id
    title_col, archive_col, delete_col = st.columns([6, 1, 1])
    with title_col:
        clicked = st.button(
            ("• " if is_current else "") + session["title"],
            key=f"session_{session_id}",
            # help=f"Updated {_format_updated_at(session['updated_at'])} · {session['turn_count']} turn(s)",
            use_container_width=True,
            disabled=is_current,
        )
        if clicked:
            _open_session(registry, session_id)
            st.rerun()
    with archive_col:
        if session["archived"]:
            if st.button(
                "", icon="↩️", key=f"unarchive_{session_id}", help="Unarchive session", use_container_width=True
            ):
                registry.set_archived(session_id, False)
                st.rerun()
        else:
            if st.button(
                "", icon="🗄️", key=f"archive_{session_id}", help="Archive session", use_container_width=True
            ):
                registry.set_archived(session_id, True)
                if is_current:
                    _switch_to_new_session()
                st.rerun()
    with delete_col:
        with st.popover("", icon="🗑️", help="Delete session", use_container_width=True):
            st.write(f"Delete **{session['title']}**? It moves to Trash, recoverable for {_retention_days()} day(s).")
            if st.button("Delete", key=f"delete_{session_id}", type="primary"):
                registry.soft_delete_session(session_id)
                if is_current:
                    _switch_to_new_session()
                st.rerun()


def _render_trashed_session_row(registry, session: dict) -> None:
    """One row of the Trash list: restore on the left, permanent delete (with
    its own confirmation, since this one purges the checkpointer thread too
    and can't be undone) on the far right."""
    session_id = session["session_id"]
    title_col, restore_col, purge_col = st.columns([6, 1, 1])
    with title_col:
        st.caption(session["title"])
    with restore_col:
        if st.button("", icon="↩️", key=f"restore_{session_id}", help="Restore session", use_container_width=True):
            registry.restore_session(session_id)
            st.rerun()
    with purge_col:
        with st.popover("", icon="🗑️", help="Delete forever", use_container_width=True):
            st.write(f"Permanently delete **{session['title']}**? This can't be undone.")
            if st.button("Delete forever", key=f"purge_{session_id}", type="primary"):
                _run_async(_delete_session_thread(session_id))
                registry.hard_delete_session(session_id)
                if session_id == st.session_state.session_id:
                    _switch_to_new_session()
                st.rerun()


def _retention_days() -> int:
    return get_settings().session_retention_days


def _render_sidebar(registry) -> None:
    with st.sidebar:
        st.header("Sessions")
        if st.button("+ New session", use_container_width=True):
            _switch_to_new_session()
            st.rerun()

        st.divider()
        all_sessions = registry.list_sessions(include_archived=True)
        active_sessions = [s for s in all_sessions if not s["archived"]]
        archived_sessions = [s for s in all_sessions if s["archived"]]

        for session in active_sessions:
            _render_active_session_row(registry, session)

        if archived_sessions:
            st.divider()
            with st.expander(f"Archived ({len(archived_sessions)})"):
                for session in archived_sessions:
                    _render_active_session_row(registry, session)

        trashed_sessions = registry.list_trashed_sessions()
        if trashed_sessions:
            st.divider()
            with st.expander(f"Trash ({len(trashed_sessions)})"):
                st.caption(f"Permanently deleted after {_retention_days()} day(s).")
                for session in trashed_sessions:
                    _render_trashed_session_row(registry, session)


def _copy_to_clipboard_button(text: str, label: str, key: str) -> None:
    # Streamlit has no native clipboard widget, so this renders a small HTML/JS
    # button in its own iframe. json.dumps encodes `text`/`label` as JS string
    # literals (handling embedded quotes/newlines/unicode), but json.dumps
    # itself uses double quotes -- nesting that directly inside a
    # double-quoted onclick="..." attribute lets the FIRST embedded `"`
    # prematurely close the attribute, corrupting the markup (verified: this
    # rendered as literal escaped text instead of a working button). Fix:
    # html.escape(..., quote=True) turns those `"` into `&quot;`, which the
    # browser correctly decodes back to `"` within the attribute value.
    text_js = html.escape(json.dumps(text), quote=True)
    label_js = html.escape(json.dumps(label), quote=True)
    components.html(
        f"""
        <button id="{key}" onclick="
            navigator.clipboard.writeText({text_js});
            var btn = document.getElementById('{key}');
            btn.innerText = 'Copied!';
            setTimeout(function() {{ btn.innerText = {label_js}; }}, 1500);
        " style="
            width: 100%; padding: 0.5rem 1rem; border-radius: 0.5rem;
            border: 1px solid rgba(49, 51, 63, 0.2); background: white;
            font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont,
                sans-serif; font-size: 1rem; cursor: pointer;
        ">{label}</button>
        """,
        height=45,
    )


def _render_dashboard(thread_id: str, state: dict) -> None:
    st.divider()
    st.header("Content Dashboard")

    quality_report = state.get("quality_report")
    human_approved = bool(state.get("human_approved_override"))
    requires_review = bool(quality_report and quality_report.get("requires_human_review"))

    if quality_report:
        if requires_review and not human_approved:
            st.error(
                "Quality gate failed after the revision cap was reached — this run "
                "has stopped. Review the draft below before deciding whether to export it."
            )
            for issue in quality_report.get("issues", []):
                st.write(f"- {issue}")
            if st.button("Approve this draft for export"):
                _run_async(_approve_draft(thread_id))
                st.rerun()
        elif quality_report.get("passed"):
            st.success("Quality gate passed.")
        elif requires_review and human_approved:
            st.warning("Quality gate did not pass, but this draft was manually approved for export.")

    blog_post = state.get("blog_post")
    if blog_post:
        st.subheader(blog_post.get("title") or "Blog Post")
        st.caption(blog_post.get("meta_description", ""))
        st.markdown(blog_post.get("body_markdown", ""))

    linkedin_post = state.get("linkedin_post")
    if linkedin_post:
        st.subheader("LinkedIn Post")
        st.text(linkedin_post.get("text", ""))
        st.caption(" ".join(linkedin_post.get("hashtags", [])))

    image_assets = state.get("image_assets") or []
    if image_assets:
        st.subheader("Image")
        for asset in image_assets:
            if asset.get("url") or asset.get("path"):
                st.image(asset.get("url") or asset.get("path"), caption=asset.get("alt_text", ""))
            else:
                st.warning("Image generation failed after retries — showing a placeholder for this draft.")

    research_findings = state.get("research_findings") or []
    if research_findings:
        with st.expander(f"Research sources ({len(research_findings)})"):
            for finding in research_findings:
                st.markdown(f"- [{finding['title']}]({finding['url']})" if finding.get("url") else f"- {finding['title']}")

    export_unlocked = bool(quality_report) and (quality_report.get("passed") or human_approved)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Export Blog (Markdown)",
            data=blog_to_markdown(blog_post) if blog_post else "",
            file_name=blog_export_filename(blog_post) if blog_post else "blog-post.md",
            mime="text/markdown",
            disabled=not export_unlocked or not blog_post,
            use_container_width=True,
        )
    with col2:
        if export_unlocked and linkedin_post:
            _copy_to_clipboard_button(
                linkedin_post_to_text(linkedin_post),
                label="Copy LinkedIn Post",
                key=f"copy_linkedin_{thread_id}",
            )
        else:
            st.button("Copy LinkedIn Post", disabled=True, use_container_width=True)


def main() -> None:
    st.title("ContentAlchemy 📝")
    st.caption("Multi-agent content marketing assistant")

    settings = get_settings()

    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    selected_provider = _render_provider_selector(settings)

    registry = get_registry_conn() if settings.session_store_backend in ("sqlite", "postgres") else None
    if registry is not None:
        _render_sidebar(registry)
        # Warm the persistent checkpointer connection here, on this (the
        # Streamlit script) thread, before any _run_async() call below can
        # reach get_persistent_graph() from inside the background loop
        # instead — see get_persistent_graph's docstring for why that ordering
        # matters.
        get_persistent_graph()

    for message in st.session_state.chat_log:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            routes = message.get("routes")
            if routes:
                with st.expander(f"Agent routing ({len(routes)} steps)"):
                    for line in routes:
                        st.write(line)

    user_query = st.chat_input("What content should I create?")
    if user_query:
        st.session_state.chat_log.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        thread_id = st.session_state.session_id
        existing_state = _run_async(_read_state(thread_id))
        # 1-based, mirrors sessions.turn_count: counts prior HumanMessages (one per
        # turn) so this turn's routing trace can be keyed and looked up the same
        # way whether or not the registry backs the session store.
        turn_index = sum(1 for m in existing_state.get("messages", []) if isinstance(m, HumanMessage)) + 1
        if not existing_state:
            input_state = initial_state(thread_id, user_query, llm_provider=selected_provider)
            if registry is not None:
                registry.record_start(thread_id, derive_session_title(user_query))
        else:
            input_state = new_turn_input(user_query)
            if registry is not None:
                registry.record_turn(thread_id)

        route_log: list[str] = []
        with st.chat_message("assistant"):
            # Created before the status block so the growing blog/LinkedIn text
            # renders above the collapsed progress pill, not hidden inside it.
            blog_placeholder = st.empty()
            linkedin_placeholder = st.empty()
            with st.status("Working...", expanded=True) as status:
                try:
                    _run_async(
                        _run_turn(
                            input_state,
                            thread_id,
                            status=status,
                            provider_label=_provider_labels(settings)[selected_provider],
                            blog_placeholder=blog_placeholder,
                            linkedin_placeholder=linkedin_placeholder,
                            ctx=get_script_run_ctx(),
                            route_log=route_log,
                        )
                    )
                except Exception:
                    status.update(label="Failed", state="error")
                    raise
                status.update(label="Done", state="complete")

        st.session_state.chat_log.append(
            {"role": "assistant", "content": "Done — see the results below.", "routes": route_log}
        )
        if registry is not None:
            registry.record_routes(thread_id, turn_index, route_log)
        st.rerun()

    state = _run_async(_read_state(st.session_state.session_id))
    if state:
        _render_dashboard(st.session_state.session_id, state)


if __name__ == "__main__":
    main()
