"""Streamlit entrypoint: chat interface, content dashboard, research panel, export tools."""

from __future__ import annotations

import html
import json
import sys
import threading
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


@st.cache_resource
def get_persistent_graph():
    """"sqlite"/"postgres" backends: builds the graph once against a
    checkpointer connection opened on async_runtime's background loop, then
    reuses both for the app's lifetime. Must be called from main() before any
    _run_async() dispatch touches it — st.cache_resource's factory runs
    synchronously on the calling thread, and async_runtime.run() would
    deadlock if the factory is entered from a coroutine already running on
    the background loop it depends on."""
    return async_runtime.run(async_runtime.build_persistent_graph(get_settings()))


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
) -> None:
    """Runs the graph. When `status` (an st.status container) is given, streams
    per-node start/finish events into it live. When blog_placeholder /
    linkedin_placeholder (st.empty() containers) are given, also streams the
    blog post body / LinkedIn post text into them token-by-token as they're
    generated, instead of only appearing once the whole run finishes.

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
                    status.write(f"▶ {label} started" + (f" — {detail}" if detail else ""))
                elif payload["type"] == "task_result":
                    status.write(f"✓ {label} done")
            elif mode == "messages":
                msg_chunk, metadata = payload
                node_name = metadata.get("langgraph_node")
                if node_name not in ("blog_writer", "linkedin_writer"):
                    continue
                raw_by_node[node_name] = raw_by_node.get(node_name, "") + (msg_chunk.content or "")
                if node_name == "blog_writer" and blog_placeholder is not None:
                    body = _extract_streaming_field(raw_by_node[node_name], "body_markdown")
                    if body:
                        blog_placeholder.markdown(body)
                elif node_name == "linkedin_writer" and linkedin_placeholder is not None:
                    text = _extract_streaming_field(raw_by_node[node_name], "text")
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


def _chat_log_from_state(state: dict) -> list[dict]:
    """Reconstructs the visible transcript when switching to a resumed session
    — st.session_state.chat_log is local to this browser tab and won't have
    that session's history otherwise."""
    log = []
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            log.append({"role": "user", "content": message.content})
        else:
            log.append({"role": "assistant", "content": "Done — see the results below."})
    return log


_PROVIDER_LABELS = {"openai": "OpenAI GPT-4o", "anthropic": "Claude", "gemini": "Gemini"}


def _render_provider_selector(settings) -> str:
    with st.sidebar:
        st.header("Model")
        options = list(_PROVIDER_LABELS)
        default_index = options.index(settings.llm_primary_provider) if settings.llm_primary_provider in options else 0
        selected = st.selectbox(
            "LLM provider",
            options=options,
            index=default_index,
            format_func=lambda key: _PROVIDER_LABELS[key],
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


def _render_sidebar(registry) -> None:
    with st.sidebar:
        st.header("Sessions")
        if st.button("+ New session", use_container_width=True):
            st.session_state.session_id = new_session_id()
            st.session_state.chat_log = []
            st.rerun()

        st.divider()
        for session in registry.list_sessions():
            is_current = session["session_id"] == st.session_state.session_id
            clicked = st.button(
                ("• " if is_current else "") + session["title"],
                key=f"session_{session['session_id']}",
                help=f"Updated {_format_updated_at(session['updated_at'])} · {session['turn_count']} turn(s)",
                use_container_width=True,
                disabled=is_current,
            )
            if clicked:
                st.session_state.session_id = session["session_id"]
                resumed_state = _run_async(_read_state(session["session_id"]))
                st.session_state.chat_log = _chat_log_from_state(resumed_state)
                st.rerun()


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

    user_query = st.chat_input("What content should I create?")
    if user_query:
        st.session_state.chat_log.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        thread_id = st.session_state.session_id
        existing_state = _run_async(_read_state(thread_id))
        if not existing_state:
            input_state = initial_state(thread_id, user_query, llm_provider=selected_provider)
            if registry is not None:
                registry.record_start(thread_id, derive_session_title(user_query))
        else:
            input_state = new_turn_input(user_query)
            if registry is not None:
                registry.record_turn(thread_id)

        with st.chat_message("assistant"):
            # Created before the status block so the growing blog/LinkedIn text
            # renders above the collapsed progress pill, not hidden inside it.
            blog_placeholder = st.empty()
            linkedin_placeholder = st.empty()
            with st.status("Working...", expanded=False) as status:
                try:
                    _run_async(
                        _run_turn(
                            input_state,
                            thread_id,
                            status=status,
                            provider_label=_PROVIDER_LABELS[selected_provider],
                            blog_placeholder=blog_placeholder,
                            linkedin_placeholder=linkedin_placeholder,
                            ctx=get_script_run_ctx(),
                        )
                    )
                except Exception:
                    status.update(label="Failed", state="error")
                    raise
                status.update(label="Done", state="complete")

        st.session_state.chat_log.append({"role": "assistant", "content": "Done — see the results below."})
        st.rerun()

    state = _run_async(_read_state(st.session_state.session_id))
    if state:
        _render_dashboard(st.session_state.session_id, state)


if __name__ == "__main__":
    main()
