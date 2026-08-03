"""Streamlit entrypoint: chat interface, content dashboard, research panel, export tools."""

from __future__ import annotations

import asyncio
import html
import json
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage

from src.core.config import get_settings
from src.utils.export_tools import blog_export_filename, blog_to_markdown, linkedin_post_to_text
from src.workflow.langgraph_workflow import build_graph
from src.workflow.state_management import (
    derive_session_title,
    get_registry_connection,
    initial_state,
    new_session_id,
    new_turn_input,
    open_checkpointer,
)

st.set_page_config(page_title="ContentAlchemy", page_icon="🧪", layout="wide")

# Hides just the "Record a screencast" entry from Streamlit's built-in main
# menu, leaving Print/theme toggle/Rerun/Deploy/etc. untouched. Streamlit has
# no official per-item toggle for this (client.toolbarMode is all-or-nothing
# for the whole menu) — this targets the item's data-testid, which Streamlit
# derives from its internal React key ("recordScreencast"). Unofficial: this
# testid isn't a public API and could change in a future Streamlit version,
# in which case this simply stops matching and the item reappears (fails
# open, not broken).
st.markdown(
    '<style>[data-testid="stMainMenuItem-recordScreencast"] { display: none; }</style>',
    unsafe_allow_html=True,
)


def _run_async(coro):
    # SelectorEventLoop, not asyncio.run()'s Windows default ProactorEventLoop —
    # required by psycopg's async mode (the postgres backend).
    return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)


@st.cache_resource
def get_graph():
    # Only used for the "memory" backend: InMemorySaver has no event-loop/OS
    # binding, so it's safe to build once and reuse across reruns — which is
    # required, since a fresh InMemorySaver would be an empty store. "sqlite"
    # and "postgres" use open_checkpointer() instead, opened fresh per call
    # (see state_management.get_checkpointer's docstring for why).
    return build_graph()


@st.cache_resource
def get_registry_conn():
    return get_registry_connection(get_settings())


async def _with_graph(op):
    """Runs op(graph) against a correctly-scoped graph + checkpointer for the
    configured backend. op: an async-returning callable taking the graph."""
    settings = get_settings()
    if settings.session_store_backend in ("sqlite", "postgres"):
        async with open_checkpointer(settings) as saver:
            return await op(build_graph(checkpointer=saver))
    return await op(get_graph())


# Static per-node description of what each graph node actually does, shown in the
# expanded progress panel. "LLM" gets the live-selected provider spliced in by
# _node_progress_line; nodes with no LLM call (templates/deterministic gates) say
# so explicitly rather than leaving the user guessing why no model shows up.
_NODE_INFO: dict[str, dict[str, str]] = {
    "orchestrator": {"label": "Orchestrator", "detail": "LLM — routes the request"},
    "research_agent": {
        "label": "Research Agent",
        "detail": "LLM + web_search tool (SerpAPI → Perplexity fallback)",
    },
    "content_strategist": {"label": "Content Strategist", "detail": "LLM — builds the content brief"},
    "blog_writer": {"label": "Blog Writer", "detail": "LLM — drafts the blog post"},
    "linkedin_writer": {"label": "LinkedIn Writer", "detail": "Template, no LLM call"},
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


async def _run_turn(input_state: dict, thread_id: str, status=None, provider_label: str = "") -> None:
    """Runs the graph. When `status` (an st.status container) is given, streams
    per-node start/finish events into it live instead of blocking silently."""
    config = {"configurable": {"thread_id": thread_id}}

    async def op(graph):
        if status is None:
            return await graph.ainvoke(input_state, config=config)
        async for event in graph.astream(input_state, config=config, stream_mode="debug"):
            node_name = event.get("payload", {}).get("name")
            if not node_name:
                continue
            label, detail = _node_progress_line(node_name, provider_label)
            if event["type"] == "task":
                status.update(label=f"Working — {label}...")
                status.write(f"▶ {label} started" + (f" — {detail}" if detail else ""))
            elif event["type"] == "task_result":
                status.write(f"✓ {label} done")
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
    st.title("ContentAlchemy")
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

        with st.status("Working...", expanded=False) as status:
            try:
                _run_async(
                    _run_turn(
                        input_state,
                        thread_id,
                        status=status,
                        provider_label=_PROVIDER_LABELS[selected_provider],
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
