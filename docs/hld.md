# ContentAlchemy — High-Level Design

Diagram: [`architecture.svg`](../architecture.svg)
Presenter reference: [`component_notes.md`](component_notes.md)

## 1. Overview

ContentAlchemy is a multi-agent content marketing assistant built on LangGraph. A single
user request (e.g. "write me a blog and LinkedIn post about X") flows through one
orchestrated graph: research → strategy → drafting (blog, then LinkedIn + image in
parallel) → synthesis → quality/enhancement → dashboard/export.

This maps to the six agents required by the problem statement (Query Handler, Deep
Research, Content Strategist, SEO Blog Writer, LinkedIn Post Writer, Image Generation),
plus two cross-cutting layers (Session Store, Integrations Layer) that aren't agents
but are required for the system to be production-grade.

## 2. Data flow

```
Interface
  → Orchestrator (Query Handler + Coordinator)
    → Research Agent  ⇄ Search tools (SERP + Perplexity)
    → Content Strategist
    → Blog Writer                      (runs first — full SEO post)
        ├→ LinkedIn Writer             (hook + link to blog)
        └→ Image Generator ⇄ Image tools (GPT Image + fallback)
                (from finished blog)
    → Synthesizer                      (assembles final cross-linked package)
    → Quality & Enhancement Pipeline   (structural · SEO · brand · facts)
        ├→ Content Dashboard           (review, edit, approve)
        └→ Export Tools                (per-platform formats, images)
```

Key sequencing decision: **Blog Writer runs before LinkedIn/Image**, because the
LinkedIn post links back to the blog and the image is generated from the finished blog
content, not from the raw brief. LinkedIn and Image generation then run as parallel
branches off the Blog Writer output and rejoin at the Synthesizer.

The dashed feedback edge from the Quality & Enhancement Pipeline back up to the
Blog Writer / Content Strategist area is the **revision loop**: failed quality gates
(structural, SEO, brand, factual) route content back for another pass rather than
failing the whole run.

## 3. Component catalog

### Interface
- **Purpose:** entry point for the user — chat, dashboard, research panel, export.
- **In:** user message / follow-up turn.
- **Out:** structured request to Orchestrator; renders final package back to the user.
- **Notes:** conversational (multi-turn) and dashboard modes share the same session.

### Orchestrator — Query Handler + Coordinator
- **Purpose:** intent classification and dispatch. Decides which downstream agents run
  and in what order (e.g. "just research" vs. "full blog + LinkedIn + image").
- **In:** user request + conversation state (from Session Store).
- **Out:** routed task(s) to Research Agent (default entry point for new content
  requests) or directly to later stages for refinement requests on existing content.
- **Notes:** this is the "intelligent routing system" called out in the problem
  statement's deliverables and rubric (Multi-Agent Architecture, 25%).

### Research Agent
- **Purpose:** web research and source synthesis for the topic.
- **In:** topic/brief from Orchestrator.
- **Out:** research findings with citations → Content Strategist.
- **Tool loop:** bidirectional, capped-retry loop against **Search tools**.

### Search tools (SERP + Perplexity)
- **Purpose:** pluggable research backends. SERP API gives more control over sources;
  Perplexity Sonar gives faster, pre-analyzed results.
- **Notes:** provider abstraction here is what lets the fallback story work — SERP as
  primary, Perplexity as fallback (or vice versa), swappable without touching the
  Research Agent's logic.

### Content Strategist
- **Purpose:** formats raw research into a structured content brief (angle, outline,
  key points, target keywords) that downstream writers consume.
- **In:** research findings.
- **Out:** brief → Blog Writer.

### Blog Writer
- **Purpose:** produces the full long-form, SEO-optimized post. Runs first because
  everything downstream (LinkedIn hook, image prompt) derives from it.
- **In:** content brief.
- **Out:** finished blog post → fans out to LinkedIn Writer and Image Generator.

### LinkedIn Writer
- **Purpose:** short-form, platform-optimized post with a hook and a link back to the
  blog. Runs in parallel with Image Generator.
- **In:** finished blog post.
- **Out:** LinkedIn post draft → Synthesizer.

### Image Generator
- **Purpose:** produces a visual derived from the finished blog (not the raw brief),
  so imagery matches the actual published content.
- **In:** finished blog post.
- **Out:** image asset(s) → Synthesizer.
- **Tool loop:** bidirectional, capped-retry loop against **Image tools**.

### Image tools (GPT Image + fallback)
- **Purpose:** pluggable image backends with a fallback chain (primary generator →
  secondary generator → placeholder/stock), matching the FAQ's guidance on image
  failure handling.

### Synthesizer
- **Purpose:** assembles the final cross-linked package — blog + LinkedIn post + image,
  with links and references reconciled across formats.
- **In:** Blog + LinkedIn + Image outputs.
- **Out:** unified content package → Quality & Enhancement Pipeline.

### Quality & Enhancement Pipeline
- **Purpose:** structural, SEO, brand-voice, and factual validation/enhancement pass
  over the assembled package.
- **In:** synthesized package.
- **Out:** on pass → Content Dashboard + Export Tools. On fail → **conditional edge**
  back into the revision loop (Blog Writer / Content Strategist) for another pass,
  capped to avoid infinite loops.
- **Notes:** this is the "Content Quality Pipeline" (5%) and a major input to
  "Content Optimization" (10%) and "Brand Consistency" (5%) in the rubric.

### Content Dashboard
- **Purpose:** human review surface — preview, edit, approve before publishing.
- **Notes:** satisfies the "human-in-the-loop" quality guidance from the FAQ and the
  UX & Interface rubric line (Interface Design, 7%).

### Export Tools
- **Purpose:** per-platform export (Markdown, HTML, plain text, PDF, images; stretch:
  WordPress XML / social-platform-optimized exports).

### Session Store (LangGraph checkpoints, conversation state)
- **Purpose:** persists conversation/workflow state across turns so multi-turn
  refinement ("make the LinkedIn post punchier") re-enters the graph with full
  context instead of starting over.
- **Notes:** implemented via LangGraph's built-in persistence layer (per FAQ Q10).
  Spans the full vertical extent of the pipeline in the diagram because any stage may
  need to read/write conversation state.
- **Backing store decision (resolved):** Postgres, via a hosted Supabase instance,
  is the default for both `development` and `production` (`config/*.yaml`'s
  `session_store.backend: postgres` + `SESSION_STORE_URL` pointing at Supabase's
  *connection pooler*, not its direct-connection host — the direct host is
  IPv6-only and unreachable from many networks). SQLite remains a supported
  fallback backend for fully-offline/no-external-DB use. Both are async-native
  (`AsyncPostgresSaver`/`AsyncSqliteSaver`) and must be opened fresh per graph
  operation rather than cached across requests — their underlying drivers
  (`psycopg`, `aiosqlite`) bind to the event loop that created them, verified
  empirically to break when reused across a second, separate `asyncio.run()`
  call (Streamlit's per-interaction execution model). A lightweight `sessions`
  registry table (title/timestamps/turn count, separate from the checkpoint
  data itself) backs the sidebar dashboard's session list. `memory`
  (`InMemorySaver`) remains available and is what test suites use, to avoid
  touching the real database. Redis was considered but never implemented and
  is no longer the intended direction.

### Integrations Layer
- **Providers:** OpenAI · Anthropic · Claude · Gemini · Perplexity · SERP · image gen.
- **Resilience:** retry / backoff / fallback — shared by all provider calls, not
  reimplemented per agent.
- **Performance:** cache · rate-limit · cost monitoring.
- **Notes:** this is a horizontal concern used by Research Agent, Blog/LinkedIn
  Writers, and Image Generator alike. Centralizing it here is what makes "Service
  Integration" (5%) and "Performance Optimization" (5%) tractable — provider
  abstraction lives in one place instead of being duplicated per agent.

## 4. Edge semantics (see diagram legend)

| Edge type | Meaning |
|---|---|
| Fixed edge (solid arrow) | Deterministic hand-off, always taken. |
| Conditional edge (dashed, revision loop) | Taken only when the Quality & Enhancement Pipeline fails a gate; routes back upstream for revision, capped retries. |
| Tool loop (bidirectional, capped retries) | Agent ⇄ tool call-and-response (Search tools, Image tools), retried up to a cap before falling back or failing gracefully. |

## 5. Rubric cross-reference

| Rubric criterion | Weight | Primarily satisfied by |
|---|---|---|
| Multi-Agent Architecture | 25% | Orchestrator routing + 6 distinct agents with clear separation of concerns |
| LangGraph Workflow | 10% | Orchestrator + Session Store (state mgmt), conditional revision loop |
| Service Integration | 5% | Integrations Layer (provider abstraction, fallback) |
| Content Quality Pipeline | 5% | Quality & Enhancement Pipeline |
| Performance Optimization | 5% | Integrations Layer (cache, rate-limit, cost) |
| Research Quality | 10% | Research Agent + Search tools (multi-source, citations) |
| Content Optimization | 10% | Content Strategist, Blog/LinkedIn Writers, Quality Pipeline (SEO/platform formatting) |
| Visual Content | 5% | Image Generator + Image tools (prompt engineering, fallback) |
| Brand Consistency | 5% | Quality & Enhancement Pipeline (brand check) applied uniformly to all formats |
| Interface Design | 7% | Interface + Content Dashboard + Export Tools |
| Conversation Flow | 3% | Session Store + Orchestrator context handling |
| Code Organization | 4% | Provider abstraction in Integrations Layer, agent separation |
| Documentation | 3% | This document + component_notes.md + README (to be written) |
| Testing | 3% | Not yet designed — see Open Questions |
| Innovation (bonus) | up to 10 | Alternative providers already designed-in via Integrations Layer; stretch export/CMS formats |

## 6. Open questions / not yet designed

- ~~Concrete tech choices for Session Store backing store~~ — **resolved**: Postgres
  (Supabase), SQLite as an offline fallback. See the Session Store component note above.
- Test strategy (unit/integration/e2e split) — problem statement calls for 80%+
  coverage; not reflected in the architecture diagram since it's cross-cutting.
- Exact quality gate thresholds (SEO keyword density, Flesch-Kincaid score, brand
  voice check method) for the Quality & Enhancement Pipeline.
- Deployment target (local vs. Docker vs. cloud) — affects whether the
  Integrations Layer needs to be externalized or can stay in-process (Session
  Store is already externalized via Supabase, independent of this question).
