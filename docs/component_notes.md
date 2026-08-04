# ContentAlchemy — Presenter Notes

Quick-reference cards for walking through [`architecture.svg`](../architecture.svg)
live — in the demo video, a review, or a stakeholder walkthrough. Full technical detail
is in [`hld.md`](hld.md); this is the "what to say when pointing at this box" version,
one card per component, in diagram order (top to bottom).

Each card: **one-liner** → **say this** → **likely question**.

---

**Interface**
Chat + dashboard + research panel + export, all in one surface.
> "Users don't pick an 'agent' — they just ask for content. Everything downstream is
> automatic."
Q: Is it conversational or form-based? → Conversational primary, dashboard for review.

---

**Orchestrator**
The Query Handler + Coordinator. Reads intent, decides what runs.
> "This is the router — it decides if you're asking for new content, a refinement of
> something already made, or just research."
Q: How does it route? → LLM-based intent classification against the conversation state.

---

**Research Agent** ⇄ **Search tools**
Web research with a swappable backend (SERP API + GPT, or Perplexity Sonar).
> "The research agent doesn't hit one search API — it goes through a tool loop with
> retries, and the backend itself is swappable."
Q: Why two search providers? → SERP gives control over sources; Perplexity gives
faster pre-analyzed answers. Provider abstraction lets us swap without touching the
agent logic — this is also the fallback path if one provider is down.

---

**Content Strategist**
Turns raw research into a structured brief — angle, outline, keywords, and an image
brief.
> "This is the hand-off point between 'here's what we found' and 'here's what to
> write' — and it also tells the Image Generator what the header image should
> actually depict, not just its title."

---

**Blog Writer**
Full SEO-optimized long-form post. **Runs first**, deliberately.
> "Everything else derives from the blog — the LinkedIn hook links back to it, and the
> image is generated from the finished post, not a guess at what the post might say."
Q: Why not generate everything in parallel? → Because LinkedIn and the image need the
*finished* blog as input, not the raw brief — sequencing this way avoids drift between
the three formats.

---

**LinkedIn Writer**
Short-form post, hook + link back to the blog. Runs parallel to Image Generator.
> "Same underlying content, re-cut for a different platform's constraints — length,
> hashtags, tone."

---

**Image Generator** → **Image tools**
Its own agent with a dedicated system prompt: turns the Content Strategist's image
brief + the finished blog title into one image-generation prompt, then calls Image
tools' fallback chain (GPT Image → fallback → placeholder).
> "This has its own system prompt so every image comes out in the same house style —
> composition, mood, no stray embedded text — instead of drifting per topic. If the
> primary image provider fails or times out, we don't just show a broken image —
> there's a fallback provider before we give up."
Q: Is this a tool loop like Research Agent? → No — one LLM call to write the image
prompt, one call to generate it. No back-and-forth tool calling.

---

**Synthesizer**
Assembles blog + LinkedIn + image into one cross-linked package.
> "This is where the three parallel outputs get reconciled — links checked, formats
> matched up — before anything goes to quality review."

---

**Quality & Enhancement Pipeline**
Structural, SEO, brand-voice, and factual checks on the assembled package.
> "If something fails a gate here — say, brand voice drifted on the LinkedIn post —
> it doesn't just get shipped with a flaw. It loops back for a revision pass."
Q: What's the revision loop look like on the diagram? → The dashed conditional edge
back up to Blog Writer / Content Strategist, capped retries so it can't loop forever.

---

**Content Dashboard**
Human review, edit, approve — before publishing.
> "We keep a human in the loop here deliberately — AI content should be reviewable,
> not auto-published."

---

**Export Tools**
Per-platform formats: Markdown, HTML, plain text, PDF, images.
> "Whatever the user is publishing into, they get a format that's ready to paste in,
> not something they have to reformat by hand."

---

**Session Store** *(side rail, spans the whole pipeline)*
LangGraph checkpoints + conversation state.
> "Every stage can read and write conversation state — that's what makes 'make the
> LinkedIn post punchier' work as a follow-up instead of starting the whole pipeline
> over."

---

**Integrations Layer** *(side panel, right)*
OpenAI · Anthropic · Gemini · Perplexity · SERP · image gen, with shared **resilience**
(retry/backoff/fallback) and **performance** (cache/rate-limit/cost) concerns.
> "This is deliberately centralized — every agent that calls out to a provider goes
> through the same retry and caching logic instead of reimplementing it."
Q: Why not just put retry logic in each agent? → Duplication and drift — a fix or a
new provider only needs to happen once here.

---

## Legend (as shown in the diagram footer)

- **Fixed edge** — solid arrow, deterministic hand-off.
- **Conditional edge (revision loop)** — dashed arrow, taken only on a failed quality
  gate.
- **Tool loop (bidirectional, capped retries)** — agent ⇄ tool, e.g. Research Agent ⇄
  Search tools (the LLM decides when to call the tool and loops). Image Generator →
  Image tools is a single-shot call, not this pattern.

## Suggested demo narration order

1. Interface → Orchestrator (show intent routing on a fresh request)
2. Research Agent ⇄ Search tools (show a research result with citations)
3. Content Strategist → Blog Writer (show the brief → finished post)
4. LinkedIn Writer + Image Generator running off the same blog (parallelism)
5. Synthesizer → Quality & Enhancement Pipeline (trigger one failure → show the
   revision loop firing)
6. Content Dashboard → Export Tools (show an edit, then an export)
7. Close on Session Store: ask a multi-turn follow-up ("shorten the LinkedIn post")
   and show it re-entering the graph with context intact.
