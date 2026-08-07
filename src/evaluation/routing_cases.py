"""Labeled routing test cases for scoring the Orchestrator's classification
accuracy in isolation from the rest of the pipeline (see scripts/eval_routing.py).
Add a case here whenever a real user query gets routed to the wrong agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingCase:
    query: str
    expected_targets: frozenset[str]
    # None means "don't check intent for this case" -- the safety-net cases below
    # deliberately leave this unset because the override only touches targets,
    # not the LLM's raw intent classification.
    expected_intent: str | None = None
    has_blog: bool = False
    has_linkedin: bool = False
    has_image: bool = False


ROUTING_CASES: list[RoutingCase] = [
    # --- new_content: nothing exists yet, should always start research ---
    RoutingCase(
        query="Write a new blog post about the future of remote work",
        expected_targets=frozenset({"research"}),
        expected_intent="new_content",
    ),
    RoutingCase(
        query="I want to create content about our new product launch",
        expected_targets=frozenset({"research"}),
        expected_intent="new_content",
    ),
    RoutingCase(
        query="Can you research sustainable packaging trends in e-commerce for me?",
        expected_targets=frozenset({"research"}),
        expected_intent="new_content",
    ),
    RoutingCase(
        query="Draft a post about how AI is changing customer support",
        expected_targets=frozenset({"research"}),
        expected_intent="new_content",
    ),

    # --- refinement: blog already exists, targeted regeneration ---
    RoutingCase(
        query="Make the LinkedIn post punchier",
        expected_targets=frozenset({"linkedin"}),
        expected_intent="refinement",
        has_blog=True,
        has_linkedin=True,
    ),
    RoutingCase(
        query="Rewrite the LinkedIn caption with a stronger hook",
        expected_targets=frozenset({"linkedin"}),
        expected_intent="refinement",
        has_blog=True,
        has_linkedin=True,
    ),
    RoutingCase(
        query="The blog post is too long, tighten it up",
        expected_targets=frozenset({"blog"}),
        expected_intent="refinement",
        has_blog=True,
    ),
    RoutingCase(
        query="Can you regenerate the header image with a brighter color palette?",
        expected_targets=frozenset({"image"}),
        expected_intent="refinement",
        has_blog=True,
        has_image=True,
    ),
    RoutingCase(
        query="Redo both the LinkedIn post and the image, they don't match the new angle",
        expected_targets=frozenset({"linkedin", "image"}),
        expected_intent="refinement",
        has_blog=True,
        has_linkedin=True,
        has_image=True,
    ),
    RoutingCase(
        query="Just package up what we have so I can review it",
        expected_targets=frozenset({"package"}),
        expected_intent="refinement",
        has_blog=True,
        has_linkedin=True,
        has_image=True,
    ),

    # --- safety-net: refinement-sounding language but nothing exists yet ---
    RoutingCase(
        query="Make the LinkedIn post punchier",
        expected_targets=frozenset({"research"}),
        has_blog=False,
    ),
    RoutingCase(
        query="Regenerate the header image",
        expected_targets=frozenset({"research"}),
        has_blog=False,
    ),
]
