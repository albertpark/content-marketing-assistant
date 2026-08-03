"""Quality & Enhancement Pipeline gates: structural, SEO, brand, and factual checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.config import get_settings

if TYPE_CHECKING:
    from src.workflow.state_management import AgentState


def _structural_gate(content_package: dict) -> list[str]:
    """The only real gate implemented in milestone 1: required fields present and
    non-trivial. Returns a list of human-readable issues (empty = pass)."""
    issues = []
    blog = content_package.get("blog", {})
    linkedin = content_package.get("linkedin", {})

    if not blog.get("title"):
        issues.append("Blog post is missing a title.")
    if not blog.get("body_markdown") or len(blog["body_markdown"].split()) < 50:
        issues.append("Blog post body is missing or too short (fewer than 50 words).")
    if not blog.get("meta_description"):
        issues.append("Blog post is missing a meta description.")
    if not linkedin.get("text"):
        issues.append("LinkedIn post is missing text.")
    return issues


def _seo_gate(content_package: dict) -> list[str]:
    """Not yet implemented (milestone 2) — auto-pass. See docs/hld.md open questions."""
    return []


def _brand_gate(content_package: dict) -> list[str]:
    """Not yet implemented (milestone 2) — auto-pass."""
    return []


def _facts_gate(content_package: dict) -> list[str]:
    """Not yet implemented (milestone 2) — auto-pass."""
    return []


def run_gates(content_package: dict) -> dict:
    """Runs every gate and returns {gates: {name: passed}, issues: [...], passed: bool}."""
    gate_results = {
        "structural": _structural_gate(content_package),
        "seo": _seo_gate(content_package),
        "brand": _brand_gate(content_package),
        "facts": _facts_gate(content_package),
    }
    issues = [issue for gate_issues in gate_results.values() for issue in gate_issues]
    return {
        "gates": {name: len(gate_issues) == 0 for name, gate_issues in gate_results.items()},
        "issues": issues,
        "passed": len(issues) == 0,
    }


async def quality_pipeline_node(state: "AgentState") -> dict:
    """The only place that mutates revision_count / quality_report.capped /
    requires_human_review — route_after_quality (src/core/router.py) only reads
    what this node already decided; LangGraph conditional-edge functions can't
    return state updates themselves."""
    content_package = state.get("content_package") or {}
    result = run_gates(content_package)

    if result["passed"]:
        return {
            "quality_report": {**result, "capped": False, "requires_human_review": False},
            "revision_feedback": None,
        }

    revision_count = state.get("revision_count", 0)
    cap = get_settings().revision_loop_cap

    if revision_count < cap:
        return {
            "quality_report": {**result, "capped": False, "requires_human_review": False},
            "revision_count": revision_count + 1,
            "revision_feedback": "\n".join(result["issues"]),
        }

    # Cap reached: hard stop, human-in-the-loop. blog_post/linkedin_post/
    # image_assets/content_package are intentionally left untouched (not returned)
    # so the last draft survives for manual review. human_approved_override is
    # explicitly reset to False: this is a NEW failing draft, so any approval a
    # human gave a previous draft must not silently carry over and unlock export
    # for a draft they never actually saw.
    return {
        "quality_report": {**result, "capped": True, "requires_human_review": True},
        "human_approved_override": False,
    }
