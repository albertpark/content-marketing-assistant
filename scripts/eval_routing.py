"""Scores the Orchestrator's routing accuracy against a labeled set of
(query, expected_targets) pairs -- src/evaluation/routing_cases.py. Unlike
run_evaluation.py (which scores generated content quality end-to-end), this
only exercises OrchestratorAgent.decide(), so it's cheap enough to re-run
after every prompt tweak to check whether routing got better or worse.

Usage:
    python scripts/eval_routing.py
    python scripts/eval_routing.py --provider anthropic
    python scripts/eval_routing.py --threshold 0.9   # exit 1 if accuracy dips below this
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Running this file directly (`python scripts/eval_routing.py`) puts scripts/
# on sys.path, not the project root -- `import src...` needs the latter.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.query_handler import OrchestratorAgent  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.evaluation.routing_cases import ROUTING_CASES, RoutingCase  # noqa: E402


class CaseResult:
    def __init__(self, case: RoutingCase, actual_targets: list[str] | None, actual_intent: str | None, error: str | None = None):
        self.case = case
        self.actual_targets = actual_targets
        self.actual_intent = actual_intent
        self.error = error

    @property
    def targets_correct(self) -> bool:
        return self.error is None and set(self.actual_targets or []) == self.case.expected_targets

    @property
    def intent_checked(self) -> bool:
        return self.case.expected_intent is not None

    @property
    def intent_correct(self) -> bool:
        return self.error is None and self.actual_intent == self.case.expected_intent


def _state_for(case: RoutingCase, provider: str | None) -> dict:
    return {
        "user_query": case.query,
        "blog_post": {"title": "Existing Post"} if case.has_blog else None,
        "linkedin_post": {"text": "Existing LinkedIn post"} if case.has_linkedin else None,
        "image_assets": [{"url": "https://example.com/image.png"}] if case.has_image else [],
        "llm_provider": provider,
    }


async def _run_case(case: RoutingCase, provider: str | None, semaphore: asyncio.Semaphore) -> CaseResult:
    async with semaphore:
        try:
            decision = await OrchestratorAgent(provider=provider).decide(_state_for(case, provider))
            return CaseResult(case, decision.targets, decision.intent)
        except Exception as exc:  # a single bad case must not abort the whole run
            return CaseResult(case, None, None, error=str(exc))


def _print_result(result: CaseResult) -> None:
    case = result.case
    if result.error is not None:
        print(f"ERROR  {case.query!r} -> {result.error}")
        return

    status = "PASS" if result.targets_correct else "FAIL"
    line = (
        f"{status}  {case.query!r}\n"
        f"       expected targets={sorted(case.expected_targets)}  "
        f"actual targets={sorted(result.actual_targets or [])}"
    )
    if result.intent_checked:
        intent_status = "ok" if result.intent_correct else "MISMATCH"
        line += f"\n       expected intent={case.expected_intent!r}  actual intent={result.actual_intent!r}  [{intent_status}]"
    print(line)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="Force a single LLM provider (openai/anthropic/gemini)")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None, help="Exit 1 if routing accuracy falls below this")
    args = parser.parse_args()

    get_settings()  # loads .env before any provider client is constructed

    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(_run_case(case, args.provider, semaphore) for case in ROUTING_CASES))

    for result in results:
        _print_result(result)

    total = len(results)
    routes_correct = sum(1 for r in results if r.targets_correct)
    route_accuracy = routes_correct / total if total else 0.0

    intent_checked = [r for r in results if r.intent_checked]
    intents_correct = sum(1 for r in intent_checked if r.intent_correct)
    intent_accuracy = intents_correct / len(intent_checked) if intent_checked else None

    print(f"\nRouting accuracy:  {routes_correct}/{total} = {route_accuracy:.1%}")
    if intent_accuracy is not None:
        print(f"Intent accuracy:   {intents_correct}/{len(intent_checked)} = {intent_accuracy:.1%}")

    if args.threshold is not None and route_accuracy < args.threshold:
        print(f"\nFAILED: routing accuracy {route_accuracy:.1%} is below threshold {args.threshold:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop))
