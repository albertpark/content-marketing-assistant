"""Perplexity Sonar research client — alternative Search tool backend."""

from __future__ import annotations

import requests

from src.core.config import get_settings
from src.integrations.observability import record_cost, traced_tool
from src.integrations.performance import cached, rate_limited
from src.integrations.resilience import ProviderError, with_retry

_PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


@traced_tool("perplexity_search")
@cached()
@with_retry(retry_on=(ProviderError,))
@rate_limited()
def search_perplexity(query: str, num_results: int = 5) -> list[dict]:
    """Runs a Perplexity Sonar query. Returns a list of {source, title, url, snippet}
    dicts: one synthesized-answer entry followed by one entry per cited source.
    Raises ProviderError on transient failures so with_retry can retry it."""
    settings = get_settings()
    model = settings.perplexity_model or "sonar"
    record_cost("perplexity_search")
    try:
        response = requests.post(
            _PERPLEXITY_URL,
            headers={"Authorization": f"Bearer {settings.perplexity_api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": query}]},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise ProviderError(str(exc)) from exc

    choices = payload.get("choices", [])
    answer = choices[0].get("message", {}).get("content", "") if choices else ""
    citations = payload.get("citations", [])[:num_results]

    results = [
        {"source": "perplexity", "title": "Perplexity synthesis", "url": "", "snippet": answer}
    ]
    results.extend(
        {"source": "perplexity", "title": url, "url": url, "snippet": ""} for url in citations
    )
    return results[: num_results + 1]
