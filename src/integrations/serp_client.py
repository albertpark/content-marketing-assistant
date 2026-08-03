"""SERP API research client — primary Search tool backend."""

from __future__ import annotations

from serpapi import GoogleSearch

from src.core.config import get_settings
from src.integrations.observability import record_cost, traced_tool
from src.integrations.performance import cached, rate_limited
from src.integrations.resilience import ProviderError, with_retry


@traced_tool("serpapi_search")
@cached()
@with_retry(retry_on=(ProviderError,))
@rate_limited()
def search_serp(query: str, num_results: int = 5) -> list[dict]:
    """Runs a Google search via SerpApi. Returns a list of
    {source, title, url, snippet} dicts. Raises ProviderError on transient failures
    (network errors, SerpApi-reported errors) so with_retry can retry it."""
    settings = get_settings()
    record_cost("serpapi_search")
    try:
        search = GoogleSearch(
            {
                "q": query,
                "api_key": settings.serpapi_api_key,
                "engine": "google",
                "num": num_results,
            }
        )
        results = search.get_dict()
    except Exception as exc:
        raise ProviderError(str(exc)) from exc

    if "error" in results:
        raise ProviderError(results["error"])

    organic = results.get("organic_results", [])[:num_results]
    return [
        {
            "source": "serpapi",
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in organic
    ]
