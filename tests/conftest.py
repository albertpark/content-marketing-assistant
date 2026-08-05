import pytest

from src.agents import base_agent
from src.integrations import performance


@pytest.fixture(autouse=True)
def _reset_admission_state():
    """performance.get_admission_limiter() and base_agent._RESPONSE_CACHE are
    process-wide singletons (admission control only works if state is shared
    across the many short-lived BaseAgent instances a real run creates — see
    their docstrings). Left alone, they'd carry token/cache state across test
    functions and files, making later tests spuriously see degraded responses
    depending on run order. Reset around every test for full-capacity, empty-
    cache isolation regardless of what ran before."""
    performance.get_admission_limiter.cache_clear()
    base_agent._RESPONSE_CACHE.clear()
    yield
    performance.get_admission_limiter.cache_clear()
    base_agent._RESPONSE_CACHE.clear()
