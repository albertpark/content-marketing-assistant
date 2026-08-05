import pytest

from src.agents import base_agent
from src.integrations import performance


@pytest.fixture(autouse=True)
def _reset_admission_state():
    """Resets the process-wide admission limiter + response cache so tests
    don't leak rate-limit state across each other depending on run order."""
    performance.get_admission_limiter.cache_clear()
    base_agent._RESPONSE_CACHE.clear()
    yield
    performance.get_admission_limiter.cache_clear()
    base_agent._RESPONSE_CACHE.clear()
