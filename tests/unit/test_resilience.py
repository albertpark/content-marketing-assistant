import time

import pytest

from src.integrations import resilience
from src.integrations.resilience import (
    AllRetriesExhaustedError,
    ProviderError,
    RetryPolicy,
    with_retry,
)


async def _noop_async_sleep(_seconds):
    return None


def test_sync_retries_exact_count_then_raises(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    calls = []

    @with_retry(policy=RetryPolicy(max_retries=3, backoff_seconds=0))
    def flaky():
        calls.append(1)
        raise ProviderError("boom")

    with pytest.raises(AllRetriesExhaustedError):
        flaky()

    assert len(calls) == 3


def test_sync_succeeds_before_exhausting_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    attempts = {"count": 0}

    @with_retry(policy=RetryPolicy(max_retries=3, backoff_seconds=0))
    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ProviderError("transient")
        return "ok"

    assert flaky() == "ok"
    assert attempts["count"] == 2


def test_non_retryable_exception_propagates_immediately(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    calls = []

    @with_retry(policy=RetryPolicy(max_retries=3, backoff_seconds=0))
    def flaky():
        calls.append(1)
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        flaky()

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_retries_exact_count_then_raises(monkeypatch):
    monkeypatch.setattr(resilience.asyncio, "sleep", _noop_async_sleep)
    calls = []

    @with_retry(policy=RetryPolicy(max_retries=3, backoff_seconds=0))
    async def flaky():
        calls.append(1)
        raise ProviderError("boom")

    with pytest.raises(AllRetriesExhaustedError):
        await flaky()

    assert len(calls) == 3
