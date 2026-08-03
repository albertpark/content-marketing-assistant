"""Shared retry/backoff wrapper used by every outbound provider call."""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from dataclasses import dataclass

from src.core.config import get_settings


class ProviderError(Exception):
    """Raised by provider clients on a retryable failure (timeout, 5xx, rate limit, ...)."""


class AllRetriesExhaustedError(ProviderError):
    """Raised once a retry-wrapped call has exhausted its retry budget."""


@dataclass
class RetryPolicy:
    max_retries: int
    backoff_seconds: float


def _default_policy() -> RetryPolicy:
    settings = get_settings()
    return RetryPolicy(max_retries=settings.max_retries, backoff_seconds=settings.backoff_seconds)


def with_retry(
    policy: RetryPolicy | None = None,
    retry_on: tuple[type[Exception], ...] = (ProviderError,),
):
    """Decorator: retries the wrapped call up to policy.max_retries total attempts,
    sleeping backoff_seconds * 2**attempt between them. Works on sync and async
    callables. Raises AllRetriesExhaustedError (wrapping the last exception) once
    the retry budget is exhausted."""

    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                active_policy = policy or _default_policy()
                last_exc: Exception | None = None
                for attempt in range(active_policy.max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except retry_on as exc:
                        last_exc = exc
                        if attempt < active_policy.max_retries - 1:
                            await asyncio.sleep(active_policy.backoff_seconds * (2**attempt))
                raise AllRetriesExhaustedError(
                    f"{func.__name__} failed after {active_policy.max_retries} attempts"
                ) from last_exc

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            active_policy = policy or _default_policy()
            last_exc: Exception | None = None
            for attempt in range(active_policy.max_retries):
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:
                    last_exc = exc
                    if attempt < active_policy.max_retries - 1:
                        time.sleep(active_policy.backoff_seconds * (2**attempt))
            raise AllRetriesExhaustedError(
                f"{func.__name__} failed after {active_policy.max_retries} attempts"
            ) from last_exc

        return sync_wrapper

    return decorator
