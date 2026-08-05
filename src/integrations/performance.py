"""In-process TTL caching and rate limiting shared by every outbound provider call."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import time
from collections import deque
from functools import lru_cache

from src.core.config import get_settings


def cached(ttl_minutes: int | None = None):
    """In-process TTL cache decorator, keyed on the call's args/kwargs. Defaults to
    get_settings().cache_ttl_minutes. Works on sync and async callables."""

    def decorator(func):
        store: dict[tuple, tuple[object, float]] = {}
        lock = threading.Lock()
        is_async = inspect.iscoroutinefunction(func)

        def ttl_seconds() -> float:
            minutes = ttl_minutes if ttl_minutes is not None else get_settings().cache_ttl_minutes
            return minutes * 60

        def cache_key(args, kwargs) -> tuple:
            return (args, tuple(sorted(kwargs.items())))

        def get_cached(key):
            with lock:
                entry = store.get(key)
                if entry is not None and time.monotonic() < entry[1]:
                    return True, entry[0]
                return False, None

        def set_cached(key, value):
            with lock:
                store[key] = (value, time.monotonic() + ttl_seconds())

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = cache_key(args, kwargs)
                hit, value = get_cached(key)
                if hit:
                    return value
                result = await func(*args, **kwargs)
                set_cached(key, result)
                return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = cache_key(args, kwargs)
            hit, value = get_cached(key)
            if hit:
                return value
            result = func(*args, **kwargs)
            set_cached(key, result)
            return result

        return sync_wrapper

    return decorator


class RateLimiter:
    """Token-bucket (sliding window) limiter. Intended as one instance per provider."""

    def __init__(self, calls_per_minute: int | None = None):
        self._calls_per_minute = calls_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    @property
    def calls_per_minute(self) -> int:
        return self._calls_per_minute or get_settings().rate_limit_per_minute

    def acquire(self) -> None:
        window = 60.0
        with self._lock:
            limit = self.calls_per_minute
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] > window:
                self._timestamps.popleft()
            if len(self._timestamps) >= limit:
                sleep_for = window - (now - self._timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > window:
                    self._timestamps.popleft()
            self._timestamps.append(now)


class TokenBucketLimiter:
    """Non-blocking token-bucket admission control: rejects immediately instead
    of sleeping/queuing like RateLimiter above."""

    def __init__(self, capacity: int | None = None, refill_per_minute: int | None = None):
        self._capacity = capacity
        self._refill_per_minute = refill_per_minute
        self._tokens: float | None = None  # lazily seeded from settings on first use
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self.stats = {"allowed": 0, "rejected": 0}

    @property
    def capacity(self) -> int:
        return self._capacity or get_settings().rate_limit_burst_capacity

    @property
    def _refill_per_second(self) -> float:
        per_minute = self._refill_per_minute or get_settings().rate_limit_per_minute
        return per_minute / 60.0

    def try_acquire(self) -> bool:
        """Consumes and returns True if a token is available, else False (no sleep)."""
        with self._lock:
            if self._tokens is None:
                self._tokens = float(self.capacity)
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self._refill_per_second)
            self._last_refill = now
            if self._tokens >= 1:
                self._tokens -= 1
                self.stats["allowed"] += 1
                return True
            self.stats["rejected"] += 1
            return False


@lru_cache
def get_admission_limiter() -> TokenBucketLimiter:
    """Process-wide singleton so admission state is shared across BaseAgent
    instances. Call get_admission_limiter.cache_clear() to reset in tests."""
    return TokenBucketLimiter()


def rate_limited(calls_per_minute: int | None = None):
    """Decorator wrapping a sync or async callable with a dedicated RateLimiter."""

    limiter = RateLimiter(calls_per_minute)

    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                await asyncio.to_thread(limiter.acquire)
                return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)

        return sync_wrapper

    return decorator
