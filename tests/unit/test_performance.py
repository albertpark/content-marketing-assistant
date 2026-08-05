import time

from src.integrations.performance import RateLimiter, TokenBucketLimiter, cached, get_admission_limiter


def test_cache_hit_avoids_second_call():
    calls = []

    @cached(ttl_minutes=10)
    def expensive(x):
        calls.append(x)
        return x * 2

    assert expensive(3) == 6
    assert expensive(3) == 6
    assert calls == [3]


def test_cache_expires_after_ttl(monkeypatch):
    calls = []
    fake_time = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: fake_time["t"])

    @cached(ttl_minutes=1)  # 60 seconds
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    fake_time["t"] += 61
    expensive(1)
    assert calls == [1, 1]


def test_rate_limiter_sleeps_once_past_cap(monkeypatch):
    sleeps = []
    fake_time = {"t": 0.0}
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(time, "monotonic", lambda: fake_time["t"])

    limiter = RateLimiter(calls_per_minute=2)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # third call within the same 60s window must wait

    assert len(sleeps) == 1


def test_rate_limiter_does_not_sleep_once_window_passes(monkeypatch):
    sleeps = []
    fake_time = {"t": 0.0}
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(time, "monotonic", lambda: fake_time["t"])

    limiter = RateLimiter(calls_per_minute=1)
    limiter.acquire()
    fake_time["t"] += 61
    limiter.acquire()

    assert sleeps == []


def test_token_bucket_rejects_once_capacity_exhausted(monkeypatch):
    fake_time = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: fake_time["t"])

    limiter = TokenBucketLimiter(capacity=2, refill_per_minute=60)  # 1/sec refill
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False  # bucket empty, no time has passed
    assert limiter.stats == {"allowed": 2, "rejected": 1}


def test_token_bucket_refills_over_time(monkeypatch):
    fake_time = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: fake_time["t"])

    limiter = TokenBucketLimiter(capacity=1, refill_per_minute=60)  # 1/sec refill
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False  # empty immediately after

    fake_time["t"] += 1.0  # exactly one token's worth of refill
    assert limiter.try_acquire() is True


def test_token_bucket_does_not_exceed_capacity(monkeypatch):
    fake_time = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: fake_time["t"])

    limiter = TokenBucketLimiter(capacity=1, refill_per_minute=60)
    fake_time["t"] += 100  # way more than enough to overflow if uncapped
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False  # still only 1 token's worth of capacity


def test_get_admission_limiter_is_a_cached_singleton():
    get_admission_limiter.cache_clear()
    try:
        assert get_admission_limiter() is get_admission_limiter()
    finally:
        get_admission_limiter.cache_clear()
