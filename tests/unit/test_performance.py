import time

from src.integrations.performance import RateLimiter, cached


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
