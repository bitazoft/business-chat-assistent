import time

import pytest

from utils.rate_limit import RateLimiter


def test_allows_up_to_capacity_then_blocks():
    limiter = RateLimiter(capacity=3, window_seconds=60)
    assert all(limiter.check("user")[0] for _ in range(3))
    allowed, retry_after = limiter.check("user")
    assert allowed is False
    assert retry_after > 0


def test_keys_are_independent():
    limiter = RateLimiter(capacity=1, window_seconds=60)
    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True
    assert limiter.check("a")[0] is False


def test_bucket_refills_over_time():
    limiter = RateLimiter(capacity=2, window_seconds=0.2)
    assert limiter.check("u")[0] is True
    assert limiter.check("u")[0] is True
    assert limiter.check("u")[0] is False
    time.sleep(0.25)
    assert limiter.check("u")[0] is True


def test_retry_after_shrinks_as_the_bucket_refills():
    limiter = RateLimiter(capacity=2, window_seconds=1.0)
    limiter.check("u")
    limiter.check("u")
    _, first = limiter.check("u")
    time.sleep(0.3)
    _, second = limiter.check("u")
    assert second < first


def test_reset_clears_a_key():
    limiter = RateLimiter(capacity=1, window_seconds=60)
    limiter.check("u")
    assert limiter.check("u")[0] is False
    limiter.reset("u")
    assert limiter.check("u")[0] is True


def test_full_buckets_are_evicted_to_bound_memory():
    limiter = RateLimiter(capacity=5, window_seconds=0.05, maxkeys=10)
    for i in range(30):
        limiter.check(f"user-{i}")
    time.sleep(0.1)          # every bucket refills to capacity
    limiter.check("trigger")  # eviction runs on the next allowed call
    assert len(limiter) <= 11


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RateLimiter(capacity=0, window_seconds=60)
    with pytest.raises(ValueError):
        RateLimiter(capacity=5, window_seconds=0)
