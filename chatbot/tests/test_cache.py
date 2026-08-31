import time

import pytest

from utils.cache import KeyedLock, TTLCache


def test_set_and_get():
    cache = TTLCache(maxsize=10, ttl=None)
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None
    assert cache.get("missing", "fallback") == "fallback"


def test_entries_expire():
    cache = TTLCache(maxsize=10, ttl=0.05)
    cache.set("a", 1)
    assert cache.get("a") == 1
    time.sleep(0.08)
    assert cache.get("a") is None


def test_per_entry_ttl_overrides_default():
    cache = TTLCache(maxsize=10, ttl=100)
    cache.set("short", 1, ttl=0.05)
    time.sleep(0.08)
    assert cache.get("short") is None


def test_lru_eviction_drops_least_recently_used():
    cache = TTLCache(maxsize=2, ttl=None)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")          # 'a' is now the most recent, so 'b' should go first
    cache.set("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_add_is_set_if_absent():
    """This is the webhook deduplication primitive."""
    cache = TTLCache(maxsize=10, ttl=None)
    assert cache.add("msg-1") is True     # first delivery
    assert cache.add("msg-1") is False    # WhatsApp redelivery
    assert cache.add("msg-2") is True


def test_add_allows_reuse_after_expiry():
    cache = TTLCache(maxsize=10, ttl=0.05)
    assert cache.add("k") is True
    time.sleep(0.08)
    assert cache.add("k") is True


def test_get_or_set_computes_once():
    cache = TTLCache(maxsize=10, ttl=None)
    calls = []

    def factory():
        calls.append(1)
        return "value"

    assert cache.get_or_set("k", factory) == "value"
    assert cache.get_or_set("k", factory) == "value"
    assert len(calls) == 1


def test_purge_expired_reports_count():
    cache = TTLCache(maxsize=10, ttl=0.05)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("keep", 3, ttl=100)
    time.sleep(0.08)
    assert cache.purge_expired() == 2
    assert len(cache) == 1


def test_items_skips_expired():
    cache = TTLCache(maxsize=10, ttl=100)
    cache.set("live", 1)
    cache.set("dead", 2, ttl=0.05)
    time.sleep(0.08)
    assert [k for k, _ in cache.items()] == ["live"]


def test_stats_tracks_hit_rate():
    cache = TTLCache(maxsize=10, ttl=None)
    cache.set("a", 1)
    cache.get("a")
    cache.get("b")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_maxsize_must_be_positive():
    with pytest.raises(ValueError):
        TTLCache(maxsize=0)


def test_keyed_lock_returns_same_lock_per_key():
    locks = KeyedLock()
    assert locks("a") is locks("a")
    assert locks("a") is not locks("b")
