"""
Small thread-safe cache with expiry and a size cap.

Why not functools.lru_cache: it never expires entries, it has no size-aware
eviction we can inspect, and when used on a method it keeps `self` alive forever
(the vector store used to leak this way). This also gives us `add()` - a
set-only-if-absent that the WhatsApp webhook uses to drop duplicate deliveries.

The Cache protocol is deliberately the subset of operations Redis also offers,
so a RedisCache can be dropped in later without touching call sites.
"""
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

_MISS = object()


class Cache(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None: ...
    def add(self, key: str, value: Any = True, ttl: Optional[float] = None) -> bool: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> None: ...
    def stats(self) -> Dict[str, Any]: ...


class TTLCache:
    """LRU cache where every entry also has an expiry time."""

    def __init__(self, maxsize: int = 512, ttl: Optional[float] = 300.0, name: str = "cache"):
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.name = name
        self.maxsize = maxsize
        self.default_ttl = ttl
        self._lock = threading.RLock()
        # key -> (expires_at or None, value); ordered by recency of use
        self._data: "OrderedDict[str, Tuple[Optional[float], Any]]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ---- internals ----------------------------------------------------
    def _expired(self, expires_at: Optional[float], now: float) -> bool:
        return expires_at is not None and expires_at <= now

    def _get_entry(self, key: str, now: float) -> Any:
        """Returns the value, or _MISS. Caller must hold the lock."""
        entry = self._data.get(key, _MISS)
        if entry is _MISS:
            return _MISS
        expires_at, value = entry
        if self._expired(expires_at, now):
            del self._data[key]
            self._expirations += 1
            return _MISS
        self._data.move_to_end(key)
        return value

    def _store(self, key: str, value: Any, ttl: Optional[float], now: float) -> None:
        """Caller must hold the lock."""
        effective_ttl = self.default_ttl if ttl is None else ttl
        expires_at = None if effective_ttl is None else now + effective_ttl
        self._data[key] = (expires_at, value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)
            self._evictions += 1

    # ---- public API ---------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        now = time.monotonic()
        with self._lock:
            value = self._get_entry(key, now)
            if value is _MISS:
                self._misses += 1
                return default
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        now = time.monotonic()
        with self._lock:
            self._store(key, value, ttl, now)

    def add(self, key: str, value: Any = True, ttl: Optional[float] = None) -> bool:
        """Set only if the key isn't already present. True if we stored it.

        This is the idempotency primitive: the first caller for a given key gets
        True, everyone after gets False until the entry expires.
        """
        now = time.monotonic()
        with self._lock:
            if self._get_entry(key, now) is not _MISS:
                return False
            self._store(key, value, ttl, now)
            return True

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        """Return the cached value, computing it with `factory` on a miss.

        `factory` runs outside the lock, so a slow factory doesn't block other
        keys. Two callers racing on the same missing key may both compute it;
        that is cheaper than serialising every lookup behind one lock.
        """
        cached = self.get(key, _MISS)
        if cached is not _MISS:
            return cached
        value = factory()
        self.set(key, value, ttl)
        return value

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, _MISS) is not _MISS

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def purge_expired(self) -> int:
        """Drop expired entries. Called by the background sweeper."""
        now = time.monotonic()
        with self._lock:
            dead = [k for k, (exp, _) in self._data.items() if self._expired(exp, now)]
            for key in dead:
                del self._data[key]
            self._expirations += len(dead)
            return len(dead)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        return self.get(key, _MISS) is not _MISS

    def items(self) -> list:
        """Snapshot of (key, value) for live entries, most recently used last.

        A copied list, not a live view, so callers can iterate without holding
        the lock. Expired-but-not-yet-purged entries are skipped.
        """
        now = time.monotonic()
        with self._lock:
            return [
                (key, value)
                for key, (expires_at, value) in self._data.items()
                if not self._expired(expires_at, now)
            ]

    def keys(self) -> list:
        return [key for key, _ in self.items()]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "name": self.name,
                "size": len(self._data),
                "maxsize": self.maxsize,
                "ttl": self.default_ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
                "evictions": self._evictions,
                "expirations": self._expirations,
            }


class KeyedLock:
    """One lock per key, so work on different keys doesn't serialise.

    Used to stop the same customer's two concurrent messages from both building
    a session, and to keep one customer's slow turn from blocking everyone else.
    """

    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def __call__(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def discard(self, key: str) -> None:
        with self._guard:
            self._locks.pop(key, None)


# ---- named caches used across the app --------------------------------
_registry: Dict[str, TTLCache] = {}
_registry_lock = threading.Lock()


def get_cache(name: str, maxsize: int = 512, ttl: Optional[float] = 300.0) -> TTLCache:
    """Fetch (or create) the shared cache registered under `name`."""
    with _registry_lock:
        cache = _registry.get(name)
        if cache is None:
            cache = TTLCache(maxsize=maxsize, ttl=ttl, name=name)
            _registry[name] = cache
        return cache


def all_cache_stats() -> Dict[str, Dict[str, Any]]:
    with _registry_lock:
        caches = list(_registry.values())
    return {c.name: c.stats() for c in caches}


def purge_all_expired() -> int:
    with _registry_lock:
        caches = list(_registry.values())
    return sum(c.purge_expired() for c in caches)
