"""
Per-key token bucket, used to stop one customer from flooding the bot.

A flood matters here because every message costs an LLM call: one script hammering
the webhook is a real bill, not just load. Buckets refill continuously rather than
resetting on a fixed window, so a customer who waits briefly can send again.
"""
import threading
import time
from typing import Dict, Optional, Tuple


class RateLimiter:
    def __init__(self, capacity: int, window_seconds: float, maxkeys: int = 10000):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.capacity = capacity
        self.window_seconds = window_seconds
        self.refill_per_second = capacity / window_seconds
        self.maxkeys = maxkeys
        self._lock = threading.Lock()
        # key -> (tokens remaining, last refill time)
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def check(self, key: str, cost: float = 1.0) -> Tuple[bool, float]:
        """Take `cost` tokens if available.

        Returns (allowed, retry_after_seconds). retry_after is 0 when allowed.
        """
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)

            if tokens >= cost:
                self._buckets[key] = (tokens - cost, now)
                self._evict_if_needed(now)
                return True, 0.0

            deficit = cost - tokens
            self._buckets[key] = (tokens, now)
            return False, round(deficit / self.refill_per_second, 2)

    def _evict_if_needed(self, now: float) -> None:
        """Drop fully-refilled buckets once the table grows too big.

        A bucket back at capacity carries no information, so forgetting it costs
        nothing. Caller must hold the lock.
        """
        if len(self._buckets) <= self.maxkeys:
            return
        stale = [
            k
            for k, (tokens, last) in self._buckets.items()
            if min(self.capacity, tokens + (now - last) * self.refill_per_second) >= self.capacity
        ]
        for key in stale:
            del self._buckets[key]

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)
