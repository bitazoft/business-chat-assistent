"""
In-process counters and latency tracking, exposed at /metrics.

Deliberately tiny: no Prometheus dependency, no external collector. Enough to
answer "how slow is the bot right now, and what is failing" without adding
infrastructure. Percentiles come from a bounded ring buffer of recent samples,
so memory is fixed no matter how long the process runs.
"""
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional

_MAX_SAMPLES = 1000


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._timings: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=_MAX_SAMPLES))
        self._gauges: Dict[str, float] = {}
        self._started_at = time.time()

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            self._timings[name].append(seconds)

    def timer(self, name: str) -> "_Timer":
        return _Timer(self, name)

    @staticmethod
    def _percentile(samples: list, pct: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        # Nearest-rank percentile - fine for operational numbers.
        idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
        return ordered[idx]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            timings = {name: list(samples) for name, samples in self._timings.items()}

        latency = {}
        for name, samples in timings.items():
            if not samples:
                continue
            latency[name] = {
                "count": len(samples),
                "avg_ms": round(sum(samples) / len(samples) * 1000, 1),
                "p50_ms": round(self._percentile(samples, 50) * 1000, 1),
                "p95_ms": round(self._percentile(samples, 95) * 1000, 1),
                "p99_ms": round(self._percentile(samples, 99) * 1000, 1),
                "max_ms": round(max(samples) * 1000, 1),
            }

        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "counters": counters,
            "gauges": gauges,
            "latency": latency,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()
            self._gauges.clear()


class _Timer:
    """`with metrics.timer("chat.turn"):` records how long the block took."""

    __slots__ = ("_metrics", "_name", "_start")

    def __init__(self, metrics: Metrics, name: str):
        self._metrics = metrics
        self._name = name
        self._start: Optional[float] = None

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._start is not None:
            self._metrics.observe(self._name, time.perf_counter() - self._start)
        # Record success/failure alongside the timing so error rate is visible.
        self._metrics.incr(f"{self._name}.{'error' if exc_type else 'ok'}")
        return False

    @property
    def elapsed(self) -> float:
        return 0.0 if self._start is None else time.perf_counter() - self._start


metrics = Metrics()
