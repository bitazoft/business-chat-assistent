import time

from utils.metrics import Metrics


def test_counters():
    m = Metrics()
    m.incr("a")
    m.incr("a", 5)
    assert m.snapshot()["counters"]["a"] == 6


def test_gauges_overwrite():
    m = Metrics()
    m.gauge("sessions", 3)
    m.gauge("sessions", 7)
    assert m.snapshot()["gauges"]["sessions"] == 7


def test_latency_percentiles():
    m = Metrics()
    for seconds in [0.01] * 90 + [1.0] * 10:
        m.observe("turn", seconds)
    latency = m.snapshot()["latency"]["turn"]
    assert latency["count"] == 100
    assert latency["p50_ms"] < latency["p95_ms"] <= latency["max_ms"]
    assert latency["max_ms"] >= 1000


def test_timer_records_duration_and_success():
    m = Metrics()
    with m.timer("op"):
        time.sleep(0.01)
    snapshot = m.snapshot()
    assert snapshot["latency"]["op"]["count"] == 1
    assert snapshot["counters"]["op.ok"] == 1


def test_timer_records_failure_and_reraises():
    m = Metrics()
    try:
        with m.timer("op"):
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("timer swallowed the exception")
    assert m.snapshot()["counters"]["op.error"] == 1


def test_sample_buffer_is_bounded():
    """Memory must stay fixed however long the process runs."""
    m = Metrics()
    for i in range(5000):
        m.observe("op", 0.001)
    assert m.snapshot()["latency"]["op"]["count"] <= 1000


def test_empty_snapshot_has_no_latency_entries():
    assert Metrics().snapshot()["latency"] == {}
