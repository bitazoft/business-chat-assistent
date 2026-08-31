"""
Sessions used to live in a dict that only grew. These cover the eviction and
concurrency behaviour that replaced it.
"""
import threading
import time

from services.session_store import SessionStore


class FakeBot:
    """Stand-in for OptimizedChatbot - building a real one binds 20 tools."""

    instances = 0

    def __init__(self, seller_id, user_id):
        self.seller_id = seller_id
        self.user_id = user_id
        self.history = []
        FakeBot.instances += 1


def _factory(seller_id, user_id):
    return FakeBot(seller_id, user_id)


def test_same_customer_reuses_one_session():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    first = store.get_or_create("7", "user-1", _factory)
    second = store.get_or_create("7", "user-1", _factory)
    assert first is second
    assert first.chatbot is second.chatbot


def test_message_count_and_activity_advance():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.get_or_create("7", "u", _factory)
    session = store.get_or_create("7", "u", _factory)
    assert session.message_count >= 1


def test_customers_are_isolated():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    a = store.get_or_create("7", "user-a", _factory)
    b = store.get_or_create("7", "user-b", _factory)
    assert a.chatbot is not b.chatbot


def test_same_customer_different_seller_is_a_different_session():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    a = store.get_or_create("7", "u", _factory)
    b = store.get_or_create("8", "u", _factory)
    assert a.chatbot is not b.chatbot


def test_idle_sessions_expire():
    store = SessionStore(ttl_seconds=0.05, max_sessions=10)
    store.get_or_create("7", "u", _factory)
    assert len(store) == 1
    time.sleep(0.08)
    assert store.get("7", "u") is None


def test_sweep_reclaims_expired_sessions():
    """This is the leak fix: nothing used to remove idle sessions."""
    store = SessionStore(ttl_seconds=0.05, max_sessions=100)
    for i in range(20):
        store.get_or_create("7", f"user-{i}", _factory)
    assert len(store) == 20
    time.sleep(0.08)
    assert store.sweep() == 20
    assert len(store) == 0


def test_activity_slides_the_expiry_forward():
    store = SessionStore(ttl_seconds=0.15, max_sessions=10)
    store.get_or_create("7", "u", _factory)
    for _ in range(3):
        time.sleep(0.06)
        store.get_or_create("7", "u", _factory)   # keeps it alive
    assert store.get("7", "u") is not None


def test_session_count_is_capped():
    store = SessionStore(ttl_seconds=600, max_sessions=5)
    for i in range(50):
        store.get_or_create("7", f"user-{i}", _factory)
    assert len(store) <= 5


def test_drop_removes_one_session():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.get_or_create("7", "u", _factory)
    assert store.drop("7", "u") is True
    assert store.drop("7", "u") is False


def test_on_create_hook_runs_once_per_session():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    calls = []
    store.get_or_create("7", "u", _factory, on_create=lambda s: calls.append(s))
    store.get_or_create("7", "u", _factory, on_create=lambda s: calls.append(s))
    assert len(calls) == 1


def test_failing_on_create_hook_still_yields_a_session():
    """A history-load failure means less context, not a dropped message."""
    store = SessionStore(ttl_seconds=60, max_sessions=10)

    def broken(session):
        raise RuntimeError("database down")

    session = store.get_or_create("7", "u", _factory, on_create=broken)
    assert session.chatbot is not None


def test_concurrent_first_messages_build_only_one_agent():
    """Two messages arriving together must not each build an agent."""
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    FakeBot.instances = 0
    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(store.get_or_create("7", "same-user", _factory))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert FakeBot.instances == 1
    assert len({id(r.chatbot) for r in results}) == 1


def test_list_and_stats():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.get_or_create("7", "u", _factory)
    listed = store.list_sessions()
    assert len(listed) == 1
    assert listed[0]["user_id"] == "u"
    assert store.stats()["active"] == 1
