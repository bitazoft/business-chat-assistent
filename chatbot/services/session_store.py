"""
Chatbot sessions with a bounded lifetime.

The old implementation was a module-level dict that only ever grew: a session was
created for every phone number that ever messaged and never removed. It recorded
last_activity but nothing read it. On a busy number that is an unbounded leak,
and each entry holds a whole agent object.

This store gives sessions a sliding TTL (activity extends the lease), caps the
total count, and takes a per-customer lock so two messages arriving together
don't build two agents and race on the same history.
"""
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config.settings import settings
from utils.cache import KeyedLock, TTLCache
from utils.logger import get_logger
from utils.metrics import metrics

logger = get_logger(__name__)


@dataclass
class ChatSession:
    """One customer's live conversation with one seller's bot."""

    seller_id: str
    user_id: str
    chatbot: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    history_loaded: bool = False

    def touch(self) -> None:
        self.last_activity = datetime.now()
        self.message_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seller_id": self.seller_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "history_loaded": self.history_loaded,
        }


class SessionStore:
    def __init__(
        self,
        ttl_seconds: Optional[int] = None,
        max_sessions: Optional[int] = None,
    ):
        self.ttl_seconds = ttl_seconds or settings.session_ttl_seconds
        self.max_sessions = max_sessions or settings.session_max_count
        self._cache: TTLCache = TTLCache(
            maxsize=self.max_sessions, ttl=self.ttl_seconds, name="sessions"
        )
        self._locks = KeyedLock()
        self._created = 0

    @staticmethod
    def key(seller_id: str, user_id: str) -> str:
        return f"{seller_id}:{user_id}"

    def get_or_create(
        self,
        seller_id: str,
        user_id: str,
        factory: Callable[[str, str], Any],
        on_create: Optional[Callable[[ChatSession], None]] = None,
    ) -> ChatSession:
        """Return the live session, building one if it is absent or expired.

        `factory(seller_id, user_id)` makes the agent. `on_create` runs once for a
        newly built session - that is where history gets loaded from the database.
        Both run under this customer's lock, so a second concurrent message waits
        rather than building a duplicate agent.
        """
        session_key = self.key(seller_id, user_id)

        # Fast path: an existing session needs no lock beyond the cache's own.
        session = self._cache.get(session_key)
        if session is not None:
            session.touch()
            self._cache.set(session_key, session)  # slide the TTL forward
            metrics.incr("sessions.hit")
            return session

        with self._locks(session_key):
            # Another thread may have built it while we waited for the lock.
            session = self._cache.get(session_key)
            if session is not None:
                session.touch()
                self._cache.set(session_key, session)
                metrics.incr("sessions.hit")
                return session

            logger.info("Creating chatbot session for %s (seller %s)", user_id, seller_id)
            session = ChatSession(
                seller_id=str(seller_id),
                user_id=str(user_id),
                chatbot=factory(str(seller_id), str(user_id)),
            )

            if on_create is not None:
                try:
                    on_create(session)
                except Exception as e:
                    # A failed history load means less context, not a failed message.
                    logger.warning("Session init hook failed for %s: %s", user_id, e)

            self._cache.set(session_key, session)
            self._created += 1
            metrics.incr("sessions.created")
            metrics.gauge("sessions.active", len(self._cache))
            return session

    def get(self, seller_id: str, user_id: str) -> Optional[ChatSession]:
        return self._cache.get(self.key(seller_id, user_id))

    def drop(self, seller_id: str, user_id: str) -> bool:
        session_key = self.key(seller_id, user_id)
        removed = self._cache.delete(session_key)
        self._locks.discard(session_key)
        if removed:
            metrics.incr("sessions.dropped")
            metrics.gauge("sessions.active", len(self._cache))
        return removed

    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        metrics.gauge("sessions.active", 0)
        return count

    def sweep(self) -> int:
        """Drop expired sessions. Called by the background sweeper task."""
        removed = self._cache.purge_expired()
        if removed:
            logger.info("Swept %d idle chatbot session(s)", removed)
        metrics.gauge("sessions.active", len(self._cache))
        return removed

    def __len__(self) -> int:
        return len(self._cache)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Snapshot for the /whatsapp/status endpoint."""
        return [
            dict(session.to_dict(), key=session_key)
            for session_key, session in self._cache.items()
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "active": len(self._cache),
            "max": self.max_sessions,
            "ttl_seconds": self.ttl_seconds,
            "created_total": self._created,
            "cache": self._cache.stats(),
        }


class SessionSweeper:
    """Background thread that expires idle sessions and stale cache entries.

    A daemon thread rather than an asyncio task so it behaves the same whether the
    app runs under uvicorn, a script, or a test.
    """

    def __init__(self, store: SessionStore, interval_seconds: Optional[int] = None):
        self.store = store
        self.interval = interval_seconds or settings.session_sweep_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="session-sweeper", daemon=True)
        self._thread.start()
        logger.info("Session sweeper started (every %ds)", self.interval)

    def _run(self) -> None:
        from utils.cache import purge_all_expired

        while not self._stop.wait(self.interval):
            try:
                self.store.sweep()
                purge_all_expired()
            except Exception as e:
                logger.error("Session sweeper error: %s", e)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            logger.info("Session sweeper stopped")


session_store = SessionStore()
session_sweeper = SessionSweeper(session_store)
