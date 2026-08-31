"""
Shared HTTP client with connection pooling and retries.

Every outbound call used to build a fresh connection, which meant a full TLS
handshake to graph.facebook.com for each WhatsApp message - roughly 100-300ms of
pure overhead per send, several times per conversation turn. A pooled Session
keeps the connection alive and reuses it.

Retries cover the transient failures worth retrying (429 and 5xx) with
exponential backoff, and only on idempotent-by-effect calls. A WhatsApp send is
retried because a dropped connection usually means the request never landed.
"""
import threading
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:  # very old urllib3
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

from utils.logger import get_logger

logger = get_logger(__name__)

_sessions = {}
_lock = threading.Lock()

# Statuses safe to retry: each means the request was rejected or never reached
# the application, so retrying cannot duplicate its effect.
#   429 - rate limited, explicitly not processed
#   502/504 - gateway could not reach or hear back from the backend
#   503 - service unavailable
# 500 is deliberately absent: it means the application itself failed *after*
# receiving the request, so a WhatsApp send that returned 500 may still have
# delivered the message.
RETRY_STATUSES = (429, 502, 503, 504)


def build_session(
    total_retries: int = 3,
    backoff_factor: float = 0.5,
    pool_maxsize: int = 32,
    allowed_methods: Optional[frozenset] = None,
) -> requests.Session:
    """A Session with a retrying, pooled adapter mounted on both schemes.

    Read retries are disabled (read=0) on purpose. A read timeout means the
    request was sent and we never heard the answer - for a non-idempotent POST
    like "send this WhatsApp message", retrying there is how a customer receives
    the same message twice. Connect errors are retried freely, because a failed
    connect means the request never arrived.
    """
    session = requests.Session()

    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=0,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=allowed_methods or frozenset(["GET", "POST", "PUT", "DELETE", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_maxsize,
        pool_maxsize=pool_maxsize,
        pool_block=False,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_session(name: str = "default", **kwargs) -> requests.Session:
    """Fetch (or create) the shared Session registered under `name`.

    Separate names keep pools independent, so a slow vision API can't starve
    WhatsApp sends of connections.
    """
    with _lock:
        session = _sessions.get(name)
        if session is None:
            session = build_session(**kwargs)
            _sessions[name] = session
            logger.debug("Created pooled HTTP session '%s'", name)
        return session


def close_all() -> None:
    """Release pooled connections on shutdown."""
    with _lock:
        for name, session in _sessions.items():
            try:
                session.close()
            except Exception as e:
                logger.warning("Error closing HTTP session '%s': %s", name, e)
        _sessions.clear()
