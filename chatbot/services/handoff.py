"""
Human handover.

Previously the bot had no way out: faced with a refund request or an angry
customer it would keep calling tools and eventually produce something unhelpful,
with nobody told. Now the agent can call escalate_to_human, which parks the
conversation here.

While a handover is open the bot stops auto-replying to that customer, so a staff
member can take over the thread without the bot talking over them. Staff clear it
through DELETE /whatsapp/handoffs/{phone_number}.

State is in-process with a TTL (a forgotten handover shouldn't silence a customer
forever). Moving it to Redis or a table means swapping the cache below.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.cache import get_cache
from utils.logger import get_logger
from utils.metrics import metrics

logger = get_logger(__name__)

# 24h: long enough for staff to pick it up next working day, short enough that a
# forgotten handover heals itself instead of muting the customer permanently.
HANDOFF_TTL_SECONDS = 24 * 60 * 60

_handoffs = get_cache("handoffs", maxsize=2000, ttl=HANDOFF_TTL_SECONDS)


@dataclass
class Handoff:
    seller_id: str
    user_id: str
    reason: str
    requested_at: datetime = field(default_factory=datetime.now)
    last_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seller_id": self.seller_id,
            "user_id": self.user_id,
            "reason": self.reason,
            "requested_at": self.requested_at.isoformat(),
            "last_message": self.last_message,
        }


def _key(seller_id: str, user_id: str) -> str:
    return f"{seller_id}:{user_id}"


def request(seller_id: str, user_id: str, reason: str, last_message: Optional[str] = None) -> Handoff:
    """Open a handover. Re-requesting keeps the original reason and timestamp."""
    key = _key(seller_id, user_id)
    existing = _handoffs.get(key)
    if existing is not None:
        existing.last_message = last_message or existing.last_message
        _handoffs.set(key, existing)
        return existing

    handoff = Handoff(
        seller_id=str(seller_id),
        user_id=str(user_id),
        reason=reason,
        last_message=last_message,
    )
    _handoffs.set(key, handoff)
    metrics.incr("handoff.requested")
    # WARNING level so it stands out in the log a human is actually reading.
    logger.warning(
        "HANDOVER REQUESTED - seller=%s customer=%s reason=%s",
        seller_id,
        user_id,
        reason,
    )
    return handoff


def is_active(seller_id: str, user_id: str) -> bool:
    return _handoffs.get(_key(seller_id, user_id)) is not None


def get(seller_id: str, user_id: str) -> Optional[Handoff]:
    return _handoffs.get(_key(seller_id, user_id))


def resolve(seller_id: str, user_id: str) -> bool:
    """Close a handover so the bot resumes replying. True if one was open."""
    removed = _handoffs.delete(_key(seller_id, user_id))
    if removed:
        metrics.incr("handoff.resolved")
        logger.info("Handover resolved - seller=%s customer=%s", seller_id, user_id)
    return removed


def list_all(seller_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Open handovers, newest first. Filtered by seller when given."""
    handoffs = [h for _, h in _handoffs.items()]
    if seller_id is not None:
        handoffs = [h for h in handoffs if h.seller_id == str(seller_id)]
    handoffs.sort(key=lambda h: h.requested_at, reverse=True)
    return [h.to_dict() for h in handoffs]


def count() -> int:
    return len(_handoffs)
