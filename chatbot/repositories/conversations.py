"""
Reading and writing conversation history.

Session context used to live only in memory, so any restart lost it. These
functions let a session be rebuilt from the database on first message after a
restart, and are cheap enough to run on the background thread that already
handles per-turn logging.
"""
from typing import Dict, List

from sqlalchemy import delete, text

from db.database import read_session, session_scope
from models.schemas import ConversationMessage
from utils.logger import get_logger
from utils.text import truncate

logger = get_logger(__name__)

# Content is trimmed before storage: a runaway tool output shouldn't bloat the
# table or the prompt it later feeds.
MAX_CONTENT_CHARS = 4000

VALID_ROLES = ("user", "assistant", "system")


def load_recent(seller_id: str, customer_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """The newest `limit` turns, oldest-first (the order a prompt wants them).

    Returns [] rather than raising: a history read failing should degrade the
    conversation to "no context", never drop the customer's message.
    """
    try:
        with read_session() as db:
            rows = (
                db.query(ConversationMessage.role, ConversationMessage.content)
                .filter(
                    ConversationMessage.seller_id == str(seller_id),
                    ConversationMessage.customer_id == str(customer_id),
                )
                .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
                .limit(limit)
                .all()
            )
        # Query is newest-first for the LIMIT to work off the index; flip it back.
        return [{"role": role, "content": content} for role, content in reversed(rows)]
    except Exception as e:
        logger.warning("Could not load conversation history for %s: %s", customer_id, e)
        return []


def append_turn(
    seller_id: str,
    customer_id: str,
    user_message: str,
    assistant_message: str,
) -> bool:
    """Store one exchange (customer message + bot reply) in a single transaction."""
    try:
        with session_scope() as db:
            db.add_all(
                [
                    ConversationMessage(
                        seller_id=str(seller_id),
                        customer_id=str(customer_id),
                        role="user",
                        content=truncate(user_message, MAX_CONTENT_CHARS),
                    ),
                    ConversationMessage(
                        seller_id=str(seller_id),
                        customer_id=str(customer_id),
                        role="assistant",
                        content=truncate(assistant_message, MAX_CONTENT_CHARS),
                    ),
                ]
            )
        return True
    except Exception as e:
        logger.warning("Could not persist conversation turn for %s: %s", customer_id, e)
        return False


def append(seller_id: str, customer_id: str, role: str, content: str) -> bool:
    """Store a single message."""
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
    try:
        with session_scope() as db:
            db.add(
                ConversationMessage(
                    seller_id=str(seller_id),
                    customer_id=str(customer_id),
                    role=role,
                    content=truncate(content, MAX_CONTENT_CHARS),
                )
            )
        return True
    except Exception as e:
        logger.warning("Could not persist message for %s: %s", customer_id, e)
        return False


def clear(seller_id: str, customer_id: str) -> int:
    """Delete a customer's stored history. Returns rows removed."""
    try:
        with session_scope() as db:
            result = db.execute(
                delete(ConversationMessage).where(
                    ConversationMessage.seller_id == str(seller_id),
                    ConversationMessage.customer_id == str(customer_id),
                )
            )
            return result.rowcount or 0
    except Exception as e:
        logger.error("Could not clear conversation history for %s: %s", customer_id, e)
        return 0


def prune(retain_days: int = 30) -> int:
    """Drop history older than `retain_days`. Returns rows removed."""
    try:
        with session_scope() as db:
            result = db.execute(
                text(
                    "DELETE FROM conversation_messages "
                    "WHERE created_at < NOW() - (:days || ' days')::INTERVAL"
                ),
                {"days": retain_days},
            )
            removed = result.rowcount or 0
        if removed:
            logger.info("Pruned %d conversation messages older than %d days", removed, retain_days)
        return removed
    except Exception as e:
        logger.warning("Could not prune conversation history: %s", e)
        return 0
