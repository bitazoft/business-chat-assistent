"""
The /chat HTTP endpoint.

Two problems this route had:

1. It was `async def` but called chatbot.process_message() directly. That is a
   blocking call making LLM and database requests, so it stalled the event loop
   for its whole duration - every other request, including WhatsApp webhooks,
   waited behind it. It now runs in a worker thread.

2. It built a brand new chatbot per request (20 Pydantic schemas, 20 tools, a
   bound LLM, an executor) and threw it away. Sessions are reused now, so that
   cost is paid once per conversation.
"""
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from agent.agent import create_optimized_chatbot
from config.settings import settings
from repositories import conversations
from services import handoff
from services.session_store import ChatSession, session_store
from services.usage_tracker import usage_tracker
from utils.logger import get_logger
from utils.metrics import metrics
from utils.rate_limit import RateLimiter

logger = get_logger(__name__)

router = APIRouter()

# Same limiter shape as the WhatsApp path: every message costs an LLM call, so
# one client looping is a real bill rather than just load.
_limiter = RateLimiter(
    capacity=settings.rate_limit_messages,
    window_seconds=settings.rate_limit_window_seconds,
)


# Pydantic model for chat request
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str
    seller_id: str
    user_id: str
    # Format: [{"role": "user", "content": "message"}, ...]
    chat_history: List[Dict[str, str]] = []


def _load_history_into(session: ChatSession) -> None:
    """Seed a fresh session from stored history so a restart keeps context."""
    if not settings.persist_conversations:
        return
    history = conversations.load_recent(
        session.seller_id, session.user_id, limit=settings.max_chat_history
    )
    if history:
        session.chatbot.load_history(history)
        session.history_loaded = True
        logger.info(
            "Restored %d message(s) of history for %s", len(history), session.user_id
        )


# Chat endpoint
@router.post("/chat")
async def chat(request: ChatRequest):
    logger.info(
        "[Chat API] seller_id=%s user_id=%s session_id=%s",
        request.seller_id,
        request.user_id,
        request.session_id,
    )

    if not request.seller_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid seller_id: must be a numeric ID")
    if not request.user_id:
        raise HTTPException(status_code=400, detail="Invalid user_id: cannot be empty")

    if settings.rate_limit_enabled:
        allowed, retry_after = _limiter.check(f"{request.seller_id}:{request.user_id}")
        if not allowed:
            metrics.incr("chat.rate_limited")
            raise HTTPException(
                status_code=429,
                detail=f"Too many messages. Try again in {retry_after:.0f}s.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    # An open handover means a person is answering this customer; the bot must
    # not talk over them.
    if handoff.is_active(request.seller_id, request.user_id):
        active = handoff.get(request.seller_id, request.user_id)
        metrics.incr("chat.handoff_skipped")
        return {
            "response": None,
            "handoff": True,
            "handoff_reason": active.reason if active else None,
            "detail": "This conversation is with a human agent. The bot is not replying.",
        }

    try:
        session = session_store.get_or_create(
            seller_id=request.seller_id,
            user_id=request.user_id,
            factory=create_optimized_chatbot,
            on_create=_load_history_into,
        )

        # The client may send its own history; when it does it wins, matching the
        # previous behaviour. An empty list falls through to the session's own.
        external_history = [
            {"role": m["role"], "content": m["content"]}
            for m in request.chat_history
            if m.get("role") and m.get("content")
        ]

        start = time.perf_counter()
        with metrics.timer("chat.request"):
            # run_in_threadpool is the whole point: process_message blocks.
            response = await run_in_threadpool(
                session.chatbot.process_message, request.message, external_history or None
            )
        processing_time = time.perf_counter() - start

        logger.info("[Chat API] Message processed in %.2fs", processing_time)

        return {
            "response": response,
            "processing_time": f"{processing_time:.2f}s",
            "model": getattr(session.chatbot, "last_model", None),
            "usage": usage_tracker.session_usage(request.seller_id, request.user_id),
            "escalated": getattr(session.chatbot, "escalated", False),
        }

    except ValueError as ve:
        logger.error("[Chat API] ValueError: %s", ve)
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        metrics.incr("chat.error")
        logger.exception("[Chat API] Unexpected error")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.delete("/chat/session/{seller_id}/{user_id}")
async def clear_chat_session(seller_id: str, user_id: str, purge_history: bool = False):
    """Drop the in-memory session, and optionally the stored history with it."""
    dropped = session_store.drop(seller_id, user_id)
    removed = (
        await run_in_threadpool(conversations.clear, seller_id, user_id)
        if purge_history
        else 0
    )
    return {
        "session_cleared": dropped,
        "history_messages_deleted": removed,
    }
