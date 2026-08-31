"""
WhatsApp Cloud API webhook and admin endpoints.

The important change is what the webhook handler does before it returns 200.
It used to download the image and run a vision classification - a multi-second
LLM call - inline in an `async def` handler. That blocked the event loop for the
whole call (stalling every other request in the process) and risked WhatsApp's
webhook timeout, which triggers a redelivery. With no deduplication, that
redelivery ran the customer's message a second time and could place a duplicate
order.

Now the handler does three cheap things - verify the signature, drop duplicates,
queue the work - and returns. Everything expensive happens on a worker thread.
"""
import json
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from agent.agent import create_optimized_chatbot
from config.settings import settings
from repositories import conversations
from repositories.tools import get_seller_id_by_whatsapp_number_id
from services import handoff, outbound_formatter
from services.image_analysis_service import vision_service as deepseek_vision_service
from services.session_store import ChatSession, session_store
from services.whatsapp_service import whatsapp_service
from templates.template_store import TemplateKey
from utils import background
from utils.cache import get_cache
from utils.logger import get_logger
from utils.metrics import metrics
from utils.rate_limit import RateLimiter

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])

# Seen message ids, so a WhatsApp redelivery is a no-op instead of a second
# order. WhatsApp retries for up to ~30 minutes; the default 15-minute window
# covers the retries that matter without holding ids forever.
_seen_messages = get_cache("whatsapp_seen", maxsize=20000, ttl=settings.dedupe_ttl_seconds)

_limiter = RateLimiter(
    capacity=settings.rate_limit_messages,
    window_seconds=settings.rate_limit_window_seconds,
)


def _load_history_into(session: ChatSession) -> None:
    """Seed a new session from stored history so a restart keeps context."""
    if not settings.persist_conversations:
        return
    history = conversations.load_recent(
        session.seller_id, session.user_id, limit=settings.max_chat_history
    )
    if history:
        session.chatbot.load_history(history)
        session.history_loaded = True
        logger.info("Restored %d message(s) of history for %s", len(history), session.user_id)


def get_or_create_chatbot(phone_number: str, seller_id: str = "default_seller"):
    """Get existing chatbot session or create new one.

    Kept for compatibility; sessions now live in services/session_store.py, which
    expires idle ones. The old module-level dict never removed anything.
    """
    session = session_store.get_or_create(
        seller_id=seller_id,
        user_id=phone_number,
        factory=create_optimized_chatbot,
        on_create=_load_history_into,
    )
    return session.chatbot


def _is_duplicate(message_id: Optional[str]) -> bool:
    """True if we've already accepted this message id.

    cache.add is set-if-absent, so the first caller for an id gets True and every
    redelivery gets False. Doing this before any work is what makes retries safe.
    """
    if not message_id:
        return False
    return not _seen_messages.add(f"msg:{message_id}")


FALLBACK_ERROR_MESSAGE = (
    "I'm experiencing technical difficulties. Please try again in a moment."
)


def _generic_error_message(seller_id: str = "") -> str:
    """The error reply, rendered without anything that could fail again.

    Used from except blocks, so it swallows its own failures: template lookup
    reads the database, and the reason we are here may well be that the database
    is down.
    """
    try:
        return outbound_formatter.render(TemplateKey.ERROR, seller_id) or FALLBACK_ERROR_MESSAGE
    except Exception:
        return FALLBACK_ERROR_MESSAGE


def _send_reply(phone_number: str, text: str, whatsapp_number_id: str) -> bool:
    """Send a reply, letting the service split it if it's over the size limit."""
    if not text:
        return False
    result = whatsapp_service.send_text_message(phone_number, text, whatsapp_number_id)
    if result.get("success"):
        logger.info("✅ Response sent to %s: %.50s...", phone_number, text)
        metrics.incr("whatsapp.sent")
        return True
    logger.error("❌ Failed to send response to %s: %s", phone_number, result.get("error"))
    metrics.incr("whatsapp.send_failed")
    return False


def _resolve_image_content(
    whatsapp_message, whatsapp_number_id: str
) -> Optional[str]:
    """Download and classify an image, returning the text to give the agent.

    Runs on a worker thread - it does a file download and a vision LLM call, both
    of which used to happen inline in the webhook handler.
    """
    download = whatsapp_service.download_image(whatsapp_message)
    if not (download.get("success") and download.get("file_path")):
        logger.error("Failed to download image: %s", download.get("error"))
        _send_reply(
            whatsapp_message.from_number,
            "Sorry, I couldn't download the image. Please try again.",
            whatsapp_number_id,
        )
        return None

    file_path = download["file_path"]
    logger.info("Image downloaded: %s", file_path)

    # Work out what kind of image this is so the agent picks the right tool.
    # If classification fails we still hand the image over - the agent can ask.
    image_hint = ""
    try:
        classification = deepseek_vision_service.classify_image(file_path)
        if classification.get("success"):
            image_type = classification.get("image_type", "other")
            summary = classification.get("summary", "")
            image_hint = f" type={image_type}. Contents: {summary}"
            logger.info("Image classified as '%s': %s", image_type, summary)
        else:
            logger.warning("Image classification failed: %s", classification.get("error"))
    except Exception as e:
        logger.error("Image classification error: %s", e)

    # Keep the customer's caption - it often says what they want
    caption = (whatsapp_message.content or "").strip()
    if caption and caption != "Image received":
        image_hint += f' Customer said: "{caption}"'

    return f"[Image received: {file_path}]{image_hint}"


def process_whatsapp_message(
    phone_number: str,
    message_content: str,
    message_id: str,
    whatsapp_number_id: str = "default_seller",
    whatsapp_message=None,
    is_image: bool = False,
):
    """Process WhatsApp message on a worker thread.

    The seller lookup, handover check and rate-limit check all live here rather
    than in the webhook handler: the seller lookup reads the database, and doing
    that on the event loop would block every other request for the round trip.
    """
    try:
        logger.info("🤖 Processing message from %s: %.50s...", phone_number, message_content)

        try:
            seller_id = get_seller_id_by_whatsapp_number_id(whatsapp_number_id)
        except Exception as e:
            # With the database unreachable nothing downstream can work either,
            # so say so once rather than letting the failure bubble up and leave
            # the customer with silence.
            metrics.incr("whatsapp.seller_lookup_failed")
            logger.error("Could not resolve seller for %s: %s", whatsapp_number_id, e)
            _send_reply(phone_number, _generic_error_message(), whatsapp_number_id)
            return

        # A person is handling this conversation - stay quiet so the bot doesn't
        # talk over them.
        if handoff.is_active(seller_id, phone_number):
            metrics.incr("whatsapp.handoff_skipped")
            logger.info("🙋 %s is with a human agent - not auto-replying", phone_number)
            return

        if settings.rate_limit_enabled:
            allowed, retry_after = _limiter.check(f"{seller_id}:{phone_number}")
            if not allowed:
                metrics.incr("whatsapp.rate_limited")
                logger.warning(
                    "🚦 Rate limited %s (retry in %.0fs)", phone_number, retry_after
                )
                _send_reply(
                    phone_number,
                    outbound_formatter.render(
                        TemplateKey.RATE_LIMITED, seller_id, retry_after=int(retry_after) + 1
                    ),
                    whatsapp_number_id,
                )
                return

        # Blue ticks + typing indicator first. A turn can take several seconds and
        # a silent chat makes customers resend; this used to happen only after the
        # reply had already been sent.
        whatsapp_service.mark_read_and_typing(message_id, whatsapp_number_id)

        if is_image:
            if whatsapp_message is None:
                logger.error("Image message arrived without its payload")
                return
            _send_reply(
                phone_number,
                outbound_formatter.render(TemplateKey.IMAGE_RECEIVED, seller_id)
                or "Image received, analyzing...",
                whatsapp_number_id,
            )
            resolved = _resolve_image_content(whatsapp_message, whatsapp_number_id)
            if resolved is None:
                return
            message_content = resolved

        chatbot = get_or_create_chatbot(phone_number, seller_id)

        with metrics.timer("whatsapp.turn"):
            response = chatbot.process_message(message_content)

        if not _send_reply(phone_number, response, whatsapp_number_id):
            return

        # Product photos go out after the text, so the text isn't held up by them.
        img_urls = chatbot.get_img_urls()
        if img_urls:
            def send_images():
                for url in img_urls:
                    whatsapp_service.send_image_message(
                        phone_number, url, "", whatsapp_number_id
                    )

            background.submit(send_images, task_name="send_product_images")

    except Exception as e:
        metrics.incr("whatsapp.processing_error")
        logger.error("❌ Error processing WhatsApp message: %s", e, exc_info=True)
        # Nothing in here may raise: this is the last chance to say anything to
        # the customer. The previous version called the seller lookup again,
        # which is exactly what had just failed.
        _send_reply(phone_number, _generic_error_message(), whatsapp_number_id)


async def process_whatsapp_message_async(
    phone_number: str,
    message_content: str,
    message_id: str,
    whatsapp_number_id: str = "default_seller",
    whatsapp_message=None,
    is_image: bool = False,
):
    """Run the blocking processing off the event loop."""
    await run_in_threadpool(
        process_whatsapp_message,
        phone_number,
        message_content,
        message_id,
        whatsapp_number_id,
        whatsapp_message,
        is_image,
    )


async def _verify_webhook_token(hub_mode: str, hub_verify_token: str, hub_challenge: str):
    logger.info("🔐 Webhook verification attempt - Mode: %s", hub_mode)

    if hub_mode != "subscribe":
        raise HTTPException(status_code=400, detail="Invalid request")

    challenge = whatsapp_service.verify_webhook(hub_verify_token, hub_challenge)
    if challenge:
        logger.info("✅ Webhook verification successful")
        return PlainTextResponse(challenge)

    logger.warning("❌ Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.get("/")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
):
    """
    Webhook verification endpoint for WhatsApp
    This endpoint is called by WhatsApp to verify your webhook URL
    """
    return await _verify_webhook_token(hub_mode, hub_verify_token, hub_challenge)


@router.get("/webhook")
async def verify_webhook_alias(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
):
    """Alias for /whatsapp - the path the startup logs advertise."""
    return await _verify_webhook_token(hub_mode, hub_verify_token, hub_challenge)


async def _handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    signature: Optional[str],
):
    try:
        raw_body = await request.body()

        # Signature check before parsing: this is a public URL that can create
        # orders, so an unsigned request should not even be decoded.
        if not whatsapp_service.verify_signature(raw_body, signature):
            raise HTTPException(status_code=403, detail="Invalid signature")

        webhook_data = json.loads(raw_body.decode("utf-8"))

        # The full payload used to be dumped at INFO with indent=2 on every
        # message - megabytes of logs and real CPU spent formatting them.
        logger.debug("📨 Webhook payload: %s", raw_body[:2000])

        whatsapp_message = whatsapp_service.parse_webhook_message(webhook_data)

        if not whatsapp_message:
            logger.debug("📝 Webhook received but no message to process (likely a status update)")
            return {"status": "received"}

        metrics.incr("whatsapp.received")

        if _is_duplicate(whatsapp_message.message_id):
            metrics.incr("whatsapp.duplicate_dropped")
            logger.info(
                "↩️ Ignoring duplicate delivery of message %s", whatsapp_message.message_id
            )
            return {"status": "duplicate_ignored"}

        is_image = whatsapp_message.message_type.value == "image"

        if not is_image and not whatsapp_message.content:
            logger.debug("📝 Message has no text content and is not an image - ignoring")
            return {"status": "received"}

        # Everything expensive - the image download, the vision call, the agent
        # turn - happens here, after we've already decided to return 200.
        background_tasks.add_task(
            process_whatsapp_message_async,
            whatsapp_message.from_number,
            whatsapp_message.content or "",
            whatsapp_message.message_id,
            whatsapp_message.to_number,
            whatsapp_message,
            is_image,
        )

        logger.info(
            "✅ Message queued from %s to %s%s",
            whatsapp_message.from_number,
            whatsapp_message.to_number,
            " (image)" if is_image else "",
        )
        return {"status": "received"}

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error("❌ Invalid JSON in webhook: %s", e)
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error("❌ Error handling webhook: %s", e, exc_info=True)
        # Still return 200 to prevent WhatsApp from retrying a message we would
        # only fail on again.
        return {"status": "error", "message": str(e)}


@router.post("/")
async def handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(default=None),
):
    """
    Handle incoming WhatsApp messages
    This endpoint receives all WhatsApp events (messages, status updates, etc.)
    """
    return await _handle_webhook(request, background_tasks, x_hub_signature_256)


@router.post("/webhook")
async def handle_webhook_alias(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(default=None),
):
    """Alias for /whatsapp - the path the startup logs advertise."""
    return await _handle_webhook(request, background_tasks, x_hub_signature_256)


@router.post("/send-message")
async def send_message(data: Dict[str, Any]):
    """
    Manual endpoint to send messages (for testing or admin use)

    Body format:
    {
        "to": "1234567890",
        "message": "Hello from the chatbot!",
        "type": "text",
        "phone_number_id": "<your business number id>"
    }
    """
    to_number = data.get("to")
    message = data.get("message")
    message_type = data.get("type", "text")
    # Was omitted entirely, so these calls raised TypeError: the service needs to
    # know which business number is sending.
    phone_number_id = data.get("phone_number_id")

    if not to_number or not message:
        raise HTTPException(status_code=400, detail="Missing 'to' or 'message' field")

    if not phone_number_id:
        configured = list(whatsapp_service.configs.keys())
        if len(configured) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing 'phone_number_id'. Configured accounts: "
                    f"{configured or 'none'}"
                ),
            )
        phone_number_id = configured[0]

    try:
        if message_type == "text":
            return await run_in_threadpool(
                whatsapp_service.send_text_message, to_number, message, phone_number_id
            )
        if message_type == "image":
            image_url = data.get("image_url")
            if not image_url:
                raise HTTPException(
                    status_code=400, detail="Missing 'image_url' for image message"
                )
            return await run_in_threadpool(
                whatsapp_service.send_image_message,
                to_number,
                image_url,
                data.get("caption", ""),
                phone_number_id,
            )
        raise HTTPException(status_code=400, detail="Unsupported message type")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Error in manual send message: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """
    Get service status and active sessions
    """
    return {
        "status": "active",
        "active_sessions": len(session_store),
        "background_queue_depth": background.queue_depth(),
        "max_threads": settings.worker_threads,
        "session_store": session_store.stats(),
        "open_handoffs": handoff.count(),
        "sessions": session_store.list_sessions(),
    }


@router.delete("/sessions/{phone_number}")
async def clear_session(phone_number: str, seller_id: str = "default_seller"):
    """
    Clear a specific user session
    """
    if session_store.drop(seller_id, phone_number):
        logger.info("🗑️ Cleared session for %s", phone_number)
        return {"status": "session_cleared", "phone_number": phone_number}
    raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/sessions")
async def clear_all_sessions():
    """
    Clear all active sessions
    """
    count = session_store.clear()
    logger.info("🗑️ Cleared %d sessions", count)
    return {"status": "all_sessions_cleared", "cleared_count": count}


# ---------------------------------------------------------------------------
# Human handover
# ---------------------------------------------------------------------------
@router.get("/handoffs")
async def list_handoffs(seller_id: Optional[str] = None):
    """Conversations waiting for a person. The bot stays silent on these."""
    return {"count": handoff.count(), "handoffs": handoff.list_all(seller_id)}


@router.delete("/handoffs/{phone_number}")
async def resolve_handoff(phone_number: str, seller_id: str = "default_seller"):
    """Close a handover so the bot starts replying to this customer again."""
    if handoff.resolve(seller_id, phone_number):
        return {"status": "handoff_resolved", "phone_number": phone_number}
    raise HTTPException(status_code=404, detail="No open handover for this customer")


@router.get("/profile/{phone_number}")
async def get_profile(phone_number: str, phone_number_id: Optional[str] = None):
    """
    Get WhatsApp profile information for a phone number
    """
    # phone_number_id was never passed, so this raised TypeError on every call.
    if not phone_number_id:
        configured = list(whatsapp_service.configs.keys())
        if len(configured) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Specify phone_number_id. Configured accounts: {configured or 'none'}",
            )
        phone_number_id = configured[0]

    try:
        return await run_in_threadpool(
            whatsapp_service.get_profile_info, phone_number, phone_number_id
        )
    except Exception as e:
        logger.error("❌ Error getting profile: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-image")
async def analyze_image_endpoint(data: Dict[str, Any]):
    """
    Manual endpoint to analyze images using the configured vision model

    Body format:
    {
        "image_path": "/path/to/image.jpg",
        "prompt": "Custom analysis prompt (optional)",
        "language": "en" or "ar",
        "business_context": "Additional business context (optional)"
    }
    """
    image_path = data.get("image_path")
    prompt = data.get("prompt")
    language = data.get("language", "en")
    business_context = data.get("business_context", "")

    if not image_path:
        raise HTTPException(status_code=400, detail="Missing 'image_path' field")
    if not deepseek_vision_service.is_configured():
        raise HTTPException(status_code=503, detail="Vision service not configured")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    if business_context and not prompt:
        prompt = (
            f"{business_context}\n\n"
            "Describe this image and anything about it a shop assistant would need to know."
        )
    elif not prompt:
        prompt = "Describe this image and anything about it a shop assistant would need to know."

    try:
        return await run_in_threadpool(
            deepseek_vision_service.analyze_image, image_path, prompt, language
        )
    except Exception as e:
        logger.error("❌ Error in image analysis endpoint: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-text")
async def extract_text_endpoint(data: Dict[str, Any]):
    """
    Manual endpoint to extract text from images using the configured vision model

    Body format:
    {
        "image_path": "/path/to/image.jpg",
        "language": "en" or "ar"
    }
    """
    image_path = data.get("image_path")
    language = data.get("language", "en")

    if not image_path:
        raise HTTPException(status_code=400, detail="Missing 'image_path' field")
    if not deepseek_vision_service.is_configured():
        raise HTTPException(status_code=503, detail="Vision service not configured")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    try:
        return await run_in_threadpool(
            deepseek_vision_service.extract_text_from_image, image_path, language
        )
    except Exception as e:
        logger.error("❌ Error in text extraction endpoint: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vision/status")
async def vision_status():
    """
    Get vision service status
    """
    return {
        "service": "Vision (OpenAI-compatible)",
        "configured": deepseek_vision_service.is_configured(),
        "supported_formats": deepseek_vision_service.get_supported_formats(),
        "model": deepseek_vision_service.model,
        "api_base": deepseek_vision_service.api_base,
        "api_available": bool(deepseek_vision_service.api_key),
    }
