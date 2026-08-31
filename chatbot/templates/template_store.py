"""
Customisable outbound message templates.

The wording the bot sends used to be hard-coded in Python (templates/message_templates.py),
so changing a greeting or a payment instruction meant a code change and a deploy.
Templates now live in the message_templates table and are edited through
/templates. A row with seller_id NULL is the default for everyone; a row with a
seller_id overrides it for that seller only. The defaults below are the final
fallback, so the bot still works with an empty table.

Rendering is deliberately restricted. Templates are edited by shop staff, and

    "{message.__class__.__mro__}"

is valid str.format syntax that would leak internals, while a typo like
"{custmer_name}" would raise KeyError mid-send. So this substitutes bare
{placeholder} names only - no attribute access, no indexing, no format specs -
and an unknown placeholder renders as empty text instead of raising.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.cache import get_cache
from utils.logger import get_logger

logger = get_logger(__name__)

# Only bare names: letters, digits, underscore. Anything else is left as-is.
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Templates change rarely and are read on every outbound message.
_template_cache = get_cache("message_templates", maxsize=512, ttl=300)


class TemplateKey:
    """The message types a seller can customise."""

    # Conversation-level
    GREETING = "greeting"
    FALLBACK = "fallback"
    ERROR = "error"
    HANDOFF = "handoff"
    RATE_LIMITED = "rate_limited"
    AWAY = "away"

    # Wraps a plain LLM answer that no other template matched
    OUTBOUND_WRAPPER = "outbound_wrapper"

    # Products
    PRODUCT_DETAILS = "product_details"
    PRODUCT_LIST = "product_list"
    PRODUCT_NOT_FOUND = "product_not_found"

    # Orders
    ORDER_CONFIRMATION = "order_confirmation"
    ORDER_DETAILS = "order_details"
    ORDER_CANCELLED = "order_cancelled"
    TRACKING_STATUS = "tracking_status"

    # Payments
    PAYMENT_REQUEST = "payment_request"
    PAYMENT_CONFIRMED = "payment_confirmed"
    PAYMENT_PENDING_REVIEW = "payment_pending_review"
    PAYMENT_MISMATCH = "payment_mismatch"

    # Media
    IMAGE_RECEIVED = "image_received"

    # Customer
    CUSTOMER_INFO = "customer_info"
    DETAILS_REQUEST = "details_request"


@dataclass(frozen=True)
class TemplateDef:
    key: str
    body: str
    description: str
    placeholders: tuple = ()


# The shipped defaults. {message} is whatever the LLM or a tool produced.
DEFAULT_TEMPLATES: Dict[str, TemplateDef] = {
    TemplateKey.OUTBOUND_WRAPPER: TemplateDef(
        key=TemplateKey.OUTBOUND_WRAPPER,
        body="{message}",
        description=(
            "Wraps every reply that no more specific template matched. Must contain "
            "{message}. Use it for a signature or shop name on every message."
        ),
        placeholders=("message", "shop_name", "customer_name"),
    ),
    TemplateKey.GREETING: TemplateDef(
        key=TemplateKey.GREETING,
        body="👋 Hello{customer_name_suffix}! Welcome to {shop_name}.\n\nHow can I help you today?",
        description="First reply in a new conversation.",
        placeholders=("shop_name", "customer_name", "customer_name_suffix"),
    ),
    TemplateKey.FALLBACK: TemplateDef(
        key=TemplateKey.FALLBACK,
        body=(
            "I'm not sure I understood that. Could you rephrase it?\n\n"
            "I can help you with products, placing an order, or tracking one you've "
            "already placed."
        ),
        description="Sent when the bot cannot work out what the customer wants.",
    ),
    TemplateKey.ERROR: TemplateDef(
        key=TemplateKey.ERROR,
        body=(
            "⚠️ Sorry, something went wrong on our side. Please try again in a moment.\n\n"
            "If it keeps happening, someone from {shop_name} will help you."
        ),
        description="Sent when the bot hits a technical error.",
        placeholders=("shop_name", "details"),
    ),
    TemplateKey.HANDOFF: TemplateDef(
        key=TemplateKey.HANDOFF,
        body=(
            "I've passed this on to our team — someone from {shop_name} will get back "
            "to you shortly. 🙏"
        ),
        description="Sent when the conversation is handed to a person.",
        placeholders=("shop_name", "reason"),
    ),
    TemplateKey.RATE_LIMITED: TemplateDef(
        key=TemplateKey.RATE_LIMITED,
        body="You're sending messages faster than I can keep up with. Please try again in {retry_after} seconds. 🙏",
        description="Sent when a customer exceeds the message rate limit.",
        placeholders=("retry_after",),
    ),
    TemplateKey.AWAY: TemplateDef(
        key=TemplateKey.AWAY,
        body="Thanks for your message! {shop_name} is currently closed. We'll reply as soon as we're back. 🌙",
        description="Out-of-hours reply (only used if you enable an away schedule).",
        placeholders=("shop_name",),
    ),
    TemplateKey.PRODUCT_NOT_FOUND: TemplateDef(
        key=TemplateKey.PRODUCT_NOT_FOUND,
        body=(
            "🔍 I couldn't find *{product_name}* in our catalogue.\n\n"
            "Would you like to see what we do have?"
        ),
        description="Sent when a product search returns nothing.",
        placeholders=("product_name",),
    ),
    TemplateKey.PRODUCT_DETAILS: TemplateDef(
        key=TemplateKey.PRODUCT_DETAILS,
        body=(
            "🛍️ *{name}*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "📝 {description}\n"
            "💰 *Price:* Rs.{price}\n"
            "📊 *Stock:* {stock} units\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "✨ Ready to order? Just let me know!"
        ),
        description="A single product's details.",
        placeholders=("product_id", "name", "description", "price", "stock"),
    ),
    TemplateKey.PRODUCT_LIST: TemplateDef(
        key=TemplateKey.PRODUCT_LIST,
        body=(
            "🛒 *OUR PRODUCTS*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "{items}\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🔍 Ask about any product for more detail."
        ),
        description="The product catalogue. {items} is the generated list.",
        placeholders=("items", "count"),
    ),
    TemplateKey.ORDER_CONFIRMATION: TemplateDef(
        key=TemplateKey.ORDER_CONFIRMATION,
        body=(
            "✅ *ORDER PLACED*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🆔 *Order:* #{order_id}\n"
            "💰 *Total:* Rs.{total_amount}\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "How would you like to pay — *COD* or *Bank Transfer*?"
        ),
        description="Sent right after an order is created.",
        placeholders=("order_id", "total_amount", "items", "shop_name"),
    ),
    TemplateKey.ORDER_DETAILS: TemplateDef(
        key=TemplateKey.ORDER_DETAILS,
        body=(
            "📋 *ORDER #{order_id}*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "{status_emoji} *Status:* {status}\n"
            "💰 *Total:* Rs.{total_amount}\n"
            "{items}\n"
            "━━━━━━━━━━━━━━━━━━━━"
        ),
        description="An existing order's details.",
        placeholders=("order_id", "status", "status_emoji", "total_amount", "items", "created_at"),
    ),
    TemplateKey.ORDER_CANCELLED: TemplateDef(
        key=TemplateKey.ORDER_CANCELLED,
        body="❌ Order #{order_id} has been cancelled. {note}",
        description="Confirmation that an order was cancelled.",
        placeholders=("order_id", "note", "reason"),
    ),
    TemplateKey.TRACKING_STATUS: TemplateDef(
        key=TemplateKey.TRACKING_STATUS,
        body=(
            "🚚 *ORDER TRACKING*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🆔 *Order:* #{order_id}\n"
            "{status_emoji} *Status:* {status}\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "{details}"
        ),
        description="Delivery / status update for an order.",
        placeholders=("order_id", "status", "status_emoji", "details"),
    ),
    TemplateKey.PAYMENT_REQUEST: TemplateDef(
        key=TemplateKey.PAYMENT_REQUEST,
        body=(
            "💳 *PAYMENT*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Order *#{order_id}* — total *Rs.{total_amount}*\n\n"
            "{bank_details}\n\n"
            "Please send a photo of your transfer slip once you've paid. 📸"
        ),
        description="Bank transfer instructions. Put your account details in {bank_details}.",
        placeholders=("order_id", "total_amount", "bank_details", "shop_name"),
    ),
    TemplateKey.PAYMENT_CONFIRMED: TemplateDef(
        key=TemplateKey.PAYMENT_CONFIRMED,
        body=(
            "✅ *PAYMENT RECEIVED*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "We received Rs.{amount} for order *#{order_id}*.\n\n"
            "Thank you! We'll start preparing it right away. 📦"
        ),
        description="Sent when a receipt is verified and matches the order total.",
        placeholders=("order_id", "amount", "currency", "reference"),
    ),
    TemplateKey.PAYMENT_PENDING_REVIEW: TemplateDef(
        key=TemplateKey.PAYMENT_PENDING_REVIEW,
        body=(
            "📄 I've saved your payment slip for order *#{order_id}*, but I couldn't "
            "read it automatically.\n\nOur team will check it and confirm shortly."
        ),
        description="Sent when a receipt is unreadable and needs a human.",
        placeholders=("order_id", "reason"),
    ),
    TemplateKey.PAYMENT_MISMATCH: TemplateDef(
        key=TemplateKey.PAYMENT_MISMATCH,
        body=(
            "⚠️ I've saved your receipt for order *#{order_id}*, but the amount on it "
            "(Rs.{amount}) doesn't match the order total (Rs.{total_amount}).\n\n"
            "Our team will review this and get back to you."
        ),
        description="Sent when the receipt amount does not match the order total.",
        placeholders=("order_id", "amount", "total_amount"),
    ),
    TemplateKey.IMAGE_RECEIVED: TemplateDef(
        key=TemplateKey.IMAGE_RECEIVED,
        body="📸 Got your image — give me a moment to look at it...",
        description="Immediate acknowledgement when a customer sends a photo.",
    ),
    TemplateKey.CUSTOMER_INFO: TemplateDef(
        key=TemplateKey.CUSTOMER_INFO,
        body=(
            "👤 *YOUR DETAILS*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "*Name:* {name}\n"
            "*Email:* {email}\n"
            "*Address:* {address}\n"
            "*Phone:* {phone}\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Let me know if anything needs updating."
        ),
        description="The customer's stored profile.",
        placeholders=("name", "email", "address", "phone"),
    ),
    TemplateKey.DETAILS_REQUEST: TemplateDef(
        key=TemplateKey.DETAILS_REQUEST,
        body=(
            "To place your order I need a few details. Please send me your:\n\n"
            "1️⃣ Full name\n2️⃣ Email address\n3️⃣ Delivery address\n4️⃣ Phone number"
        ),
        description="Asks a new customer for their details before an order.",
    ),
}


def render_template(body: str, context: Dict[str, Any]) -> str:
    """Substitute {placeholder} names in `body` from `context`.

    Unknown placeholders become empty strings - a seller's typo should produce
    slightly odd copy, not a failed message. Values are str()-ed; None is empty.
    """

    def _replace(match: "re.Match") -> str:
        value = context.get(match.group(1))
        return "" if value is None else str(value)

    return _PLACEHOLDER.sub(_replace, body or "")


def placeholders_in(body: str) -> List[str]:
    """Placeholder names used by a template, for validation and preview."""
    seen: List[str] = []
    for name in _PLACEHOLDER.findall(body or ""):
        if name not in seen:
            seen.append(name)
    return seen


class TemplateStore:
    """Resolves a template key to a body: seller override, else global, else default."""

    def _cache_key(self, key: str, seller_id: Optional[str]) -> str:
        return f"{seller_id or '*'}|{key}"

    def get_body(self, key: str, seller_id: Optional[str] = None) -> str:
        """The effective template body for this key and seller."""
        cache_key = self._cache_key(key, seller_id)
        cached = _template_cache.get(cache_key)
        if cached is not None:
            return cached

        body = self._load_body(key, seller_id)
        _template_cache.set(cache_key, body)
        return body

    def _load_body(self, key: str, seller_id: Optional[str]) -> str:
        """Read from the database, falling back to the shipped default.

        A database problem here must not stop the bot replying, so any failure
        falls through to the default body.
        """
        default = DEFAULT_TEMPLATES.get(key)
        default_body = default.body if default else ""

        try:
            from db.database import read_session
            from models.schemas import MessageTemplate

            with read_session() as db:
                query = db.query(MessageTemplate.body, MessageTemplate.seller_id).filter(
                    MessageTemplate.template_key == key,
                    MessageTemplate.enabled.is_(True),
                )
                if seller_id is not None:
                    query = query.filter(
                        MessageTemplate.seller_id.in_([str(seller_id), None])
                    )
                else:
                    query = query.filter(MessageTemplate.seller_id.is_(None))
                rows = query.all()

            if rows:
                # A seller-specific row wins over the global one.
                for body, row_seller in rows:
                    if row_seller is not None:
                        return body
                return rows[0][0]

        except Exception as e:
            logger.debug("Template lookup for %r fell back to default: %s", key, e)

        return default_body

    def render(self, key: str, seller_id: Optional[str] = None, /, **context) -> str:
        """Render a template. Returns "" if the key is unknown and undefined.

        `key` and `seller_id` are positional-only (the `/`) on purpose: callers
        splat a context dict in here, and without it a dict carrying a
        "seller_id" entry raises TypeError at the call site - which is exactly
        what happened, breaking every templated reply. Now such an entry is just
        another placeholder value.
        """
        body = self.get_body(key, seller_id)
        if not body:
            logger.warning("No template body for key %r", key)
            return ""
        return render_template(body, context).strip()

    def has(self, key: str) -> bool:
        return key in DEFAULT_TEMPLATES or bool(self.get_body(key))

    # -- editing ---------------------------------------------------------
    def upsert(
        self,
        key: str,
        body: str,
        seller_id: Optional[str] = None,
        enabled: bool = True,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or replace a template, then drop it from the cache."""
        from db.database import session_scope
        from models.schemas import MessageTemplate

        with session_scope() as db:
            existing = (
                db.query(MessageTemplate)
                .filter(
                    MessageTemplate.template_key == key,
                    MessageTemplate.seller_id == (str(seller_id) if seller_id else None),
                )
                .first()
            )
            if existing:
                existing.body = body
                existing.enabled = enabled
                if description is not None:
                    existing.description = description
            else:
                db.add(
                    MessageTemplate(
                        template_key=key,
                        body=body,
                        seller_id=str(seller_id) if seller_id else None,
                        enabled=enabled,
                        description=description,
                    )
                )

        self.invalidate(key, seller_id)
        logger.info("Template %r updated for seller %s", key, seller_id or "(all)")
        return {
            "template_key": key,
            "seller_id": seller_id,
            "enabled": enabled,
            "placeholders": placeholders_in(body),
        }

    def delete(self, key: str, seller_id: Optional[str] = None) -> bool:
        """Remove a customisation so the default applies again."""
        from db.database import session_scope
        from models.schemas import MessageTemplate

        with session_scope() as db:
            deleted = (
                db.query(MessageTemplate)
                .filter(
                    MessageTemplate.template_key == key,
                    MessageTemplate.seller_id == (str(seller_id) if seller_id else None),
                )
                .delete()
            )

        self.invalidate(key, seller_id)
        return bool(deleted)

    def invalidate(self, key: Optional[str] = None, seller_id: Optional[str] = None) -> None:
        """Drop cached bodies. Called after any edit."""
        if key is None:
            _template_cache.clear()
            return
        _template_cache.delete(self._cache_key(key, seller_id))
        # A changed global default affects every seller's cached resolution.
        if seller_id is None:
            _template_cache.clear()

    def list_all(self, seller_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every known key with its effective body and where it came from."""
        from db.database import read_session
        from models.schemas import MessageTemplate

        overrides: Dict[str, Dict[str, Any]] = {}
        try:
            with read_session() as db:
                rows = db.query(MessageTemplate).all()
            for row in rows:
                scope = "seller" if row.seller_id else "global"
                if row.seller_id and str(row.seller_id) != str(seller_id or ""):
                    continue
                # A seller row outranks a global row for the same key.
                if key_existing := overrides.get(row.template_key):
                    if key_existing["scope"] == "seller":
                        continue
                overrides[row.template_key] = {
                    "scope": scope,
                    "body": row.body,
                    "enabled": row.enabled,
                    "description": row.description,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
        except Exception as e:
            logger.warning("Could not list template overrides: %s", e)

        out: List[Dict[str, Any]] = []
        for key in sorted(set(DEFAULT_TEMPLATES) | set(overrides)):
            default = DEFAULT_TEMPLATES.get(key)
            override = overrides.get(key)
            body = override["body"] if override else (default.body if default else "")
            out.append(
                {
                    "template_key": key,
                    "body": body,
                    "source": override["scope"] if override else "default",
                    "enabled": override["enabled"] if override else True,
                    "description": (override or {}).get("description")
                    or (default.description if default else ""),
                    "placeholders": placeholders_in(body),
                    "supported_placeholders": list(default.placeholders) if default else [],
                    "default_body": default.body if default else None,
                    "updated_at": (override or {}).get("updated_at"),
                }
            )
        return out


template_store = TemplateStore()
