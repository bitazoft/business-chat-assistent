"""
Turns a finished agent turn into the text we actually send the customer.

This is the layer the seller's templates apply at. Two paths:

1. A tool produced structured data (a product, an order, a receipt result). The
   matching template is rendered from that data, so the seller's wording wins over
   whatever the model decided to write. The model's paraphrase is discarded here,
   which is also what the system prompt asks it not to produce.

2. No structured data - a free-text answer. It goes through the
   outbound_wrapper template, which is "{message}" by default but is where a
   seller can add a shop name or signature to every reply.

Formatting is separate from sending: this returns one string, and the WhatsApp
service splits it into as many messages as the 4096-character limit needs.
"""
from typing import Any, Dict, List, Optional

from templates.template_store import TemplateKey, template_store
from utils.logger import get_logger

logger = get_logger(__name__)

STATUS_EMOJI = {
    "pending": "⏳",
    "confirmed": "✅",
    "processing": "🔄",
    "shipped": "🚚",
    "delivered": "📦",
    "cancelled": "❌",
    "paid": "💰",
}


def _status_emoji(status: Any) -> str:
    return STATUS_EMOJI.get(str(status or "").strip().lower(), "📋")


def _money(value: Any) -> str:
    """Format an amount the way the templates expect (no currency symbol)."""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value if value is not None else "")


def _base_context(seller_id: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from repositories.tools import get_shop_name

    customer_name = (extra or {}).get("customer_name") or ""
    # Deliberately no "seller_id" key: it would collide with the seller_id
    # parameter of TemplateStore.render when this dict is splatted, and a
    # customer has no use for the shop's internal id anyway.
    context: Dict[str, Any] = {
        "shop_name": get_shop_name(seller_id),
        "customer_name": customer_name,
        # Lets a greeting read "Hello Nimal!" or just "Hello!" from one template.
        "customer_name_suffix": f" {customer_name}" if customer_name else "",
    }
    if extra:
        context.update({k: v for k, v in extra.items() if v is not None})
    return context


def _render_product_details(data: Dict[str, Any], seller_id: str, ctx: Dict[str, Any]) -> str:
    return template_store.render(
        TemplateKey.PRODUCT_DETAILS,
        seller_id,
        **ctx,
        product_id=data.get("product_id", ""),
        name=data.get("product") or data.get("name") or "",
        description=data.get("description") or "No description available",
        price=_money(data.get("price")),
        stock=data.get("stock", 0),
    )


def _render_product_list(data: Any, seller_id: str, ctx: Dict[str, Any]) -> str:
    products = [p for p in (data or []) if isinstance(p, dict)]
    if not products:
        return ""
    lines = []
    for i, product in enumerate(products[:20], 1):
        lines.append(
            f"{i}. *{product.get('name', 'Unknown')}* — Rs.{_money(product.get('price'))}"
            f" ({product.get('stock', 0)} in stock)"
        )
    return template_store.render(
        TemplateKey.PRODUCT_LIST,
        seller_id,
        **ctx,
        items="\n".join(lines),
        count=len(products),
    )


def _render_order_details(data: Dict[str, Any], seller_id: str, ctx: Dict[str, Any]) -> str:
    items = data.get("items") or []
    item_lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("product") or item.get("name") or f"Product {item.get('product_id', '')}"
        item_lines.append(
            f"• {name} × {item.get('quantity', 1)} — Rs.{_money(item.get('price'))}"
        )

    return template_store.render(
        TemplateKey.ORDER_DETAILS,
        seller_id,
        **ctx,
        order_id=data.get("order_id", ""),
        status=str(data.get("status") or "pending").title(),
        status_emoji=_status_emoji(data.get("status")),
        total_amount=_money(data.get("total_amount")),
        items="\n".join(item_lines),
        created_at=data.get("created_at", ""),
    )


def _order_items_text(order_id: Any, fallback_items: Any) -> str:
    """A readable item list for an order confirmation.

    place_order only knows the ids and quantities the model passed in, so the
    names and prices are read back from the order we just created. One indexed
    query, and only when an order is placed. Falls back to ids if that fails -
    an ugly line is better than a failed confirmation.
    """
    try:
        from repositories.tools import get_order_details

        details = get_order_details(order_id=int(order_id))
        lines = [
            f"• {item.get('product')} × {item.get('quantity')} — Rs.{_money(item.get('price'))}"
            for item in (details.get("items") or [])
            if isinstance(item, dict)
        ]
        if lines:
            return "\n".join(lines)
    except Exception as e:
        logger.debug("Could not read items for order %s: %s", order_id, e)

    return "\n".join(
        f"• {item.get('quantity', 1)} × item {item.get('product_id')}"
        for item in (fallback_items or [])
        if isinstance(item, dict)
    )


def _render_order_confirmation(data: Dict[str, Any], seller_id: str, ctx: Dict[str, Any]) -> str:
    order_id = data.get("order_id", "")
    return template_store.render(
        TemplateKey.ORDER_CONFIRMATION,
        seller_id,
        **ctx,
        order_id=order_id,
        total_amount=_money(data.get("total_amount")),
        items=_order_items_text(order_id, data.get("items")),
    )


def _render_tracking(data: Dict[str, Any], seller_id: str, ctx: Dict[str, Any]) -> str:
    return template_store.render(
        TemplateKey.TRACKING_STATUS,
        seller_id,
        **ctx,
        order_id=data.get("order_id", ""),
        status=str(data.get("status") or "pending").title(),
        status_emoji=_status_emoji(data.get("status")),
        details=data.get("details", ""),
    )


def _render_payment(data: Dict[str, Any], seller_id: str, ctx: Dict[str, Any]) -> str:
    verification = str(data.get("verification") or "").lower()

    if verification == "verified":
        return template_store.render(
            TemplateKey.PAYMENT_CONFIRMED,
            seller_id,
            **ctx,
            order_id=data.get("order_id", ""),
            amount=_money(data.get("amount")),
            currency=data.get("currency", ""),
            reference=data.get("reference", ""),
        )

    if verification == "amount_mismatch":
        return template_store.render(
            TemplateKey.PAYMENT_MISMATCH,
            seller_id,
            **ctx,
            order_id=data.get("order_id", ""),
            amount=_money(data.get("amount")),
            total_amount=_money(data.get("total_amount")),
        )

    # not_a_receipt / unreadable / anything else needs a human to look.
    return template_store.render(
        TemplateKey.PAYMENT_PENDING_REVIEW,
        seller_id,
        **ctx,
        order_id=data.get("order_id", ""),
        reason=data.get("flag_reason", ""),
    )


# Which tool result feeds which template, most specific first. A payment result
# matters more than the product lookup that may have happened earlier in the turn.
_RENDERERS = (
    ("verify_and_save_payment_proof", _render_payment),
    ("place_order", _render_order_confirmation),
    ("get_order_details", _render_order_details),
    ("track_order", _render_tracking),
    ("get_product_info", _render_product_details),
    ("get_all_products", _render_product_list),
)


def format_reply(
    response: str,
    tool_results: Optional[List[Dict[str, Any]]] = None,
    seller_id: str = "",
    context: Optional[Dict[str, Any]] = None,
    escalated: bool = False,
) -> str:
    """The final text to send.

    Falls back to the model's own reply whenever a template renders empty, so a
    seller who blanks a template gets the raw answer rather than silence.
    """
    ctx = _base_context(str(seller_id), context)

    if escalated:
        rendered = template_store.render(
            TemplateKey.HANDOFF, seller_id, **ctx, reason=ctx.get("reason", "")
        )
        if rendered:
            return rendered

    by_tool: Dict[str, Any] = {}
    for entry in tool_results or []:
        name = entry.get("tool_name")
        data = entry.get("data")
        if name and isinstance(data, (dict, list)) and name not in by_tool:
            by_tool[name] = data

    for tool_name, renderer in _RENDERERS:
        if tool_name not in by_tool:
            continue
        try:
            rendered = renderer(by_tool[tool_name], str(seller_id), ctx)
        except Exception as e:
            logger.warning("Template render failed for %s: %s", tool_name, e)
            continue
        if rendered:
            return rendered

    # No structured data - wrap the model's own words.
    wrapper = template_store.get_body(TemplateKey.OUTBOUND_WRAPPER, str(seller_id))
    if not wrapper or "{message}" not in wrapper:
        # A wrapper without {message} would throw the reply away.
        if wrapper:
            logger.warning(
                "outbound_wrapper template for seller %s has no {message} placeholder - "
                "sending the reply unwrapped",
                seller_id,
            )
        return response

    from templates.template_store import render_template

    return render_template(wrapper, dict(ctx, message=response)).strip()


def render(key: str, seller_id: str = "", /, **context) -> str:
    """Render one template directly, for the fixed messages the routes send.

    Positional-only for the same reason as TemplateStore.render: the context is
    splatted, so a "seller_id" or "key" entry must not collide with a parameter.
    """
    return template_store.render(key, str(seller_id), **_base_context(str(seller_id), context))
