"""
Endpoints for editing the outbound message templates.

A seller (or the Admin Portal on their behalf) can change what the bot says
without a code change. seller_id is optional everywhere: omit it to edit the
global default that applies to every seller, pass it to override just that one.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from templates.template_store import (
    DEFAULT_TEMPLATES,
    TemplateKey,
    placeholders_in,
    render_template,
    template_store,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/templates", tags=["Templates"])

# Supplied to every template by services/outbound_formatter._base_context.
ALWAYS_AVAILABLE = {"shop_name", "customer_name", "customer_name_suffix"}

# Sample values so a preview looks like a real message.
PREVIEW_CONTEXT: Dict[str, Any] = {
    "message": "Yes, we have that in stock!",
    "shop_name": "Sample Shop",
    "customer_name": "Nimal",
    "customer_name_suffix": " Nimal",
    "product_name": "Ceylon Tea 500g",
    "name": "Ceylon Tea 500g",
    "description": "Premium loose-leaf black tea",
    "price": "1250.00",
    "stock": 42,
    "product_id": 7,
    "items": "1. *Ceylon Tea 500g* — Rs.1250.00 (42 in stock)\n2. *Spice Box* — Rs.890.00 (12 in stock)",
    "count": 2,
    "order_id": 1042,
    "total_amount": "2140.00",
    "status": "Pending",
    "status_emoji": "⏳",
    "created_at": "2026-08-31 14:05",
    "details": "Order ID: 1042, Status: Pending",
    "amount": "2140.00",
    "currency": "LKR",
    "reference": "TXN889210",
    "bank_details": "Bank: Sample Bank\nAccount: 1234567890\nName: Sample Shop",
    "reason": "Customer asked for a refund",
    "flag_reason": "Amount could not be read",
    "note": "Stock has been restored.",
    "retry_after": 30,
    "email": "nimal@example.com",
    "address": "45/2 Galle Road, Colombo 03",
    "phone": "+94771234567",
    "details_text": "",
}


class TemplateUpdate(BaseModel):
    body: str = Field(..., min_length=1, max_length=8000)
    seller_id: Optional[str] = None
    enabled: bool = True
    description: Optional[str] = None


class TemplatePreview(BaseModel):
    body: Optional[str] = None
    seller_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


def _known_keys() -> List[str]:
    return sorted(DEFAULT_TEMPLATES.keys())


@router.get("")
async def list_templates(seller_id: Optional[str] = None):
    """Every template, its effective body, and where that body came from.

    `source` is "seller" (this seller's override), "global" (an override for
    everyone), or "default" (shipped with the app).
    """
    templates = await run_in_threadpool(template_store.list_all, seller_id)
    return {"count": len(templates), "seller_id": seller_id, "templates": templates}


@router.get("/keys")
async def list_template_keys():
    """The customisable message types and what each is for."""
    return {
        "keys": [
            {
                "template_key": key,
                "description": definition.description,
                "supported_placeholders": list(definition.placeholders),
                "default_body": definition.body,
            }
            for key, definition in sorted(DEFAULT_TEMPLATES.items())
        ]
    }


@router.get("/{template_key}")
async def get_template(template_key: str, seller_id: Optional[str] = None):
    """One template's effective body for this seller."""
    body = await run_in_threadpool(template_store.get_body, template_key, seller_id)
    if not body:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown template '{template_key}'. Known keys: {_known_keys()}",
        )
    default = DEFAULT_TEMPLATES.get(template_key)
    return {
        "template_key": template_key,
        "seller_id": seller_id,
        "body": body,
        "placeholders": placeholders_in(body),
        "supported_placeholders": list(default.placeholders) if default else [],
        "description": default.description if default else None,
        "default_body": default.body if default else None,
    }


@router.put("/{template_key}")
async def upsert_template(template_key: str, update: TemplateUpdate):
    """Create or replace a template.

    Unknown placeholders are allowed but reported back in `unknown_placeholders`
    so a typo is visible: at send time they render as empty text rather than
    raising, which would otherwise silently produce odd-looking messages.
    """
    if template_key not in DEFAULT_TEMPLATES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown template '{template_key}'. Known keys: {_known_keys()}",
        )

    used = placeholders_in(update.body)
    supported = set(DEFAULT_TEMPLATES[template_key].placeholders) | ALWAYS_AVAILABLE
    unknown = [p for p in used if p not in supported]

    # outbound_wrapper is the one template where a missing placeholder throws the
    # reply away entirely, so it's rejected rather than warned about.
    if template_key == TemplateKey.OUTBOUND_WRAPPER and "{message}" not in update.body:
        raise HTTPException(
            status_code=400,
            detail="The outbound_wrapper template must contain {message}, or replies would be discarded.",
        )

    try:
        result = await run_in_threadpool(
            template_store.upsert,
            template_key,
            update.body,
            update.seller_id,
            update.enabled,
            update.description,
        )
    except Exception as e:
        logger.error("Could not save template %s: %s", template_key, e)
        raise HTTPException(status_code=500, detail=f"Could not save template: {e}")

    return {
        **result,
        "unknown_placeholders": unknown,
        "preview": render_template(update.body, PREVIEW_CONTEXT).strip(),
    }


@router.delete("/{template_key}")
async def delete_template(template_key: str, seller_id: Optional[str] = None):
    """Remove a customisation so the shipped default applies again."""
    removed = await run_in_threadpool(template_store.delete, template_key, seller_id)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No stored override for '{template_key}'"
            + (f" and seller {seller_id}" if seller_id else " (global)"),
        )
    return {"status": "reverted_to_default", "template_key": template_key, "seller_id": seller_id}


@router.post("/{template_key}/preview")
async def preview_template(template_key: str, preview: TemplatePreview):
    """Render a template with sample data - to check wording before saving.

    Pass `body` to preview unsaved text, or omit it to preview what's stored.
    """
    body = preview.body
    if body is None:
        body = await run_in_threadpool(
            template_store.get_body, template_key, preview.seller_id
        )
    if not body:
        raise HTTPException(status_code=404, detail=f"Unknown template '{template_key}'")

    context = {**PREVIEW_CONTEXT, **preview.context}
    return {
        "template_key": template_key,
        "placeholders": placeholders_in(body),
        "rendered": render_template(body, context).strip(),
    }


@router.post("/reload")
async def reload_templates():
    """Drop the template cache, so edits made directly in the database show up."""
    template_store.invalidate()
    return {"status": "template_cache_cleared"}
