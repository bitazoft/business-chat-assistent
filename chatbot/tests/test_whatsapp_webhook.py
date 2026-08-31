"""
Webhook safety: deduplication and signature verification.

Deduplication is the one that mattered most - WhatsApp redelivers a webhook it
thinks failed, and without this a redelivery re-ran the customer's message and
could place a second order.
"""
import hashlib
import hmac
import json

import pytest

from routes import whatsapp_routes
from services.whatsapp_service import whatsapp_service


@pytest.fixture(autouse=True)
def _clear_seen():
    whatsapp_routes._seen_messages.clear()
    whatsapp_routes._limiter.reset()
    yield


# ---- deduplication ----------------------------------------------------
def test_first_delivery_is_accepted():
    assert whatsapp_routes._is_duplicate("wamid.ABC123") is False


def test_redelivery_is_rejected():
    assert whatsapp_routes._is_duplicate("wamid.ABC123") is False
    assert whatsapp_routes._is_duplicate("wamid.ABC123") is True
    assert whatsapp_routes._is_duplicate("wamid.ABC123") is True


def test_different_messages_are_independent():
    assert whatsapp_routes._is_duplicate("wamid.A") is False
    assert whatsapp_routes._is_duplicate("wamid.B") is False


def test_missing_message_id_is_never_treated_as_duplicate():
    """No id means we cannot tell - better to process than to silently drop."""
    assert whatsapp_routes._is_duplicate(None) is False
    assert whatsapp_routes._is_duplicate("") is False


def test_duplicate_check_survives_many_ids():
    for i in range(5000):
        assert whatsapp_routes._is_duplicate(f"wamid.{i}") is False
    # The most recent id must still be remembered.
    assert whatsapp_routes._is_duplicate("wamid.4999") is True


# ---- signature verification -------------------------------------------
def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_disabled_by_default_lets_requests_through():
    assert whatsapp_service.verify_signature(b"{}", None) is True


def test_valid_signature_accepted(override_settings):
    secret = "shhh"
    override_settings(verify_webhook_signature=True, whatsapp_app_secret=secret)
    body = json.dumps({"entry": []}).encode()
    assert whatsapp_service.verify_signature(body, _sign(body, secret)) is True


def test_wrong_secret_rejected(override_settings):
    override_settings(verify_webhook_signature=True, whatsapp_app_secret="right")
    body = b'{"entry": []}'
    assert whatsapp_service.verify_signature(body, _sign(body, "wrong")) is False


def test_tampered_body_rejected(override_settings):
    """The point of signing: a modified payload must not verify."""
    secret = "shhh"
    override_settings(verify_webhook_signature=True, whatsapp_app_secret=secret)
    original = b'{"amount": 100}'
    signature = _sign(original, secret)
    assert whatsapp_service.verify_signature(b'{"amount": 999}', signature) is False


def test_missing_header_rejected(override_settings):
    override_settings(verify_webhook_signature=True, whatsapp_app_secret="shhh")
    assert whatsapp_service.verify_signature(b"{}", None) is False


def test_malformed_header_rejected(override_settings):
    override_settings(verify_webhook_signature=True, whatsapp_app_secret="shhh")
    for header in ["garbage", "sha1=abc", "sha256=", "=abc"]:
        assert whatsapp_service.verify_signature(b"{}", header) is False


def test_enabled_without_a_secret_rejects_rather_than_allows(override_settings):
    """Failing closed: a misconfiguration must not silently disable the check."""
    override_settings(verify_webhook_signature=True, whatsapp_app_secret=None)
    assert whatsapp_service.verify_signature(b"{}", "sha256=abc") is False


# ---- resilience -------------------------------------------------------
def test_a_database_outage_still_gets_the_customer_an_answer(monkeypatch):
    """The seller lookup reads the database. If it fails the customer must still
    hear something - the previous error handler called that same lookup again,
    so a database blip produced total silence."""
    from routes import whatsapp_routes

    sent = []
    monkeypatch.setattr(
        whatsapp_routes,
        "_send_reply",
        lambda phone, text, number_id: sent.append(text) or True,
    )

    def _explode(*args, **kwargs):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(whatsapp_routes, "get_seller_id_by_whatsapp_number_id", _explode)

    # Must not raise.
    whatsapp_routes.process_whatsapp_message(
        phone_number="94771234567",
        message_content="hello",
        message_id="wamid.X",
        whatsapp_number_id="111",
    )

    assert len(sent) == 1
    assert sent[0]


def test_generic_error_message_never_raises(monkeypatch):
    from routes import whatsapp_routes
    from services import outbound_formatter

    monkeypatch.setattr(
        outbound_formatter,
        "render",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("templates down")),
    )
    assert whatsapp_routes._generic_error_message() == whatsapp_routes.FALLBACK_ERROR_MESSAGE
