"""
End-to-end checks through the ASGI app.

TestClient is built without `with`, so the lifespan (model preload, sweeper)
does not run - these exercise routing and validation, not startup.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import main

    return TestClient(main.app)


def test_shallow_health_is_always_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_deep_health_reports_unhealthy_when_the_database_is_down(client):
    """The old /health returned healthy unconditionally, even with no DB."""
    response = client.get("/health?deep=true")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["checks"]["database"]["ok"] is False


def test_metrics_endpoint_shape(client):
    body = client.get("/metrics").json()
    for key in ("counters", "latency", "caches", "sessions", "db_pool", "uptime_seconds"):
        assert key in body


def test_template_keys_are_listed(client):
    body = client.get("/templates/keys").json()
    keys = {k["template_key"] for k in body["keys"]}
    assert {"greeting", "outbound_wrapper", "payment_confirmed", "order_details"} <= keys


def test_get_a_single_template_falls_back_to_the_default(client):
    body = client.get("/templates/greeting").json()
    assert body["template_key"] == "greeting"
    assert body["body"]
    assert "shop_name" in body["placeholders"]


def test_unknown_template_is_404(client):
    assert client.get("/templates/no_such_template").status_code == 404


def test_template_preview_renders_sample_data(client):
    response = client.post(
        "/templates/greeting/preview",
        json={"body": "Hi {customer_name}, welcome to {shop_name}!"},
    )
    assert response.status_code == 200
    assert response.json()["rendered"] == "Hi Nimal, welcome to Sample Shop!"


def test_preview_does_not_raise_on_an_unknown_placeholder(client):
    response = client.post(
        "/templates/greeting/preview", json={"body": "Hello {nonexistent}!"}
    )
    assert response.status_code == 200
    assert response.json()["rendered"] == "Hello !"


def test_wrapper_without_message_placeholder_is_rejected(client):
    """Saving this would throw away every reply, so it must not be storable."""
    response = client.put(
        "/templates/outbound_wrapper",
        json={"body": "Thanks for contacting us!"},
    )
    assert response.status_code == 400
    assert "message" in response.json()["detail"]


def test_saving_an_unknown_template_key_is_rejected(client):
    response = client.put("/templates/made_up_key", json={"body": "hello"})
    assert response.status_code == 400


def test_usage_endpoints_respond_without_data(client):
    assert client.get("/usage/summary").status_code == 200
    assert client.get("/usage/models").status_code == 200
    assert client.get("/usage/optimization").status_code == 200
    assert client.get("/usage/seller/7").status_code == 200
    assert client.get("/usage/session/7/94771234567").status_code == 200


def test_usage_history_reports_a_clear_error_without_the_migration(client):
    response = client.get("/usage/history")
    assert response.status_code == 503
    assert "migration" in response.json()["detail"].lower()


def test_usage_history_rejects_a_bad_group_by(client):
    assert client.get("/usage/history?group_by=drop_table").status_code == 422


def test_chat_rejects_a_non_numeric_seller_id(client):
    response = client.post(
        "/chat",
        json={"message": "hi", "session_id": "s", "seller_id": "abc", "user_id": "u"},
    )
    assert response.status_code == 400


def test_chat_rejects_an_empty_message(client):
    response = client.post(
        "/chat",
        json={"message": "", "session_id": "s", "seller_id": "7", "user_id": "u"},
    )
    assert response.status_code == 422


def test_whatsapp_status(client):
    body = client.get("/whatsapp/status").json()
    assert body["status"] == "active"
    assert "session_store" in body


def test_webhook_verification_requires_the_right_token(client):
    response = client.get(
        "/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "123"},
    )
    assert response.status_code == 403


def test_webhook_ignores_a_status_update_payload(client):
    """Delivery receipts must not be mistaken for customer messages."""
    response = client.post(
        "/whatsapp/webhook",
        json={"entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]},
    )
    assert response.status_code == 200
    assert response.json()["status"] in ("received", "error")


def test_webhook_drops_a_duplicate_delivery(client):
    """The duplicate-order bug, end to end."""
    from routes import whatsapp_routes

    whatsapp_routes._seen_messages.clear()
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "metadata": {"phone_number_id": "111"},
                            "messages": [
                                {
                                    "from": "94771234567",
                                    "id": "wamid.DUPLICATE",
                                    "timestamp": "1",
                                    "type": "text",
                                    "text": {"body": "I want to order 2 boxes"},
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    }

    first = client.post("/whatsapp/webhook", json=payload).json()
    second = client.post("/whatsapp/webhook", json=payload).json()
    assert second["status"] == "duplicate_ignored"
    assert first["status"] != "duplicate_ignored"


def test_webhook_rejects_a_bad_signature_when_enabled(client, override_settings):
    override_settings(verify_webhook_signature=True, whatsapp_app_secret="shhh")
    response = client.post(
        "/whatsapp/webhook",
        json={"entry": []},
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert response.status_code == 403


def test_handoff_endpoints(client):
    from services import handoff

    assert client.get("/whatsapp/handoffs").json()["count"] == 0

    handoff.request("7", "94771234567", "wants a refund")
    body = client.get("/whatsapp/handoffs").json()
    assert body["count"] == 1
    assert body["handoffs"][0]["reason"] == "wants a refund"

    assert client.delete("/whatsapp/handoffs/94771234567?seller_id=7").status_code == 200
    assert client.delete("/whatsapp/handoffs/94771234567?seller_id=7").status_code == 404


def test_chat_is_silent_while_a_handover_is_open(client):
    from services import handoff

    handoff.request("7", "silent-user", "complaint")
    try:
        response = client.post(
            "/chat",
            json={
                "message": "hello?",
                "session_id": "s",
                "seller_id": "7",
                "user_id": "silent-user",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["handoff"] is True
        assert body["response"] is None
    finally:
        handoff.resolve("7", "silent-user")


def test_send_message_requires_a_phone_number_id(client):
    """Was omitted entirely, so this endpoint raised TypeError on every call."""
    response = client.post("/whatsapp/send-message", json={"to": "947", "message": "hi"})
    assert response.status_code == 400
    assert "phone_number_id" in response.json()["detail"]
