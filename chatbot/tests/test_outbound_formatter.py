"""
The outbound formatter is what makes the seller's templates the final word on
what gets sent. It must never lose the reply.
"""
from services import outbound_formatter
from templates.template_store import TemplateKey


def test_plain_answer_passes_through_the_default_wrapper():
    out = outbound_formatter.format_reply("Yes, we deliver.", [], seller_id="7")
    assert out == "Yes, we deliver."


def test_product_data_is_rendered_from_the_template():
    tool_results = [
        {
            "tool_name": "get_product_info",
            "result": "whatever the tool formatted",
            "data": {
                "product_id": 3,
                "product": "Ceylon Tea 500g",
                "description": "Loose leaf",
                "price": 1250.0,
                "stock": 42,
            },
        }
    ]
    out = outbound_formatter.format_reply("The tea costs 1250.", tool_results, seller_id="7")
    assert "Ceylon Tea 500g" in out
    assert "1250.00" in out
    assert "42" in out


def test_payment_verified_uses_the_confirmation_template():
    tool_results = [
        {
            "tool_name": "verify_and_save_payment_proof",
            "result": "Payment verified.",
            "data": {"order_id": 12, "verification": "verified", "amount": 2140.0},
        }
    ]
    out = outbound_formatter.format_reply("ok", tool_results, seller_id="7")
    assert "12" in out
    assert "2140.00" in out


def test_payment_mismatch_uses_the_mismatch_template():
    tool_results = [
        {
            "tool_name": "verify_and_save_payment_proof",
            "result": "mismatch",
            "data": {
                "order_id": 12,
                "verification": "amount_mismatch",
                "amount": 1000.0,
                "total_amount": 2140.0,
            },
        }
    ]
    out = outbound_formatter.format_reply("ok", tool_results, seller_id="7")
    assert "1000.00" in out and "2140.00" in out


def test_unreadable_receipt_uses_the_review_template():
    tool_results = [
        {
            "tool_name": "verify_and_save_payment_proof",
            "result": "saved",
            "data": {"order_id": 9, "verification": "unreadable", "flag_reason": "blurry"},
        }
    ]
    out = outbound_formatter.format_reply("ok", tool_results, seller_id="7")
    assert "9" in out


def test_payment_outranks_an_earlier_product_lookup():
    """A turn can touch several tools; the payment result is what matters."""
    tool_results = [
        {"tool_name": "get_product_info", "result": "x", "data": {"product": "Tea", "price": 10}},
        {
            "tool_name": "verify_and_save_payment_proof",
            "result": "y",
            "data": {"order_id": 5, "verification": "verified", "amount": 100.0},
        },
    ]
    out = outbound_formatter.format_reply("ok", tool_results, seller_id="7")
    assert "Tea" not in out


def test_escalation_uses_the_handoff_template():
    out = outbound_formatter.format_reply(
        "I'll pass this on.", [], seller_id="7", escalated=True
    )
    assert "team" in out.lower()


def test_product_list_is_rendered():
    tool_results = [
        {
            "tool_name": "get_all_products",
            "result": "x",
            "data": [
                {"name": "Tea", "price": 1250.0, "stock": 42},
                {"name": "Spice Box", "price": 890.0, "stock": 12},
            ],
        }
    ]
    out = outbound_formatter.format_reply("here they are", tool_results, seller_id="7")
    assert "Tea" in out and "Spice Box" in out


def test_non_dict_tool_data_is_ignored_not_fatal():
    tool_results = [{"tool_name": "get_product_info", "result": "x", "data": "not structured"}]
    out = outbound_formatter.format_reply("fallback answer", tool_results, seller_id="7")
    assert out == "fallback answer"


def test_reply_survives_a_wrapper_missing_its_message_placeholder(monkeypatch):
    """A seller who blanks {message} must not silence the bot."""
    from templates.template_store import template_store

    monkeypatch.setattr(
        template_store,
        "get_body",
        lambda key, seller_id=None: "Thanks for contacting us!"
        if key == TemplateKey.OUTBOUND_WRAPPER
        else "",
    )
    out = outbound_formatter.format_reply("The real answer", [], seller_id="7")
    assert out == "The real answer"


def test_wrapper_can_add_a_signature():
    from templates.template_store import template_store

    original = template_store.get_body

    def patched(key, seller_id=None):
        if key == TemplateKey.OUTBOUND_WRAPPER:
            return "{message}\n\n— {shop_name}"
        return original(key, seller_id)

    template_store.get_body = patched
    try:
        out = outbound_formatter.format_reply("Hello there", [], seller_id="7")
        assert out.startswith("Hello there")
        assert "—" in out
    finally:
        template_store.get_body = original


def test_money_formatting_tolerates_bad_input():
    assert outbound_formatter._money(None) == ""
    assert outbound_formatter._money("abc") == "abc"
    assert outbound_formatter._money(5) == "5.00"


def test_status_emoji_falls_back_for_unknown_status():
    assert outbound_formatter._status_emoji("shipped") == "🚚"
    assert outbound_formatter._status_emoji("banana") == "📋"
    assert outbound_formatter._status_emoji(None) == "📋"


def test_order_confirmation_renders_ids_when_the_lookup_fails():
    """With the database down the confirmation must still go out."""
    tool_results = [
        {
            "tool_name": "place_order",
            "result": "x",
            "data": {
                "order_id": 42,
                "total_amount": 2140.0,
                "items": [{"product_id": 3, "quantity": 2}],
            },
        }
    ]
    out = outbound_formatter.format_reply("ok", tool_results, seller_id="7")
    assert "42" in out
    assert "2140.00" in out


def test_order_items_text_falls_back_to_ids():
    text = outbound_formatter._order_items_text(999, [{"product_id": 3, "quantity": 2}])
    assert "3" in text and "2" in text


def test_order_items_text_tolerates_junk():
    assert outbound_formatter._order_items_text(None, None) == ""
    assert outbound_formatter._order_items_text("abc", ["not a dict"]) == ""
