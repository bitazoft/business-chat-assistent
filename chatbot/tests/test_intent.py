"""
Intent detection is rule-based so it costs no LLM call. The old version listed
'order' under three intents and checked order_tracking first, so anything
mentioning an order was classified as tracking.
"""
import pytest

from agent.agent import fast_intent_detection


@pytest.mark.parametrize(
    "message",
    [
        "track my order",
        "where is my order",
        "what is the status of order 12",
        "has it shipped yet",
    ],
)
def test_tracking(message):
    assert fast_intent_detection(message) == "order_tracking"


@pytest.mark.parametrize(
    "message",
    [
        "I want to order a laptop",
        "I want to buy 2 boxes of tea",
        "can I purchase this",
        "place an order please",
        "checkout now",
    ],
)
def test_ordering_is_not_misread_as_tracking(message):
    """The regression this fixes: 'I want to order X' was order_tracking."""
    assert fast_intent_detection(message) == "place_order"


@pytest.mark.parametrize(
    "message",
    ["what is the price of tea", "how much is it", "do you have this in stock", "show me products"],
)
def test_product_info(message):
    assert fast_intent_detection(message) == "product_info"


@pytest.mark.parametrize(
    "message", ["update my address", "change my email", "my profile details"]
)
def test_user_management(message):
    assert fast_intent_detection(message) == "user_management"


@pytest.mark.parametrize("message", ["hello", "are you open today", "", "asdfgh"])
def test_general_inquiry(message):
    assert fast_intent_detection(message) == "general_inquiry"


def test_bare_order_mention_still_maps_to_tracking():
    """The catch-all: 'my orders' with no other signal means tracking."""
    assert fast_intent_detection("my orders") == "order_tracking"


def test_none_is_safe():
    assert fast_intent_detection(None) == "general_inquiry"


def test_word_boundaries_prevent_substring_false_positives():
    """'reorder' should not be matched by a bare 'order' substring scan."""
    assert fast_intent_detection("borders and frames") == "general_inquiry"
