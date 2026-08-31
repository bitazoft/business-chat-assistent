import pytest

from config.settings import settings
from services import cost_optimizer


@pytest.fixture(autouse=True)
def _reset():
    cost_optimizer._response_cache.clear()
    cost_optimizer._rotation_cycle = None
    yield


# ---- shortcut replies -------------------------------------------------
@pytest.mark.parametrize("message", ["hi", "Hello!", "hey", "  Hi  ", "good morning", "ආයුබෝවන්"])
def test_greeting_is_answered_without_an_llm_call(message):
    assert cost_optimizer.try_shortcut_reply(message, 0, "7") is not None


def test_greeting_only_shortcuts_at_the_start_of_a_conversation():
    """Mid-conversation 'hi' usually means the customer is chasing a reply."""
    assert cost_optimizer.try_shortcut_reply("hi", 6, "7") is None


@pytest.mark.parametrize("message", ["thanks", "Thank you!", "ස්තූතියි", "thx"])
def test_thanks_is_shortcut(message):
    assert cost_optimizer.try_shortcut_reply(message, 4, "7") is not None


@pytest.mark.parametrize(
    "message",
    [
        "hi, do you have Ceylon tea?",
        "hello I want to order 2 boxes",
        "thanks but where is my order",
        "hi hi hi what is the price",
    ],
)
def test_real_questions_never_shortcut(message):
    """A shortcut here would skip the agent and drop the customer's request."""
    assert cost_optimizer.try_shortcut_reply(message, 0, "7") is None


def test_shortcuts_can_be_disabled(override_settings):
    override_settings(shortcut_replies_enabled=False)
    assert cost_optimizer.try_shortcut_reply("hi", 0, "7") is None


# ---- response cache ---------------------------------------------------
def test_identical_opening_question_is_reused():
    cost_optimizer.remember_response("7", "do you deliver?", "Yes we do", 0, used_tools=False)
    assert cost_optimizer.cached_response("7", "do you deliver?", 0) == "Yes we do"


def test_cache_key_ignores_case_and_extra_spacing():
    cost_optimizer.remember_response("7", "Do You  Deliver?", "Yes", 0, used_tools=False)
    assert cost_optimizer.cached_response("7", "do you deliver?", 0) == "Yes"


def test_never_caches_a_turn_that_used_tools():
    """Tool output depends on this customer's orders - it must not be replayed."""
    cost_optimizer.remember_response(
        "7", "where is my order", "Order 5 is shipped", 0, used_tools=True
    )
    assert cost_optimizer.cached_response("7", "where is my order", 0) is None


def test_never_caches_mid_conversation():
    cost_optimizer.remember_response("7", "yes", "Okay", 4, used_tools=False)
    assert cost_optimizer.cached_response("7", "yes", 4) is None


def test_cache_is_scoped_per_seller():
    cost_optimizer.remember_response("7", "do you deliver?", "Yes", 0, used_tools=False)
    assert cost_optimizer.cached_response("8", "do you deliver?", 0) is None


def test_cache_only_serves_the_first_turn():
    cost_optimizer.remember_response("7", "hours?", "9-5", 0, used_tools=False)
    assert cost_optimizer.cached_response("7", "hours?", 3) is None


# ---- model routing ----------------------------------------------------
def test_fixed_strategy_always_returns_chat_model(override_settings):
    override_settings(cost_strategy="fixed")
    assert cost_optimizer.choose_model("anything", "place_order", 20, "7") == settings.chat_model


def test_unknown_strategy_falls_back_to_fixed(override_settings):
    override_settings(cost_strategy="nonsense")
    assert cost_optimizer.choose_model("hi", "", 0, "7") == settings.chat_model


def test_tiered_uses_cheap_model_for_a_simple_turn(override_settings):
    override_settings(cost_strategy="tiered", model_cheap="cheap-model", model_strong="strong-model")
    assert cost_optimizer.choose_model("what are your hours?", "general_inquiry", 0, "7") == "cheap-model"


@pytest.mark.parametrize(
    "message,intent",
    [
        ("I want to order 3 boxes", "place_order"),
        ("[Image received: /tmp/a.jpg] type=receipt", "general_inquiry"),
        ("I need a refund", "general_inquiry"),
        ("this is the wrong item", "general_inquiry"),
    ],
)
def test_tiered_escalates_where_mistakes_cost_money(override_settings, message, intent):
    override_settings(cost_strategy="tiered", model_cheap="cheap-model", model_strong="strong-model")
    assert cost_optimizer.choose_model(message, intent, 0, "7") == "strong-model"


def test_tiered_escalates_on_a_long_conversation(override_settings):
    override_settings(cost_strategy="tiered", model_cheap="cheap-model", model_strong="strong-model")
    assert cost_optimizer.choose_model("ok and then?", "general_inquiry", 14, "7") == "strong-model"


def test_rotation_cycles_through_the_configured_list(override_settings):
    override_settings(cost_strategy="rotation", model_rotation=["m1", "m2", "m3"])
    picks = [cost_optimizer.choose_model("hi", "", 0, "7") for _ in range(6)]
    assert picks == ["m1", "m2", "m3", "m1", "m2", "m3"]


def test_rotation_with_no_list_falls_back_to_chat_model(override_settings):
    override_settings(cost_strategy="rotation", model_rotation=[])
    assert cost_optimizer.choose_model("hi", "", 0, "7") == settings.chat_model


# ---- budget cap -------------------------------------------------------
def test_no_budget_means_never_over(override_settings):
    override_settings(daily_budget_usd=0.0)
    assert cost_optimizer.over_budget("7") == (False, 0.0)


def test_over_budget_downgrades_to_the_cheapest_model(override_settings):
    from services.usage_tracker import TurnUsage, usage_tracker

    # Every model field is pinned: cheapest_model() considers all of them, so
    # leaving one at its ambient .env value made this test order-dependent.
    override_settings(
        cost_strategy="tiered",
        daily_budget_usd=0.001,
        model_cheap="gpt-4o-mini",
        model_standard="gpt-4o",
        model_strong="gpt-4o",
        model_rotation=[],
        chat_model="gpt-4o",
    )

    usage = TurnUsage()
    usage.add("gpt-4o", 1_000_000, 100_000)   # comfortably over $0.001
    usage_tracker.record("budget-test-seller", "c", usage, persist=False)

    # Would normally escalate to the strong model; the cap overrides that.
    chosen = cost_optimizer.choose_model("I need a refund", "", 0, "budget-test-seller")
    assert chosen == cost_optimizer.cheapest_model()
    assert chosen == "gpt-4o-mini"


def test_cheapest_model_picks_the_lowest_priced_configured_model(override_settings):
    override_settings(
        model_cheap="gpt-4.1-nano",
        model_standard="gpt-4o-mini",
        model_strong="gpt-4o",
        model_rotation=[],
        chat_model="gpt-4o",
    )
    assert cost_optimizer.cheapest_model() == "gpt-4.1-nano"


def test_status_reports_configuration():
    status = cost_optimizer.status()
    assert "strategy" in status
    assert "models" in status
    assert "response_cache" in status
