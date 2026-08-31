"""
Token usage is read from whatever shape the provider returned, so each shape
gets a test. Getting this wrong means silently reporting zero cost.
"""
from types import SimpleNamespace

from services.usage_tracker import TokenUsageCallback, TurnUsage, UsageTracker


def _response_with_llm_output(prompt=100, completion=50, model="gpt-4o-mini"):
    return SimpleNamespace(
        llm_output={
            "model_name": model,
            "token_usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
        generations=[],
    )


def _response_with_usage_metadata(prompt=200, completion=80, model="gpt-4o-mini"):
    message = SimpleNamespace(
        usage_metadata={"input_tokens": prompt, "output_tokens": completion},
        response_metadata={"model_name": model},
    )
    return SimpleNamespace(llm_output={}, generations=[[SimpleNamespace(message=message)]])


def test_reads_openai_style_token_usage():
    cb = TokenUsageCallback()
    cb.on_llm_end(_response_with_llm_output())
    assert cb.usage.prompt_tokens == 100
    assert cb.usage.completion_tokens == 50
    assert cb.usage.total_tokens == 150
    assert cb.usage.cost_usd > 0


def test_reads_langchain_usage_metadata():
    cb = TokenUsageCallback()
    cb.on_llm_end(_response_with_usage_metadata())
    assert cb.usage.prompt_tokens == 200
    assert cb.usage.completion_tokens == 80


def test_accumulates_across_tool_round_trips():
    """A turn makes one LLM call per tool round trip; all of them must count."""
    cb = TokenUsageCallback()
    cb.on_llm_end(_response_with_llm_output(100, 20))
    cb.on_llm_end(_response_with_llm_output(300, 40))
    cb.on_llm_end(_response_with_llm_output(500, 60))
    assert cb.usage.llm_calls == 3
    assert cb.usage.prompt_tokens == 900
    assert cb.usage.completion_tokens == 120


def test_malformed_response_does_not_raise():
    """Usage accounting must never break a customer's reply."""
    cb = TokenUsageCallback()
    cb.on_llm_end(SimpleNamespace())          # no attributes at all
    cb.on_llm_end(None)
    assert cb.usage.total_tokens == 0


def test_model_hint_used_when_provider_omits_the_name():
    cb = TokenUsageCallback(model_hint="openai/gpt-4o-mini")
    cb.on_llm_end(
        SimpleNamespace(
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            generations=[],
        )
    )
    assert cb.usage.model == "openai/gpt-4o-mini"
    assert cb.usage.cost_usd > 0


def test_tracks_multiple_models_in_one_turn():
    usage = TurnUsage()
    usage.add("gpt-4o-mini", 100, 10)
    usage.add("gpt-4o", 100, 10)
    assert usage.models_used == ["gpt-4o-mini", "gpt-4o"]


def test_seller_and_session_totals_accumulate():
    tracker = UsageTracker()
    for _ in range(3):
        usage = TurnUsage()
        usage.add("gpt-4o-mini", 1000, 200)
        tracker.record("7", "94771234567", usage, persist=False)

    session = tracker.session_usage("7", "94771234567")
    assert session["turns"] == 3
    assert session["prompt_tokens"] == 3000
    assert session["cost_usd"] > 0

    seller = tracker.seller_usage("7")
    assert seller["total_tokens"] == 3 * 1200


def test_sessions_are_isolated_per_customer():
    tracker = UsageTracker()
    a, b = TurnUsage(), TurnUsage()
    a.add("gpt-4o-mini", 1000, 100)
    b.add("gpt-4o-mini", 5000, 500)
    tracker.record("7", "customer-a", a, persist=False)
    tracker.record("7", "customer-b", b, persist=False)

    assert tracker.session_usage("7", "customer-a")["total_tokens"] == 1100
    assert tracker.session_usage("7", "customer-b")["total_tokens"] == 5500
    # ...but both roll up to the seller, which is who gets billed.
    assert tracker.seller_usage("7")["total_tokens"] == 6600


def test_zero_usage_is_not_recorded():
    tracker = UsageTracker()
    tracker.record("7", "c", TurnUsage(), persist=False)
    assert tracker.session_usage("7", "c")["turns"] == 0


def test_spend_today_feeds_the_budget_check():
    tracker = UsageTracker()
    usage = TurnUsage()
    usage.add("gpt-4o", 1_000_000, 1_000_000)
    tracker.record("9", "c", usage, persist=False)
    assert tracker.spend_today("9") > 0
    assert tracker.spend_today("other-seller") == 0.0


# ---- model name normalisation -----------------------------------------
def test_doubled_model_name_is_collapsed():
    """Observed for real: langchain-openai against OpenRouter reports
    response_metadata["model_name"] as the name twice over, which would show up
    as its own row in every per-model cost report."""
    from services.usage_tracker import normalise_model_name

    assert normalise_model_name("openai/gpt-4o-miniopenai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert normalise_model_name("gpt-4o-minigpt-4o-mini") == "gpt-4o-mini"


def test_normal_names_are_untouched():
    from services.usage_tracker import normalise_model_name

    for name in ("gpt-4o-mini", "openai/gpt-4o-mini", "deepseek-chat", "a"):
        assert normalise_model_name(name) == name


def test_normalisation_handles_empty():
    from services.usage_tracker import normalise_model_name

    assert normalise_model_name(None) == ""
    assert normalise_model_name("") == ""
    assert normalise_model_name("   ") == ""


def test_callback_reports_the_clean_name():
    doubled = "openai/gpt-4o-miniopenai/gpt-4o-mini"
    cb = TokenUsageCallback()
    cb.on_llm_end(
        SimpleNamespace(
            llm_output={
                "model_name": doubled,
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
            generations=[],
        )
    )
    assert cb.usage.model == "openai/gpt-4o-mini"
    assert cb.usage.models_used == ["openai/gpt-4o-mini"]


def test_doubled_name_from_response_metadata_is_cleaned():
    doubled = "openai/gpt-4o-miniopenai/gpt-4o-mini"
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 50, "output_tokens": 10},
        response_metadata={"model_name": doubled},
    )
    cb = TokenUsageCallback()
    cb.on_llm_end(
        SimpleNamespace(llm_output=None, generations=[[SimpleNamespace(message=message)]])
    )
    assert cb.usage.model == "openai/gpt-4o-mini"
