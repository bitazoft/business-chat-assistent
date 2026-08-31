from config.pricing import estimate_cost, get_price


def test_exact_model_name():
    price = get_price("gpt-4o-mini")
    assert price.input > 0 and price.output > 0


def test_provider_prefix_is_stripped():
    """OpenRouter-style names must resolve to the same price."""
    assert get_price("openai/gpt-4o-mini") == get_price("gpt-4o-mini")


def test_dated_release_falls_back_to_base_model():
    assert get_price("gpt-4o-mini-2024-07-18") == get_price("gpt-4o-mini")


def test_case_insensitive():
    assert get_price("GPT-4O-MINI") == get_price("gpt-4o-mini")


def test_unknown_model_costs_zero_rather_than_guessing():
    price = get_price("some-model-nobody-configured")
    assert price.input == 0 and price.output == 0


def test_none_model():
    assert get_price(None).input == 0


def test_cost_math():
    price = get_price("gpt-4o-mini")   # 0.15 in / 0.60 out per 1M
    cost = price.cost(1_000_000, 1_000_000)
    assert abs(cost - (price.input + price.output)) < 1e-9


def test_estimate_cost_scales_linearly():
    one = estimate_cost("gpt-4o-mini", 1000, 500)
    ten = estimate_cost("gpt-4o-mini", 10000, 5000)
    assert abs(ten - one * 10) < 1e-9


def test_longest_prefix_wins():
    """'gpt-4.1-nano' must not be priced as 'gpt-4.1'."""
    assert get_price("gpt-4.1-nano") != get_price("gpt-4.1")
