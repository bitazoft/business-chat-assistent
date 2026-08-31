"""
What each model costs, so token counts can be turned into money.

Prices are USD per 1 million tokens and are editable without touching code:
point MODEL_PRICING_FILE at a JSON file to override or extend the defaults.

    {
      "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
      "my-provider/some-model": {"input": 1.0, "output": 3.0}
    }

Provider prices change; treat the defaults as a starting point and confirm them
against your provider's current price list. An unknown model bills at 0 and logs
a warning rather than guessing a number that would be wrong in the reports.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input / 1_000_000
            + completion_tokens * self.output / 1_000_000
        )


# Defaults cover the models this project is configured with out of the box.
# Keys are matched case-insensitively, and a provider prefix ("openai/...") is
# stripped before falling back, so OpenRouter names resolve too.
DEFAULT_PRICING: Dict[str, Price] = {
    # OpenAI
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4.1-mini": Price(0.40, 1.60),
    "gpt-4.1": Price(2.00, 8.00),
    "gpt-4.1-nano": Price(0.10, 0.40),
    "gpt-3.5-turbo": Price(0.50, 1.50),
    # DeepSeek
    "deepseek-chat": Price(0.27, 1.10),
    "deepseek-reasoner": Price(0.55, 2.19),
    # Anthropic (via an OpenAI-compatible gateway)
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-sonnet-4-5": Price(3.00, 15.00),
    # Open models commonly used as the cheap tier on OpenRouter
    "llama-3.1-8b-instruct": Price(0.02, 0.03),
    "gemini-2.0-flash": Price(0.10, 0.40),
    "gemini-2.5-flash": Price(0.30, 2.50),
}

_ZERO = Price(0.0, 0.0)
_warned = set()


def _load_overrides() -> Dict[str, Price]:
    path = os.getenv("MODEL_PRICING_FILE")
    if not path:
        return {}
    if not os.path.exists(path):
        logger.warning("MODEL_PRICING_FILE=%s does not exist - using default prices", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        overrides = {
            str(name).lower(): Price(float(v["input"]), float(v["output"]))
            for name, v in raw.items()
        }
        logger.info("Loaded %d model price override(s) from %s", len(overrides), path)
        return overrides
    except Exception as e:
        logger.error("Could not read MODEL_PRICING_FILE=%s: %s", path, e)
        return {}


PRICING: Dict[str, Price] = {**{k.lower(): v for k, v in DEFAULT_PRICING.items()}, **_load_overrides()}


def get_price(model: Optional[str]) -> Price:
    """Price for a model name, tolerating provider prefixes and version suffixes."""
    if not model:
        return _ZERO

    name = str(model).strip().lower()

    # Exact match, then without the provider prefix ("openai/gpt-4o" -> "gpt-4o").
    for candidate in (name, name.split("/")[-1]):
        if candidate in PRICING:
            return PRICING[candidate]

    # Then longest known key that prefixes the name, so dated releases
    # ("gpt-4o-mini-2024-07-18") match their base model.
    bare = name.split("/")[-1]
    matches = [key for key in PRICING if bare.startswith(key)]
    if matches:
        return PRICING[max(matches, key=len)]

    if name not in _warned:
        _warned.add(name)
        logger.warning(
            "No price configured for model %r - its cost will be reported as 0. "
            "Add it via MODEL_PRICING_FILE.",
            model,
        )
    return _ZERO


def estimate_cost(model: Optional[str], prompt_tokens: int, completion_tokens: int) -> float:
    return get_price(model).cost(prompt_tokens or 0, completion_tokens or 0)


def known_models() -> Dict[str, Dict[str, float]]:
    return {name: {"input": p.input, "output": p.output} for name, p in sorted(PRICING.items())}
