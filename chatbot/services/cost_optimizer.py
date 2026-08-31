"""
Keeping the LLM bill down.

Four mechanisms, cheapest first:

1. Shortcut replies. "hi", "thanks", "ok" are a real share of WhatsApp traffic and
   don't need a model at all - they're answered from a template for zero tokens.

2. Response cache. The same opening question ("do you deliver?") from different
   customers produces the same answer. Cached only when the turn used no tools and
   had no history, so nothing customer-specific can leak between people.

3. Model routing. A greeting and a five-item order do not need the same model.
   'tiered' sends simple turns to a cheap model and escalates on the signals that
   actually predict difficulty; 'rotation' round-robins a list to spread spend and
   provider rate limits.

4. Daily budget. Past the per-seller cap everything drops to the cheapest model,
   so a runaway conversation degrades instead of billing without limit.

Default strategy is 'fixed' - identical behaviour to before. Opt in with
COST_STRATEGY.
"""
import itertools
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from config.pricing import get_price
from config.settings import settings
from templates.template_store import TemplateKey
from utils.cache import get_cache
from utils.logger import get_logger
from utils.metrics import metrics

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. Shortcut replies
# ---------------------------------------------------------------------------
# Deliberately narrow: only messages that are *nothing but* a greeting or thanks.
# Anything with a question or a product in it must reach the agent, so these are
# anchored and allow only trailing punctuation and emoji.
_GREETING = re.compile(
    r"^\s*(hi|hii+|hey+|hello+|helo|good\s*(morning|afternoon|evening)|"
    r"ayubowan|ayubovan|හෙලෝ|ආයුබෝවන්)"
    r"[\s!.,?😊👋🙏🇱🇰]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(
    r"^\s*(thanks?|thank\s*you|thank\s*u|ty|tnx|thx|bohoma\s*sthuthi|ස්තූතියි|බොහොම\s*ස්තූතියි)"
    r"[\s!.,?😊👍🙏❤️]*$",
    re.IGNORECASE,
)
_ACK = re.compile(r"^\s*(ok+|okay|k|sure|got\s*it|noted|fine|hari|හරි)[\s!.,?👍]*$", re.IGNORECASE)


def try_shortcut_reply(message: str, history_length: int, seller_id: str) -> Optional[str]:
    """A template answer for a message that needs no reasoning, else None."""
    if not settings.shortcut_replies_enabled or not message:
        return None

    from services import outbound_formatter

    # Only greet at the start. Mid-conversation "hi" usually means the customer
    # is chasing a reply, which the agent should see.
    if history_length == 0 and _GREETING.match(message):
        reply = outbound_formatter.render(TemplateKey.GREETING, seller_id)
        if reply:
            metrics.incr("cost.shortcut.greeting")
            return reply

    if _THANKS.match(message):
        metrics.incr("cost.shortcut.thanks")
        return "You're welcome! 🙏 Let me know if there's anything else."

    if _ACK.match(message):
        metrics.incr("cost.shortcut.ack")
        return "👍"

    return None


# ---------------------------------------------------------------------------
# 2. Response cache
# ---------------------------------------------------------------------------
_response_cache = get_cache(
    "llm_responses", maxsize=1000, ttl=settings.response_cache_ttl
)


def _response_key(seller_id: str, message: str) -> str:
    return f"{seller_id}|{' '.join(message.lower().split())}"


def cached_response(seller_id: str, message: str, history_length: int) -> Optional[str]:
    """A previous answer to this exact opening question, if we have one."""
    if not settings.response_cache_enabled or history_length != 0:
        return None
    hit = _response_cache.get(_response_key(seller_id, message))
    if hit is not None:
        metrics.incr("cost.response_cache.hit")
    return hit


def remember_response(
    seller_id: str, message: str, response: str, history_length: int, used_tools: bool
) -> None:
    """Cache an answer, but only when it cannot be customer-specific.

    used_tools is the important guard: any tool call means the answer depended on
    this customer's orders, profile or stock at that moment, and must never be
    replayed to someone else.
    """
    if not settings.response_cache_enabled:
        return
    if history_length != 0 or used_tools or not response:
        return
    _response_cache.set(_response_key(seller_id, message), response)


# ---------------------------------------------------------------------------
# 3 + 4. Model routing and the budget cap
# ---------------------------------------------------------------------------
# Signals that a turn is worth a better model. Order placement and payment are
# where a cheap model's mistakes cost real money.
_COMPLEX_INTENTS = frozenset({"place_order"})
_COMPLEX_MARKERS = re.compile(
    r"(\[Image received:|refund|complain|cancel|wrong|mistake|urgent|manager|"
    r"receipt|payment|transfer|invoice)",
    re.IGNORECASE,
)

_rotation_lock = threading.Lock()
_rotation_cycle = None


def _rotation_models() -> List[str]:
    return [m for m in settings.model_rotation if m] or [settings.chat_model]


def _next_rotation_model() -> str:
    global _rotation_cycle
    with _rotation_lock:
        if _rotation_cycle is None:
            _rotation_cycle = itertools.cycle(_rotation_models())
        return next(_rotation_cycle)


def cheapest_model() -> str:
    """The lowest-priced model among those configured.

    Sorted before min() so equal-priced candidates always resolve the same way.
    Iterating the set directly made the answer depend on string hashing, so the
    same configuration could pick a different model from one restart to the next.
    """
    candidates = sorted(
        {
            c
            for c in (
                settings.model_cheap,
                settings.model_standard,
                settings.model_strong,
                settings.chat_model,
                *_rotation_models(),
            )
            if c
        }
    )
    if not candidates:
        return settings.chat_model
    return min(candidates, key=lambda m: get_price(m).input + get_price(m).output)


def over_budget(seller_id: str) -> Tuple[bool, float]:
    """Whether this seller has passed today's cap, and what they've spent."""
    if settings.daily_budget_usd <= 0:
        return False, 0.0
    from services.usage_tracker import usage_tracker

    spent = usage_tracker.spend_today(seller_id)
    return spent >= settings.daily_budget_usd, spent


def choose_model(
    message: str,
    intent: str = "",
    history_length: int = 0,
    seller_id: str = "",
) -> str:
    """Which model to use for this turn."""
    strategy = settings.cost_strategy

    exceeded, spent = over_budget(seller_id)
    if exceeded:
        model = cheapest_model()
        metrics.incr("cost.budget_exceeded")
        logger.warning(
            "Seller %s has spent $%.4f today (cap $%.2f) - downgrading to %s",
            seller_id,
            spent,
            settings.daily_budget_usd,
            model,
        )
        return model

    if strategy == "rotation":
        model = _next_rotation_model()
        metrics.incr("cost.route.rotation")
        return model

    if strategy == "tiered":
        complex_turn = (
            intent in _COMPLEX_INTENTS
            or bool(_COMPLEX_MARKERS.search(message or ""))
            # A long conversation is usually one that has gone wrong or is
            # mid-order; both benefit from the better model.
            or history_length >= 12
        )
        model = settings.model_strong if complex_turn else settings.model_cheap
        metrics.incr(f"cost.route.{'strong' if complex_turn else 'cheap'}")
        logger.debug("Routed to %s (complex=%s, intent=%s)", model, complex_turn, intent)
        return model

    if strategy != "fixed":
        logger.warning(
            "Unknown COST_STRATEGY=%r - falling back to 'fixed'. Use fixed, tiered or rotation.",
            strategy,
        )
    return settings.chat_model


def status() -> Dict[str, Any]:
    """What's configured and how well it's working, for /usage/optimization."""
    return {
        "strategy": settings.cost_strategy,
        "models": {
            "fixed": settings.chat_model,
            "cheap": settings.model_cheap,
            "standard": settings.model_standard,
            "strong": settings.model_strong,
            "rotation": _rotation_models(),
            "cheapest_configured": cheapest_model(),
        },
        "daily_budget_usd": settings.daily_budget_usd,
        "shortcut_replies_enabled": settings.shortcut_replies_enabled,
        "response_cache": _response_cache.stats(),
    }
