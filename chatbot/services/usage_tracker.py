"""
Token usage and cost, per session and per seller.

Nothing measured token use before, so there was no way to tell what a
conversation cost or which seller was driving the bill. A LangChain callback
collects usage from every LLM call in a turn; the totals are aggregated in
memory for live reads and written to token_usage for reporting.

Why a callback rather than reading the agent's return value: an AgentExecutor
turn makes several model calls (one per tool round trip) and only surfaces the
last message, so usage has to be collected as it happens.
"""
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from config.pricing import estimate_cost
from utils.cache import get_cache
from utils.logger import get_logger
from utils.metrics import metrics

logger = get_logger(__name__)


def normalise_model_name(name: Optional[str]) -> str:
    """Clean up the model name a provider reports.

    Observed with langchain-openai pointed at OpenRouter: response_metadata
    carries the name doubled -

        "openai/gpt-4o-miniopenai/gpt-4o-mini"

    which would show up as its own row in any per-model cost report. Only an
    exact doubling is collapsed, so a legitimately repetitive name is left alone.
    """
    if not name:
        return ""
    name = str(name).strip()
    half, remainder = divmod(len(name), 2)
    if remainder == 0 and half > 0 and name[:half] == name[half:]:
        return name[:half]
    return name


@dataclass
class TurnUsage:
    """Everything one conversation turn spent, across all its model calls."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    models_used: List[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.llm_calls += 1
        self.cost_usd += estimate_cost(model, prompt_tokens, completion_tokens)
        if model:
            self.model = model
            if model not in self.models_used:
                self.models_used.append(model)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "models_used": list(self.models_used),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "cost_usd": round(self.cost_usd, 6),
        }


class TokenUsageCallback(BaseCallbackHandler):
    """Collects token counts for a single turn.

    Create one per turn and pass it in the invoke config's callbacks. Different
    providers report usage in different places, so several shapes are checked.
    """

    def __init__(self, model_hint: str = ""):
        self.usage = TurnUsage()
        self._model_hint = normalise_model_name(model_hint)

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001 - LangChain signature
        try:
            model, prompt_tokens, completion_tokens = self._extract(response)
            if prompt_tokens or completion_tokens:
                self.usage.add(model or self._model_hint, prompt_tokens, completion_tokens)
        except Exception as e:
            # Usage accounting must never break a customer's reply.
            logger.debug("Could not read token usage: %s", e)

    def _extract(self, response) -> tuple:  # noqa: ANN001
        llm_output = getattr(response, "llm_output", None) or {}
        model = normalise_model_name(
            llm_output.get("model_name") or llm_output.get("model")
        )

        # Shape 1: llm_output["token_usage"] - OpenAI-compatible providers.
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
        completion_tokens = int(token_usage.get("completion_tokens") or 0)
        if prompt_tokens or completion_tokens:
            return model, prompt_tokens, completion_tokens

        # Shape 2: usage_metadata on the generated message - LangChain's
        # normalised form, and the only one present when streaming.
        for generation_list in getattr(response, "generations", None) or []:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                usage_metadata = getattr(message, "usage_metadata", None) or {}
                if usage_metadata:
                    metadata = getattr(message, "response_metadata", None) or {}
                    return (
                        model
                        or normalise_model_name(
                            metadata.get("model_name") or metadata.get("model")
                        ),
                        int(usage_metadata.get("input_tokens") or 0),
                        int(usage_metadata.get("output_tokens") or 0),
                    )
                metadata = getattr(message, "response_metadata", None) or {}
                nested = metadata.get("token_usage") or metadata.get("usage") or {}
                if nested:
                    return (
                        model or normalise_model_name(metadata.get("model_name")),
                        int(nested.get("prompt_tokens") or nested.get("input_tokens") or 0),
                        int(nested.get("completion_tokens") or nested.get("output_tokens") or 0),
                    )

        return model, 0, 0


@dataclass
class Totals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    turns: int = 0
    cost_usd: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage: TurnUsage) -> None:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.llm_calls += usage.llm_calls
        self.cost_usd += usage.cost_usd
        self.turns += 1
        self.last_seen = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "turns": self.turns,
            "cost_usd": round(self.cost_usd, 6),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


class UsageTracker:
    """Live per-session and per-seller totals, plus persistence.

    Session totals expire with the session; seller totals are kept for the life
    of the process and are what the daily budget check reads. Anything needed
    beyond a restart comes from the token_usage table.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = get_cache("usage_sessions", maxsize=10000, ttl=24 * 3600)
        self._sellers: Dict[str, Totals] = {}
        self._daily: Dict[str, Totals] = {}  # keyed "seller|YYYY-MM-DD"

    @staticmethod
    def _session_key(seller_id: str, user_id: str) -> str:
        return f"{seller_id}:{user_id}"

    @staticmethod
    def _day_key(seller_id: str, when: Optional[datetime] = None) -> str:
        day = (when or datetime.now()).strftime("%Y-%m-%d")
        return f"{seller_id}|{day}"

    def record(
        self,
        seller_id: str,
        user_id: str,
        usage: TurnUsage,
        persist: bool = True,
    ) -> None:
        """Fold one turn's usage into the live totals and queue the DB write."""
        if usage.total_tokens == 0:
            return

        seller_id, user_id = str(seller_id), str(user_id)

        session_key = self._session_key(seller_id, user_id)
        session_totals = self._sessions.get(session_key) or Totals()
        session_totals.add(usage)
        self._sessions.set(session_key, session_totals)

        with self._lock:
            self._sellers.setdefault(seller_id, Totals()).add(usage)
            self._daily.setdefault(self._day_key(seller_id), Totals()).add(usage)

        metrics.incr("llm.tokens.prompt", usage.prompt_tokens)
        metrics.incr("llm.tokens.completion", usage.completion_tokens)
        metrics.incr("llm.calls", usage.llm_calls)

        logger.info(
            "[Usage] seller=%s customer=%s model=%s tokens=%d/%d calls=%d cost=$%.6f",
            seller_id,
            user_id,
            usage.model or "unknown",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.llm_calls,
            usage.cost_usd,
        )

        if persist:
            from utils import background

            background.submit(
                self._persist, seller_id, user_id, usage, task_name="persist_usage"
            )

    @staticmethod
    def _persist(seller_id: str, user_id: str, usage: TurnUsage) -> None:
        from db.database import session_scope
        from models.schemas import TokenUsage

        with session_scope() as db:
            db.add(
                TokenUsage(
                    seller_id=seller_id,
                    customer_id=user_id,
                    model=usage.model or "unknown",
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    llm_calls=usage.llm_calls,
                    cost_usd=usage.cost_usd,
                )
            )

    # -- reads --------------------------------------------------------------
    def session_usage(self, seller_id: str, user_id: str) -> Dict[str, Any]:
        totals = self._sessions.get(self._session_key(str(seller_id), str(user_id)))
        return totals.to_dict() if totals else Totals().to_dict()

    def seller_usage(self, seller_id: str) -> Dict[str, Any]:
        with self._lock:
            totals = self._sellers.get(str(seller_id))
        return totals.to_dict() if totals else Totals().to_dict()

    def spend_today(self, seller_id: str) -> float:
        """USD spent by this seller today, for the budget check."""
        with self._lock:
            totals = self._daily.get(self._day_key(str(seller_id)))
        return totals.cost_usd if totals else 0.0

    def all_sellers(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {seller: totals.to_dict() for seller, totals in self._sellers.items()}

    def top_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = [
            dict(totals.to_dict(), session=key) for key, totals in self._sessions.items()
        ]
        rows.sort(key=lambda r: r["cost_usd"], reverse=True)
        return rows[:limit]


usage_tracker = UsageTracker()
