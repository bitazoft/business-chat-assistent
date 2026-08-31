"""
One place where environment variables are read.

Everything here is read once, at import time, into a frozen object. Hot paths
(every chat message, every webhook) used to call os.getenv repeatedly; now they
read an attribute. It also means a typo in a value fails at startup instead of
halfway through a customer conversation.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}")


def _csv(name: str, default: str) -> List[str]:
    return [p.strip() for p in os.getenv(name, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    # ---- environment ----
    environment: str = os.getenv("ENVIRONMENT", "development").strip().lower()
    debug_mode: bool = _bool("DEBUG_MODE", False)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _int("PORT", 8001)

    # ---- LLM ----
    ai_provider: str = os.getenv("AI_PROVIDER", "GPT").strip().upper()
    api_key: Optional[str] = os.getenv("API_KEY")
    api_base: str = os.getenv("API_BASE", "https://api.deepseek.com/v1")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.1)
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 512)
    llm_timeout: int = _int("LLM_TIMEOUT", 60)
    llm_max_retries: int = _int("LLM_MAX_RETRIES", 2)

    # ---- agent ----
    agent_max_iterations: int = _int("AGENT_MAX_ITERATIONS", 8)
    max_chat_history: int = _int("MAX_CHAT_HISTORY", 20)
    rag_enabled: bool = _bool("RAG_ENABLED", False)
    language_detection_enabled: bool = _bool("LANGUAGE_DETECTION_ENABLED", False)
    preload_models: bool = _bool("PRELOAD_MODELS", True)

    # ---- cost control ----
    # fixed    : always CHAT_MODEL (previous behaviour)
    # tiered   : cheap model for simple turns, stronger one when it matters
    # rotation : round-robin across MODEL_ROTATION, to spread spend and rate limits
    cost_strategy: str = os.getenv("COST_STRATEGY", "fixed").strip().lower()
    model_cheap: str = os.getenv("MODEL_CHEAP", "") or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    model_standard: str = os.getenv("MODEL_STANDARD", "") or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    model_strong: str = os.getenv("MODEL_STRONG", "") or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    model_rotation: List[str] = field(default_factory=lambda: _csv("MODEL_ROTATION", ""))
    # Per seller, per day. 0 disables the cap.
    daily_budget_usd: float = _float("DAILY_BUDGET_USD", 0.0)
    # Answer greetings and thanks from a template instead of paying for a turn.
    shortcut_replies_enabled: bool = _bool("SHORTCUT_REPLIES_ENABLED", True)
    # Reuse the reply for an identical opening question (no tools, no history).
    response_cache_enabled: bool = _bool("RESPONSE_CACHE_ENABLED", True)
    response_cache_ttl: int = _int("RESPONSE_CACHE_TTL", 900)
    track_token_usage: bool = _bool("TRACK_TOKEN_USAGE", True)

    # ---- vision ----
    vision_model: str = os.getenv("VISION_MODEL", "") or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    vision_timeout: int = _int("VISION_TIMEOUT", 60)
    payment_amount_tolerance: float = _float("PAYMENT_AMOUNT_TOLERANCE", 1.0)

    # ---- database ----
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/business_db"
    )
    db_pool_size: int = _int("DB_POOL_SIZE", 10)
    db_max_overflow: int = _int("DB_MAX_OVERFLOW", 20)
    db_pool_timeout: int = _int("DB_POOL_TIMEOUT", 30)
    db_pool_recycle: int = _int("DB_POOL_RECYCLE", 1800)
    db_echo: bool = _bool("DB_ECHO", False)
    # Cap how long a new connection may take. Without this, an unreachable
    # database holds a worker thread for the OS TCP timeout (~30s+) and the
    # customer just waits.
    db_connect_timeout: int = _int("DB_CONNECT_TIMEOUT", 10)
    persist_conversations: bool = _bool("PERSIST_CONVERSATIONS", True)

    # ---- server / API ----
    cors_origins: List[str] = field(default_factory=lambda: _csv("CORS_ORIGINS", "*"))
    worker_threads: int = _int("WORKER_THREADS", 16)

    # ---- sessions ----
    session_ttl_seconds: int = _int("SESSION_TTL_SECONDS", 3600)
    session_max_count: int = _int("SESSION_MAX_COUNT", 5000)
    session_sweep_interval: int = _int("SESSION_SWEEP_INTERVAL", 300)

    # ---- WhatsApp ----
    whatsapp_app_secret: Optional[str] = os.getenv("WHATSAPP_APP_SECRET") or None
    verify_webhook_signature: bool = _bool("VERIFY_WEBHOOK_SIGNATURE", False)
    whatsapp_max_message_chars: int = _int("WHATSAPP_MAX_MESSAGE_CHARS", 4000)
    whatsapp_send_retries: int = _int("WHATSAPP_SEND_RETRIES", 3)
    dedupe_ttl_seconds: int = _int("DEDUPE_TTL_SECONDS", 900)
    typing_indicator: bool = _bool("TYPING_INDICATOR", True)

    # ---- rate limiting (per customer phone number / user id) ----
    rate_limit_enabled: bool = _bool("RATE_LIMIT_ENABLED", True)
    rate_limit_messages: int = _int("RATE_LIMIT_MESSAGES", 20)
    rate_limit_window_seconds: int = _int("RATE_LIMIT_WINDOW_SECONDS", 60)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()
