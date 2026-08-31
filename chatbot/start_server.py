#!/usr/bin/env python3
"""
Startup script for the chatbot application.

Applies process-level settings that must be in place before the heavy imports
happen, then hands over to uvicorn.
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize custom logger FIRST
from utils.logger import GlobalLogger  # noqa: E402

GlobalLogger()  # This sets up the logging configuration


def setup_performance_optimizations():
    """Apply process-level settings before the app imports anything heavy."""

    # Tokenizers forks per call otherwise, which is slower than useless here and
    # prints a warning on every request.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "cache/huggingface")

    # LangSmith tracing sends every LLM call and tool result to an external
    # service. It is off unless explicitly enabled, but LangChain reads several
    # spellings of the flag, so all of them are pinned unless the operator has
    # deliberately set one.
    for flag in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
        os.environ.setdefault(flag, "false")

    if os.getenv("ENVIRONMENT") == "production":
        # Only suppress uvicorn access logs in production, keep other logs visible
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    sys.dont_write_bytecode = False  # Enable bytecode caching

    print("✅ Performance optimizations applied")


def create_cache_directories():
    """Create necessary cache directories"""
    cache_dirs = [
        "cache",
        "cache/transformers",
        "cache/huggingface",
        "cache/vector_store",
        "cache/embeddings",
        "logs",
    ]

    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)

    print("✅ Cache directories created")


def main():
    """Main startup function"""
    print("🚀 Starting Chatbot Server")
    print("=" * 40)

    setup_performance_optimizations()
    create_cache_directories()

    # Imported after the env vars above are set.
    import uvicorn

    from config.settings import settings

    # Sessions, caches, rate limits and webhook deduplication are per-process,
    # so more than one worker means a customer's session and their duplicate
    # message can land on different processes. Keep this at 1 until that state
    # moves to Redis - the app is already concurrent within a process via its
    # worker threads.
    workers = int(os.getenv("WEB_CONCURRENCY", "1"))
    if workers > 1:
        print(
            f"⚠️  WEB_CONCURRENCY={workers}: sessions, rate limits and webhook\n"
            "    deduplication are per-process and will not be shared between\n"
            "    workers. Duplicate WhatsApp deliveries may be processed twice."
        )

    config = {
        "app": "main:app",
        "host": settings.host,
        "port": settings.port,
        "reload": not settings.is_production,
        "workers": workers if workers > 1 else None,
        "access_log": True,
        "log_level": "info",
        # Keep connections alive a little longer than the default 5s: the Admin
        # Portal polls, and reconnecting each time wastes a TLS handshake.
        "timeout_keep_alive": 30,
    }
    config = {k: v for k, v in config.items() if v is not None}

    # reload and workers are mutually exclusive in uvicorn.
    if config.get("reload") and config.get("workers"):
        config.pop("workers")

    print(f"🌐 Server starting on {config['host']}:{config['port']}")
    print(f"🤖 Model: {settings.chat_model}  |  cost strategy: {settings.cost_strategy}")
    print("📊 Health: /health?deep=true   Metrics: /metrics")
    print("✏️  Templates: /templates      Cost: /usage/summary")
    print("=" * 40)

    uvicorn.run(**config)


if __name__ == "__main__":
    main()
