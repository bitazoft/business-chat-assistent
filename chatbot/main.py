import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from config.storage import LOCAL_STORAGE_DIR, S3_ENABLED

# Initialize custom logger
from utils.logger import GlobalLogger, get_logger

GlobalLogger()  # This sets up the logging configuration
logger = get_logger(__name__)

from routes.chat import router as chat_router  # noqa: E402
from routes.template_routes import router as template_router  # noqa: E402
from routes.usage_routes import router as usage_router  # noqa: E402
from routes.whatsapp_routes import router as whatsapp_router  # noqa: E402


def preload_models():
    """Warm up the vector store and LLM so the first real request isn't slow."""
    try:
        logger.info("🔄 Preloading models and vector store...")

        if settings.rag_enabled:
            from vector_store.vector_store import fast_vector_store

            fast_vector_store._lazy_load()
        else:
            # Loading the embedding model costs seconds and hundreds of MB of RAM.
            # With RAG off nothing will ever query it.
            logger.info("⏭️ RAG disabled - skipping vector store load")

        from agent.agent import llm

        llm.invoke("Hello")

        logger.info("✅ Models preloaded successfully")
    except Exception as e:
        logger.warning(
            "⚠️ Could not preload models: %s. First response may be slower.", e
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown.

    Replaces @app.on_event, which is deprecated, and adds an actual shutdown:
    the background executor is drained so queued log and usage writes complete
    instead of being killed mid-write with the daemon threads.
    """
    logger.info("🚀 Business Chat Assistant starting up...")

    from services.session_store import session_sweeper

    if settings.preload_models:
        # Runs on the startup thread on purpose: better to be slow to accept the
        # first request than to serve it before the model client is ready.
        preload_models()
    else:
        logger.info("⏭️ Skipping model preloading (PRELOAD_MODELS=false)")

    # Nothing expired idle sessions before, so they accumulated for the life of
    # the process - each holding a full agent object.
    session_sweeper.start()

    logger.info("🎉 Business Chat Assistant with WhatsApp Integration is ready!")
    logger.info("📱 WhatsApp webhook endpoint: /whatsapp (alias: /whatsapp/webhook)")
    logger.info("📊 Status: /whatsapp/status | Health: /health | Metrics: /metrics")
    logger.info("✏️ Message templates: /templates | 💰 Cost & usage: /usage/summary")

    try:
        yield
    finally:
        logger.info("👋 Business Chat Assistant is shutting down...")

        from utils import async_bridge, background, http

        session_sweeper.stop()
        background.shutdown(wait=True)
        async_bridge.shutdown()
        http.close_all()

        from db.database import dispose

        dispose()
        logger.info("👋 Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Business Chat Assistant with WhatsApp Integration",
    lifespan=lifespan,
)

# CORS. allow_origins=["*"] together with allow_credentials=True is rejected by
# browsers (a wildcard origin cannot be used with credentials), so credentials
# are only enabled when actual origins are configured via CORS_ORIGINS.
allow_all_origins = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
if allow_all_origins and settings.is_production:
    logger.warning(
        "⚠️ CORS is open to all origins in production. Set CORS_ORIGINS to your "
        "Admin Portal's URL(s)."
    )

# Product lists and order histories are repetitive text; compressing them cuts
# response size a lot for very little CPU.
app.add_middleware(GZipMiddleware, minimum_size=1000)

logger.info("✅ CORS middleware configured (origins=%s)", settings.cors_origins)

# Serve locally-stored uploads (payment proofs etc.) when STORAGE_BACKEND=local
if not S3_ENABLED:
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=LOCAL_STORAGE_DIR), name="uploads")
    logger.info("✅ Local storage backend active - serving '%s' at /uploads", LOCAL_STORAGE_DIR)

app.include_router(chat_router)
app.include_router(whatsapp_router)
app.include_router(template_router)
app.include_router(usage_router)
logger.info("✅ Routes registered: chat, whatsapp, templates, usage")


@app.get("/health")
async def health(deep: bool = False):
    """Liveness, and with ?deep=true a real readiness check.

    The old version returned {"status": "healthy"} unconditionally - it stayed
    green with the database unreachable, so a load balancer kept sending traffic
    to an instance that could not answer a single message.
    """
    from fastapi.concurrency import run_in_threadpool

    payload = {"status": "healthy", "environment": settings.environment}

    if not deep:
        return payload

    from db.database import check_connection
    from services.whatsapp_service import whatsapp_service

    db_status = await run_in_threadpool(check_connection)

    checks = {
        "database": db_status,
        "llm": {"ok": bool(settings.api_key), "model": settings.chat_model},
        "whatsapp": {
            "ok": whatsapp_service.is_configured(),
            "accounts": len(whatsapp_service.configs),
        },
        "storage": {"ok": True, "backend": "s3" if S3_ENABLED else "local"},
    }

    # The database is the only hard dependency: without it nothing works.
    payload["status"] = "healthy" if db_status.get("ok") else "unhealthy"
    payload["checks"] = checks
    return payload


@app.get("/metrics")
async def get_metrics():
    """Counters, latency percentiles, and pool/cache occupancy."""
    from db.database import pool_status
    from utils import background
    from utils.cache import all_cache_stats
    from utils.metrics import metrics
    from services.session_store import session_store

    snapshot = metrics.snapshot()
    snapshot["db_pool"] = pool_status()
    snapshot["caches"] = all_cache_stats()
    snapshot["sessions"] = session_store.stats()
    snapshot["background_queue_depth"] = background.queue_depth()

    if settings.rag_enabled:
        from vector_store.vector_store import fast_vector_store

        snapshot["vector_store"] = fast_vector_store.stats()

    return snapshot


# uvicorn main:app --reload
