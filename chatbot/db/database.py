"""
Database engine, session factory, and the session_scope helper.

The engine previously took no pool arguments at all, so it ran on SQLAlchemy's
defaults (5 connections, 10 overflow, no liveness check). With a 16-thread worker
pool that runs out under load, and without pool_pre_ping a connection dropped by
Postgres or a NAT timeout surfaces as a failed customer message rather than a
silent reconnect. Those knobs are now set and configurable.
"""
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

DATABASE_URL = settings.database_url

Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    # Recycle before Postgres/NAT idle timeouts can close a pooled connection.
    pool_recycle=settings.db_pool_recycle,
    # Cheap liveness check on checkout; turns "server closed the connection"
    # errors into a transparent reconnect.
    pool_pre_ping=True,
    echo=settings.db_echo,
    future=True,
    # A database that isn't answering should fail fast rather than pin the
    # worker thread for the OS-level TCP timeout.
    connect_args={"connect_timeout": settings.db_connect_timeout},
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

logger.info(
    "Database engine ready (pool_size=%d, max_overflow=%d, recycle=%ds)",
    settings.db_pool_size,
    settings.db_max_overflow,
    settings.db_pool_recycle,
)


@contextmanager
def session_scope(commit: bool = True) -> Iterator[Session]:
    """Transactional session that always closes and rolls back on error.

    Replaces the try/except/finally-db.close() block repeated in every
    repository function, where a missing rollback would leave a poisoned
    connection back in the pool.

    Pass commit=False for read-only work to skip the round-trip.
    """
    db = SessionLocal()
    try:
        yield db
        if commit:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def read_session() -> Iterator[Session]:
    """Read-only session - no commit on the way out."""
    with session_scope(commit=False) as db:
        yield db


def check_connection() -> dict:
    """Ping the database. Used by the /health endpoint."""
    import time

    start = time.perf_counter()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def pool_status() -> dict:
    """Pool occupancy, surfaced at /metrics so exhaustion is visible."""
    pool = engine.pool
    try:
        return {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
        }
    except AttributeError:
        # Some pool implementations (NullPool in tests) don't expose these.
        return {}


def dispose() -> None:
    """Close every pooled connection. Called on shutdown."""
    engine.dispose()
    logger.info("Database connection pool disposed")
