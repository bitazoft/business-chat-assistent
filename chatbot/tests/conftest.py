"""
Shared test setup.

Env vars are set before any app module is imported, because config/settings.py
reads the environment once at import time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("API_KEY", "test-key-not-used")
os.environ.setdefault("PRELOAD_MODELS", "false")
os.environ.setdefault("RAG_ENABLED", "false")
os.environ.setdefault("LANGUAGE_DETECTION_ENABLED", "false")
os.environ.setdefault("PERSIST_CONVERSATIONS", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
# Point at a database that isn't there: the tests must not need one, and this
# proves the fallbacks work rather than quietly relying on a live server.
os.environ.setdefault("DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/nodb")
os.environ.setdefault("DB_CONNECT_TIMEOUT", "1")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_caches():
    """Stop cached state leaking between tests."""
    from utils.cache import _registry

    for cache in list(_registry.values()):
        cache.clear()
    yield


@pytest.fixture
def override_settings():
    """Temporarily change frozen Settings fields.

    Settings is a frozen dataclass on purpose - config shouldn't be mutable at
    runtime - so monkeypatch.setattr raises FrozenInstanceError. This goes
    through object.__setattr__ and restores the originals afterwards.
    """
    from config.settings import settings

    saved = {}

    def _apply(**changes):
        for name, value in changes.items():
            if not hasattr(settings, name):
                raise AttributeError(f"Settings has no field {name!r}")
            saved.setdefault(name, getattr(settings, name))
            object.__setattr__(settings, name, value)

    yield _apply

    for name, value in saved.items():
        object.__setattr__(settings, name, value)


@pytest.fixture(autouse=True)
def _no_database(request, monkeypatch):
    """Make database access fail instantly.

    There is no Postgres in the test environment, and letting each call wait out
    a real TCP connect attempt made the suite take minutes. Failing immediately
    exercises exactly the same fallback paths (default templates, "our shop",
    empty history) that a production outage would, only faster.

    Opt out with @pytest.mark.needs_db for a test that wants the real thing.
    """
    if request.node.get_closest_marker("needs_db"):
        yield
        return

    from sqlalchemy.exc import OperationalError

    def _unavailable(*args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("test: database unavailable"))

    import db.database as database_module
    import repositories.tools as tools_module

    # Modules that imported these helpers by name hold their own references, so
    # each namespace needs patching. raising=False because not every module
    # imports all three, and which ones it imports is an implementation detail
    # this fixture should not be coupled to.
    for module in (database_module, tools_module):
        for name in ("read_session", "session_scope", "SessionLocal"):
            monkeypatch.setattr(module, name, _unavailable, raising=False)
    yield
