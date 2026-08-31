"""
Running a coroutine from synchronous code, safely.

The tool functions are synchronous (LangChain calls them that way) but the
storage services expose `async def` methods, so the tools used `asyncio.run(...)`.
That works only on a thread with no event loop running. Reached from the
/chat endpoint - an `async def` handler that called the agent inline - it raised

    RuntimeError: asyncio.run() cannot be called from a running event loop

and the customer's payment receipt failed to save. The WhatsApp path happened to
work because it runs in a worker thread, so the bug only showed on one route.

This runs the coroutine on a dedicated loop in its own thread and blocks for the
result. Because that loop is never the caller's loop, it is safe from anywhere:
a plain thread, a worker thread, or the event loop thread itself.
"""
import asyncio
import atexit
import threading
from typing import Any, Coroutine, Optional, TypeVar

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()

# A storage upload shouldn't be able to wedge a worker thread forever.
DEFAULT_TIMEOUT = 120.0


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is not None and not _loop.is_closed():
            return _loop

        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run, name="async-bridge", daemon=True)
        thread.start()

        _loop, _thread = loop, thread
        logger.debug("Async bridge loop started")
        return loop


def run_sync(coro: Coroutine[Any, Any, T], timeout: Optional[float] = DEFAULT_TIMEOUT) -> T:
    """Run `coro` to completion and return its result.

    Raises whatever the coroutine raises, and TimeoutError if it outlives
    `timeout`. Safe to call from any thread, including one running an event loop.
    """
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise TimeoutError(f"Async operation exceeded {timeout}s and was cancelled")


def shutdown() -> None:
    """Stop the bridge loop."""
    global _loop, _thread
    with _lock:
        loop, thread = _loop, _thread
        _loop, _thread = None, None

    if loop is None:
        return
    loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5.0)
    if not loop.is_closed():
        loop.close()
    logger.debug("Async bridge loop stopped")


atexit.register(shutdown)
