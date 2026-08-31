"""
One shared thread pool for fire-and-forget work.

Previously each finished turn did `threading.Thread(target=..., daemon=True).start()`
to write its log row. That creates and tears down an OS thread per message, and
nothing bounds it - a traffic spike spawns unbounded threads, and daemon threads
are killed mid-write at shutdown, losing the row.

A single bounded pool reuses threads, caps concurrency, and can be drained on
shutdown so in-flight writes finish.
"""
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from config.settings import settings
from utils.logger import get_logger
from utils.metrics import metrics

logger = get_logger(__name__)

_executor: Optional[ThreadPoolExecutor] = None
_lock = threading.Lock()
_shutting_down = False


def get_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=settings.worker_threads,
                thread_name_prefix="worker",
            )
            logger.info("Background executor started (%d threads)", settings.worker_threads)
        return _executor


def submit(fn: Callable[..., Any], *args, task_name: str = "", **kwargs):
    """Run `fn` off the caller's thread. Exceptions are logged, never raised.

    Returns the Future, or None if we're shutting down. Callers are
    fire-and-forget, so a failure here must not surface as a failed customer
    message.
    """
    if _shutting_down:
        logger.debug("Ignoring background task %s during shutdown", task_name or fn.__name__)
        return None

    label = task_name or getattr(fn, "__name__", "task")

    def _wrapped():
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            metrics.incr(f"background.{label}.error")
            logger.error("Background task '%s' failed: %s", label, e, exc_info=True)
            return None

    try:
        return get_executor().submit(_wrapped)
    except RuntimeError:
        # Pool already shut down (interpreter tearing down).
        logger.debug("Background executor unavailable for task %s", label)
        return None


def shutdown(wait: bool = True, timeout: Optional[float] = 10.0) -> None:
    """Stop accepting work and let queued tasks finish."""
    global _executor, _shutting_down
    _shutting_down = True
    with _lock:
        executor = _executor
        _executor = None
    if executor is None:
        return
    logger.info("Draining background executor...")
    # cancel_futures leaves running tasks alone but drops the queued backlog,
    # so shutdown can't be held open indefinitely by a growing queue.
    executor.shutdown(wait=wait, cancel_futures=not wait)
    logger.info("Background executor stopped")


def queue_depth() -> int:
    """Approximate pending task count, for /metrics."""
    with _lock:
        executor = _executor
    if executor is None:
        return 0
    try:
        return executor._work_queue.qsize()  # noqa: SLF001 - no public accessor exists
    except Exception:
        return 0


atexit.register(shutdown, wait=False)
