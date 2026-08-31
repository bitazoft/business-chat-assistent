"""
The bridge exists because tool functions are sync but the storage services are
async. asyncio.run() crashed when a tool was reached from the async /chat
handler; these tests cover the cases that used to break.
"""
import asyncio

import pytest

from utils.async_bridge import run_sync


async def _double(x):
    await asyncio.sleep(0)
    return x * 2


async def _boom():
    raise ValueError("inner failure")


async def _forever():
    await asyncio.sleep(30)


def test_runs_a_coroutine_from_a_plain_thread():
    assert run_sync(_double(21)) == 42


def test_works_from_inside_a_running_event_loop():
    """This is the exact case asyncio.run() raised RuntimeError on."""

    async def caller():
        # Blocking inside a coroutine is not something to do in production code,
        # but it must not raise "cannot be called from a running event loop".
        return run_sync(_double(5))

    assert asyncio.run(caller()) == 10


def test_exceptions_propagate_to_the_caller():
    with pytest.raises(ValueError, match="inner failure"):
        run_sync(_boom())


def test_timeout_is_enforced():
    with pytest.raises(TimeoutError):
        run_sync(_forever(), timeout=0.05)


def test_reuses_the_same_loop_across_calls():
    assert run_sync(_double(1)) == 2
    assert run_sync(_double(2)) == 4
    assert run_sync(_double(3)) == 6


def test_works_from_a_worker_thread():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: run_sync(_double(i)), range(10)))
    assert results == [i * 2 for i in range(10)]
