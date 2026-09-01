from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Lock
from typing import TypeVar

_ItemT = TypeVar("_ItemT")
_ResultT = TypeVar("_ResultT")


class ByteRateLimiter:
    """Thread-safe aggregate byte-rate limiter for streamed transfers."""

    def __init__(self, max_mib_per_second: float) -> None:
        if max_mib_per_second <= 0:
            raise ValueError("max_mib_per_second must be positive")
        self.bytes_per_second = max_mib_per_second * 2**20
        self._lock = Lock()
        self._next_slot = time.perf_counter()

    def wait_for(self, byte_count: int) -> None:
        """Reserve bandwidth for byte_count bytes and wait for its slot."""

        if byte_count <= 0:
            return
        duration = byte_count / self.bytes_per_second
        with self._lock:
            now = time.perf_counter()
            slot = max(now, self._next_slot)
            self._next_slot = slot + duration
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def bounded_thread_map(
    executor: ThreadPoolExecutor,
    function: Callable[[_ItemT], _ResultT],
    items: Iterable[_ItemT],
    *,
    max_pending: int,
) -> Iterator[_ResultT]:
    """Yield completed results while keeping only a bounded future window."""

    if max_pending < 1:
        raise ValueError("max_pending must be >= 1")
    iterator = iter(items)
    pending: set[Future[_ResultT]] = set()
    exhausted = False

    while len(pending) < max_pending:
        try:
            item = next(iterator)
        except StopIteration:
            exhausted = True
            break
        pending.add(executor.submit(function, item))

    while pending:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            yield future.result()
        while not exhausted and len(pending) < max_pending:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.add(executor.submit(function, item))


def bounded_thread_map_ordered(
    executor: ThreadPoolExecutor,
    function: Callable[[_ItemT], _ResultT],
    items: Iterable[_ItemT],
    *,
    max_pending: int,
) -> Iterator[_ResultT]:
    """Yield results in input order with only a bounded future window."""

    if max_pending < 1:
        raise ValueError("max_pending must be >= 1")
    iterator = iter(items)
    pending: deque[Future[_ResultT]] = deque()

    for _ in range(max_pending):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending.append(executor.submit(function, item))

    while pending:
        yield pending.popleft().result()
        try:
            item = next(iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(function, item))
