"""Bot-wide Telegram rate limit: sliding windows, and a hard stop on flood control.

Every topic lives in one private chat, so one bot-wide limit is the per-chat
limit. TelegramRateLimiter replaces python-telegram-bot's AIORateLimiter on the
Application (BaseRateLimiter is the library's plug-in point): every Bot API
request that targets a chat — sends, edits, deletes, topic calls — waits for a
slot in the window, then goes out and is logged (request_log).

Key components:
  - SlidingWindow: for each of WINDOW_LIMITS, at most that many requests in
    any that many seconds — 15 per 30s lets a burst out at once, 20 per 60s
    holds a sustained flow to Telegram's 20 a minute (15 per 30s alone, run
    flat out for 4 minutes, earned a ban). A request that finds a limit full
    waits for its oldest entry to expire. On startup each limit is filled as
    if the bot had been sending at its full rate (Telegram's own counter
    outlives a restart): the first ~60s pace at one request per 3s.
  - TelegramRateLimiter: flood control (429 RetryAfter) means the window let
    too much through. Instead of retrying, it refuses every later request and
    calls on_flood (Application.stop_running) so ccbot stops; main posts
    flood_message to the WeCom alert webhook (alert.py) and exits with it.
    Undelivered messages are re-sent after the restart.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable, Coroutine
from datetime import timedelta
from typing import Any

from telegram.error import RetryAfter
from telegram.ext import BaseRateLimiter

from .request_log import log_request

logger = logging.getLogger(__name__)

# (max requests, in any this many seconds)
WINDOW_LIMITS: tuple[tuple[int, float], ...] = ((15, 30.0), (20, 60.0))


class SlidingWindow:
    """At most max_requests acquisitions in any period seconds, for each
    (max_requests, period) limit; waiters queue FIFO."""

    def __init__(self, *limits: tuple[int, float]) -> None:
        self.limits = limits
        # Per limit: monotonic times of the requests it counts, oldest first
        self._sent: list[deque[float]] = [deque() for _ in limits]
        self._lock = asyncio.Lock()

    def _expire(self, now: float) -> None:
        for (_, period), sent in zip(self.limits, self._sent):
            while sent and sent[0] <= now - period:
                sent.popleft()

    def free(self) -> int:
        """How many requests could go out right now."""
        self._expire(time.monotonic())
        return min(n - len(sent) for (n, _), sent in zip(self.limits, self._sent))

    async def acquire(self) -> None:
        """Take a slot, waiting for the oldest entries of full limits to expire."""
        async with self._lock:
            while True:
                now = time.monotonic()
                self._expire(now)
                waits = [
                    sent[0] + period - now
                    for (n, period), sent in zip(self.limits, self._sent)
                    if len(sent) >= n
                ]
                if not waits:
                    for sent in self._sent:
                        sent.append(now)
                    return
                await asyncio.sleep(max(waits))

    def fill_evenly(self) -> None:
        """Fill each limit as if requests had gone out at its full rate until now."""
        now = time.monotonic()
        self._sent = [
            deque(now - period + period / n * (i + 1) for i in range(n))
            for n, period in self.limits
        ]

    def describe(self) -> str:
        return " and ".join(f"{n} per {period:.0f}s" for n, period in self.limits)


class TelegramRateLimiter(BaseRateLimiter[None]):
    """Sliding-window limiter that logs every request and stops ccbot on a 429."""

    def __init__(self, limits: tuple[tuple[int, float], ...] = WINDOW_LIMITS) -> None:
        self.window = SlidingWindow(*limits)
        self.on_flood: Callable[[], None] | None = None
        self.flood_message: str | None = None
        self._ban: timedelta | None = None

    async def initialize(self) -> None:
        self.window.fill_evenly()

    async def shutdown(self) -> None:
        pass

    def _refuse_after_flood(self) -> None:
        """Once Telegram has banned us, nothing more goes out."""
        if self._ban is not None:
            raise RetryAfter(self._ban)

    def _flooded(self, endpoint: str, error: RetryAfter) -> None:
        if self._ban is not None:
            return
        self._ban = error._retry_after  # timedelta; .retry_after warns
        seconds = self._ban.total_seconds()
        lifted = time.strftime("%H:%M:%S", time.localtime(time.time() + seconds))
        self.flood_message = (
            f"Telegram flood control on {endpoint}: banned for "
            f"{seconds:.0f}s (until {lifted}). The rate limit "
            f"({self.window.describe()}) let too much through — ccbot stopped; "
            "restart once the ban is over"
        )
        logger.error(self.flood_message)
        if self.on_flood is not None:
            self.on_flood()

    async def process_request(
        self,
        callback: Callable[..., Coroutine[Any, Any, bool | dict[str, Any] | list]],
        args: Any,
        kwargs: dict[str, Any],
        endpoint: str,
        data: dict[str, Any],
        rate_limit_args: None,
    ) -> bool | dict[str, Any] | list:
        self._refuse_after_flood()
        queued = time.monotonic()
        if "chat_id" in data:
            await self.window.acquire()
        self._refuse_after_flood()  # the ban may have come while we waited

        started = time.monotonic()
        result: Any = None
        error: BaseException | None = None
        try:
            result = await callback(*args, **kwargs)
            return result
        except RetryAfter as e:
            error = e
            self._flooded(endpoint, e)
            raise
        except BaseException as e:
            error = e
            raise
        finally:
            done = time.monotonic()
            log_request(endpoint, data, result, started - queued, done - started, error)
