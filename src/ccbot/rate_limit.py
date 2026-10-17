"""Bot-wide Telegram rate limit: a sliding window, and a hard stop on flood control.

Every topic lives in one private chat, so one bot-wide limit is the per-chat
limit. TelegramRateLimiter replaces python-telegram-bot's AIORateLimiter on the
Application (BaseRateLimiter is the library's plug-in point): every Bot API
request that targets a chat — sends, edits, deletes, topic calls — waits for a
slot in the window, then goes out and is logged (request_log).

Key components:
  - SlidingWindow: at most WINDOW_MAX_REQUESTS requests in any WINDOW_SECONDS
    seconds. A request that finds the window full waits for the oldest entry
    to expire, so a burst goes out at once and only a sustained flow is slowed.
    On startup the window is filled as if the bot had been sending at full
    rate (Telegram's own counter outlives a restart): the first ~30s pace at
    one request per WINDOW_SECONDS / WINDOW_MAX_REQUESTS seconds.
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

WINDOW_MAX_REQUESTS = 15
WINDOW_SECONDS = 30.0


class SlidingWindow:
    """At most max_requests acquisitions in any period seconds; waiters queue FIFO."""

    def __init__(self, max_requests: int, period: float) -> None:
        self.max_requests = max_requests
        self.period = period
        self._sent: deque[float] = deque()  # monotonic times, oldest first
        self._lock = asyncio.Lock()

    def _expire(self, now: float) -> None:
        while self._sent and self._sent[0] <= now - self.period:
            self._sent.popleft()

    def free(self) -> int:
        """How many requests could go out right now."""
        self._expire(time.monotonic())
        return self.max_requests - len(self._sent)

    async def acquire(self) -> None:
        """Take a slot, waiting for the oldest entry to expire if there is none."""
        async with self._lock:
            while True:
                now = time.monotonic()
                self._expire(now)
                if len(self._sent) < self.max_requests:
                    self._sent.append(now)
                    return
                await asyncio.sleep(self._sent[0] + self.period - now)

    def fill_evenly(self) -> None:
        """Fill the window as if requests had gone out at full rate until now."""
        now = time.monotonic()
        step = self.period / self.max_requests
        self._sent = deque(
            now - self.period + step * (i + 1) for i in range(self.max_requests)
        )


class TelegramRateLimiter(BaseRateLimiter[None]):
    """Sliding-window limiter that logs every request and stops ccbot on a 429."""

    def __init__(
        self,
        max_requests: int = WINDOW_MAX_REQUESTS,
        period: float = WINDOW_SECONDS,
    ) -> None:
        self.window = SlidingWindow(max_requests, period)
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
            f"{seconds:.0f}s (until {lifted}). The "
            f"{self.window.max_requests} per {self.window.period:.0f}s window "
            "let too much through — ccbot stopped; restart once the ban is over"
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
