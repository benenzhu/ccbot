"""Tests for rate_limit — the bot-wide sliding window and the stop on a 429."""

import asyncio
import time
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from telegram.error import RetryAfter

from ccbot.rate_limit import SlidingWindow, TelegramRateLimiter


class _Clock:
    """Fake monotonic clock; sleeping just moves it forward."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Iterator[_Clock]:
    fake = _Clock()
    with (
        patch(
            "ccbot.rate_limit.time",
            SimpleNamespace(
                monotonic=fake.monotonic,
                time=time.time,
                localtime=time.localtime,
                strftime=time.strftime,
            ),
        ),
        patch(
            "ccbot.rate_limit.asyncio",
            SimpleNamespace(Lock=asyncio.Lock, sleep=fake.sleep),
        ),
    ):
        yield fake


async def test_a_burst_goes_out_at_once(clock: _Clock):
    window = SlidingWindow(3, 30.0)
    for _ in range(3):
        await window.acquire()
    assert clock.now == 1000.0
    assert window.free() == 0


async def test_a_full_window_waits_for_its_oldest_entry(clock: _Clock):
    window = SlidingWindow(3, 30.0)
    for _ in range(3):  # at 1000, 1010, 1020
        await window.acquire()
        clock.now += 10
    clock.now -= 10

    await window.acquire()  # waits until the 1000 entry is 30s old
    assert clock.now == 1030.0
    assert window.free() == 0
    clock.now += 10  # the 1010 entry expires
    assert window.free() == 1


async def test_startup_fill_frees_one_slot_per_step(clock: _Clock):
    window = SlidingWindow(15, 30.0)
    window.fill_evenly()
    assert window.free() == 0
    clock.now += 2.0
    assert window.free() == 1
    clock.now += 28.0
    assert window.free() == 15


async def test_requests_without_a_chat_skip_the_window(clock: _Clock):
    limiter = TelegramRateLimiter()
    await limiter.initialize()  # window starts full

    async def call() -> bool:
        return True

    assert await limiter.process_request(call, (), {}, "getMe", {}, None)
    assert clock.now == 1000.0
    await limiter.process_request(call, (), {}, "sendMessage", {"chat_id": 1}, None)
    assert clock.now == 1002.0


async def test_flood_control_stops_the_bot_and_refuses_later_requests(clock: _Clock):
    limiter = TelegramRateLimiter()
    limiter.on_flood = MagicMock()
    sent: list[str] = []

    async def banned() -> bool:
        sent.append("banned")
        raise RetryAfter(162)

    async def ok() -> bool:
        sent.append("ok")
        return True

    data = {"chat_id": 1, "text": "x"}
    with pytest.raises(RetryAfter):
        await limiter.process_request(banned, (), {}, "sendMessage", data, None)
    limiter.on_flood.assert_called_once_with()
    assert limiter.flood_message is not None
    assert "sendMessage" in limiter.flood_message
    assert "162s" in limiter.flood_message

    with pytest.raises(RetryAfter):
        await limiter.process_request(ok, (), {}, "editMessageText", data, None)
    assert sent == ["banned"]  # nothing reached Telegram after the ban
    limiter.on_flood.assert_called_once_with()
