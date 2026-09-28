"""Unit tests for message_sender — transient vs permanent send failures."""

from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TimedOut

from ccbot.handlers import message_sender
from ccbot.handlers.message_sender import (
    is_topic_gone,
    is_topic_gone_error,
    is_transient_error,
    send_photo,
    send_rich_markdown,
    send_with_fallback,
)


@pytest.fixture(autouse=True)
def _no_gone_topics():
    message_sender._gone_topics.clear()
    yield
    message_sender._gone_topics.clear()


@pytest.mark.parametrize(
    "error, expected",
    [
        pytest.param(RetryAfter(5), True, id="flood_control"),
        pytest.param(TimedOut(), True, id="timeout"),
        pytest.param(NetworkError("reset"), True, id="network"),
        pytest.param(BadRequest("can't parse entities"), False, id="bad_request"),
        pytest.param(Forbidden("bot was blocked"), False, id="forbidden"),
        pytest.param(ValueError("bug"), False, id="non_telegram"),
    ],
)
def test_is_transient_error(error: Exception, expected: bool):
    assert is_transient_error(error) is expected


async def test_bad_markdown_falls_back_to_plain_text():
    bot = AsyncMock()
    bot.send_message.side_effect = [BadRequest("can't parse entities"), "sent"]

    assert await send_with_fallback(bot, 1, "*oops", raise_transient=True) == "sent"
    assert "parse_mode" not in bot.send_message.await_args.kwargs


async def test_network_error_is_raised_for_the_queue_to_retry():
    bot = AsyncMock()
    bot.send_message.side_effect = TimedOut()

    with pytest.raises(TimedOut):
        await send_with_fallback(bot, 1, "hello", raise_transient=True)
    # No pointless plain-text attempt over a dead connection
    bot.send_message.assert_awaited_once()


async def test_network_error_returns_none_for_other_callers():
    bot = AsyncMock()
    bot.send_message.side_effect = TimedOut()

    assert await send_with_fallback(bot, 1, "hello") is None


async def test_retry_after_is_always_raised():
    bot = AsyncMock()
    bot.send_message.side_effect = RetryAfter(5)

    with pytest.raises(RetryAfter):
        await send_with_fallback(bot, 1, "hello")


async def test_permanent_failure_returns_none():
    bot = AsyncMock()
    bot.send_message.side_effect = BadRequest("message thread not found")

    assert await send_with_fallback(bot, 1, "hello", raise_transient=True) is None


async def test_send_photo_raises_transient_and_swallows_permanent():
    bot = AsyncMock()
    bot.send_photo.side_effect = NetworkError("reset")
    with pytest.raises(NetworkError):
        await send_photo(bot, 1, [("image/png", b"png")])

    bot.send_photo.side_effect = BadRequest("PHOTO_INVALID_DIMENSIONS")
    await send_photo(bot, 1, [("image/png", b"png")])


async def test_send_rich_markdown_raises_transient_and_reports_permanent():
    bot = AsyncMock()
    bot.do_api_request.side_effect = TimedOut()
    with pytest.raises(TimedOut):
        await send_rich_markdown(bot, 1, "| a |")

    bot.do_api_request.side_effect = BadRequest("method not found")
    assert await send_rich_markdown(bot, 1, "| a |") is False


# --- Deleted topics: Telegram sends no update, the first send finds out ---


@pytest.mark.parametrize(
    "error, expected",
    [
        pytest.param(BadRequest("Message thread not found"), True, id="thread"),
        pytest.param(BadRequest("Bad Request: TOPIC_ID_INVALID"), True, id="topic"),
        pytest.param(BadRequest("chat not found"), False, id="chat"),
        pytest.param(NetworkError("message thread not found"), False, id="network"),
    ],
)
def test_is_topic_gone_error(error: Exception, expected: bool):
    assert is_topic_gone_error(error) is expected


async def test_send_to_deleted_topic_marks_it_and_skips_plain_text_retry():
    bot = AsyncMock()
    bot.send_message.side_effect = BadRequest("Message thread not found")

    sent = await send_with_fallback(
        bot, 777, "hello", raise_transient=True, message_thread_id=42
    )

    assert sent is None
    bot.send_message.assert_awaited_once()  # plain text can't reach it either
    assert is_topic_gone(777, 42)
    assert not is_topic_gone(777, 43)
    assert not is_topic_gone(777, None)


async def test_other_permanent_failure_does_not_mark_topic():
    bot = AsyncMock()
    bot.send_message.side_effect = BadRequest("can't parse entities")

    await send_with_fallback(bot, 777, "*x", message_thread_id=42)

    assert bot.send_message.await_count == 2
    assert not is_topic_gone(777, 42)
