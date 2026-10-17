"""Unit tests for request_log — one log line per Bot API request.

Requests go through TelegramRateLimiter, which logs each one when it finishes.
"""

import json
import logging
import time

import pytest
from telegram.error import BadRequest

from ccbot.rate_limit import TelegramRateLimiter
from ccbot.request_log import request_purpose


def _limiter() -> TelegramRateLimiter:
    # Not initialized, so the window starts empty: no waiting in these tests
    return TelegramRateLimiter()


async def test_logs_method_topic_message_and_text(caplog: pytest.LogCaptureFixture):
    async def send(*_args, **_kwargs):
        return {"message_id": 123}

    data = {"chat_id": 777, "message_thread_id": 42, "text": 'say "hi"\nthere'}
    with (
        caplog.at_level(logging.INFO, logger="ccbot.request_log"),
        request_purpose("status"),
    ):
        result = await _limiter().process_request(
            send, (), {}, "sendMessage", data, None
        )

    assert result == {"message_id": 123}
    line = caplog.records[-1].getMessage()
    assert line.startswith("api sendMessage purpose=status chat=777 thread=42 msg=123 ")
    assert "len=14" in line
    assert line.endswith("text=" + json.dumps('say "hi"\nthere'))
    assert "\n" not in line


async def test_logs_failed_request_and_reraises(caplog: pytest.LogCaptureFixture):
    async def edit(*_args, **_kwargs):
        raise BadRequest("Message is not modified")

    data = {"chat_id": 777, "message_id": 9, "text": "same"}
    with (
        caplog.at_level(logging.INFO, logger="ccbot.request_log"),
        pytest.raises(BadRequest),
    ):
        await _limiter().process_request(edit, (), {}, "editMessageText", data, None)

    line = caplog.records[-1].getMessage()
    assert line.startswith("api editMessageText purpose=- chat=777 thread=- msg=9 ")
    assert 'error="Message is not modified"' in line


async def test_logs_lag_since_the_source_was_read(caplog: pytest.LogCaptureFixture):
    async def send(*_args, **_kwargs):
        return {"message_id": 1}

    read = time.time() - 12.5
    with (
        caplog.at_level(logging.INFO, logger="ccbot.request_log"),
        request_purpose("text", read_at=read),
    ):
        await _limiter().process_request(
            send, (), {}, "sendMessage", {"chat_id": 1, "text": "x"}, None
        )
        with request_purpose("status"):  # status carries no source time
            await _limiter().process_request(
                send, (), {}, "sendMessage", {"chat_id": 1, "text": "y"}, None
            )

    text_line, status_line = (r.getMessage() for r in caplog.records[-2:])
    lag = float(text_line.split("lag=")[1].split()[0])
    assert 12.5 <= lag < 14
    assert "lag=" not in status_line
