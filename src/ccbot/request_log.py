"""Per-request Telegram API log — one line for every Bot API call that goes out.

The bot-wide rate limit (one request per 2.2s) is shared by content, status
edits, deletes and command replies alike; this log shows where the quota goes.

Key components:
  - LoggingRateLimiter: AIORateLimiter subclass that logs each request after
    it finishes: method, chat, topic, message id, the time spent waiting for
    a rate-limit slot, the time Telegram took, and a preview of the text.
    Requests carrying Claude output also log lag: seconds from the bot
    reading the entry from the JSONL to Telegram accepting the request — the
    bot's own delay (queue, rate limit, sending), not Claude's: Claude Code
    can write an entry well after its timestamp (a thinking block waits for
    the tool call after it). getUpdates bypasses the rate limiter, unlogged.
  - request_purpose: context manager labelling the requests made inside it
    (e.g. "status", "tool_use") and when the bot read their source; the
    queue worker sets it per task.

scripts/api_log.py turns these lines into a readable timeline.
"""

import json
import logging
import time
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from telegram.ext import AIORateLimiter

logger = logging.getLogger(__name__)

# Characters of message text kept in each log line
TEXT_PREVIEW_CHARS = 200

_purpose: ContextVar[str] = ContextVar("ccbot_request_purpose", default="-")
_read_at: ContextVar[float | None] = ContextVar("ccbot_request_read_at", default=None)


@contextmanager
def request_purpose(label: str, read_at: float | None = None) -> Iterator[None]:
    """Label every Bot API request made inside this block (per asyncio task).

    read_at is when the bot read the message these requests deliver from the
    JSONL (epoch seconds); requests made inside get logged with their lag.
    """
    purpose_token = _purpose.set(label)
    read_token = _read_at.set(read_at)
    try:
        yield
    finally:
        _read_at.reset(read_token)
        _purpose.reset(purpose_token)


def _text_of(data: dict[str, Any]) -> str | None:
    """The human-readable payload of a request, if it has one."""
    for key in ("text", "caption", "action"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    rich = data.get("rich_message")
    if isinstance(rich, dict) and isinstance(rich.get("markdown"), str):
        return rich["markdown"]
    return None


def _format_line(
    endpoint: str,
    data: dict[str, Any],
    result: Any,
    wait: float,
    took: float,
    error: BaseException | None,
) -> str:
    fields = [f"api {endpoint}", f"purpose={_purpose.get()}"]
    fields.append(f"chat={data.get('chat_id', '-')}")
    fields.append(f"thread={data.get('message_thread_id', '-')}")
    msg_id = data.get("message_id")
    if msg_id is None and isinstance(result, dict):
        msg_id = result.get("message_id")
    fields.append(f"msg={msg_id if msg_id is not None else '-'}")
    fields.append(f"wait={wait:.2f}")
    fields.append(f"took={took:.2f}")
    read_at = _read_at.get()
    if read_at is not None:
        fields.append(f"lag={time.time() - read_at:.2f}")
    text = _text_of(data)
    if text is not None:
        fields.append(f"len={len(text)}")
    if error is not None:
        fields.append(f"error={json.dumps(str(error), ensure_ascii=False)}")
    if text is not None:
        preview = json.dumps(text[:TEXT_PREVIEW_CHARS], ensure_ascii=False)
        fields.append(f"text={preview}")
    return " ".join(fields)


class LoggingRateLimiter(AIORateLimiter):
    """AIORateLimiter that logs every request it lets through."""

    async def process_request(
        self,
        callback: Callable[..., Coroutine[Any, Any, bool | dict[str, Any] | list]],
        args: Any,
        kwargs: dict[str, Any],
        endpoint: str,
        data: dict[str, Any],
        rate_limit_args: int | None,
    ) -> bool | dict[str, Any] | list:
        queued = time.monotonic()
        started: float | None = None

        async def timed(*a: Any, **kw: Any) -> bool | dict[str, Any] | list:
            nonlocal started
            if started is None:  # AIORateLimiter retries after a flood ban
                started = time.monotonic()
            return await callback(*a, **kw)

        result: Any = None
        error: BaseException | None = None
        try:
            result = await super().process_request(
                timed, args, kwargs, endpoint, data, rate_limit_args
            )
            return result
        except BaseException as e:
            error = e
            raise
        finally:
            done = time.monotonic()
            begin = started if started is not None else done
            logger.info(
                _format_line(
                    endpoint, data, result, begin - queued, done - begin, error
                )
            )
