"""Per-user message queue management for ordered message delivery.

Provides a queue-based message processing system that ensures:
  - Messages are sent in receive order (FIFO)
  - Status messages always follow content messages
  - Consecutive content messages can be merged for efficiency
  - Thread-aware sending: each MessageTask carries an optional thread_id
    for Telegram topic support
  - Nothing but status is dropped: a task that hits flood control or a
    network error is retried in place until it goes through. Tasks are
    consumed as they are sent (parts, tables, images), so a retry only sends
    what is still missing. Delivery releases the task's DeliveryTickets,
    which lets the monitor's persisted offset move past the message.

Rate limiting: AIORateLimiter on the Application handles the global limit and
groups, but has no per-chat limiter for private chats. This module therefore
paces every chat itself (_pace / _try_pace): content waits for a send slot,
status is skipped when none is free.

Key components:
  - MessageTask: Dataclass representing a queued message task (with thread_id)
  - get_or_create_queue: Get or create queue and worker for a user
  - Message queue worker: Background task processing user's queue
  - Content task processing with tool_use/tool_result handling
  - Status message tracking and conversion (keyed by (user_id, thread_id))
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from telegram import Bot
from telegram.constants import ChatAction
from telegram.error import RetryAfter

from ..config import config
from ..markdown_v2 import ParsedTable, convert_markdown, table_to_markdown
from ..monitor_state import DeliveryTicket
from ..screenshot import render_table_image
from ..session import session_manager
from ..terminal_parser import parse_status_line
from ..tmux_manager import tmux_manager
from ..tts import prepare_tts_segments, synthesize_prepared
from .message_sender import (
    NO_LINK_PREVIEW,
    PARSE_MODE,
    is_transient_error,
    send_photo,
    send_rich_markdown,
    send_with_fallback,
    strip_sentinels,
)

logger = logging.getLogger(__name__)


def _ensure_formatted(text: str) -> str:
    """Convert markdown to MarkdownV2."""
    return convert_markdown(text)


# Merge limit for content messages
MERGE_MAX_LENGTH = 3800  # Leave room for markdown conversion overhead


@dataclass
class MessageTask:
    """Message task for queue processing."""

    task_type: Literal["content", "status_update", "status_clear", "voice", "table"]
    text: str | None = None
    window_id: str | None = None
    # content type fields
    parts: list[str] = field(default_factory=list)
    tool_use_id: str | None = None
    content_type: str = "text"
    thread_id: int | None = None  # Telegram topic thread_id for targeted send
    image_data: list[tuple[str, bytes]] | None = None  # From tool_result images
    # Markdown tables stripped from the text, sent after it as native
    # Telegram tables (PNG fallback). Each is (headers, rows).
    tables: list[ParsedTable] | None = None
    # Receipts released once the task is delivered (or permanently failed),
    # letting the monitor's persisted offset move past the source messages.
    tickets: list[DeliveryTicket] = field(default_factory=list)
    # Retry bookkeeping: the first part has been handled (status conversion
    # must not run again), voice segments already sent, and how often the
    # attachments/voice failed transiently.
    started: bool = False
    voice_sent: int = 0
    upload_failures: int = 0


# Per-user message queues and worker tasks
_message_queues: dict[int, asyncio.Queue[MessageTask]] = {}
_queue_workers: dict[int, asyncio.Task[None]] = {}
_queue_locks: dict[int, asyncio.Lock] = {}  # Protect drain/refill operations

# Map (tool_use_id, user_id, thread_id_or_0) -> telegram message_id
# for editing tool_use messages with results
_tool_msg_ids: dict[tuple[str, int, int], int] = {}

# Status message tracking: (user_id, thread_id_or_0) -> (message_id, window_id, last_text)
_status_msg_info: dict[tuple[int, int], tuple[int, str, str]] = {}

# Flood control: user_id -> monotonic time when ban expires
_flood_until: dict[int, float] = {}

# Longest RetryAfter that is simply slept through. A longer ban also pauses
# status traffic (_flood_until) while the content task waits it out.
FLOOD_CONTROL_MAX_WAIT = 10

# Transient network errors back off exponentially up to this many seconds
RETRY_BACKOFF_MAX = 60.0

# Telegram allows about one message per second per chat, with short bursts.
# Going faster (a history replay easily produces 100+ messages a minute) earns
# flood bans that grow to half an hour, during which nothing is delivered.
CHAT_SEND_INTERVAL = 1.0  # sustained seconds per message to one chat
CHAT_SEND_BURST = 3  # messages that may go out back to back

# chat_id -> monotonic time the next non-burst send is due
_chat_next_send: dict[int, float] = {}

# Uploads (images, table PNGs, voice) get a bounded number of attempts: a
# payload that always times out would otherwise block the queue forever, and
# at-least-once replay would bring it right back after a restart.
UPLOAD_MAX_ATTEMPTS = 5


def _reserve_send_slot(chat_id: int, *, wait: bool) -> float | None:
    """Book the next send slot for a chat.

    Returns how long to sleep before sending. With wait=False, returns None
    (booking nothing) when a slot isn't free right now.
    """
    now = time.monotonic()
    due = max(_chat_next_send.get(chat_id, 0.0), now)
    delay = due - now - (CHAT_SEND_BURST - 1) * CHAT_SEND_INTERVAL
    if delay > 0 and not wait:
        return None
    _chat_next_send[chat_id] = due + CHAT_SEND_INTERVAL
    return max(delay, 0.0)


async def _pace(chat_id: int) -> None:
    """Wait for this chat's next send slot (content, tables, images, voice)."""
    delay = _reserve_send_slot(chat_id, wait=True)
    if delay:
        await asyncio.sleep(delay)


def _try_pace(chat_id: int) -> bool:
    """Take a send slot only if one is free now (status: skip, never wait)."""
    return _reserve_send_slot(chat_id, wait=False) is not None


def get_message_queue(user_id: int) -> asyncio.Queue[MessageTask] | None:
    """Get the message queue for a user (if exists)."""
    return _message_queues.get(user_id)


def get_or_create_queue(bot: Bot, user_id: int) -> asyncio.Queue[MessageTask]:
    """Get or create message queue and worker for a user."""
    if user_id not in _message_queues:
        _message_queues[user_id] = asyncio.Queue()
        _queue_locks[user_id] = asyncio.Lock()
        # Start worker task for this user
        _queue_workers[user_id] = asyncio.create_task(
            _message_queue_worker(bot, user_id)
        )
    return _message_queues[user_id]


def _inspect_queue(queue: asyncio.Queue[MessageTask]) -> list[MessageTask]:
    """Non-destructively inspect all items in queue.

    Drains the queue and returns all items. Caller must refill.
    """
    items: list[MessageTask] = []
    while not queue.empty():
        try:
            item = queue.get_nowait()
            items.append(item)
        except asyncio.QueueEmpty:
            break
    return items


def _can_merge_tasks(base: MessageTask, candidate: MessageTask) -> bool:
    """Check if two content tasks can be merged."""
    if base.window_id != candidate.window_id:
        return False
    if candidate.task_type != "content":
        return False
    # tool_use/tool_result break merge chain
    # - tool_use: will be edited later by tool_result
    # - tool_result: edits previous message, merging would cause order issues
    if base.content_type in ("tool_use", "tool_result"):
        return False
    if candidate.content_type in ("tool_use", "tool_result"):
        return False
    return True


async def _merge_content_tasks(
    queue: asyncio.Queue[MessageTask],
    first: MessageTask,
    lock: asyncio.Lock,
) -> tuple[MessageTask, int]:
    """Merge consecutive content tasks from queue.

    Returns: (merged_task, merge_count) where merge_count is the number of
    additional tasks merged (0 if no merging occurred).

    Note on queue counter management:
        When we put items back, we call task_done() to compensate for the
        internal counter increment caused by put_nowait(). This is necessary
        because the items were already counted when originally enqueued.
        Without this compensation, queue.join() would wait indefinitely.
    """
    merged_parts = list(first.parts)
    merged_images: list[tuple[str, bytes]] = list(first.image_data or [])
    merged_tables: list[ParsedTable] = list(first.tables or [])
    merged_tickets: list[DeliveryTicket] = list(first.tickets)
    current_length = sum(len(p) for p in merged_parts)
    merge_count = 0

    async with lock:
        items = _inspect_queue(queue)
        remaining: list[MessageTask] = []

        for i, task in enumerate(items):
            if not _can_merge_tasks(first, task):
                # Can't merge, keep this and all remaining items
                remaining = items[i:]
                break

            # Check length before merging
            task_length = sum(len(p) for p in task.parts)
            if current_length + task_length > MERGE_MAX_LENGTH:
                # Too long, stop merging
                remaining = items[i:]
                break

            merged_parts.extend(task.parts)
            merged_images.extend(task.image_data or [])
            merged_tables.extend(task.tables or [])
            merged_tickets.extend(task.tickets)
            current_length += task_length
            merge_count += 1

        # Put remaining items back into the queue
        for item in remaining:
            queue.put_nowait(item)
            # Compensate: this item was already counted when first enqueued,
            # put_nowait adds a duplicate count that must be removed
            queue.task_done()

    if merge_count == 0:
        return first, 0

    return (
        MessageTask(
            task_type="content",
            window_id=first.window_id,
            parts=merged_parts,
            tool_use_id=first.tool_use_id,
            content_type=first.content_type,
            thread_id=first.thread_id,
            image_data=merged_images or None,
            tables=merged_tables or None,
            tickets=merged_tickets,
        ),
        merge_count,
    )


async def _message_queue_worker(bot: Bot, user_id: int) -> None:
    """Process message tasks for a user sequentially."""
    queue = _message_queues[user_id]
    lock = _queue_locks[user_id]
    logger.info(f"Message queue worker started for user {user_id}")

    while True:
        try:
            task = await queue.get()
            try:
                # Flood control: drop status, wait for content
                flood_end = _flood_until.get(user_id, 0)
                if flood_end > 0:
                    remaining = flood_end - time.monotonic()
                    if remaining > 0:
                        if task.task_type in ("status_update", "status_clear"):
                            # Status is ephemeral — safe to drop
                            continue
                        # Content/voice is actual Claude output — wait then send
                        logger.debug(
                            "Flood controlled: waiting %.0fs for content (user %d)",
                            remaining,
                            user_id,
                        )
                        await asyncio.sleep(remaining)
                    # Ban expired
                    _flood_until.pop(user_id, None)
                    logger.info("Flood control lifted for user %d", user_id)

                if task.task_type == "content":
                    # Try to merge consecutive content tasks
                    task, merge_count = await _merge_content_tasks(queue, task, lock)
                    if merge_count > 0:
                        logger.debug(f"Merged {merge_count} tasks for user {user_id}")
                        # Mark merged tasks as done
                        for _ in range(merge_count):
                            queue.task_done()
                try:
                    await _deliver_task(bot, user_id, task)
                except Exception as e:
                    # Permanent failure: retrying the same request can't help
                    logger.error(
                        f"Error processing message task for user {user_id}: {e}"
                    )
                # Delivered or permanently failed — either way this task no
                # longer holds back the monitor offset. A cancelled task never
                # gets here, so a restart re-sends it.
                for ticket in task.tickets:
                    ticket.release()
            finally:
                queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"Message queue worker cancelled for user {user_id}")
            break
        except Exception as e:
            logger.error(f"Unexpected error in queue worker for user {user_id}: {e}")


def _retry_after_seconds(error: RetryAfter) -> int:
    """Seconds Telegram asked us to wait (retry_after is int or timedelta)."""
    retry_after = error.retry_after
    if isinstance(retry_after, int):
        return retry_after
    return int(retry_after.total_seconds())


async def _run_task(bot: Bot, user_id: int, task: MessageTask) -> None:
    """Dispatch one task to its processor."""
    if task.task_type == "content":
        await _process_content_task(bot, user_id, task)
    elif task.task_type == "status_update":
        await _process_status_update_task(bot, user_id, task)
    elif task.task_type == "status_clear":
        await _do_clear_status_message(bot, user_id, task.thread_id or 0)
    elif task.task_type == "voice":
        await _process_voice_task(bot, user_id, task)
    elif task.task_type == "table":
        await _process_table_task(bot, user_id, task)


async def _deliver_task(bot: Bot, user_id: int, task: MessageTask) -> None:
    """Run a task, retrying transient failures until it is delivered.

    Flood control (RetryAfter) waits as long as Telegram asks; network errors
    back off exponentially. Status tasks are ephemeral: the next poll
    supersedes them, so they get one attempt and are then dropped.
    Permanent errors propagate to the caller.
    """
    ephemeral = task.task_type in ("status_update", "status_clear")
    attempt = 0
    while True:
        try:
            await _run_task(bot, user_id, task)
            return
        except Exception as e:
            if not is_transient_error(e):
                raise
            flood = isinstance(e, RetryAfter)
            if flood:
                delay = float(_retry_after_seconds(e))
                if delay > FLOOD_CONTROL_MAX_WAIT:
                    # Long ban: keep status traffic off until it expires
                    _flood_until[user_id] = time.monotonic() + delay
            else:
                delay = min(2.0**attempt, RETRY_BACKOFF_MAX)

            if ephemeral:
                logger.warning("Status dropped for user %d: %s", user_id, e)
                # Still sit out a short ban so the next request isn't early
                if flood and delay <= FLOOD_CONTROL_MAX_WAIT:
                    await asyncio.sleep(delay)
                return

            attempt += 1
            logger.warning(
                "%s for user %d (%s): retry %d in %.0fs",
                "Flood control" if flood else "Send failed",
                user_id,
                e,
                attempt,
                delay,
            )
            await asyncio.sleep(delay)


def _send_kwargs(thread_id: int | None) -> dict[str, int]:
    """Build message_thread_id kwargs for bot.send_message()."""
    if thread_id is not None:
        return {"message_thread_id": thread_id}
    return {}


async def _send_task_images(bot: Bot, chat_id: int, task: MessageTask) -> None:
    """Send images attached to a task, if any."""
    if not task.image_data:
        return
    logger.info(
        "Sending %d image(s) in thread %s",
        len(task.image_data),
        task.thread_id,
    )
    await _pace(chat_id)
    await send_photo(
        bot,
        chat_id,
        task.image_data,
        **_send_kwargs(task.thread_id),  # type: ignore[arg-type]
    )
    task.image_data = None  # sent — a retry must not send them again


async def _send_table_as_image(
    bot: Bot, chat_id: int, table: ParsedTable, thread_id: int | None
) -> None:
    """PNG fallback for a single table."""
    headers, rows = table
    try:
        png = await render_table_image(headers, rows)
    except Exception as e:
        logger.warning("Failed to render table as image: %s", e)
        return
    await send_photo(
        bot,
        chat_id,
        [("table.png", png)],
        **_send_kwargs(thread_id),  # type: ignore[arg-type]
    )


async def _send_task_tables(bot: Bot, chat_id: int, task: MessageTask) -> None:
    """Send tables attached to a task as native rich-message tables.

    Falls back to a PNG rendering per table when native tables are disabled
    or the sendRichMessage call fails (old client, API error).
    """
    if not task.tables:
        return
    logger.info(
        "Sending %d table(s) in thread %s (native=%s)",
        len(task.tables),
        task.thread_id,
        config.native_tables,
    )
    # Tables are popped as they go out, so a retry resumes at the failed one
    while task.tables:
        table = task.tables[0]
        await _pace(chat_id)
        ok = config.native_tables and await send_rich_markdown(
            bot,
            chat_id,
            table_to_markdown(table),
            **_send_kwargs(task.thread_id),
        )
        if not ok:
            await _send_table_as_image(bot, chat_id, table, task.thread_id)
        task.tables.pop(0)


async def _process_table_task(bot: Bot, user_id: int, task: MessageTask) -> None:
    """Send a standalone table task (a table kept at its place in the prose).

    Table tasks are never merged with content, so a message split as
    prose → table → prose arrives in that order.
    """
    chat_id = session_manager.resolve_chat_id(user_id, task.thread_id)
    await _send_task_tables(bot, chat_id, task)
    await _check_and_send_status(bot, user_id, task.window_id or "", task.thread_id)


def _give_up_on_upload(task: MessageTask, error: Exception) -> bool:
    """Count a transient upload failure; True once the attempts are used up.

    Flood control doesn't count — Telegram says exactly when to come back.
    """
    if isinstance(error, RetryAfter) or not is_transient_error(error):
        return False
    task.upload_failures += 1
    return task.upload_failures >= UPLOAD_MAX_ATTEMPTS


async def _send_task_attachments(bot: Bot, chat_id: int, task: MessageTask) -> None:
    """Send everything that follows a task's text: tables, then images."""
    try:
        await _send_task_tables(bot, chat_id, task)
        await _send_task_images(bot, chat_id, task)
    except Exception as e:
        if not _give_up_on_upload(task, e):
            raise
        logger.error(
            "Giving up on attachments in thread %s after %d attempts: %s",
            task.thread_id,
            task.upload_failures,
            e,
        )
        task.tables = None
        task.image_data = None


async def _edit_tool_message(
    bot: Bot, chat_id: int, message_id: int, task: MessageTask
) -> bool:
    """Edit a tool_use message in place to show its tool_result.

    Returns False when the edit permanently fails (the caller then sends the
    result as a new message). Transient errors are re-raised for a retry.
    """
    # Join all parts for editing (merged content goes together)
    full_text = "\n\n".join(task.parts)
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=_ensure_formatted(full_text),
            parse_mode=PARSE_MODE,
            link_preview_options=NO_LINK_PREVIEW,
        )
        return True
    except Exception as e:
        if is_transient_error(e):
            raise
    try:
        # Fallback: plain text with sentinels stripped
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=strip_sentinels(task.text or full_text),
            link_preview_options=NO_LINK_PREVIEW,
        )
        return True
    except Exception as e:
        if is_transient_error(e):
            raise
        logger.debug(f"Failed to edit tool msg {message_id}, sending new")
        return False


async def _process_content_task(bot: Bot, user_id: int, task: MessageTask) -> None:
    """Process a content message task."""
    wid = task.window_id or ""
    tid = task.thread_id or 0
    chat_id = session_manager.resolve_chat_id(user_id, task.thread_id)

    # The task is consumed as it is sent (parts, then tables and images), so
    # when a transient error makes the worker run it again, only what is
    # still missing goes out.

    # 1. Handle tool_result editing (merged parts are edited together)
    if task.content_type == "tool_result" and task.tool_use_id and task.parts:
        _tkey = (task.tool_use_id, user_id, tid)
        # Looked up, not popped: a transient error must leave it for the retry
        edit_msg_id = _tool_msg_ids.get(_tkey)
        if edit_msg_id is not None:
            # Clear status message first
            await _do_clear_status_message(bot, user_id, tid)
            await _pace(chat_id)
            if await _edit_tool_message(bot, chat_id, edit_msg_id, task):
                task.parts = []
            # Edited, or permanently failed and sent as a new message below
            _tool_msg_ids.pop(_tkey, None)

    # 2. Send content messages, converting status message to first content part
    last_msg_id: int | None = None
    while task.parts:
        part = task.parts[0]
        sent_msg_id: int | None = None
        await _pace(chat_id)

        # For first part, try to convert status message to content (edit instead of delete)
        if not task.started:
            sent_msg_id = await _convert_status_to_content(
                bot,
                user_id,
                tid,
                wid,
                part,
            )

        if sent_msg_id is None:
            sent = await send_with_fallback(
                bot,
                chat_id,
                part,
                raise_transient=True,
                **_send_kwargs(task.thread_id),  # type: ignore[arg-type]
            )
            if sent:
                sent_msg_id = sent.message_id

        task.started = True
        task.parts.pop(0)
        if sent_msg_id is not None:
            last_msg_id = sent_msg_id

    # 3. Record tool_use message ID for later editing
    if last_msg_id and task.tool_use_id and task.content_type == "tool_use":
        _tool_msg_ids[(task.tool_use_id, user_id, tid)] = last_msg_id

    # 4. Send attachments: tables (native, PNG fallback), then images
    await _send_task_attachments(bot, chat_id, task)

    # 5. After content, check and send status
    await _check_and_send_status(bot, user_id, wid, task.thread_id)


async def _synthesize_segment(segment: str, user_id: int) -> bytes | None:
    """Synthesize one segment, retrying once on transient provider errors.

    Volcano occasionally resets streams server-side, so a single retry
    recovers most failures. Returns None when synthesis is hopeless —
    callers skip that segment rather than aborting the whole reply.
    """
    for attempt in range(2):
        try:
            return await synthesize_prepared(segment)
        except ValueError as e:
            # Nothing speakable (e.g. segment was all code) — skip quietly
            logger.debug("Skipping voice segment: %s", e)
            return None
        except Exception as e:
            if attempt == 0:
                logger.warning("TTS synthesis failed, retrying once: %s", e)
                await asyncio.sleep(2.0)
            else:
                logger.error(
                    "TTS synthesis failed after retry for user %d: %s", user_id, e
                )
    return None


async def _process_voice_task(bot: Bot, user_id: int, task: MessageTask) -> None:
    """Synthesize the reply text and send it as voice message(s).

    Long replies are split into segments; segment N+1 is synthesized while
    N is being sent, so the first voice bubble arrives after one short
    synthesis instead of the whole reply. Segments are still *sent* in
    order. TTS failures are logged and swallowed — the text reply was
    already delivered, so a missing voice clip must never crash the worker.

    Transient send errors are re-raised for a retry, which resumes after the
    segments already delivered (task.voice_sent).
    """
    if not task.text:
        return
    segments = prepare_tts_segments(task.text)[task.voice_sent :]
    if not segments:
        return
    if len(segments) > 1:
        logger.debug(
            "Voice reply split into %d segments for user %d", len(segments), user_id
        )
    chat_id = session_manager.resolve_chat_id(user_id, task.thread_id)

    pending: asyncio.Task[bytes | None] | None = asyncio.create_task(
        _synthesize_segment(segments[0], user_id)
    )
    try:
        for i in range(len(segments)):
            current = pending
            assert current is not None  # one task is always queued per iteration
            # Kick off the next synthesis before awaiting the current send
            pending = (
                asyncio.create_task(_synthesize_segment(segments[i + 1], user_id))
                if i + 1 < len(segments)
                else None
            )
            audio = await current
            if audio is None:
                task.voice_sent += 1
                continue
            try:
                await _pace(chat_id)
                await bot.send_voice(
                    chat_id=chat_id,
                    voice=audio,
                    **_send_kwargs(task.thread_id),  # type: ignore[arg-type]
                )
            except Exception as e:
                if is_transient_error(e) and not _give_up_on_upload(task, e):
                    raise
                logger.error("Failed to send voice message to %d: %s", user_id, e)
                return
            task.voice_sent += 1
    finally:
        # Don't leak an in-flight prefetch if we bailed out early
        if pending is not None and not pending.done():
            pending.cancel()


async def _convert_status_to_content(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
    window_id: str,
    content_text: str,
) -> int | None:
    """Convert status message to content message by editing it.

    Returns the message_id if converted successfully, None otherwise.
    """
    skey = (user_id, thread_id_or_0)
    info = _status_msg_info.pop(skey, None)
    if not info:
        return None

    msg_id, stored_wid, _ = info
    chat_id = session_manager.resolve_chat_id(user_id, thread_id_or_0 or None)
    if stored_wid != window_id:
        # Different window, just delete the old status
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception:
            pass
        return None

    # Edit status message to show content
    try:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=_ensure_formatted(content_text),
                parse_mode=PARSE_MODE,
                link_preview_options=NO_LINK_PREVIEW,
            )
            return msg_id
        except Exception as e:
            if is_transient_error(e):
                raise
        # Fallback to plain text with sentinels stripped
        plain = strip_sentinels(content_text)
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text=plain,
            link_preview_options=NO_LINK_PREVIEW,
        )
        return msg_id
    except Exception as e:
        if is_transient_error(e):
            # The task will be retried: keep tracking the status message so
            # the retry converts it instead of leaving it behind as an orphan
            _status_msg_info[skey] = info
            raise
        logger.debug(f"Failed to convert status to content: {e}")
        # Message might be deleted or too old, caller will send new message
        return None


async def _process_status_update_task(
    bot: Bot, user_id: int, task: MessageTask
) -> None:
    """Process a status update task."""
    wid = task.window_id or ""
    tid = task.thread_id or 0
    chat_id = session_manager.resolve_chat_id(user_id, task.thread_id)
    skey = (user_id, tid)
    status_text = task.text or ""

    if not status_text:
        # No status text means clear status
        await _do_clear_status_message(bot, user_id, tid)
        return

    current_info = _status_msg_info.get(skey)
    if current_info and current_info[1] == wid and current_info[2] == status_text:
        return  # Same content, nothing to send
    if not _try_pace(chat_id):
        return  # Chat is busy; the next poll brings a fresher status anyway

    if current_info:
        msg_id, stored_wid, last_text = current_info

        if stored_wid != wid:
            # Window changed - delete old and send new
            await _do_clear_status_message(bot, user_id, tid)
            await _do_send_status_message(bot, user_id, tid, wid, status_text)
        elif status_text == last_text:
            # Same content, skip edit
            return
        else:
            # Same window, text changed - edit in place
            # Send typing indicator when Claude is working
            if "esc to interrupt" in status_text.lower():
                try:
                    await bot.send_chat_action(
                        chat_id=chat_id, action=ChatAction.TYPING
                    )
                except RetryAfter:
                    raise
                except Exception:
                    pass
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=_ensure_formatted(status_text),
                    parse_mode=PARSE_MODE,
                    link_preview_options=NO_LINK_PREVIEW,
                )
                _status_msg_info[skey] = (msg_id, wid, status_text)
            except RetryAfter:
                raise
            except Exception:
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text=status_text,
                        link_preview_options=NO_LINK_PREVIEW,
                    )
                    _status_msg_info[skey] = (msg_id, wid, status_text)
                except RetryAfter:
                    raise
                except Exception as e:
                    logger.debug(f"Failed to edit status message: {e}")
                    _status_msg_info.pop(skey, None)
                    await _do_send_status_message(bot, user_id, tid, wid, status_text)
    else:
        # No existing status message, send new
        await _do_send_status_message(bot, user_id, tid, wid, status_text)


async def _do_send_status_message(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int,
    window_id: str,
    text: str,
) -> None:
    """Send a new status message and track it (internal, called from worker)."""
    skey = (user_id, thread_id_or_0)
    thread_id: int | None = thread_id_or_0 if thread_id_or_0 != 0 else None
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    # Safety net: delete any orphaned status message before sending a new one.
    # This catches edge cases where tracking was cleared without deleting the message.
    old = _status_msg_info.pop(skey, None)
    if old:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=old[0])
        except Exception:
            pass
    # Send typing indicator when Claude is working
    if "esc to interrupt" in text.lower():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except RetryAfter:
            raise
        except Exception:
            pass
    sent = await send_with_fallback(
        bot,
        chat_id,
        text,
        **_send_kwargs(thread_id),  # type: ignore[arg-type]
    )
    if sent:
        _status_msg_info[skey] = (sent.message_id, window_id, text)


async def _do_clear_status_message(
    bot: Bot,
    user_id: int,
    thread_id_or_0: int = 0,
) -> None:
    """Delete the status message for a user (internal, called from worker)."""
    skey = (user_id, thread_id_or_0)
    info = _status_msg_info.pop(skey, None)
    if info:
        msg_id = info[0]
        chat_id = session_manager.resolve_chat_id(user_id, thread_id_or_0 or None)
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception as e:
            logger.debug(f"Failed to delete status message {msg_id}: {e}")


async def _check_and_send_status(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
) -> None:
    """Check terminal for status line and send status message if present."""
    # Skip if there are more messages pending in the queue
    queue = _message_queues.get(user_id)
    if queue and not queue.empty():
        return
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        return

    pane_text = await tmux_manager.capture_pane(w.window_id)
    if not pane_text:
        return

    tid = thread_id or 0
    status_line = parse_status_line(pane_text)
    if status_line and _try_pace(session_manager.resolve_chat_id(user_id, thread_id)):
        await _do_send_status_message(bot, user_id, tid, window_id, status_line)


async def enqueue_content_message(
    bot: Bot,
    user_id: int,
    window_id: str,
    parts: list[str],
    tool_use_id: str | None = None,
    content_type: str = "text",
    text: str | None = None,
    thread_id: int | None = None,
    image_data: list[tuple[str, bytes]] | None = None,
    tables: list[ParsedTable] | None = None,
    ticket: DeliveryTicket | None = None,
) -> None:
    """Enqueue a content message task.

    `ticket` is the source message's delivery receipt; the task holds a
    reference until it has been sent.
    """
    logger.debug(
        "Enqueue content: user=%d, window_id=%s, content_type=%s",
        user_id,
        window_id,
        content_type,
    )
    queue = get_or_create_queue(bot, user_id)

    task = MessageTask(
        task_type="content",
        text=text,
        window_id=window_id,
        parts=parts,
        tool_use_id=tool_use_id,
        content_type=content_type,
        thread_id=thread_id,
        image_data=image_data,
        tables=tables,
        tickets=[ticket.hold()] if ticket else [],
    )
    queue.put_nowait(task)


async def enqueue_table_message(
    bot: Bot,
    user_id: int,
    window_id: str,
    table: ParsedTable,
    thread_id: int | None = None,
    ticket: DeliveryTicket | None = None,
) -> None:
    """Enqueue one table as its own message, preserving its position."""
    logger.debug("Enqueue table: user=%d, window_id=%s", user_id, window_id)
    queue = get_or_create_queue(bot, user_id)
    queue.put_nowait(
        MessageTask(
            task_type="table",
            window_id=window_id,
            thread_id=thread_id,
            tables=[table],
            tickets=[ticket.hold()] if ticket else [],
        )
    )


async def enqueue_voice_message(
    bot: Bot,
    user_id: int,
    window_id: str,
    text: str,
    thread_id: int | None = None,
) -> None:
    """Enqueue a TTS voice reply of the final assistant message."""
    logger.debug("Enqueue voice: user=%d, window_id=%s", user_id, window_id)
    queue = get_or_create_queue(bot, user_id)
    queue.put_nowait(
        MessageTask(
            task_type="voice",
            text=text,
            window_id=window_id,
            thread_id=thread_id,
        )
    )


async def enqueue_status_update(
    bot: Bot,
    user_id: int,
    window_id: str,
    status_text: str | None,
    thread_id: int | None = None,
) -> None:
    """Enqueue status update. Skipped if text unchanged or during flood control."""
    # Don't enqueue during flood control — they'd just be dropped
    flood_end = _flood_until.get(user_id, 0)
    if flood_end > time.monotonic():
        return

    tid = thread_id or 0

    # Deduplicate: skip if text matches what's already displayed
    if status_text:
        skey = (user_id, tid)
        info = _status_msg_info.get(skey)
        if info and info[1] == window_id and info[2] == status_text:
            return

    queue = get_or_create_queue(bot, user_id)

    if status_text:
        task = MessageTask(
            task_type="status_update",
            text=status_text,
            window_id=window_id,
            thread_id=thread_id,
        )
    else:
        task = MessageTask(task_type="status_clear", thread_id=thread_id)

    queue.put_nowait(task)


def clear_status_msg_info(user_id: int, thread_id: int | None = None) -> None:
    """Clear status message tracking for a user (and optionally a specific thread)."""
    skey = (user_id, thread_id or 0)
    _status_msg_info.pop(skey, None)


def clear_tool_msg_ids_for_topic(user_id: int, thread_id: int | None = None) -> None:
    """Clear tool message ID tracking for a specific topic.

    Removes all entries in _tool_msg_ids that match the given user and thread.
    """
    tid = thread_id or 0
    # Find and remove all matching keys
    keys_to_remove = [
        key for key in _tool_msg_ids if key[1] == user_id and key[2] == tid
    ]
    for key in keys_to_remove:
        _tool_msg_ids.pop(key, None)


async def shutdown_workers() -> None:
    """Stop all queue workers (called during bot shutdown)."""
    for _, worker in list(_queue_workers.items()):
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    _queue_workers.clear()
    _message_queues.clear()
    _queue_locks.clear()
    logger.info("Message queue workers stopped")
