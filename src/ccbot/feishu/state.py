"""Feishu-specific chat state tracking.

Manages Feishu-specific message state that parallels the Telegram handlers'
tracking (status messages, tool message IDs, interactive mode).

Since Feishu doesn't have forum topics like Telegram, the mapping is:
  - Each Feishu group chat acts as a single session context
  - Thread/reply chains within a group can optionally isolate sessions
  - Direct messages (P2P) each map to one session

State dicts are keyed by (user_open_id, chat_id).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal

from .sender import delete_message, edit_text, send_text

logger = logging.getLogger(__name__)


@dataclass
class FeishuMessageTask:
    """Message task for queue processing."""

    task_type: Literal["content", "status_update", "status_clear"]
    text: str | None = None
    window_id: str | None = None
    parts: list[str] = field(default_factory=list)
    tool_use_id: str | None = None
    content_type: str = "text"
    chat_id: str = ""
    image_data: list[tuple[str, bytes]] | None = None


# Per-user message queues and workers
_message_queues: dict[str, asyncio.Queue[FeishuMessageTask]] = {}
_queue_workers: dict[str, asyncio.Task[None]] = {}

# Status message tracking: chat_id -> (message_id, window_id, last_text)
_status_msg_info: dict[str, tuple[str, str, str]] = {}

# Tool message tracking: (tool_use_id, chat_id) -> feishu_message_id
_tool_msg_ids: dict[tuple[str, str], str] = {}

# Merge limit for content messages
MERGE_MAX_LENGTH = 10000


def get_message_queue(chat_id: str) -> asyncio.Queue[FeishuMessageTask] | None:
    """Get the message queue for a chat (if exists)."""
    return _message_queues.get(chat_id)


def get_or_create_queue(chat_id: str) -> asyncio.Queue[FeishuMessageTask]:
    """Get or create message queue and worker for a chat."""
    if chat_id not in _message_queues:
        _message_queues[chat_id] = asyncio.Queue()
        _queue_workers[chat_id] = asyncio.create_task(_message_queue_worker(chat_id))
    return _message_queues[chat_id]


async def _message_queue_worker(chat_id: str) -> None:
    """Process message tasks for a chat sequentially."""
    queue = _message_queues[chat_id]
    logger.info("Feishu message queue worker started for chat %s", chat_id)

    while True:
        try:
            task = await queue.get()
            try:
                if task.task_type == "content":
                    await _process_content_task(chat_id, task)
                elif task.task_type == "status_update":
                    await _process_status_update_task(chat_id, task)
                elif task.task_type == "status_clear":
                    await _do_clear_status_message(chat_id)
            except Exception as e:
                logger.error("Error processing Feishu task for chat %s: %s", chat_id, e)
            finally:
                queue.task_done()
        except asyncio.CancelledError:
            logger.info("Feishu queue worker cancelled for chat %s", chat_id)
            break
        except Exception as e:
            logger.error(
                "Unexpected error in Feishu queue worker for %s: %s", chat_id, e
            )


async def _process_content_task(chat_id: str, task: FeishuMessageTask) -> None:
    """Process a content message task."""
    wid = task.window_id or ""

    # Handle tool_result editing
    if task.content_type == "tool_result" and task.tool_use_id:
        tkey = (task.tool_use_id, chat_id)
        edit_msg_id = _tool_msg_ids.pop(tkey, None)
        if edit_msg_id is not None:
            await _do_clear_status_message(chat_id)
            full_text = "\n\n".join(task.parts)
            ok = await edit_text(edit_msg_id, full_text)
            if ok:
                if task.image_data:
                    await _send_task_images(chat_id, task)
                return

    # Send content messages
    last_msg_id: str | None = None
    first_part = True
    for part in task.parts:
        if first_part:
            first_part = False
            converted = await _convert_status_to_content(chat_id, wid, part)
            if converted:
                last_msg_id = converted
                continue

        msg_id = await send_text(chat_id, part)
        if msg_id:
            last_msg_id = msg_id

    # Record tool_use message ID for later editing
    if last_msg_id and task.tool_use_id and task.content_type == "tool_use":
        _tool_msg_ids[(task.tool_use_id, chat_id)] = last_msg_id

    # Send images if present
    if task.image_data:
        await _send_task_images(chat_id, task)


async def _send_task_images(chat_id: str, task: FeishuMessageTask) -> None:
    """Send images attached to a task."""
    if not task.image_data:
        return
    from .sender import send_image

    for _media_type, raw_bytes in task.image_data:
        await send_image(chat_id, raw_bytes)


async def _convert_status_to_content(
    chat_id: str,
    window_id: str,
    content_text: str,
) -> str | None:
    """Convert status message to content by editing it. Returns message_id if ok."""
    info = _status_msg_info.pop(chat_id, None)
    if not info:
        return None

    msg_id, stored_wid, _ = info
    if stored_wid != window_id:
        await delete_message(msg_id)
        return None

    ok = await edit_text(msg_id, content_text)
    return msg_id if ok else None


async def _process_status_update_task(chat_id: str, task: FeishuMessageTask) -> None:
    """Process a status update task."""
    wid = task.window_id or ""
    status_text = task.text or ""

    if not status_text:
        await _do_clear_status_message(chat_id)
        return

    current_info = _status_msg_info.get(chat_id)
    if current_info:
        msg_id, stored_wid, last_text = current_info
        if stored_wid != wid:
            await _do_clear_status_message(chat_id)
            await _do_send_status_message(chat_id, wid, status_text)
        elif status_text == last_text:
            return
        else:
            ok = await edit_text(msg_id, status_text)
            if ok:
                _status_msg_info[chat_id] = (msg_id, wid, status_text)
            else:
                _status_msg_info.pop(chat_id, None)
                await _do_send_status_message(chat_id, wid, status_text)
    else:
        await _do_send_status_message(chat_id, wid, status_text)


async def _do_send_status_message(
    chat_id: str,
    window_id: str,
    text: str,
) -> None:
    """Send a new status message and track it."""
    old = _status_msg_info.pop(chat_id, None)
    if old:
        await delete_message(old[0])

    msg_id = await send_text(chat_id, text)
    if msg_id:
        _status_msg_info[chat_id] = (msg_id, window_id, text)


async def _do_clear_status_message(chat_id: str) -> None:
    """Delete and clear the status message for a chat."""
    info = _status_msg_info.pop(chat_id, None)
    if info:
        await delete_message(info[0])


async def enqueue_content_message(
    chat_id: str,
    window_id: str,
    parts: list[str],
    tool_use_id: str | None = None,
    content_type: str = "text",
    text: str | None = None,
    image_data: list[tuple[str, bytes]] | None = None,
) -> None:
    """Enqueue a content message task."""
    queue = get_or_create_queue(chat_id)
    task = FeishuMessageTask(
        task_type="content",
        text=text,
        window_id=window_id,
        parts=parts,
        tool_use_id=tool_use_id,
        content_type=content_type,
        chat_id=chat_id,
        image_data=image_data,
    )
    queue.put_nowait(task)


async def enqueue_status_update(
    chat_id: str,
    window_id: str,
    status_text: str | None,
) -> None:
    """Enqueue a status update."""
    if status_text:
        info = _status_msg_info.get(chat_id)
        if info and info[1] == window_id and info[2] == status_text:
            return

    queue = get_or_create_queue(chat_id)
    if status_text:
        task = FeishuMessageTask(
            task_type="status_update",
            text=status_text,
            window_id=window_id,
            chat_id=chat_id,
        )
    else:
        task = FeishuMessageTask(task_type="status_clear", chat_id=chat_id)
    queue.put_nowait(task)


def clear_status_msg_info(chat_id: str) -> None:
    """Clear status message tracking for a chat."""
    _status_msg_info.pop(chat_id, None)


async def shutdown_workers() -> None:
    """Stop all queue workers."""
    for _, worker in list(_queue_workers.items()):
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    _queue_workers.clear()
    _message_queues.clear()
    logger.info("Feishu message queue workers stopped")
