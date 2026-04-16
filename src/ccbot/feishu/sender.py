"""Feishu message sending helpers.

Provides utility functions for sending messages via Feishu API:
  - Text messages (plain and markdown)
  - Interactive card messages for rich formatting
  - Message editing and deletion
  - Image/file sending

All sends go through lark-oapi's synchronous client wrapped in
asyncio.to_thread() for non-blocking operation.

Key functions:
  - send_text: Send a plain text or markdown message
  - send_card: Send an interactive card message
  - edit_text: Edit an existing message
  - delete_message: Delete a message
  - reply_text: Reply to a specific message
"""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Feishu message length limit (characters). Post messages can be longer
# than text, but we use a conservative limit for card markdown elements.
FEISHU_MAX_MESSAGE_LENGTH = 30000


def _get_client() -> Any:
    """Get the shared Feishu client (lazy import to avoid circular deps)."""
    from .bot import get_lark_client

    return get_lark_client()


def _send_message_sync(
    receive_id: str,
    msg_type: str,
    content: str,
    receive_id_type: str = "chat_id",
) -> str | None:
    """Send a message synchronously. Returns message_id on success."""
    from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

    client = _get_client()
    if not client:
        logger.error("Feishu client not initialized")
        return None

    request = (
        CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(receive_id)
            .msg_type(msg_type)
            .content(content)
            .build()
        )
        .build()
    )

    response = client.im.v1.message.create(request)
    if not response.success():
        logger.error(
            "Failed to send message: code=%s, msg=%s", response.code, response.msg
        )
        return None

    msg_id: str = response.data.message_id if response.data else ""
    return msg_id or None


def _reply_message_sync(
    message_id: str,
    msg_type: str,
    content: str,
) -> str | None:
    """Reply to a message synchronously. Returns new message_id on success."""
    from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody

    client = _get_client()
    if not client:
        return None

    request = (
        ReplyMessageRequest.builder()
        .message_id(message_id)
        .request_body(
            ReplyMessageRequestBody.builder()
            .msg_type(msg_type)
            .content(content)
            .build()
        )
        .build()
    )

    response = client.im.v1.message.reply(request)
    if not response.success():
        logger.error("Failed to reply: code=%s, msg=%s", response.code, response.msg)
        return None

    new_id: str = response.data.message_id if response.data else ""
    return new_id or None


def _edit_message_sync(message_id: str, msg_type: str, content: str) -> bool:
    """Edit a message synchronously."""
    from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody

    client = _get_client()
    if not client:
        return False

    request = (
        PatchMessageRequest.builder()
        .message_id(message_id)
        .request_body(PatchMessageRequestBody.builder().content(content).build())
        .build()
    )

    response = client.im.v1.message.patch(request)
    if not response.success():
        logger.error(
            "Failed to edit message: code=%s, msg=%s", response.code, response.msg
        )
        return False
    return True


def _delete_message_sync(message_id: str) -> bool:
    """Delete a message synchronously."""
    from lark_oapi.api.im.v1 import DeleteMessageRequest

    client = _get_client()
    if not client:
        return False

    request = DeleteMessageRequest.builder().message_id(message_id).build()
    response = client.im.v1.message.delete(request)
    if not response.success():
        logger.debug(
            "Failed to delete message %s: code=%s, msg=%s",
            message_id,
            response.code,
            response.msg,
        )
        return False
    return True


def _build_text_content(text: str) -> str:
    """Build text message content JSON."""
    return json.dumps({"text": text})


def _build_post_content(text: str) -> str:
    """Build post (rich text) message content with markdown element."""
    return json.dumps(
        {
            "zh_cn": {
                "content": [
                    [{"tag": "md", "text": text}],
                ],
            },
        }
    )


def _build_markdown_card(text: str, title: str | None = None) -> str:
    """Build an interactive card with a markdown element."""
    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": text},
    ]

    card: dict[str, Any] = {
        "schema": "2.0",
        "body": {"elements": elements},
    }
    if title:
        card["header"] = {
            "title": {"tag": "plain_text", "content": title},
        }

    return json.dumps(card)


def split_message(text: str, max_length: int = FEISHU_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split long text into chunks for Feishu's limits.

    Similar to telegram_sender.split_message but with Feishu's limits.
    Tries to split on newlines and preserves code block integrity.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    current_chunk = ""
    in_code_block = False
    code_fence = ""

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_fence = stripped
            else:
                in_code_block = False

        if len(line) > max_length:
            if current_chunk:
                chunk_text = current_chunk.rstrip("\n")
                if in_code_block:
                    chunk_text += "\n```"
                chunks.append(chunk_text)
                current_chunk = (code_fence + "\n") if in_code_block else ""
            for i in range(0, len(line), max_length):
                chunks.append(line[i : i + max_length])
        elif len(current_chunk) + len(line) + 1 > max_length:
            chunk_text = current_chunk.rstrip("\n")
            if in_code_block:
                chunk_text += "\n```"
            chunks.append(chunk_text)
            if in_code_block:
                current_chunk = code_fence + "\n" + line + "\n"
            else:
                current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.rstrip("\n"))

    return chunks


async def send_text(
    chat_id: str,
    text: str,
    *,
    use_card: bool = False,
) -> str | None:
    """Send a text message to a Feishu chat.

    Args:
        chat_id: Feishu chat_id (group or p2p)
        text: Message text (markdown supported in cards/post)
        use_card: Use interactive card for rich markdown rendering

    Returns:
        message_id on success, None on failure
    """
    if use_card:
        content = _build_markdown_card(text)
        return await asyncio.to_thread(
            _send_message_sync, chat_id, "interactive", content
        )
    else:
        content = _build_post_content(text)
        return await asyncio.to_thread(_send_message_sync, chat_id, "post", content)


async def reply_text(
    message_id: str,
    text: str,
    *,
    use_card: bool = False,
) -> str | None:
    """Reply to a Feishu message.

    Args:
        message_id: The message to reply to
        text: Reply text
        use_card: Use interactive card

    Returns:
        New message_id on success, None on failure
    """
    if use_card:
        content = _build_markdown_card(text)
        return await asyncio.to_thread(
            _reply_message_sync, message_id, "interactive", content
        )
    else:
        content = _build_post_content(text)
        return await asyncio.to_thread(_reply_message_sync, message_id, "post", content)


async def edit_text(message_id: str, text: str) -> bool:
    """Edit an existing Feishu message.

    Note: Feishu only allows editing messages sent by the bot.
    Editing changes the content to post type with markdown.
    """
    content = _build_post_content(text)
    return await asyncio.to_thread(_edit_message_sync, message_id, "post", content)


async def delete_message(message_id: str) -> bool:
    """Delete a Feishu message."""
    return await asyncio.to_thread(_delete_message_sync, message_id)


async def send_image(
    chat_id: str,
    image_data: bytes,
) -> str | None:
    """Upload and send an image to a Feishu chat.

    Returns message_id on success, None on failure.
    """
    from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

    client = _get_client()
    if not client:
        return None

    import io

    def _upload_and_send() -> str | None:
        # Upload image first
        upload_req = (
            CreateImageRequest.builder()
            .request_body(
                CreateImageRequestBody.builder()
                .image_type("message")
                .image(io.BytesIO(image_data))
                .build()
            )
            .build()
        )
        upload_resp = client.im.v1.image.create(upload_req)
        if not upload_resp.success() or not upload_resp.data:
            logger.error(
                "Failed to upload image: code=%s, msg=%s",
                upload_resp.code,
                upload_resp.msg,
            )
            return None
        image_key = upload_resp.data.image_key
        if not image_key:
            return None

        content = json.dumps({"image_key": image_key})
        return _send_message_sync(chat_id, "image", content)

    return await asyncio.to_thread(_upload_and_send)
