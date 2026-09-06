"""Unit tests for message_queue — voice task retry and segmentation behavior."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ccbot.handlers import message_queue
from ccbot.handlers.message_queue import MessageTask, _process_voice_task
from ccbot.tts import TtsApiError


def _voice_task(text: str = "你好") -> MessageTask:
    return MessageTask(task_type="voice", text=text, window_id="@1", thread_id=42)


@pytest.mark.asyncio
async def test_voice_transient_error_retried_once():
    bot = AsyncMock()
    synth = AsyncMock(side_effect=[TtsApiError("stream reset"), b"OggS-audio"])
    with (
        patch.object(message_queue, "synthesize_prepared", synth),
        patch.object(message_queue, "prepare_tts_segments", return_value=["你好"]),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock),
    ):
        await _process_voice_task(bot, 1, _voice_task())

    assert synth.call_count == 2
    bot.send_voice.assert_called_once()
    assert bot.send_voice.call_args.kwargs["voice"] == b"OggS-audio"


@pytest.mark.asyncio
async def test_voice_persistent_error_gives_up_after_retry():
    bot = AsyncMock()
    synth = AsyncMock(side_effect=TtsApiError("still broken"))
    with (
        patch.object(message_queue, "synthesize_prepared", synth),
        patch.object(message_queue, "prepare_tts_segments", return_value=["你好"]),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock),
    ):
        await _process_voice_task(bot, 1, _voice_task())

    assert synth.call_count == 2
    bot.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_unspeakable_text_not_retried():
    bot = AsyncMock()
    synth = AsyncMock(side_effect=ValueError("No speakable text"))
    with (
        patch.object(message_queue, "synthesize_prepared", synth),
        patch.object(message_queue, "prepare_tts_segments", return_value=["x"]),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
    ):
        await _process_voice_task(bot, 1, _voice_task("```code```"))

    assert synth.call_count == 1
    bot.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_empty_text_skipped():
    bot = AsyncMock()
    synth = AsyncMock()
    with patch.object(message_queue, "synthesize_prepared", synth):
        await _process_voice_task(bot, 1, _voice_task(""))

    synth.assert_not_called()
    bot.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_nothing_speakable_skipped():
    """prepare_tts_segments returning [] means the reply was all code."""
    bot = AsyncMock()
    synth = AsyncMock()
    with (
        patch.object(message_queue, "synthesize_prepared", synth),
        patch.object(message_queue, "prepare_tts_segments", return_value=[]),
    ):
        await _process_voice_task(bot, 1, _voice_task("```code```"))

    synth.assert_not_called()
    bot.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_segments_sent_in_order():
    bot = AsyncMock()
    segments = ["第一段。", "第二段。", "第三段。"]

    async def synth(seg: str) -> bytes:
        # Later segments finish faster — ordering must not follow completion
        await asyncio.sleep(0.03 if seg == segments[0] else 0.001)
        return f"audio:{seg}".encode()

    with (
        patch.object(
            message_queue, "synthesize_prepared", AsyncMock(side_effect=synth)
        ),
        patch.object(message_queue, "prepare_tts_segments", return_value=segments),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
    ):
        await _process_voice_task(bot, 1, _voice_task("long reply"))

    sent = [c.kwargs["voice"] for c in bot.send_voice.call_args_list]
    assert sent == [f"audio:{s}".encode() for s in segments]


@pytest.mark.asyncio
async def test_voice_failed_segment_skipped_rest_delivered():
    bot = AsyncMock()
    segments = ["一。", "二。", "三。"]

    async def synth(seg: str) -> bytes:
        if seg == "二。":
            raise ValueError("No speakable text")
        return f"audio:{seg}".encode()

    with (
        patch.object(
            message_queue, "synthesize_prepared", AsyncMock(side_effect=synth)
        ),
        patch.object(message_queue, "prepare_tts_segments", return_value=segments),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
    ):
        await _process_voice_task(bot, 1, _voice_task("reply"))

    sent = [c.kwargs["voice"] for c in bot.send_voice.call_args_list]
    assert sent == ["audio:一。".encode(), "audio:三。".encode()]


@pytest.mark.asyncio
async def test_voice_prefetch_overlaps_synthesis():
    """Segment N+1 synthesizes while N is sending, beating serial timing."""
    bot = AsyncMock()
    segments = ["一。", "二。", "三。", "四。"]
    delay = 0.05

    async def synth(seg: str) -> bytes:
        await asyncio.sleep(delay)
        return b"audio"

    with (
        patch.object(
            message_queue, "synthesize_prepared", AsyncMock(side_effect=synth)
        ),
        patch.object(message_queue, "prepare_tts_segments", return_value=segments),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
    ):
        loop = asyncio.get_running_loop()
        start = loop.time()
        await _process_voice_task(bot, 1, _voice_task("long"))
        elapsed = loop.time() - start

    assert bot.send_voice.call_count == len(segments)
    # Serial synthesis alone would cost len(segments) * delay
    assert elapsed < len(segments) * delay


# --- Table attachments: native rich message with PNG fallback ---


def _table_task() -> MessageTask:
    return MessageTask(
        task_type="content",
        window_id="@1",
        thread_id=42,
        parts=["text"],
        tables=[(["h"], [["v"]])],
    )


async def test_tables_sent_natively_when_rich_send_succeeds():
    bot = AsyncMock()
    with (
        patch.object(message_queue.config, "native_tables", True),
        patch.object(
            message_queue, "send_rich_markdown", AsyncMock(return_value=True)
        ) as rich,
        patch.object(message_queue, "send_photo", AsyncMock()) as photo,
        patch.object(message_queue, "render_table_image", AsyncMock()) as render,
    ):
        await message_queue._send_task_tables(bot, 777, _table_task())

    rich.assert_awaited_once()
    args, kwargs = rich.await_args
    assert args[:2] == (bot, 777)
    assert args[2].startswith("| h |")
    assert kwargs == {"message_thread_id": 42}
    render.assert_not_awaited()
    photo.assert_not_awaited()


async def test_tables_fall_back_to_png_when_rich_send_fails():
    bot = AsyncMock()
    with (
        patch.object(message_queue.config, "native_tables", True),
        patch.object(
            message_queue, "send_rich_markdown", AsyncMock(return_value=False)
        ),
        patch.object(message_queue, "send_photo", AsyncMock()) as photo,
        patch.object(
            message_queue, "render_table_image", AsyncMock(return_value=b"png")
        ) as render,
    ):
        await message_queue._send_task_tables(bot, 777, _table_task())

    render.assert_awaited_once_with(["h"], [["v"]])
    photo.assert_awaited_once()
    args, kwargs = photo.await_args
    assert args[:2] == (bot, 777)
    assert args[2] == [("table.png", b"png")]
    assert kwargs == {"message_thread_id": 42}


async def test_tables_go_straight_to_png_when_native_disabled():
    bot = AsyncMock()
    with (
        patch.object(message_queue.config, "native_tables", False),
        patch.object(message_queue, "send_rich_markdown", AsyncMock()) as rich,
        patch.object(message_queue, "send_photo", AsyncMock()) as photo,
        patch.object(
            message_queue, "render_table_image", AsyncMock(return_value=b"png")
        ),
    ):
        await message_queue._send_task_tables(bot, 777, _table_task())

    rich.assert_not_awaited()
    photo.assert_awaited_once()


async def test_merge_carries_tables_and_images_from_all_tasks():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    first = MessageTask(
        task_type="content",
        window_id="@1",
        parts=["a"],
        image_data=[("img", b"1")],
    )
    second = MessageTask(
        task_type="content",
        window_id="@1",
        parts=["b"],
        tables=[(["h"], [["v"]])],
        image_data=[("img", b"2")],
    )
    queue.put_nowait(second)

    merged, count = await message_queue._merge_content_tasks(
        queue, first, asyncio.Lock()
    )

    assert count == 1
    assert merged.parts == ["a", "b"]
    assert merged.image_data == [("img", b"1"), ("img", b"2")]
    assert merged.tables == [(["h"], [["v"]])]
