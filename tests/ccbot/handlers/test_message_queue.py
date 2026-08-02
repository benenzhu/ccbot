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
