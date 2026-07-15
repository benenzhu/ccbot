"""Unit tests for message_queue — voice task retry behavior."""

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
        patch.object(message_queue, "synthesize_speech", synth),
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
        patch.object(message_queue, "synthesize_speech", synth),
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
        patch.object(message_queue, "synthesize_speech", synth),
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
    with patch.object(message_queue, "synthesize_speech", synth):
        await _process_voice_task(bot, 1, _voice_task(""))

    synth.assert_not_called()
    bot.send_voice.assert_not_called()
