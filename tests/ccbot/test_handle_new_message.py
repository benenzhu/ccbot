"""Tests for handle_new_message — how monitor output becomes queued sends."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.bot import handle_new_message
from ccbot.session_monitor import NewMessage
from ccbot.transcript_parser import TranscriptParser

EXP_START = TranscriptParser.EXPANDABLE_QUOTE_START
EXP_END = TranscriptParser.EXPANDABLE_QUOTE_END
TABLE = "| op | ms |\n|---|---|\n| gemm | 1.5 |"


@contextmanager
def _routed() -> Iterator[tuple[AsyncMock, AsyncMock]]:
    """Route every message to user 5, topic 42; yields (content, table) enqueues."""
    sm = MagicMock()
    sm.find_users_for_session = AsyncMock(return_value=[(5, "@1", 42)])
    content, table = AsyncMock(), AsyncMock()
    with (
        patch("ccbot.bot.session_manager", sm),
        patch("ccbot.bot.get_interactive_msg_id", return_value=None),
        patch("ccbot.bot.enqueue_content_message", content),
        patch("ccbot.bot.enqueue_table_message", table),
    ):
        yield content, table


@pytest.mark.parametrize(
    "content_type, text, shown",
    [
        pytest.param(
            "tool_result",
            f"**Bash**(cat README.md)\n  ⎿  Output 3 lines\n{EXP_START}{TABLE}{EXP_END}",
            "| gemm | 1.5 |",  # verbatim inside the collapsed output
            id="tool_result",
        ),
        pytest.param(
            "tool_use", f"**Write**(notes.md)\n{TABLE}", "**op**: gemm", id="tool_use"
        ),
    ],
)
async def test_tool_call_tables_stay_inline(content_type: str, text: str, shown: str):
    msg = NewMessage(
        session_id="s", text=text, is_complete=True, content_type=content_type
    )
    with _routed() as (content, table):
        await handle_new_message(msg, AsyncMock())

    table.assert_not_awaited()
    kwargs = content.await_args.kwargs
    assert kwargs["tables"] is None
    assert shown in "".join(kwargs["parts"])


async def test_assistant_prose_tables_still_go_out_as_tables():
    msg = NewMessage(session_id="s", text=f"Results:\n{TABLE}", is_complete=True)
    with _routed() as (content, table):
        await handle_new_message(msg, AsyncMock())

    table.assert_awaited_once()
    assert table.await_args.args[3] == (["op", "ms"], [["gemm", "1.5"]])
    content.assert_awaited_once()


async def test_a_subagent_message_goes_to_the_subagent_topic():
    msg = NewMessage(
        session_id="s",
        text="Job 7 finished.",
        is_complete=True,
        agent_id="a1",
        agent_description="Watch job 7",
    )
    topics = AsyncMock(return_value=[(5, "agent:a1", 99)])
    with (
        _routed() as (content, _),
        patch("ccbot.bot.topics_for_message", topics),
        patch("ccbot.bot.enqueue_voice_message") as voice,
        patch("ccbot.bot.config.tts_enabled", True),
    ):
        await handle_new_message(msg, AsyncMock())

    topics.assert_awaited_once()
    kwargs = content.await_args.kwargs
    assert (kwargs["user_id"], kwargs["window_id"], kwargs["thread_id"]) == (
        5,
        "agent:a1",
        99,
    )
    voice.assert_not_called()  # only the main session is read aloud


async def test_messages_in_a_subagent_topic_are_refused():
    from ccbot.bot import SUBAGENT_TOPIC_REPLY, _refuse_in_subagent_topic

    message = MagicMock()
    sub = MagicMock(parent_name="ccbot dev")
    with (
        patch("ccbot.bot.subagent_topic", return_value=sub),
        patch("ccbot.bot.safe_reply", AsyncMock()) as reply,
    ):
        assert await _refuse_in_subagent_topic(message, 5, 99)
    reply.assert_awaited_once_with(
        message, SUBAGENT_TOPIC_REPLY.format(parent="ccbot dev")
    )

    with patch("ccbot.bot.subagent_topic", return_value=None):
        assert not await _refuse_in_subagent_topic(message, 5, 42)
