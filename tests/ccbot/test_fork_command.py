"""Tests for /fork — fork a topic's session into a new topic with a task."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot import bot
from ccbot.bot import FORK_TASK_PROMPT, _parse_fork_command, fork_command


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            "/fork moe-profile\n看一下 gemm2\n为什么慢",
            ("moe-profile", "看一下 gemm2\n为什么慢"),
            id="name_and_task",
        ),
        pytest.param("/fork@ccbot  two  words ", ("two words", ""), id="mention"),
        pytest.param("/fork\njust a task", ("", "just a task"), id="no_name"),
    ],
)
def test_parse_fork_command(text: str, expected: tuple[str, str]):
    assert _parse_fork_command(text) == expected


def _update(text: str) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = 5
    update.message.text = text
    update.effective_chat.type = "private"
    return update


def _context() -> MagicMock:
    context = MagicMock()
    context.bot = AsyncMock()
    context.bot.create_forum_topic.return_value = MagicMock(message_thread_id=99)
    return context


def _session_manager() -> MagicMock:
    sm = MagicMock()
    sm.resolve_window_for_thread.return_value = "@4"
    sm.get_window_state.return_value = MagicMock(session_id="src-sid", cwd="/w")
    sm.session_file_for_window.return_value = Path("/p/src-sid.jsonl")
    sm.resolve_chat_id.return_value = 777
    sm.get_display_name.return_value = "attention"
    sm.send_to_window = AsyncMock(return_value=(True, ""))
    return sm


@contextmanager
def _patched(
    sm: MagicMock, launch: AsyncMock, ready: bool = True
) -> Iterator[tuple[MagicMock, AsyncMock, AsyncMock]]:
    """Everything fork_command touches outside itself.

    Yields (tmux, safe_reply, safe_send); ready is whether Claude's input box
    comes up.
    """
    tmux = MagicMock()
    tmux.rename_window = AsyncMock()
    reply = AsyncMock()
    send = AsyncMock()
    with (
        patch("ccbot.bot.is_user_allowed", return_value=True),
        patch("ccbot.bot._get_thread_id", return_value=42),
        patch("ccbot.bot.session_manager", sm),
        patch("ccbot.bot.tmux_manager", tmux),
        patch("ccbot.bot._launch_session_window", launch),
        patch("ccbot.bot._await_session_start", AsyncMock()),
        patch("ccbot.bot._wait_for_input", AsyncMock(return_value=ready)),
        patch("ccbot.bot.safe_reply", reply),
        patch("ccbot.bot.safe_send", send),
        patch("ccbot.bot.asyncio.sleep", AsyncMock()),
    ):
        yield tmux, reply, send


async def _finished() -> None:
    """Let the forks' background halves run."""
    await asyncio.gather(*bot._fork_tasks)


async def test_fork_opens_named_topic_and_hands_over_the_task():
    sm = _session_manager()
    context = _context()
    launch = AsyncMock(return_value=(True, "ok", "w", "@9", "fork-sid"))
    with _patched(sm, launch) as (tmux, _reply, _send):
        await fork_command(_update("/fork moe-profile\n看一下 gemm2"), context)
        await _finished()

    context.bot.create_forum_topic.assert_awaited_once_with(
        chat_id=777, name="moe-profile"
    )
    launch.assert_awaited_once_with(
        "/w",
        resume_session_id="src-sid",
        resume_file_path="/p/src-sid.jsonl",
        fork_session=True,
    )
    sm.bind_thread.assert_called_once_with(5, 99, "@9", window_name="moe-profile")
    tmux.rename_window.assert_awaited_once_with("@9", "moe-profile")
    sent = [call.args for call in sm.send_to_window.await_args_list]
    assert sent == [
        ("@9", "/rename moe-profile"),
        ("@9", FORK_TASK_PROMPT.format(task="看一下 gemm2")),
    ]


async def test_fork_without_a_name_only_explains_usage():
    sm = _session_manager()
    context = _context()
    launch = AsyncMock()
    with _patched(sm, launch) as (_tmux, reply, _send):
        await fork_command(_update("/fork"), context)

    assert "Usage" in reply.await_args.args[1]
    context.bot.create_forum_topic.assert_not_awaited()
    launch.assert_not_awaited()


async def test_failed_launch_removes_the_new_topic():
    sm = _session_manager()
    context = _context()
    launch = AsyncMock(return_value=(False, "tmux error", "", "", None))
    with _patched(sm, launch) as (_tmux, reply, _send):
        await fork_command(_update("/fork x\ntask"), context)

    context.bot.delete_forum_topic.assert_awaited_once_with(
        chat_id=777, message_thread_id=99
    )
    sm.bind_thread.assert_not_called()
    sm.send_to_window.assert_not_awaited()
    assert "tmux error" in reply.await_args.args[1]


async def test_fork_types_nothing_while_claude_is_not_ready():
    sm = _session_manager()
    context = _context()
    launch = AsyncMock(return_value=(True, "ok", "w", "@9", "fork-sid"))
    with _patched(sm, launch, ready=False) as (_tmux, _reply, send):
        await fork_command(_update("/fork moe-profile\n看一下 gemm2"), context)
        await _finished()

    sm.send_to_window.assert_not_awaited()
    notice = send.await_args
    assert notice.kwargs["message_thread_id"] == 99
    assert "/rename moe-profile and the task not sent" in notice.args[2]
