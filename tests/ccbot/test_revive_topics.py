"""Tests for _revive_topics — resuming topics whose windows died with tmux."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ccbot.bot import _revive_topics
from ccbot.session import DeadBinding


async def test_dead_topics_resume_their_sessions_in_new_windows():
    sm = MagicMock()
    sm._locate_session_file.side_effect = lambda sid, cwd: (
        Path(f"/p/{sid}.jsonl") if sid != "gone" else None
    )
    launch = AsyncMock(return_value=(True, "ok", "w", "@7", "s15"))
    start = AsyncMock()
    tmux = MagicMock(rename_window=AsyncMock())
    with (
        patch("ccbot.bot.session_manager", sm),
        patch("ccbot.bot.tmux_manager", tmux),
        patch("ccbot.bot._launch_session_window", launch),
        patch("ccbot.bot._await_session_start", start),
    ):
        await _revive_topics(
            [
                DeadBinding(5, 42, "vllm#713", "s15", "/w"),
                DeadBinding(5, 43, "old", "gone", "/w"),  # transcript deleted
            ]
        )

    launch.assert_awaited_once_with(
        "/w", resume_session_id="s15", resume_file_path="/p/s15.jsonl"
    )
    tmux.rename_window.assert_awaited_once_with("@7", "vllm#713")
    sm.bind_thread.assert_called_once_with(5, 42, "@7", window_name="vllm#713")
    start.assert_awaited_once_with("@7", "s15", resumed=True)
