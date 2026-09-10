"""Session naming, fork launch, topic routing, and inherited-history regression tests."""

import json
import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import CallbackQuery, User

from ccbot import bot
from ccbot.config import config
from ccbot.handlers.callback_data import (
    CB_SESSION_FORK,
    CB_SESSION_REPLAY,
    CB_SESSION_SELECT,
)
from ccbot.handlers.directory_browser import (
    REPLAY_KEY,
    SESSIONS_KEY,
    build_session_picker,
)
from ccbot.monitor_state import TrackedSession
from ccbot.session import ClaudeSession, SessionManager
from ccbot.session_monitor import SessionInfo, SessionMonitor
from ccbot.tmux_manager import TmuxManager


def write_entries(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ([], "last user message"),
        ([{"type": "summary", "summary": "generated summary"}], "generated summary"),
        (
            [
                {"type": "custom-title", "customTitle": "old name"},
                {"type": "custom-title", "customTitle": "m3_1", "sessionId": "source"},
                {"type": "summary", "summary": "generated summary"},
                {
                    "type": "custom-title",
                    "customTitle": "foreign",
                    "sessionId": "other",
                },
            ],
            "m3_1",
        ),
    ],
)
async def test_picker_uses_latest_session_name(
    tmp_path, metadata, expected, monkeypatch
):
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    mgr = SessionManager()
    path = tmp_path / "source.jsonl"
    write_entries(
        path, [{"type": "user", "message": {"content": "last user message"}}, *metadata]
    )
    monkeypatch.setattr(mgr, "_build_session_file_path", lambda *_: path)

    session = await mgr._get_session_direct("source", str(tmp_path))
    assert session is not None
    assert session.display_name == expected
    text, keyboard = build_session_picker([session])
    assert f"`{expected}`" in text
    toggle = keyboard.inline_keyboard[0][0]
    assert toggle.callback_data == CB_SESSION_REPLAY
    assert toggle.text.endswith("OFF")
    resume, fork = keyboard.inline_keyboard[1]
    assert resume.callback_data == f"{CB_SESSION_SELECT}0"
    assert fork.callback_data == f"{CB_SESSION_FORK}0"
    if expected == "m3_1":
        assert "m3_1" in resume.text


@pytest.mark.parametrize("fork", [False, True])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("replay", [False, True])
async def test_session_selection_routes_action(tmp_path, fork, stale, replay):
    update = MagicMock()
    update.effective_user.id = 12345
    update.effective_chat = None
    prefix = CB_SESSION_FORK if fork else CB_SESSION_SELECT
    update.callback_query.data = f"{prefix}0"
    update.callback_query.answer = AsyncMock()
    context = MagicMock()
    context.user_data = {
        "_pending_thread_id": 42,
        "_selected_path": str(tmp_path),
        SESSIONS_KEY: [ClaudeSession("source", "summary", 1, "source.jsonl")],
        REPLAY_KEY: False,
    }
    with (
        patch.object(bot, "_get_thread_id", return_value=99 if stale else 42),
        patch.object(bot, "_create_and_bind_window", new_callable=AsyncMock) as create,
        patch.object(bot, "safe_edit", new_callable=AsyncMock),
    ):
        if replay:
            update.callback_query.data = CB_SESSION_REPLAY
            await bot.callback_handler(update, context)
            assert context.user_data[REPLAY_KEY] is (not stale)
            update.callback_query.data = f"{prefix}0"
        await bot.callback_handler(update, context)
    if stale:
        create.assert_not_awaited()
        assert SESSIONS_KEY in context.user_data
    else:
        create.assert_awaited_once_with(
            update.callback_query,
            context,
            update.effective_user,
            str(tmp_path),
            42,
            resume_session_id="source",
            resume_file_path="source.jsonl",
            replay_history=replay,
            fork_session=fork,
        )
        assert SESSIONS_KEY not in context.user_data


@pytest.mark.parametrize("fork", [False, True])
async def test_launch_preserves_configured_command_and_quotes_session(
    tmp_path, monkeypatch, fork
):
    mgr = TmuxManager()
    tmux_session = MagicMock()
    window = tmux_session.new_window.return_value
    window.window_id = "@7"
    monkeypatch.setattr(mgr, "find_window_by_name", AsyncMock(return_value=None))
    monkeypatch.setattr(mgr, "get_or_create_session", lambda: tmux_session)
    monkeypatch.setattr(
        config, "claude_command", "IS_SANDBOX=1 claude --dangerously-skip-permissions"
    )
    source = "m3_1; $(touch should-not-run)"
    success, _, _, _ = await mgr.create_window(
        str(tmp_path),
        resume_session_id=source,
        fork_session_id="fork-id" if fork else None,
    )
    assert success
    command = window.active_pane.send_keys.call_args.args[0]
    expected = [
        "IS_SANDBOX=1",
        "claude",
        "--dangerously-skip-permissions",
        "--resume",
        source,
    ]
    if fork:
        expected += ["--fork-session", "--session-id", "fork-id"]
    assert shlex.split(command) == expected


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("hook_ok", [False, True])
async def test_fork_binds_new_id_and_preserves_source(
    tmp_path, monkeypatch, replay, hook_ok
):
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    mgr = SessionManager()
    monitor = SessionMonitor(state_file=tmp_path / "monitor.json")
    source = tmp_path / "source.jsonl"
    write_entries(
        source,
        [{"uuid": "old", "type": "assistant", "message": {"content": "history"}}],
    )
    original = source.read_bytes()
    monitor.state.update_session(
        TrackedSession("source", str(source), last_byte_offset=17)
    )
    mgr.get_window_state("@1").session_id = "source"
    mgr.bind_thread(12345, 41, "@1")
    query = MagicMock(spec=CallbackQuery)
    query.answer = AsyncMock()
    user = User(12345, "Test", False)
    context = MagicMock()
    context.bot = AsyncMock()
    context.user_data = {
        "_pending_thread_id": 42,
        "_pending_thread_text": "next prompt",
    }
    target_id = "c0d4a4d3-2308-497a-8ea2-c742bb5b1ab9"

    async def create(*_args, **kwargs):
        assert kwargs == {"resume_session_id": "source", "fork_session_id": target_id}
        tracked = monitor.state.get_session(target_id)
        assert tracked is not None and tracked.last_byte_offset == 0
        assert Path(tracked.file_path).name == f"{target_id}.jsonl"
        assert monitor.state.get_session("source").last_byte_offset == 17
        return True, "Window created", "project-2", "@2"

    async def hook(*_args, **_kwargs):
        assert mgr.get_window_for_thread(12345, 42) == "@2"
        assert mgr.get_window_state("@2").session_id == target_id
        return hook_ok

    monkeypatch.setattr(bot, "session_manager", mgr)
    monkeypatch.setattr(bot, "session_monitor", monitor)
    monkeypatch.setattr(bot, "uuid4", lambda: target_id)
    monkeypatch.setattr(
        bot.tmux_manager, "create_window", AsyncMock(side_effect=create)
    )
    monkeypatch.setattr(mgr, "wait_for_session_map_entry", AsyncMock(side_effect=hook))
    send = AsyncMock(return_value=(True, "sent"))
    monkeypatch.setattr(mgr, "send_to_window", send)
    monkeypatch.setattr(bot, "safe_edit", AsyncMock())
    await bot._create_and_bind_window(
        query,
        context,
        user,
        str(tmp_path),
        42,
        resume_session_id="source",
        resume_file_path=str(source),
        replay_history=replay,
        fork_session=True,
    )
    assert mgr.get_window_state("@1").session_id == "source"
    assert mgr.get_window_state("@2").session_id == target_id
    assert source.read_bytes() == original
    assert (target_id in monitor._replay_sessions) == replay
    if replay:
        send.assert_not_awaited()
    else:
        send.assert_awaited_once_with("@2", "next prompt")
    assert not context.user_data


async def test_fork_skips_rewritten_history_across_polls(tmp_path, monkeypatch):
    monitor = SessionMonitor(state_file=tmp_path / "monitor.json")
    source = tmp_path / "source.jsonl"
    old = [
        {
            "uuid": f"old-{i}",
            "sessionId": "source",
            "type": "assistant",
            "message": {"content": f"old {i}"},
        }
        for i in range(2)
    ]
    write_entries(source, old)
    await monitor.skip_fork_history("fork", str(source))
    target = tmp_path / "fork.jsonl"
    monitor.state.update_session(
        TrackedSession("fork", str(target), last_byte_offset=0)
    )
    monkeypatch.setattr(
        monitor, "scan_projects", AsyncMock(return_value=[SessionInfo("fork", target)])
    )
    # Claude rewrites the transcript; its byte layout differs from the source.
    inherited = [
        dict(entry, sessionId="fork", extraMetadata="changed bytes") for entry in old
    ]
    write_entries(target, inherited[:1])
    assert await monitor.check_for_updates({"fork"}) == []
    assert "fork" in monitor._fork_history_uuids
    new = {
        "uuid": "new",
        "type": "assistant",
        "message": {"content": "first new answer"},
    }
    write_entries(target, [*inherited, new])
    messages = await monitor.check_for_updates({"fork"})
    assert [m.text for m in messages] == ["first new answer"]
    assert not messages[0].is_replay
    assert "fork" not in monitor._fork_history_uuids
    assert await monitor.check_for_updates({"fork"}) == []
