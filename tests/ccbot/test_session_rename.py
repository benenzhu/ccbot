"""Tests for handle_session_renamed — Claude's /rename names window and topic."""

from unittest.mock import AsyncMock, MagicMock, patch

from ccbot.bot import handle_session_renamed


def _patches(display_name: str, bindings: list[tuple[int, str, int]]):
    sm = MagicMock()
    sm.find_users_for_session = AsyncMock(return_value=bindings)
    sm.get_display_name.return_value = display_name
    sm.resolve_chat_id.return_value = 777
    tmux = MagicMock()
    tmux.rename_window = AsyncMock()
    return (
        patch("ccbot.bot.session_manager", sm),
        patch("ccbot.bot.tmux_manager", tmux),
        sm,
        tmux,
    )


async def test_new_title_renames_window_and_topic():
    bot = AsyncMock()
    p_sm, p_tmux, sm, tmux = _patches("taoyuzhu-2", [(5, "@4", 84921)])
    with p_sm, p_tmux:
        await handle_session_renamed("sid", "ccbot_rename", bot)

    tmux.rename_window.assert_awaited_once_with("@4", "ccbot_rename")
    sm.update_display_name.assert_called_once_with("@4", "ccbot_rename")
    bot.edit_forum_topic.assert_awaited_once_with(
        chat_id=777, message_thread_id=84921, name="ccbot_rename"
    )


async def test_unchanged_title_makes_no_requests():
    bot = AsyncMock()
    p_sm, p_tmux, _sm, tmux = _patches("ccbot_rename", [(5, "@4", 84921)])
    with p_sm, p_tmux:
        await handle_session_renamed("sid", "ccbot_rename", bot)

    tmux.rename_window.assert_not_awaited()
    bot.edit_forum_topic.assert_not_awaited()


async def test_long_title_is_cut_to_the_topic_name_limit():
    bot = AsyncMock()
    p_sm, p_tmux, _sm, tmux = _patches("old", [(5, "@4", 84921)])
    with p_sm, p_tmux:
        await handle_session_renamed("sid", "x" * 200, bot)

    assert bot.edit_forum_topic.await_args.kwargs["name"] == "x" * 128
    tmux.rename_window.assert_awaited_once_with("@4", "x" * 128)


async def test_failed_topic_rename_still_renames_the_window():
    bot = AsyncMock()
    bot.edit_forum_topic.side_effect = Exception("TOPIC_NOT_MODIFIED")
    p_sm, p_tmux, _sm, tmux = _patches("old", [(5, "@4", 84921)])
    with p_sm, p_tmux:
        await handle_session_renamed("sid", "new", bot)

    tmux.rename_window.assert_awaited_once_with("@4", "new")
