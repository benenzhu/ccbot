"""Tests for the /screenshot control keys."""

from unittest.mock import AsyncMock, MagicMock, call, patch

from ccbot import bot
from ccbot.handlers.callback_data import CB_KEYS_PREFIX


def test_keyboard_offers_ctrl_b_twice_instead_of_tab():
    keyboard = bot._build_screenshot_keyboard("@1")
    buttons = {b.text: b.callback_data for row in keyboard.inline_keyboard for b in row}
    assert buttons["^B ^B"] == f"{CB_KEYS_PREFIX}bg:@1"
    assert "⇥ Tab" not in buttons


async def test_ctrl_b_is_sent_twice_to_background_the_tool_call():
    update = MagicMock()
    update.effective_user.id = 12345
    update.effective_chat = None
    update.callback_query.data = f"{CB_KEYS_PREFIX}bg:@1"
    update.callback_query.answer = AsyncMock()
    tmux = MagicMock()
    tmux.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@1"))
    tmux.send_keys = AsyncMock()
    tmux.capture_pane = AsyncMock(return_value="")
    with (
        patch.object(bot, "tmux_manager", tmux),
        patch.object(bot.asyncio, "sleep", AsyncMock()),
    ):
        await bot.callback_handler(update, MagicMock())

    assert (
        tmux.send_keys.await_args_list
        == [call("@1", "C-b", enter=False, literal=False)] * 2
    )
    update.callback_query.answer.assert_awaited_once_with("^B ^B: to background")
