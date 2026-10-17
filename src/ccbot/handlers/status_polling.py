"""Terminal status line polling for thread-bound windows.

Provides background polling of terminal status lines for all active users:
  - Detects Claude Code status (working, waiting, etc.)
  - Detects interactive UIs (permission prompts) not triggered via JSONL
  - Updates status messages in Telegram
  - Keeps each topic's status light (🟢 working / 🟡 background) current,
    subagent topics' too
  - Polls thread_bindings (each topic = one window)
  - Cleans up deleted topics (kills tmux window + unbinds thread) once a send
    to them has failed — Telegram sends no update on deletion, and probing
    every topic would spend the bot-wide send quota

Key components:
  - STATUS_POLL_INTERVAL: Polling frequency (5 seconds)
  - status_poll_loop: Background polling task
  - update_status_message: Poll and enqueue status updates
"""

import asyncio
import logging

from telegram import Bot

from ..session import session_manager
from ..terminal_parser import is_interactive_ui, parse_status_line
from ..tmux_manager import tmux_manager
from .cleanup import clear_topic_state
from .interactive_ui import (
    clear_interactive_msg,
    get_interactive_window,
    handle_interactive_ui,
)
from .message_queue import end_turn, enqueue_status_update, get_message_queue
from .message_sender import is_topic_gone
from .subagent_topics import update_subagent_lights
from .topic_light import light_state, update_topic_light

logger = logging.getLogger(__name__)

# Status polling interval
STATUS_POLL_INTERVAL = 5.0  # seconds


async def update_status_message(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    skip_status: bool = False,
) -> None:
    """Poll terminal and check for interactive UIs and status updates.

    UI detection always happens regardless of skip_status. When skip_status=True,
    only UI detection runs (used when message queue is non-empty to avoid
    flooding the queue with status updates).

    Also detects permission prompt UIs (not triggered via JSONL) and enters
    interactive mode when found.
    """
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        # Window gone, enqueue clear (unless skipping status)
        if not skip_status:
            await enqueue_status_update(
                bot, user_id, window_id, None, thread_id=thread_id
            )
        return

    pane_text = await tmux_manager.capture_pane(w.window_id)
    if not pane_text:
        # Transient capture failure - keep existing status message
        return

    interactive_window = get_interactive_window(user_id, thread_id)
    should_check_new_ui = True

    if interactive_window == window_id:
        # User is in interactive mode for THIS window
        if is_interactive_ui(pane_text):
            # Interactive UI still showing — skip status update (user is interacting)
            return
        # Interactive UI gone — clear interactive mode, fall through to status check.
        # Don't re-check for new UI this cycle (the old one just disappeared).
        await clear_interactive_msg(user_id, bot, thread_id)
        should_check_new_ui = False
    elif interactive_window is not None:
        # User is in interactive mode for a DIFFERENT window (window switched)
        # Clear stale interactive mode
        await clear_interactive_msg(user_id, bot, thread_id)

    # Check for permission prompt (interactive UI not triggered via JSONL)
    # ALWAYS check UI, regardless of skip_status
    if should_check_new_ui and is_interactive_ui(pane_text):
        logger.debug(
            "Interactive UI detected in polling (user=%d, window=%s, thread=%s)",
            user_id,
            window_id,
            thread_id,
        )
        await handle_interactive_ui(bot, user_id, window_id, thread_id)
        return

    status_line = parse_status_line(pane_text)
    if thread_id is not None:
        await update_topic_light(
            bot,
            user_id,
            thread_id,
            session_manager.get_display_name(window_id),
            light_state(status_line, pane_text),
        )

    # Normal status line check — skip if queue is non-empty
    if skip_status:
        return

    if status_line:
        await enqueue_status_update(
            bot,
            user_id,
            window_id,
            status_line,
            thread_id=thread_id,
        )
    else:
        # Claude is idle: the turn is over. Keep any status message (don't
        # clear on transient state), but let the next turn show one again
        end_turn(user_id, thread_id)


async def status_poll_loop(bot: Bot) -> None:
    """Background task to poll terminal status for all thread-bound windows."""
    logger.info("Status polling started (interval: %ss)", STATUS_POLL_INTERVAL)
    while True:
        try:
            for user_id, thread_id, wid in list(session_manager.iter_thread_bindings()):
                try:
                    # Topic deleted (a send to it failed) — kill window, unbind
                    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
                    if is_topic_gone(chat_id, thread_id):
                        w = await tmux_manager.find_window_by_id(wid)
                        if w:
                            await tmux_manager.kill_window(w.window_id)
                        session_manager.unbind_thread(user_id, thread_id)
                        await clear_topic_state(user_id, thread_id, bot)
                        logger.info(
                            "Topic deleted: killed window_id '%s' and "
                            "unbound thread %d for user %d",
                            wid,
                            thread_id,
                            user_id,
                        )
                        continue

                    # Clean up stale bindings (window no longer exists)
                    w = await tmux_manager.find_window_by_id(wid)
                    if not w:
                        session_manager.unbind_thread(user_id, thread_id)
                        await clear_topic_state(user_id, thread_id, bot)
                        logger.info(
                            "Cleaned up stale binding: user=%d thread=%d window_id=%s",
                            user_id,
                            thread_id,
                            wid,
                        )
                        continue

                    # UI detection happens unconditionally in update_status_message.
                    # Status enqueue is skipped inside update_status_message when
                    # interactive UI is detected (returns early) or when queue is non-empty.
                    queue = get_message_queue(user_id)
                    skip_status = queue is not None and not queue.empty()

                    await update_status_message(
                        bot,
                        user_id,
                        wid,
                        thread_id=thread_id,
                        skip_status=skip_status,
                    )
                except Exception as e:
                    logger.debug(
                        f"Status update error for user {user_id} "
                        f"thread {thread_id}: {e}"
                    )
            await update_subagent_lights(bot)
        except Exception as e:
            logger.error(f"Status poll loop error: {e}")

        await asyncio.sleep(STATUS_POLL_INTERVAL)
