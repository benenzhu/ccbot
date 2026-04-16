"""Feishu bot — the main UI layer for the Feishu backend.

Handles the Feishu bot lifecycle using lark-oapi SDK with WebSocket
long-connection. Each Feishu group chat maps to a tmux window (Claude session).

Core responsibilities:
  - WebSocket event handling: receives im.message.receive_v1 events
  - Message routing: maps Feishu chats to tmux windows
  - Command handling: /start, /history, /screenshot, /esc, /kill, etc.
  - Session monitor integration: routes Claude output back to Feishu
  - Status polling: shows Claude Code terminal status in Feishu

Key functions: create_feishu_bot(), run_feishu_bot().
"""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from ..config import config
from ..session import session_manager
from ..session_monitor import NewMessage, SessionMonitor
from ..terminal_parser import (
    extract_interactive_content,
    is_interactive_ui,
    parse_status_line,
)
from ..tmux_manager import tmux_manager

logger = logging.getLogger(__name__)

# Global Feishu client reference
_lark_client: Any = None

# Session monitor
_session_monitor: SessionMonitor | None = None

# Status polling task
_status_poll_task: asyncio.Task[None] | None = None

# Chat → window_id bindings (simple mapping for Feishu)
# In Feishu, each group chat or P2P chat maps to one tmux window.
# For group chats, the chat_id is used as the key.
_chat_bindings: dict[str, str] = {}

# Window display names
_window_display_names: dict[str, str] = {}

# Event loop reference for cross-thread dispatch
_main_loop: asyncio.AbstractEventLoop | None = None


def get_lark_client() -> Any:
    """Get the shared Feishu lark client."""
    return _lark_client


def _is_user_allowed(open_id: str) -> bool:
    """Check if a Feishu user is allowed."""
    return config.is_user_allowed(open_id)


def _extract_text_from_content(msg_type: str, content_str: str) -> str:
    """Extract plain text from Feishu message content JSON."""
    try:
        content = json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        return content_str

    if msg_type == "text":
        return content.get("text", "")
    elif msg_type == "post":
        return _extract_post_text(content)
    elif msg_type == "interactive":
        return ""
    else:
        return ""


def _extract_post_text(content: dict[str, Any]) -> str:
    """Extract text from Feishu post (rich text) content."""
    parts: list[str] = []

    root = content
    if isinstance(root.get("post"), dict):
        root = root["post"]

    block: dict[str, Any] | None = None
    if "content" in root:
        block = root
    else:
        for key in ("zh_cn", "en_us", "ja_jp"):
            if key in root:
                block = root[key]
                break
        if block is None:
            for val in root.values():
                if isinstance(val, dict):
                    block = val
                    break

    if not block or not isinstance(block.get("content"), list):
        return ""

    if title := block.get("title"):
        parts.append(str(title))

    for row in block["content"]:
        if not isinstance(row, list):
            continue
        for el in row:
            if not isinstance(el, dict):
                continue
            tag = el.get("tag", "")
            if tag in ("text", "a"):
                parts.append(el.get("text", ""))
            elif tag == "at":
                pass
            elif tag == "md":
                parts.append(el.get("text", ""))

    return " ".join(parts).strip()


def _get_chat_binding(chat_id: str) -> str | None:
    """Get the window_id bound to a Feishu chat."""
    return _chat_bindings.get(chat_id)


def _set_chat_binding(chat_id: str, window_id: str, display_name: str = "") -> None:
    """Bind a Feishu chat to a tmux window."""
    _chat_bindings[chat_id] = window_id
    if display_name:
        _window_display_names[window_id] = display_name
    logger.info("Bound chat %s -> window %s (%s)", chat_id, window_id, display_name)


def _remove_chat_binding(chat_id: str) -> str | None:
    """Remove a chat binding. Returns the previously bound window_id."""
    return _chat_bindings.pop(chat_id, None)


def _find_chats_for_window(window_id: str) -> list[str]:
    """Find all chat_ids bound to a window."""
    return [cid for cid, wid in _chat_bindings.items() if wid == window_id]


async def _handle_message_event(data: Any) -> None:
    """Handle an incoming Feishu message event."""
    from .sender import send_text
    from .state import enqueue_status_update

    event = data.event
    if not event or not event.message:
        return

    message = event.message
    sender = event.sender
    if not sender or not sender.sender_id:
        return

    open_id = sender.sender_id.open_id or ""
    chat_id = message.chat_id or ""
    msg_type = message.message_type or ""
    content_str = message.content or ""
    feishu_msg_id = message.message_id or ""

    if not open_id or not chat_id:
        return

    # Authorization check
    if not _is_user_allowed(open_id):
        logger.debug("Unauthorized user: %s", open_id)
        return

    # Extract text content
    text = _extract_text_from_content(msg_type, content_str)
    if not text:
        if msg_type not in ("text", "post"):
            logger.debug("Unsupported message type: %s", msg_type)
        return

    text = text.strip()
    if not text:
        return

    # Handle commands
    if text.startswith("/"):
        await _handle_command(chat_id, open_id, text, feishu_msg_id)
        return

    # Get or create window binding
    wid = _get_chat_binding(chat_id)
    if wid is None:
        # No binding — auto-create a new session in cwd
        await _auto_create_session(chat_id, text)
        return

    # Check window still exists
    w = await tmux_manager.find_window_by_id(wid)
    if not w:
        display = _window_display_names.get(wid, wid)
        _remove_chat_binding(chat_id)
        await send_text(
            chat_id,
            f"❌ Window '{display}' no longer exists. Binding removed.\n"
            "Send a message to start a new session.",
        )
        return

    # Forward text to Claude Code via tmux
    await enqueue_status_update(chat_id, wid, None)
    success, message_text = await session_manager.send_to_window(wid, text)
    if not success:
        await send_text(chat_id, f"❌ {message_text}")


async def _handle_command(chat_id: str, open_id: str, text: str, msg_id: str) -> None:
    """Handle slash commands from Feishu."""
    from .sender import send_text

    cmd = text.split()[0].lower().rstrip("@")
    args = text[len(cmd) :].strip()

    if cmd == "/start" or cmd == "/help":
        await send_text(
            chat_id,
            "🤖 **Claude Code Monitor (Feishu)**\n\n"
            "Send any text to forward it to Claude Code.\n\n"
            "**Commands:**\n"
            "- `/start` — Show this help\n"
            "- `/esc` — Send Escape to interrupt Claude\n"
            "- `/screenshot` — Capture terminal screenshot\n"
            "- `/unbind` — Unbind this chat from its session\n"
            "- `/new [path]` — Create new session (optional path)\n"
            "- `/clear`, `/compact`, `/cost`, `/model` — Forward to Claude Code",
        )
        return

    if cmd == "/esc":
        wid = _get_chat_binding(chat_id)
        if not wid:
            await send_text(chat_id, "❌ No session bound to this chat.")
            return
        w = await tmux_manager.find_window_by_id(wid)
        if not w:
            await send_text(chat_id, "❌ Window no longer exists.")
            return
        await tmux_manager.send_keys(w.window_id, "\x1b", enter=False)
        await send_text(chat_id, "⎋ Sent Escape")
        return

    if cmd == "/screenshot":
        wid = _get_chat_binding(chat_id)
        if not wid:
            await send_text(chat_id, "❌ No session bound to this chat.")
            return
        w = await tmux_manager.find_window_by_id(wid)
        if not w:
            await send_text(chat_id, "❌ Window no longer exists.")
            return
        pane_text = await tmux_manager.capture_pane(w.window_id)
        if not pane_text:
            await send_text(chat_id, "❌ Failed to capture pane content.")
            return
        # Send as code block
        if len(pane_text) > 4000:
            pane_text = "…" + pane_text[-4000:]
        await send_text(chat_id, f"```\n{pane_text}\n```")
        return

    if cmd == "/unbind":
        wid = _get_chat_binding(chat_id)
        if not wid:
            await send_text(chat_id, "❌ No session bound to this chat.")
            return
        display = _window_display_names.get(wid, wid)
        _remove_chat_binding(chat_id)
        await send_text(
            chat_id,
            f"✅ Chat unbound from window '{display}'.\n"
            "The Claude session is still running in tmux.\n"
            "Send a message to bind to a new session.",
        )
        return

    if cmd == "/new":
        path = args or str(Path.cwd())
        await _auto_create_session(chat_id, None, path=path)
        return

    # Forward other slash commands to Claude Code
    cc_commands = {"/clear", "/compact", "/cost", "/help", "/memory", "/model"}
    if cmd in cc_commands:
        wid = _get_chat_binding(chat_id)
        if not wid:
            await send_text(chat_id, "❌ No session bound to this chat.")
            return
        w = await tmux_manager.find_window_by_id(wid)
        if not w:
            await send_text(chat_id, "❌ Window no longer exists.")
            return
        display = _window_display_names.get(wid, wid)
        success, msg = await session_manager.send_to_window(wid, text)
        if success:
            await send_text(chat_id, f"⚡ [{display}] Sent: {text}")
            if cmd == "/clear":
                session_manager.clear_window_session(wid)
        else:
            await send_text(chat_id, f"❌ {msg}")
        return

    # Unknown command
    await send_text(chat_id, f"❓ Unknown command: {cmd}")


async def _auto_create_session(
    chat_id: str,
    pending_text: str | None,
    path: str | None = None,
) -> None:
    """Auto-create a new tmux window and bind it to the chat."""
    from .sender import send_text

    selected_path = path or str(Path.cwd())

    # Check for unbound windows first
    all_windows = await tmux_manager.list_windows()
    bound_ids = set(_chat_bindings.values())
    unbound = [
        (w.window_id, w.window_name, w.cwd)
        for w in all_windows
        if w.window_id not in bound_ids
    ]

    if unbound and pending_text:
        # Bind to first unbound window
        wid, wname, wcwd = unbound[0]
        _set_chat_binding(chat_id, wid, wname)

        # Also register in session_manager for monitor routing
        session_manager.bind_thread(0, hash(chat_id) & 0x7FFFFFFF, wid, wname)

        await send_text(chat_id, f"✅ Bound to existing window: {wname}")

        if pending_text:
            success, msg = await session_manager.send_to_window(wid, pending_text)
            if not success:
                await send_text(chat_id, f"❌ Failed to send message: {msg}")
        return

    # Create new window
    success, message, created_wname, created_wid = await tmux_manager.create_window(
        selected_path
    )
    if not success:
        await send_text(chat_id, f"❌ {message}")
        return

    logger.info(
        "Window created: %s (id=%s) at %s for chat %s",
        created_wname,
        created_wid,
        selected_path,
        chat_id,
    )

    # Wait for hook to register session
    await session_manager.wait_for_session_map_entry(created_wid, timeout=5.0)

    _set_chat_binding(chat_id, created_wid, created_wname or "")

    # Also register in session_manager for monitor routing
    session_manager.bind_thread(
        0, hash(chat_id) & 0x7FFFFFFF, created_wid, created_wname or ""
    )

    status = "Created"
    await send_text(chat_id, f"✅ {message}\n\n{status}. Send messages here.")

    # Forward pending text
    if pending_text:
        send_ok, send_msg = await session_manager.send_to_window(
            created_wid, pending_text
        )
        if not send_ok:
            await send_text(chat_id, f"❌ Failed to send: {send_msg}")


async def handle_new_message(msg: NewMessage) -> None:
    """Handle a new assistant message from the session monitor.

    Routes the message to all Feishu chats bound to the session's window.
    """
    from ..handlers.response_builder import build_response_parts
    from .state import enqueue_content_message

    active_users = await session_manager.find_users_for_session(msg.session_id)
    if not active_users:
        logger.debug("No active users for session %s", msg.session_id)
        return

    for _user_id, wid, _thread_id in active_users:
        # Find Feishu chats bound to this window
        chat_ids = _find_chats_for_window(wid)
        if not chat_ids:
            continue

        # Skip tool calls if disabled
        if not config.show_tool_calls and msg.content_type in (
            "tool_use",
            "tool_result",
        ):
            continue

        parts = build_response_parts(
            msg.text,
            msg.is_complete,
            msg.content_type,
            msg.role,
        )

        if msg.is_complete:
            for cid in chat_ids:
                await enqueue_content_message(
                    chat_id=cid,
                    window_id=wid,
                    parts=parts,
                    tool_use_id=msg.tool_use_id,
                    content_type=msg.content_type,
                    text=msg.text,
                    image_data=msg.image_data,
                )

            # Update read offset
            session = await session_manager.resolve_session_for_window(wid)
            if session and session.file_path:
                try:
                    file_size = Path(session.file_path).stat().st_size
                    session_manager.update_user_window_offset(0, wid, file_size)
                except OSError:
                    pass


async def _status_poll_loop() -> None:
    """Background task to poll terminal status for all bound chats."""
    from .state import enqueue_status_update, get_message_queue

    logger.info("Feishu status polling started")
    while True:
        try:
            for chat_id, wid in list(_chat_bindings.items()):
                try:
                    w = await tmux_manager.find_window_by_id(wid)
                    if not w:
                        _remove_chat_binding(chat_id)
                        continue

                    queue = get_message_queue(chat_id)
                    if queue and not queue.empty():
                        continue

                    pane_text = await tmux_manager.capture_pane(w.window_id)
                    if not pane_text:
                        continue

                    # Check for interactive UI
                    if is_interactive_ui(pane_text):
                        content = extract_interactive_content(pane_text)
                        if content:
                            from .sender import send_text

                            await send_text(chat_id, content.content)
                        continue

                    status_line = parse_status_line(pane_text)
                    if status_line:
                        await enqueue_status_update(chat_id, wid, status_line)

                except Exception as e:
                    logger.debug("Status poll error for chat %s: %s", chat_id, e)
        except Exception as e:
            logger.error("Status poll loop error: %s", e)

        await asyncio.sleep(1.0)


def _on_message_sync(data: Any) -> None:
    """Synchronous message handler called from WebSocket thread.

    Dispatches to the async handler on the main event loop.
    """
    if _main_loop is None:
        logger.warning("Main loop not set, dropping message")
        return

    asyncio.run_coroutine_threadsafe(_handle_message_event(data), _main_loop)


async def _post_init() -> None:
    """Initialize session monitor and status polling after bot starts."""
    global _session_monitor, _status_poll_task

    await session_manager.resolve_stale_ids()

    monitor = SessionMonitor()

    async def message_callback(msg: NewMessage) -> None:
        await handle_new_message(msg)

    monitor.set_message_callback(message_callback)
    monitor.start()
    _session_monitor = monitor
    logger.info("Session monitor started (Feishu)")

    _status_poll_task = asyncio.create_task(_status_poll_loop())
    logger.info("Status polling started (Feishu)")


async def _post_shutdown() -> None:
    """Clean up on shutdown."""
    global _status_poll_task
    from .state import shutdown_workers

    if _status_poll_task:
        _status_poll_task.cancel()
        try:
            await _status_poll_task
        except asyncio.CancelledError:
            pass
        _status_poll_task = None

    await shutdown_workers()

    if _session_monitor:
        _session_monitor.stop()
        logger.info("Session monitor stopped (Feishu)")


async def run_feishu_bot() -> None:
    """Start and run the Feishu bot.

    Sets up the lark-oapi WebSocket client and runs until interrupted.
    """
    global _lark_client, _main_loop

    try:
        import lark_oapi as lark
        from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN
    except ImportError:
        logger.error(
            "lark-oapi is not installed. Install with: pip install 'ccbot[feishu]'"
        )
        raise SystemExit(1)

    _main_loop = asyncio.get_running_loop()

    # Create the Feishu client
    domain = LARK_DOMAIN if config.feishu_domain == "lark" else FEISHU_DOMAIN
    _lark_client = (
        lark.Client.builder()
        .app_id(config.feishu_app_id)
        .app_secret(config.feishu_app_secret)
        .domain(domain)
        .log_level(lark.LogLevel.INFO)
        .build()
    )

    # Build event dispatcher
    event_handler = (
        lark.EventDispatcherHandler.builder(
            config.feishu_encrypt_key,
            config.feishu_verification_token,
        )
        .register_p2_im_message_receive_v1(_on_message_sync)
        .build()
    )

    # Create WebSocket client
    ws_client = lark.ws.Client(
        config.feishu_app_id,
        config.feishu_app_secret,
        domain=domain,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    # Start session monitor and status polling
    await _post_init()

    # Run WebSocket in a separate thread (lark SDK uses its own event loop)
    def run_ws() -> None:
        import lark_oapi.ws.client as _lark_ws_mod

        ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(ws_loop)
        _lark_ws_mod.loop = ws_loop
        try:
            while True:
                try:
                    ws_client.start()
                except Exception as e:
                    logger.warning("Feishu WebSocket error: %s", e)
                    time.sleep(5)
        finally:
            ws_loop.close()

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    logger.info("Feishu bot started with WebSocket long-connection")

    # Fetch bot info
    try:

        def _fetch_bot_info() -> dict[str, str]:
            import lark_oapi as lark_mod

            request = (
                lark_mod.BaseRequest.builder()
                .http_method(lark_mod.HttpMethod.GET)
                .uri("/open-apis/bot/v3/info")
                .token_types({lark_mod.AccessTokenType.APP})
                .build()
            )
            response = _lark_client.request(request)
            if response.success():
                resp_data = json.loads(response.raw.content)
                bot = (resp_data.get("data") or resp_data).get("bot") or {}
                return {
                    "open_id": bot.get("open_id", ""),
                    "app_name": bot.get("app_name", ""),
                }
            return {}

        bot_info = await asyncio.to_thread(_fetch_bot_info)
        if bot_info:
            logger.info(
                "Feishu bot info: name=%s, open_id=%s",
                bot_info.get("app_name"),
                bot_info.get("open_id"),
            )
    except Exception as e:
        logger.warning("Failed to fetch bot info: %s", e)

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Feishu bot shutting down...")
    finally:
        await _post_shutdown()
