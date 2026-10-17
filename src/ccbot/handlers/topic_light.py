"""Topic status light — 🟢 / 🟡 / ✅ in front of a topic's name.

🟢 Claude is working (its spinner is up); 🟡 Claude is idle but background
shells or agents it started still run; ✅ the work finished, for DONE_FOR
(24h) after it did; no light otherwise. Telegram has no other per-topic
status, so the light is part of the topic name and a change costs one
editForumTopic request. A new state must hold for a while before it shows
(LIGHT_DELAY): lighting up waits one poll, finishing 30s, so the pauses
between tool calls don't flicker. Like status, lights are optional traffic:
they wait while the bot-wide window is short of room. Each lit topic's
state, the name it was given and since when are persisted
(topic_lights.json): a restart neither repaints every topic nor leaves a
stale 🟢 behind, ✅ keeps its 24h, and a lit topic whose window got another
name is renamed again right away.

Key functions:
  - light_state: the state a pane shows (working / background / idle)
  - update_topic_light: called by the status poller for each bound topic
  - topic_name / strip_light: a display name with / without its light
  - forget_topic_light: the topic's name no longer carries a light
  - note_topic_name: the bot renamed a topic itself (it kept the light)
  - mark_lit: the bot created a topic with a light in its name
"""

import json
import logging
import time

from telegram import Bot
from telegram.constants import ForumTopicLimit
from telegram.error import BadRequest, TelegramError

from ..config import config
from ..rate_limit import TelegramRateLimiter
from ..session import session_manager
from ..terminal_parser import count_background_tasks
from ..utils import atomic_write_json
from .message_queue import STATUS_RESERVE

logger = logging.getLogger(__name__)

LIGHTS = {"working": "🟢", "background": "🟡", "done": "✅"}  # idle: no light

# Seconds a new state must hold before the name changes
LIGHT_DELAY = {"working": 4.0, "background": 25.0, "done": 25.0, "idle": 25.0}

# How long ✅ stays after the work finished
DONE_FOR = 24 * 3600.0

LIGHTS_FILE = config.config_dir / "topic_lights.json"

TopicKey = tuple[int, int]  # (user_id, thread_id)


# A lit topic: its state, the name it was given, since when (epoch seconds)
Shown = tuple[str, str, float]


def _load() -> dict[TopicKey, Shown]:
    try:
        raw = json.loads(LIGHTS_FILE.read_text())
    except (OSError, ValueError):
        return {}
    shown: dict[TopicKey, Shown] = {}
    for key, entry in raw.items():
        user, _, thread = key.partition(":")
        if isinstance(entry, str):  # older file: state only, name unknown
            entry = {"state": entry, "name": ""}
        if not isinstance(entry, dict) or not (user.isdigit() and thread.isdigit()):
            continue
        state, name = entry.get("state"), entry.get("name")
        since = entry.get("since", time.time())
        if state in LIGHTS and isinstance(name, str):
            shown[(int(user), int(thread))] = (state, name, float(since))
    return shown


# Lit topics, and states waiting out their delay: (state, first seen, monotonic)
_shown: dict[TopicKey, Shown] = _load()
_pending: dict[TopicKey, tuple[str, float]] = {}


def _save() -> None:
    atomic_write_json(
        LIGHTS_FILE,
        {
            f"{u}:{t}": {"state": state, "name": name, "since": since}
            for (u, t), (state, name, since) in _shown.items()
        },
    )


def light_state(status_line: str | None, pane_text: str) -> str:
    """working / background / idle, from the status line and the bottom bar."""
    if status_line:
        return "working"
    if count_background_tasks(pane_text):
        return "background"
    return "idle"


def strip_light(name: str) -> str:
    """A topic name without its status light."""
    for light in LIGHTS.values():
        if name.startswith(light):
            return name[len(light) :].lstrip()
    return name


def topic_name(display_name: str, state: str) -> str:
    """The topic name for a window name and state."""
    light = LIGHTS.get(state)
    if not light:
        return display_name[: ForumTopicLimit.MAX_NAME_LENGTH]
    return f"{light} {display_name}"[: ForumTopicLimit.MAX_NAME_LENGTH]


def shown_state(user_id: int, thread_id: int) -> str:
    """The state the topic's name shows now."""
    return _shown.get((user_id, thread_id), ("idle", "", 0.0))[0]


def note_topic_name(user_id: int, thread_id: int, name: str) -> None:
    """The bot just gave the topic this name (with its light, if it has one)."""
    key = (user_id, thread_id)
    if key in _shown:
        state, _, since = _shown[key]
        _shown[key] = (state, name, since)
        _save()


def _wanted(key: TopicKey, observed: str) -> str:
    """The light for what the pane shows: idle after work is ✅ for a day."""
    if observed != "idle" or key not in _shown:
        return observed
    state, _, since = _shown[key]
    if state != "done":
        return "done"  # the work just finished
    return "done" if time.time() - since < DONE_FOR else "idle"


def forget_topic_light(user_id: int, thread_id: int) -> None:
    """The topic's name was set without a light (new binding, closed topic)."""
    key = (user_id, thread_id)
    _pending.pop(key, None)
    if _shown.pop(key, None) is not None:
        _save()


def _window_has_room(bot: Bot) -> bool:
    limiter = getattr(bot, "rate_limiter", None)
    if isinstance(limiter, TelegramRateLimiter):
        return limiter.window.free() > STATUS_RESERVE
    return True


def mark_lit(user_id: int, thread_id: int, state: str, name: str) -> None:
    """The bot just created the topic with this name, lit for state."""
    _pending.pop((user_id, thread_id), None)
    _shown[(user_id, thread_id)] = (state, name, time.time())
    _save()


async def update_topic_light(
    bot: Bot, user_id: int, thread_id: int, display_name: str, state: str
) -> None:
    """Show state (what the pane shows) in front of display_name.

    A new light shows once it has held for its delay; a lit topic whose
    display name changed is renamed without delay.
    """
    key = (user_id, thread_id)
    state = _wanted(key, state)
    name = topic_name(display_name, state)
    shown, shown_name, since = _shown.get(key, ("idle", "", 0.0))
    if state == shown:
        _pending.pop(key, None)
        if state not in LIGHTS or name == shown_name:
            return
    else:
        now = time.monotonic()
        pending = _pending.get(key)
        if pending is None or pending[0] != state:
            _pending[key] = (state, now)
            return
        if now - pending[1] < LIGHT_DELAY[state]:
            return
    if not _window_has_room(bot):
        return

    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    try:
        await bot.edit_forum_topic(
            chat_id=chat_id, message_thread_id=thread_id, name=name
        )
    except BadRequest as e:
        if "not_modified" not in str(e).lower().replace(" ", "_"):
            logger.debug("Topic light not set (thread=%d): %s", thread_id, e)
            return
    except TelegramError as e:
        logger.debug("Topic light not set (thread=%d): %s", thread_id, e)
        return
    _pending.pop(key, None)
    if state not in LIGHTS:
        _shown.pop(key, None)
    else:
        _shown[key] = (state, name, since if state == shown else time.time())
    _save()
