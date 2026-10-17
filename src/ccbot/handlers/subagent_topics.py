"""Subagent topics — each subagent Claude starts is shown in a topic of its own.

The SessionMonitor reads subagent transcripts and tags their messages with
agent_id; topics_for_message routes them to the subagent's topic, which it
creates on the subagent's first message, in the chat of the parent's topic.
The topic is named sub-<parent window>-<task>, the task being the 3-5 words
Claude described the subagent with. Its messages go through the same queue as
any topic's (tool calls included), under the pseudo window id "agent:<id>".

The topic is read-only: the bot answers anything sent there with a pointer to
the parent topic (subagent_topic tells the handlers). Its light: created 🟢,
✅ for a day once the subagent's reply ended its work (or its transcript has
been quiet for SUBAGENT_STALE_SECONDS: it was stopped). The bot never deletes
these topics. They are persisted (subagent_topics.json), so after a restart a
running subagent keeps its topic.

Key functions:
  - topics_for_message: where a subagent's message goes (creates the topic)
  - subagent_topic: the subagent shown in a topic, if any
  - update_subagent_lights: called by the status poller
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from telegram import Bot
from telegram.constants import ForumTopicLimit
from telegram.error import TelegramError

from ..config import config
from ..session import session_manager
from ..session_monitor import NewMessage
from ..utils import atomic_write_json
from .message_sender import is_topic_gone
from .topic_light import mark_lit, shown_state, topic_name, update_topic_light

logger = logging.getLogger(__name__)

SUBAGENT_TOPICS_FILE = config.config_dir / "subagent_topics.json"

# A running subagent whose transcript is quiet this long was stopped
SUBAGENT_STALE_SECONDS = 3600.0

# Queue tasks for a subagent's topic carry this pseudo window id + agent id
AGENT_WINDOW_PREFIX = "agent:"


@dataclass
class SubagentTopic:
    """A topic showing one subagent."""

    user_id: int
    thread_id: int
    agent_id: str
    parent_session_id: str
    parent_name: str  # the parent's window name when the topic was made
    name: str  # the topic name, without its light
    file_path: str  # the subagent's transcript
    running: bool = True


def _load() -> dict[str, SubagentTopic]:
    try:
        raw = json.loads(SUBAGENT_TOPICS_FILE.read_text())
    except (OSError, ValueError):
        return {}
    topics: dict[str, SubagentTopic] = {}
    for key, entry in raw.items() if isinstance(raw, dict) else ():
        try:
            topics[key] = SubagentTopic(**entry)
        except TypeError:
            continue
    return topics


# "user_id:agent_id" -> topic; and subagents whose topic couldn't be created
# (not retried until a restart, so a refusal doesn't cost a request each time)
_topics: dict[str, SubagentTopic] = _load()
_failed: set[str] = set()


def _save() -> None:
    atomic_write_json(
        SUBAGENT_TOPICS_FILE, {key: asdict(t) for key, t in _topics.items()}
    )


def subagent_topic_name(parent_name: str, description: str, agent_id: str) -> str:
    """sub-<parent>-<task>, cut to fit a topic name."""
    task = description or agent_id[:8]
    return f"sub-{parent_name}-{task}"[: ForumTopicLimit.MAX_NAME_LENGTH]


def subagent_topic(user_id: int, thread_id: int | None) -> SubagentTopic | None:
    """The subagent this topic shows, if it is a subagent topic."""
    for topic in _topics.values():
        if topic.user_id == user_id and topic.thread_id == thread_id:
            return topic
    return None


async def _create_topic(
    bot: Bot, user_id: int, parent_wid: str, parent_thread: int, msg: NewMessage
) -> SubagentTopic | None:
    agent_id = msg.agent_id or ""
    parent_name = session_manager.get_display_name(parent_wid)
    name = subagent_topic_name(parent_name, msg.agent_description, agent_id)
    lit = topic_name(name, "working")  # it is working: saves a rename
    chat_id = session_manager.resolve_chat_id(user_id, parent_thread)
    try:
        created = await bot.create_forum_topic(chat_id=chat_id, name=lit)
    except TelegramError as e:
        logger.warning("Could not create a topic for subagent %s: %s", agent_id, e)
        return None
    thread_id = created.message_thread_id
    if chat_id != user_id:
        session_manager.set_group_chat_id(user_id, thread_id, chat_id)
    mark_lit(user_id, thread_id, "working", lit)
    logger.info(
        "Subagent %s of '%s' shown in new topic %d ('%s')",
        agent_id,
        parent_name,
        thread_id,
        name,
    )
    return SubagentTopic(
        user_id=user_id,
        thread_id=thread_id,
        agent_id=agent_id,
        parent_session_id=msg.session_id,
        parent_name=parent_name,
        name=name,
        file_path=msg.agent_file,
    )


async def topics_for_message(bot: Bot, msg: NewMessage) -> list[tuple[int, str, int]]:
    """(user_id, window_id, thread_id) of each topic showing msg's subagent.

    The topic is created on the subagent's first message, for each user whose
    topic shows the parent session. window_id is the pseudo "agent:<id>".
    """
    agent_id = msg.agent_id
    if not agent_id:
        return []
    window_id = f"{AGENT_WINDOW_PREFIX}{agent_id}"
    targets: list[tuple[int, str, int]] = []
    changed = False
    for (
        user_id,
        parent_wid,
        parent_thread,
    ) in await session_manager.find_users_for_session(msg.session_id):
        key = f"{user_id}:{agent_id}"
        if key in _failed or any(user_id == t[0] for t in targets):
            continue
        topic = _topics.get(key)
        if topic is None:
            topic = await _create_topic(bot, user_id, parent_wid, parent_thread, msg)
            if topic is None:
                _failed.add(key)
                continue
            _topics[key] = topic
            changed = True
        if topic.running == msg.agent_finished:
            topic.running = not msg.agent_finished
            changed = True
        chat_id = session_manager.resolve_chat_id(user_id, topic.thread_id)
        if not is_topic_gone(chat_id, topic.thread_id):
            targets.append((user_id, window_id, topic.thread_id))
    if changed:
        _save()
    return targets


def _stale(topic: SubagentTopic, now: float) -> bool:
    """The subagent's transcript has been quiet too long (or is gone)."""
    if not topic.file_path:
        return False
    try:
        return now - Path(topic.file_path).stat().st_mtime > SUBAGENT_STALE_SECONDS
    except OSError:
        return True


async def update_subagent_lights(bot: Bot) -> None:
    """🟢 while a subagent works, ✅ for a day after (topic_light)."""
    now = time.time()
    for topic in list(_topics.values()):
        user_id, thread_id = topic.user_id, topic.thread_id
        if is_topic_gone(
            session_manager.resolve_chat_id(user_id, thread_id), thread_id
        ):
            continue
        working = topic.running and not _stale(topic, now)
        if not working and shown_state(user_id, thread_id) == "idle":
            continue  # finished long ago: nothing to show
        await update_topic_light(
            bot, user_id, thread_id, topic.name, "working" if working else "idle"
        )
