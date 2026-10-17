"""Tests for subagent topics — a read-only topic per subagent."""

import os
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

from ccbot.handlers import subagent_topics, topic_light
from ccbot.session_monitor import NewMessage


@pytest.fixture
def topics(tmp_path) -> Iterator[MagicMock]:
    """Fresh topic and light state on temp files; yields the session_manager mock."""
    sm = MagicMock()
    sm.find_users_for_session = AsyncMock(return_value=[(5, "@1", 42)])
    sm.get_display_name.return_value = "ccbot"
    sm.resolve_chat_id.side_effect = lambda user_id, thread_id=None: user_id
    with (
        patch.object(subagent_topics, "SUBAGENT_TOPICS_FILE", tmp_path / "sub.json"),
        patch.object(subagent_topics, "_topics", {}),
        patch.object(subagent_topics, "_failed", set()),
        patch.object(subagent_topics, "session_manager", sm),
        patch.object(subagent_topics, "is_topic_gone", return_value=False),
        patch.object(topic_light, "LIGHTS_FILE", tmp_path / "lights.json"),
        patch.object(topic_light, "_shown", {}),
        patch.object(topic_light, "_pending", {}),
        patch.object(topic_light, "session_manager", sm),
    ):
        yield sm


def _msg(*, finished: bool = False, agent_file: str = "") -> NewMessage:
    return NewMessage(
        session_id="s",
        text="hi",
        is_complete=True,
        agent_id="a1b2c3d4e5",
        agent_description="Watch job 7",
        agent_file=agent_file,
        agent_finished=finished,
    )


def _bot(thread_id: int = 99) -> AsyncMock:
    bot = AsyncMock()
    bot.create_forum_topic.return_value = SimpleNamespace(message_thread_id=thread_id)
    return bot


async def test_first_message_opens_a_lit_topic_next_to_the_parent(topics: MagicMock):
    bot = _bot()
    targets = await subagent_topics.topics_for_message(bot, _msg())

    bot.create_forum_topic.assert_awaited_once_with(
        chat_id=5, name="🟢 sub-ccbot-Watch job 7"
    )
    assert targets == [(5, "agent:a1b2c3d4e5", 99)]
    assert topic_light.shown_state(5, 99) == "working"  # no rename needed
    assert subagent_topics._load()["5:a1b2c3d4e5"].name == "sub-ccbot-Watch job 7"

    # Later messages reuse the topic
    assert await subagent_topics.topics_for_message(bot, _msg()) == targets
    bot.create_forum_topic.assert_awaited_once()


async def test_topic_name_without_a_description_uses_the_agent_id():
    assert (
        subagent_topics.subagent_topic_name("p", "", "a1b2c3d4e5") == "sub-p-a1b2c3d4"
    )
    assert len(subagent_topics.subagent_topic_name("p" * 200, "x", "a")) == 128


async def test_the_topic_is_known_as_a_subagent_topic(topics: MagicMock):
    await subagent_topics.topics_for_message(_bot(), _msg())
    sub = subagent_topics.subagent_topic(5, 99)
    assert sub is not None and sub.parent_name == "ccbot"
    assert subagent_topics.subagent_topic(5, 42) is None  # the parent's topic


async def test_a_refused_topic_is_not_asked_for_again(topics: MagicMock):
    bot = _bot()
    bot.create_forum_topic.side_effect = BadRequest("Not enough rights")
    assert await subagent_topics.topics_for_message(bot, _msg()) == []
    assert await subagent_topics.topics_for_message(bot, _msg()) == []
    bot.create_forum_topic.assert_awaited_once()


async def test_a_deleted_topic_gets_nothing(topics: MagicMock):
    bot = _bot()
    await subagent_topics.topics_for_message(bot, _msg())
    with patch.object(subagent_topics, "is_topic_gone", return_value=True):
        assert await subagent_topics.topics_for_message(bot, _msg()) == []
    bot.create_forum_topic.assert_awaited_once()  # not made again


async def test_light_turns_done_once_the_subagent_finished(topics: MagicMock, tmp_path):
    transcript = tmp_path / "agent-a1b2c3d4e5.jsonl"
    transcript.write_text("{}\n")
    bot = _bot()
    await subagent_topics.topics_for_message(bot, _msg(agent_file=str(transcript)))

    await subagent_topics.update_subagent_lights(bot)
    bot.edit_forum_topic.assert_not_awaited()  # still working: 🟢 as created

    await subagent_topics.topics_for_message(bot, _msg(finished=True))
    assert subagent_topics._topics["5:a1b2c3d4e5"].running is False
    now = MagicMock(return_value=1000.0)
    with patch.object(
        topic_light, "time", SimpleNamespace(monotonic=now, time=lambda: 2e9)
    ):
        await subagent_topics.update_subagent_lights(bot)
        now.return_value = 1030.0  # finishing holds 25s first
        await subagent_topics.update_subagent_lights(bot)
    bot.edit_forum_topic.assert_awaited_once_with(
        chat_id=5, message_thread_id=99, name="✅ sub-ccbot-Watch job 7"
    )


async def test_a_quiet_subagent_counts_as_stopped(topics: MagicMock, tmp_path):
    transcript = tmp_path / "agent-a1b2c3d4e5.jsonl"
    transcript.write_text("{}\n")
    await subagent_topics.topics_for_message(_bot(), _msg(agent_file=str(transcript)))
    topic = subagent_topics._topics["5:a1b2c3d4e5"]
    now = os.stat(transcript).st_mtime
    assert not subagent_topics._stale(topic, now + 60)
    assert subagent_topics._stale(
        topic, now + subagent_topics.SUBAGENT_STALE_SECONDS + 1
    )
