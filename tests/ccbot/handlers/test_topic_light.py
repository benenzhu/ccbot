"""Tests for topic_light — 🟢 / 🟡 / ✅ in front of a topic's name."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.handlers import topic_light
from ccbot.terminal_parser import count_background_tasks

RULE = "─" * 40


def _pane(above: str, bar: str) -> str:
    return f"{above}\n\n{RULE} name ─\n❯ \n{RULE}\n  user@host  Opus\n{bar}\n"


@pytest.mark.parametrize(
    "bar, count",
    [
        ("  ⏵⏵ bypass permissions on · 2 shells · ← for agents", 2),
        ("  ⏵⏵ bypass permissions on · 1 shell · 1 local agent", 2),
        ("  ⏵⏵ bypass permissions on (shift+tab to cycle) · ← for agents", 0),
    ],
)
def test_background_tasks_come_from_the_bottom_bar(bar: str, count: int):
    # The finished turn's "2 shells still running" above the chrome is stale
    pane = _pane("✻ Brewed for 18s · done 1:55 AM · 2 shells still running", bar)
    assert count_background_tasks(pane) == count


def test_light_state():
    busy = _pane("", "  ⏵⏵ on · 1 shell")
    assert topic_light.light_state("Zesting…", busy) == "working"
    assert topic_light.light_state(None, busy) == "background"
    assert topic_light.light_state(None, _pane("", "  ⏵⏵ on")) == "idle"


def test_names_with_and_without_a_light():
    assert topic_light.topic_name("vllm#713", "working") == "🟢 vllm#713"
    assert topic_light.topic_name("vllm#713", "idle") == "vllm#713"
    assert topic_light.strip_light("🟡 vllm#713") == "vllm#713"
    assert topic_light.strip_light("✅ vllm#713") == "vllm#713"
    assert topic_light.strip_light("vllm#713") == "vllm#713"


@pytest.fixture
def lights(tmp_path) -> Iterator[MagicMock]:
    """Fresh light state on a temp file; yields the session_manager mock."""
    now = MagicMock(return_value=1000.0)  # topic_light's clocks only
    wall = MagicMock(return_value=2_000_000_000.0)
    sm = MagicMock()
    sm.resolve_chat_id.return_value = 777
    with (
        patch.object(topic_light, "LIGHTS_FILE", tmp_path / "lights.json"),
        patch.object(topic_light, "_shown", {}),
        patch.object(topic_light, "_pending", {}),
        patch.object(topic_light, "session_manager", sm),
        patch.object(topic_light, "time", SimpleNamespace(monotonic=now, time=wall)),
    ):
        sm.now, sm.wall = now, wall
        yield sm


async def test_a_state_shows_once_it_has_held(lights: MagicMock):
    bot = AsyncMock()
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "working")
    bot.edit_forum_topic.assert_not_awaited()  # first sighting only

    lights.now.return_value = 1005.0  # next poll
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "working")
    bot.edit_forum_topic.assert_awaited_once_with(
        chat_id=777, message_thread_id=42, name="🟢 vllm#713"
    )
    assert topic_light.shown_state(5, 42) == "working"
    assert topic_light._load() == {
        (5, 42): ("working", "🟢 vllm#713", 2_000_000_000.0)
    }  # persisted

    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "working")
    assert bot.edit_forum_topic.await_count == 1  # nothing changed


async def test_finished_work_is_done_for_a_day(lights: MagicMock):
    bot = AsyncMock()
    topic_light._shown[(5, 42)] = ("working", "🟢 vllm#713", 0.0)
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "idle")
    lights.now.return_value = 1010.0
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "idle")
    bot.edit_forum_topic.assert_not_awaited()  # a pause between tool calls

    lights.now.return_value = 1030.0
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "idle")
    assert bot.edit_forum_topic.await_args.kwargs["name"] == "✅ vllm#713"
    assert topic_light.shown_state(5, 42) == "done"

    # Still idle 23h later: ✅ stays
    lights.wall.return_value += 23 * 3600
    lights.now.return_value = 2000.0
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "idle")
    lights.now.return_value = 2030.0
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "idle")
    assert bot.edit_forum_topic.await_count == 1

    # A day after it finished, the ✅ goes
    lights.wall.return_value += 3600
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "idle")
    lights.now.return_value = 2060.0
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "idle")
    assert bot.edit_forum_topic.await_args.kwargs["name"] == "vllm#713"
    assert topic_light.shown_state(5, 42) == "idle"


async def test_never_worked_topic_gets_no_done_light(lights: MagicMock):
    bot = AsyncMock()
    for t in (1000.0, 1030.0, 1060.0):
        lights.now.return_value = t
        await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "idle")
    bot.edit_forum_topic.assert_not_awaited()


async def test_light_waits_while_the_window_is_short_of_room(lights: MagicMock):
    from ccbot.rate_limit import TelegramRateLimiter

    bot = AsyncMock()
    bot.rate_limiter = TelegramRateLimiter()
    for _ in range(bot.rate_limiter.window.max_requests):
        await bot.rate_limiter.window.acquire()
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "background")
    lights.now.return_value = 1100.0
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "background")
    bot.edit_forum_topic.assert_not_awaited()
    assert topic_light.shown_state(5, 42) == "idle"


async def test_lit_topic_follows_a_new_window_name_at_once(lights: MagicMock):
    """E.g. a name drifted before; or the old file didn't know the name."""
    bot = AsyncMock()
    topic_light._shown[(5, 42)] = ("background", "🟡 100tok speed test", 0.0)
    await topic_light.update_topic_light(bot, 5, 42, "indexer cp", "background")
    assert bot.edit_forum_topic.await_args.kwargs["name"] == "🟡 indexer cp"

    await topic_light.update_topic_light(bot, 5, 42, "indexer cp", "background")
    assert bot.edit_forum_topic.await_count == 1


def test_older_lights_file_still_loads(lights: MagicMock):
    topic_light.LIGHTS_FILE.write_text('{"5:42": "working", "bad": "working"}')
    assert topic_light._load() == {(5, 42): ("working", "", 2_000_000_000.0)}


async def test_a_topic_created_lit_needs_no_rename(lights: MagicMock):
    bot = AsyncMock()
    topic_light.mark_lit(5, 42, "working", "🟢 vllm#713")
    await topic_light.update_topic_light(bot, 5, 42, "vllm#713", "working")
    bot.edit_forum_topic.assert_not_awaited()
    assert topic_light._load()[(5, 42)][:2] == ("working", "🟢 vllm#713")
