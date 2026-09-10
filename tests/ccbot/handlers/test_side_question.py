"""Tests for /btw relay — panel parsing and the send/poll/copy/close flow."""

from unittest.mock import AsyncMock, patch

from ccbot.handlers import side_question
from ccbot.handlers.side_question import parse_btw_panel, run_side_question

# Captured from Claude Code 2.1.263 while the main turn was still running.
PENDING = """\
● Running something…

✻ Mustering… (16s · ↓ 712 tokens)
▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

    /btw 刚刚做了点啥？
    /btw 简单的提问. 简单回答一下就行.

      ✢ Answering…

    ⇧←/→ to browse · x to clear history · Esc to close
"""

SETTLED = """\
* Mustering… (43s · ↓ 807 tokens)
▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

    /btw 刚刚做了点啥？
    /btw 简单的提问. 简单回答一下就行.

      好，简单说：这条路是可行的。

      第二段。

    ⇧←/→ to browse · c to copy · f to fork · x to clear history · Esc to close
"""

FAILED = """\
    /btw 简单的提问.

      Failed to get response

    ↑/↓ to scroll · x to clear history · Esc to close
"""


class TestParseBtwPanel:
    def test_absent(self):
        st = parse_btw_panel("❯ hello\n\n  some prose\n")
        assert not st.present

    def test_pending(self):
        st = parse_btw_panel(PENDING)
        assert st.present and not st.settled and st.error is None

    def test_settled_scrapes_dedented_body(self):
        st = parse_btw_panel(SETTLED)
        assert st.present and st.settled
        assert st.answer == "好，简单说：这条路是可行的。\n\n第二段。"

    def test_error(self):
        st = parse_btw_panel(FAILED)
        assert st.present and st.error == "Failed to get response"

    def test_copied_hint_still_counts_as_settled(self):
        st = parse_btw_panel(SETTLED.replace("c to copy", "Copied to clipboard"))
        assert st.settled


def _patches(captures, buffers, send_ok=True):
    """Patch tmux/session touchpoints; captures/buffers are consumed in order."""
    capture = AsyncMock(side_effect=list(captures))
    newest = AsyncMock(side_effect=list(buffers))
    return (
        patch.object(
            side_question.session_manager,
            "send_to_window",
            AsyncMock(return_value=(send_ok, "sent" if send_ok else "boom")),
        ),
        patch.object(side_question.tmux_manager, "capture_pane", capture),
        patch.object(
            side_question.tmux_manager, "send_keys", AsyncMock(return_value=True)
        ),
        patch.object(side_question, "_newest_buffer_name", newest),
        patch.object(side_question, "_tmux", AsyncMock(return_value="full **md**\n")),
        patch.object(side_question.asyncio, "sleep", AsyncMock()),
    )


async def test_happy_path_reads_buffer_and_closes_panel():
    patches = _patches(
        captures=["no panel yet", PENDING, SETTLED],
        buffers=["buffer1", "buffer2"],  # before pressing c, then the new one
    )
    with (
        patches[0] as send,
        patches[1],
        patches[2] as keys,
        patches[3],
        patches[4] as tmux,
        patches[5],
    ):
        ok, answer = await run_side_question("@1", "what?", timeout=30)

    assert ok and answer == "full **md**"
    send.assert_awaited_once_with("@1", "/btw what?")
    pressed = [c.args[1] for c in keys.await_args_list]
    assert pressed == ["c", "\x1b"]
    called = [c.args for c in tmux.await_args_list]
    assert ("show-buffer", "-b", "buffer2") in called
    assert ("delete-buffer", "-b", "buffer2") in called


async def test_falls_back_to_scraped_text_when_no_buffer_appears():
    patches = _patches(captures=[SETTLED], buffers=["buffer1"] * 8)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        ok, answer = await run_side_question("@1", "q", timeout=30)
    assert ok and answer == "好，简单说：这条路是可行的。\n\n第二段。"


async def test_error_panel_reports_and_closes():
    patches = _patches(captures=[FAILED], buffers=[])
    with patches[0], patches[1], patches[2] as keys, patches[3], patches[4], patches[5]:
        ok, answer = await run_side_question("@1", "q", timeout=30)
    assert not ok and answer == "Failed to get response"
    assert [c.args[1] for c in keys.await_args_list] == ["\x1b"]


async def test_timeout_when_panel_never_appears():
    patches = _patches(captures=["nothing"] * 50, buffers=[])
    clock = iter(range(0, 1000, 10))
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch.object(side_question.time, "monotonic", side_effect=lambda: next(clock)),
    ):
        ok, answer = await run_side_question("@1", "q", timeout=30)
    assert not ok and "panel never appeared" in answer


async def test_send_failure_short_circuits():
    patches = _patches(captures=[], buffers=[], send_ok=False)
    with (
        patches[0],
        patches[1] as capture,
        patches[2],
        patches[3],
        patches[4],
        patches[5],
    ):
        ok, answer = await run_side_question("@1", "q", timeout=30)
    assert not ok and answer == "boom"
    capture.assert_not_awaited()


async def test_second_question_on_same_window_is_rejected():
    side_question._active_windows.add("@9")
    try:
        ok, answer = await run_side_question("@9", "q", timeout=1)
    finally:
        side_question._active_windows.discard("@9")
    assert not ok and "already in progress" in answer
