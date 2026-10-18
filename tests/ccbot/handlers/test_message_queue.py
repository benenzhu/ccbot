"""Unit tests for message_queue — voice, tables, merging, and retry-until-delivered."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from ccbot.handlers import message_queue, message_sender
from ccbot.handlers.message_queue import MessageTask, _process_voice_task
from ccbot.monitor_state import DeliveryTicket
from ccbot.rate_limit import TelegramRateLimiter
from ccbot.tts import TtsApiError


@pytest.fixture(autouse=True)
def _fresh_chat_pacing():
    """Every test starts with all per-chat send slots free and no turn begun."""
    message_queue._chat_next_send.clear()
    message_queue._turn_has_output.clear()
    yield
    message_queue._turn_has_output.clear()
    message_queue._chat_next_send.clear()


def _voice_task(text: str = "你好") -> MessageTask:
    return MessageTask(task_type="voice", text=text, window_id="@1", thread_id=42)


@pytest.mark.asyncio
async def test_voice_transient_error_retried_once():
    bot = AsyncMock()
    synth = AsyncMock(side_effect=[TtsApiError("stream reset"), b"OggS-audio"])
    with (
        patch.object(message_queue, "synthesize_prepared", synth),
        patch.object(message_queue, "prepare_tts_segments", return_value=["你好"]),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock),
    ):
        await _process_voice_task(bot, 1, _voice_task())

    assert synth.call_count == 2
    bot.send_voice.assert_called_once()
    assert bot.send_voice.call_args.kwargs["voice"] == b"OggS-audio"


@pytest.mark.asyncio
async def test_voice_persistent_error_gives_up_after_retry():
    bot = AsyncMock()
    synth = AsyncMock(side_effect=TtsApiError("still broken"))
    with (
        patch.object(message_queue, "synthesize_prepared", synth),
        patch.object(message_queue, "prepare_tts_segments", return_value=["你好"]),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock),
    ):
        await _process_voice_task(bot, 1, _voice_task())

    assert synth.call_count == 2
    bot.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_unspeakable_text_not_retried():
    bot = AsyncMock()
    synth = AsyncMock(side_effect=ValueError("No speakable text"))
    with (
        patch.object(message_queue, "synthesize_prepared", synth),
        patch.object(message_queue, "prepare_tts_segments", return_value=["x"]),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
    ):
        await _process_voice_task(bot, 1, _voice_task("```code```"))

    assert synth.call_count == 1
    bot.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_empty_text_skipped():
    bot = AsyncMock()
    synth = AsyncMock()
    with patch.object(message_queue, "synthesize_prepared", synth):
        await _process_voice_task(bot, 1, _voice_task(""))

    synth.assert_not_called()
    bot.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_nothing_speakable_skipped():
    """prepare_tts_segments returning [] means the reply was all code."""
    bot = AsyncMock()
    synth = AsyncMock()
    with (
        patch.object(message_queue, "synthesize_prepared", synth),
        patch.object(message_queue, "prepare_tts_segments", return_value=[]),
    ):
        await _process_voice_task(bot, 1, _voice_task("```code```"))

    synth.assert_not_called()
    bot.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_segments_sent_in_order():
    bot = AsyncMock()
    segments = ["第一段。", "第二段。", "第三段。"]

    async def synth(seg: str) -> bytes:
        # Later segments finish faster — ordering must not follow completion
        await asyncio.sleep(0.03 if seg == segments[0] else 0.001)
        return f"audio:{seg}".encode()

    with (
        patch.object(
            message_queue, "synthesize_prepared", AsyncMock(side_effect=synth)
        ),
        patch.object(message_queue, "prepare_tts_segments", return_value=segments),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
    ):
        await _process_voice_task(bot, 1, _voice_task("long reply"))

    sent = [c.kwargs["voice"] for c in bot.send_voice.call_args_list]
    assert sent == [f"audio:{s}".encode() for s in segments]


@pytest.mark.asyncio
async def test_voice_failed_segment_skipped_rest_delivered():
    bot = AsyncMock()
    segments = ["一。", "二。", "三。"]

    async def synth(seg: str) -> bytes:
        if seg == "二。":
            raise ValueError("No speakable text")
        return f"audio:{seg}".encode()

    with (
        patch.object(
            message_queue, "synthesize_prepared", AsyncMock(side_effect=synth)
        ),
        patch.object(message_queue, "prepare_tts_segments", return_value=segments),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
    ):
        await _process_voice_task(bot, 1, _voice_task("reply"))

    sent = [c.kwargs["voice"] for c in bot.send_voice.call_args_list]
    assert sent == ["audio:一。".encode(), "audio:三。".encode()]


@pytest.mark.asyncio
async def test_voice_prefetch_overlaps_synthesis():
    """Segment N+1 synthesizes while N is sending, beating serial timing."""
    bot = AsyncMock()
    segments = ["一。", "二。", "三。", "四。"]
    delay = 0.05

    async def synth(seg: str) -> bytes:
        await asyncio.sleep(delay)
        return b"audio"

    with (
        patch.object(
            message_queue, "synthesize_prepared", AsyncMock(side_effect=synth)
        ),
        patch.object(message_queue, "prepare_tts_segments", return_value=segments),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=100
        ),
        patch.object(message_queue, "_pace", AsyncMock()),  # timing is about TTS
    ):
        loop = asyncio.get_running_loop()
        start = loop.time()
        await _process_voice_task(bot, 1, _voice_task("long"))
        elapsed = loop.time() - start

    assert bot.send_voice.call_count == len(segments)
    # Serial synthesis alone would cost len(segments) * delay
    assert elapsed < len(segments) * delay


# --- Table attachments: native rich message with PNG fallback ---


def _table_task() -> MessageTask:
    return MessageTask(
        task_type="content",
        window_id="@1",
        thread_id=42,
        parts=["text"],
        tables=[(["h"], [["v"]])],
    )


async def test_tables_sent_natively_when_rich_send_succeeds():
    bot = AsyncMock()
    with (
        patch.object(message_queue.config, "native_tables", True),
        patch.object(
            message_queue, "send_rich_markdown", AsyncMock(return_value=True)
        ) as rich,
        patch.object(message_queue, "send_photo", AsyncMock()) as photo,
        patch.object(message_queue, "render_table_image", AsyncMock()) as render,
    ):
        await message_queue._send_task_tables(bot, 777, _table_task())

    rich.assert_awaited_once()
    args, kwargs = rich.await_args
    assert args[:2] == (bot, 777)
    assert args[2].startswith("| h |")
    assert kwargs == {"message_thread_id": 42}
    render.assert_not_awaited()
    photo.assert_not_awaited()


async def test_tables_fall_back_to_png_when_rich_send_fails():
    bot = AsyncMock()
    with (
        patch.object(message_queue.config, "native_tables", True),
        patch.object(
            message_queue, "send_rich_markdown", AsyncMock(return_value=False)
        ),
        patch.object(message_queue, "send_photo", AsyncMock()) as photo,
        patch.object(
            message_queue, "render_table_image", AsyncMock(return_value=b"png")
        ) as render,
    ):
        await message_queue._send_task_tables(bot, 777, _table_task())

    render.assert_awaited_once_with(["h"], [["v"]])
    photo.assert_awaited_once()
    args, kwargs = photo.await_args
    assert args[:2] == (bot, 777)
    assert args[2] == [("table.png", b"png")]
    assert kwargs == {"message_thread_id": 42}


async def test_tables_go_straight_to_png_when_native_disabled():
    bot = AsyncMock()
    with (
        patch.object(message_queue.config, "native_tables", False),
        patch.object(message_queue, "send_rich_markdown", AsyncMock()) as rich,
        patch.object(message_queue, "send_photo", AsyncMock()) as photo,
        patch.object(
            message_queue, "render_table_image", AsyncMock(return_value=b"png")
        ),
    ):
        await message_queue._send_task_tables(bot, 777, _table_task())

    rich.assert_not_awaited()
    photo.assert_awaited_once()


async def test_merge_carries_tables_and_images_from_all_tasks():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    first = MessageTask(
        task_type="content",
        window_id="@1",
        parts=["a"],
        image_data=[("img", b"1")],
    )
    second = MessageTask(
        task_type="content",
        window_id="@1",
        parts=["b"],
        tables=[(["h"], [["v"]])],
        image_data=[("img", b"2")],
    )
    queue.put_nowait(second)

    merged, count = await message_queue._merge_content_tasks(
        queue, first, asyncio.Lock()
    )

    assert count == 1
    assert merged.parts == ["a\n\nb"]  # one message, one request
    assert merged.image_data == [("img", b"1"), ("img", b"2")]
    assert merged.tables == [(["h"], [["v"]])]


# --- Standalone table tasks keep their place between prose messages ---


async def test_table_task_breaks_content_merge():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    first = MessageTask(task_type="content", window_id="@1", parts=["before"])
    table = MessageTask(task_type="table", window_id="@1", tables=[(["h"], [["v"]])])
    after = MessageTask(task_type="content", window_id="@1", parts=["after"])
    queue.put_nowait(table)
    queue.put_nowait(after)

    merged, count = await message_queue._merge_content_tasks(
        queue, first, asyncio.Lock()
    )

    assert count == 0
    assert merged.parts == ["before"]
    # Queue order untouched: table still comes before the trailing prose
    assert queue.get_nowait() is table
    assert queue.get_nowait() is after


async def test_table_task_sends_table_and_counts_as_turn_output():
    bot = AsyncMock()
    task = MessageTask(
        task_type="table", window_id="@1", thread_id=42, tables=[(["h"], [["v"]])]
    )
    with (
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(message_queue, "_send_task_tables", AsyncMock()) as send,
        patch.object(message_queue, "_turn_has_output", set()) as turns,
    ):
        await message_queue._process_table_task(bot, 5, task)

    send.assert_awaited_once_with(bot, 777, task)
    assert turns == {(5, 42)}


async def test_enqueue_table_message_creates_table_task():
    bot = AsyncMock()
    with patch.object(message_queue, "get_or_create_queue") as goc:
        q: asyncio.Queue[MessageTask] = asyncio.Queue()
        goc.return_value = q
        await message_queue.enqueue_table_message(bot, 5, "@1", (["h"], [["v"]]), 42)

    task = q.get_nowait()
    assert task.task_type == "table"
    assert task.window_id == "@1"
    assert task.thread_id == 42
    assert task.tables == [(["h"], [["v"]])]


# --- Nothing but status is dropped: transient failures are retried ---


def _ticket() -> tuple[DeliveryTicket, list[str]]:
    """A ticket whose opener already let go, plus its delivery log."""
    delivered: list[str] = []
    ticket = DeliveryTicket(lambda: delivered.append("delivered"))
    held = ticket.hold()
    ticket.release()
    return held, delivered


def _content_patches(send: AsyncMock):
    return (
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(message_queue, "send_with_fallback", send),
        patch.object(message_queue, "_turn_has_output", set()),
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock),
    )


async def test_flood_control_retries_content_instead_of_dropping_it():
    bot = AsyncMock()
    task = MessageTask(task_type="content", window_id="@1", parts=["hello"])
    send = AsyncMock(side_effect=[RetryAfter(30), AsyncMock(message_id=9)])
    resolve, fallback, status, sleep = _content_patches(send)
    with resolve, fallback, status, sleep as slept:
        await message_queue._deliver_task(bot, 5, task)

    assert send.await_count == 2
    slept.assert_awaited_once_with(30.0)
    assert task.parts == []
    message_queue._flood_until.pop(5, None)


async def test_network_error_retries_only_the_parts_still_missing():
    bot = AsyncMock()
    task = MessageTask(task_type="content", window_id="@1", parts=["one", "two"])
    send = AsyncMock(
        side_effect=[
            AsyncMock(message_id=1),
            TimedOut(),
            NetworkError("connection reset"),
            AsyncMock(message_id=2),
        ]
    )
    resolve, fallback, status, sleep = _content_patches(send)
    pace = patch.object(message_queue, "_pace", AsyncMock())  # backoff only
    with resolve, fallback, status, pace, sleep as slept:
        await message_queue._deliver_task(bot, 5, task)

    sent = [call.args[2] for call in send.await_args_list]
    assert sent == ["one", "two", "two", "two"]  # "one" is never sent twice
    assert [call.args[0] for call in slept.await_args_list] == [1.0, 2.0]
    assert all(call.kwargs["raise_transient"] for call in send.await_args_list)


async def test_permanent_error_propagates_without_retry():
    bot = AsyncMock()
    task = MessageTask(task_type="content", window_id="@1", parts=["hello"])
    send = AsyncMock(side_effect=BadRequest("chat not found"))
    resolve, fallback, status, sleep = _content_patches(send)
    with resolve, fallback, status, sleep, pytest.raises(BadRequest):
        await message_queue._deliver_task(bot, 5, task)

    send.assert_awaited_once()


async def test_status_task_is_dropped_on_flood_control():
    bot = AsyncMock()
    task = MessageTask(task_type="status_update", window_id="@1", text="Working…")
    with (
        patch.object(
            message_queue,
            "_process_status_update_task",
            AsyncMock(side_effect=RetryAfter(300)),
        ) as process,
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock) as slept,
    ):
        await message_queue._deliver_task(bot, 6, task)

    process.assert_awaited_once()
    slept.assert_not_awaited()
    assert message_queue._flood_until.pop(6) > 0  # status traffic paused


async def test_retry_after_failed_edit_keeps_tool_message_id():
    bot = AsyncMock()
    bot.edit_message_text.side_effect = [RetryAfter(3), None]
    message_queue._tool_msg_ids[("tool-1", 5, 0)] = 1234
    task = MessageTask(
        task_type="content",
        window_id="@1",
        parts=["result"],
        tool_use_id="tool-1",
        content_type="tool_result",
    )
    send = AsyncMock()
    resolve, fallback, status, sleep = _content_patches(send)
    with resolve, fallback, status, sleep:
        await message_queue._deliver_task(bot, 5, task)

    # Second attempt still edited the tool_use message instead of sending anew
    assert bot.edit_message_text.await_count == 2
    assert bot.edit_message_text.await_args.kwargs["message_id"] == 1234
    send.assert_not_awaited()
    assert ("tool-1", 5, 0) not in message_queue._tool_msg_ids


async def test_attachments_are_not_resent_on_retry():
    bot = AsyncMock()
    task = MessageTask(
        task_type="content",
        window_id="@1",
        parts=["text"],
        image_data=[("image/png", b"png")],
    )
    send = AsyncMock(return_value=AsyncMock(message_id=1))
    photo = AsyncMock(side_effect=[RetryAfter(2), None])
    with (
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(message_queue, "send_with_fallback", send),
        patch.object(message_queue, "send_photo", photo),
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock),
    ):
        await message_queue._deliver_task(bot, 5, task)

    send.assert_awaited_once()  # the text went out before the flood ban
    assert photo.await_count == 2


async def test_upload_that_keeps_timing_out_is_given_up():
    bot = AsyncMock()
    task = MessageTask(
        task_type="content",
        window_id="@1",
        thread_id=42,
        image_data=[("image/png", b"png")],
    )
    with (
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(
            message_queue, "send_photo", AsyncMock(side_effect=TimedOut())
        ) as photo,
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock),
    ):
        await message_queue._deliver_task(bot, 5, task)

    assert photo.await_count == message_queue.UPLOAD_MAX_ATTEMPTS
    assert task.image_data is None


async def test_worker_releases_tickets_after_delivery_not_before():
    bot = AsyncMock()
    held, delivered = _ticket()
    gate = asyncio.Event()

    async def slow_send(*_args, **_kwargs):
        await gate.wait()
        return AsyncMock(message_id=1)

    resolve, fallback, status, _sleep = _content_patches(
        AsyncMock(side_effect=slow_send)
    )
    with resolve, fallback, status:
        queue = message_queue.get_or_create_queue(bot, 4242)
        try:
            queue.put_nowait(
                MessageTask(
                    task_type="content", window_id="@1", parts=["x"], tickets=[held]
                )
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert delivered == []  # queued / in flight: offset stays pinned
            gate.set()
            await queue.join()
            assert delivered == ["delivered"]
        finally:
            message_queue._queue_workers.pop(4242).cancel()
            message_queue._message_queues.pop(4242, None)
            message_queue._queue_locks.pop(4242, None)


async def test_cancelled_worker_keeps_ticket_so_restart_resends():
    bot = AsyncMock()
    held, delivered = _ticket()

    async def never_sent(*_args, **_kwargs):
        await asyncio.Event().wait()

    resolve, fallback, status, _sleep = _content_patches(
        AsyncMock(side_effect=never_sent)
    )
    with resolve, fallback, status:
        queue = message_queue.get_or_create_queue(bot, 4343)
        queue.put_nowait(
            MessageTask(
                task_type="content", window_id="@1", parts=["x"], tickets=[held]
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        worker = message_queue._queue_workers.pop(4343)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        message_queue._message_queues.pop(4343, None)
        message_queue._queue_locks.pop(4343, None)

    assert delivered == []


async def test_merge_carries_tickets_from_all_tasks():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    first_ticket, _ = _ticket()
    second_ticket, _ = _ticket()
    first = MessageTask(
        task_type="content", window_id="@1", parts=["a"], tickets=[first_ticket]
    )
    queue.put_nowait(
        MessageTask(
            task_type="content", window_id="@1", parts=["b"], tickets=[second_ticket]
        )
    )

    merged, _count = await message_queue._merge_content_tasks(
        queue, first, asyncio.Lock()
    )

    assert merged.tickets == [first_ticket, second_ticket]


async def test_enqueue_content_holds_the_message_ticket():
    bot = AsyncMock()
    delivered: list[str] = []
    ticket = DeliveryTicket(lambda: delivered.append("delivered"))
    with patch.object(message_queue, "get_or_create_queue") as goc:
        q: asyncio.Queue[MessageTask] = asyncio.Queue()
        goc.return_value = q
        await message_queue.enqueue_content_message(
            bot, 5, "@1", ["part"], ticket=ticket
        )

    ticket.release()  # the monitor lets go once the callback returns
    assert delivered == []
    q.get_nowait().tickets[0].release()
    assert delivered == ["delivered"]


# --- Per-chat pacing: Telegram bans chats that get more than ~1 message/s ---


async def test_pacing_allows_a_burst_then_one_message_per_interval():
    with (
        patch.object(message_queue.time, "monotonic", return_value=1000.0),
        patch.object(message_queue.asyncio, "sleep", new_callable=AsyncMock) as slept,
    ):
        for _ in range(message_queue.CHAT_SEND_BURST + 3):
            await message_queue._pace(777)

    assert [call.args[0] for call in slept.await_args_list] == [1.0, 2.0, 3.0]


async def test_pacing_is_per_chat_and_recovers_when_idle():
    clock = patch.object(message_queue.time, "monotonic", return_value=1000.0)
    with clock as now:
        for _ in range(message_queue.CHAT_SEND_BURST):
            assert message_queue._try_pace(777)
        assert not message_queue._try_pace(777)  # burst used up
        assert message_queue._try_pace(888)  # another chat is unaffected

        now.return_value += message_queue.CHAT_SEND_INTERVAL
        assert message_queue._try_pace(777)  # one slot per interval comes back


async def test_replay_sized_backlog_is_spread_out_not_blasted():
    bot = AsyncMock()
    task = MessageTask(
        task_type="content", window_id="@1", parts=[f"msg {i}" for i in range(10)]
    )
    send = AsyncMock(return_value=AsyncMock(message_id=1))
    resolve, fallback, status, sleep = _content_patches(send)
    with (
        patch.object(message_queue.time, "monotonic", return_value=1000.0),
        resolve,
        fallback,
        status,
        sleep as slept,
    ):
        await message_queue._deliver_task(bot, 5, task)

    assert send.await_count == 10
    # 3 go out at once, the other 7 are each booked one interval later
    assert [call.args[0] for call in slept.await_args_list] == [
        float(n) for n in range(1, 8)
    ]


async def test_status_update_is_skipped_while_chat_has_no_free_slot():
    bot = AsyncMock()
    task = MessageTask(
        task_type="status_update", window_id="@1", text="Working…", thread_id=42
    )
    with (
        patch.object(message_queue.time, "monotonic", return_value=1000.0),
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(message_queue, "_do_send_status_message", AsyncMock()) as send,
    ):
        for _ in range(message_queue.CHAT_SEND_BURST):
            message_queue._try_pace(777)
        await message_queue._process_status_update_task(bot, 5, task)
        send.assert_not_awaited()

        message_queue._chat_next_send.clear()
        await message_queue._process_status_update_task(bot, 5, task)
        send.assert_awaited_once()


# --- Backlog: a tool_use whose result is already queued goes out as the result ---


@pytest.fixture
def _clean_status_state():
    """Start and end with no status message, tool message or gone topic tracked."""
    for state in (
        message_queue._status_msg_info,
        message_queue._turn_has_output,
        message_queue._tool_msg_ids,
        message_sender._gone_topics,
    ):
        state.clear()
    yield
    for state in (
        message_queue._status_msg_info,
        message_queue._turn_has_output,
        message_queue._tool_msg_ids,
        message_sender._gone_topics,
    ):
        state.clear()


def _tool_task(kind: str, tool_id: str, thread_id: int | None = 42) -> MessageTask:
    return MessageTask(
        task_type="content",
        window_id="@1",
        thread_id=thread_id,
        parts=[f"{kind} {tool_id}"],
        tool_use_id=tool_id,
        content_type=kind,
    )


async def test_queued_tool_result_is_sent_in_place_of_its_tool_use():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    use_ticket, _ = _ticket()
    result_ticket, _ = _ticket()
    tool_use = _tool_task("tool_use", "A")
    tool_use.tickets = [use_ticket]
    other_use = _tool_task("tool_use", "B")
    result = _tool_task("tool_result", "A")
    result.tickets = [result_ticket]
    other_result = _tool_task("tool_result", "B")
    for item in (other_use, result, other_result):
        queue.put_nowait(item)

    sent, absorbed = await message_queue._absorb_queued_tool_result(
        queue, tool_use, asyncio.Lock()
    )

    assert absorbed == 1
    assert sent is result
    assert sent.tickets == [use_ticket, result_ticket]
    # Everything else keeps its order
    assert queue.get_nowait() is other_use
    assert queue.get_nowait() is other_result
    assert queue.empty()


async def test_tool_use_is_sent_as_is_without_its_result_queued():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    tool_use = _tool_task("tool_use", "A")
    other_thread = _tool_task("tool_result", "A", thread_id=43)
    other_tool = _tool_task("tool_result", "B")
    queue.put_nowait(other_thread)
    queue.put_nowait(other_tool)

    sent, absorbed = await message_queue._absorb_queued_tool_result(
        queue, tool_use, asyncio.Lock()
    )

    assert (sent, absorbed) == (tool_use, 0)
    assert queue.get_nowait() is other_thread
    assert queue.get_nowait() is other_tool


@pytest.mark.usefixtures("_clean_status_state")
async def test_worker_sends_backlogged_tool_call_once_instead_of_send_plus_edit():
    bot = AsyncMock()
    send = AsyncMock(return_value=AsyncMock(message_id=1))
    resolve, fallback, status, sleep = _content_patches(send)
    with resolve, fallback, status, sleep:
        queue = message_queue.get_or_create_queue(bot, 4444)
        try:
            queue.put_nowait(_tool_task("tool_use", "A"))
            queue.put_nowait(_tool_task("tool_result", "A"))
            await asyncio.wait_for(queue.join(), 1)  # both tasks accounted for
        finally:
            message_queue._queue_workers.pop(4444).cancel()
            message_queue._message_queues.pop(4444, None)
            message_queue._queue_locks.pop(4444, None)

    assert [call.args[2] for call in send.await_args_list] == ["tool_result A"]
    bot.edit_message_text.assert_not_awaited()
    assert ("A", 4444, 42) not in message_queue._tool_msg_ids


def _tool_call(tool_id: str, thread_id: int | None = 42) -> list[MessageTask]:
    """A tool_use and its tool_result; the result's text is the folded call."""
    tool_use = _tool_task("tool_use", tool_id, thread_id)
    result = _tool_task("tool_result", tool_id, thread_id)
    result.text = message_queue.TranscriptParser._fold_tool_message(
        f"Bash({tool_id}) -- Output 1 line\nout {tool_id}"
    )
    return [tool_use, result]


async def test_queued_tool_calls_of_a_topic_go_out_as_one_message():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    first_use, first = _tool_call("A")
    first_use.tickets = [_ticket()[0]]
    other_topic = _tool_task("tool_use", "X", thread_id=43)
    use_b, result_b = _tool_call("B")
    use_c, result_c = _tool_call("C")
    reply = MessageTask(task_type="content", window_id="@1", thread_id=42)
    use_d, result_d = _tool_call("D")
    for item in (use_b, other_topic, use_c, result_b, result_c, reply, use_d, result_d):
        queue.put_nowait(item)
    first = message_queue._as_one_call(first_use, first)

    sent, taken = await message_queue._batch_queued_tool_calls(
        queue, first, asyncio.Lock()
    )

    assert taken == 4
    assert sent.tool_use_id is None  # sent as it is, never edited
    assert sent.tickets == first_use.tickets
    lines = (sent.text or "").split("\n")
    assert lines[0].endswith("...")  # a dot per call, in the collapsed preview
    assert "1. Bash(A) -- Output 1 line" in lines
    assert "3. Bash(C) -- Output 1 line" in lines
    assert "── 3 ──\nout C" in (sent.text or "")
    # The topic's reply ends the batch; everything else keeps its order
    assert [queue.get_nowait() for _ in range(4)] == [
        other_topic,
        reply,
        use_d,
        result_d,
    ]


async def test_a_batch_stops_at_a_running_tool_and_at_five_calls():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    running = _tool_task("tool_use", "R")
    queue.put_nowait(running)
    sent, taken = await message_queue._batch_queued_tool_calls(
        queue, _tool_call("A")[1], asyncio.Lock()
    )
    assert taken == 0 and sent.tool_use_id == "A"
    assert queue.get_nowait() is running

    for tool_id in "BCDEFG":
        for item in _tool_call(tool_id):
            queue.put_nowait(item)
    sent, taken = await message_queue._batch_queued_tool_calls(
        queue, _tool_call("A")[1], asyncio.Lock()
    )
    assert taken == 8  # B..E with A makes five
    assert (sent.text or "").split("\n")[0].endswith(".....")
    assert queue.qsize() == 4  # F and G


@pytest.mark.usefixtures("_clean_status_state")
async def test_worker_batches_tool_calls_only_while_the_window_is_busy():
    async def sends(free: int) -> list[str]:
        bot = AsyncMock()
        bot.rate_limiter = await _limiter_with_free(free)
        send = AsyncMock(return_value=AsyncMock(message_id=1))
        resolve, fallback, status, sleep = _content_patches(send)
        with resolve, fallback, status, sleep:
            queue = message_queue.get_or_create_queue(bot, 4445)
            try:
                for tool_id in "AB":
                    for item in _tool_call(tool_id):
                        queue.put_nowait(item)
                await asyncio.wait_for(queue.join(), 1)
            finally:
                message_queue._queue_workers.pop(4445).cancel()
                message_queue._message_queues.pop(4445, None)
                message_queue._queue_locks.pop(4445, None)
        return [call.args[2] for call in send.await_args_list]

    assert await sends(message_queue.STATUS_RESERVE + 1) == [
        "tool_result A",
        "tool_result B",
    ]
    (batch,) = await sends(message_queue.STATUS_RESERVE)
    assert "1. Bash(A)" in batch and "2. Bash(B)" in batch


# --- Status is optional traffic: it yields to everything else ---


@pytest.mark.parametrize(
    "line, label",
    [
        pytest.param("Zesting… (2m 24s · ↓ 11.7k tokens)", "Zesting…", id="counters"),
        pytest.param("Zesting… (9s · ↓ 640 tok", "Zesting…", id="cut_off"),
        pytest.param("Brewed for 3m 14s · done 12:34 AM", None, id="no_details"),
    ],
)
def test_status_message_drops_details_that_would_go_stale(line: str, label: str | None):
    assert message_queue._status_label(line) == (label or line)


async def _limiter_with_free(free: int) -> TelegramRateLimiter:
    limiter = TelegramRateLimiter()
    for _ in range(limiter.window.free() - free):
        await limiter.window.acquire()
    return limiter


@pytest.mark.usefixtures("_clean_status_state")
async def test_status_leaves_the_last_window_slots_to_content():
    task = MessageTask(
        task_type="status_update", window_id="@1", text="Working…", thread_id=42
    )
    bot = AsyncMock()
    bot.rate_limiter = await _limiter_with_free(message_queue.STATUS_RESERVE)
    with (
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(message_queue, "_do_send_status_message", AsyncMock()) as send,
    ):
        await message_queue._process_status_update_task(bot, 5, task)
        send.assert_not_awaited()

        bot.rate_limiter = await _limiter_with_free(message_queue.STATUS_RESERVE + 1)
        await message_queue._process_status_update_task(bot, 5, task)
        send.assert_awaited_once()


@pytest.mark.usefixtures("_clean_status_state")
async def test_status_is_sent_once_and_never_edited():
    bot = AsyncMock()
    bot.send_message.return_value = MagicMock(message_id=9)

    def task(text: str) -> MessageTask:
        return MessageTask(
            task_type="status_update", window_id="@1", text=text, thread_id=42
        )

    with patch.object(
        message_queue.session_manager, "resolve_chat_id", return_value=777
    ):
        await message_queue._process_status_update_task(
            bot, 5, task("Zesting… (3s · thinking with high effort)")
        )
        await message_queue._process_status_update_task(
            bot, 5, task("Zesting… (40s · ↓ 2.1k tokens · still thinking)")
        )

    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.kwargs["text"].strip() == "Zesting…"
    bot.edit_message_text.assert_not_awaited()
    bot.send_chat_action.assert_not_awaited()
    assert message_queue._status_msg_info[(5, 42)] == (9, "@1", "Zesting…")


@pytest.mark.usefixtures("_clean_status_state")
async def test_no_status_once_the_turn_has_output_until_the_next_turn():
    bot = AsyncMock()
    bot.send_message.return_value = MagicMock(message_id=9)
    status = MessageTask(
        task_type="status_update", window_id="@1", text="Zesting…", thread_id=42
    )
    with (
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(message_queue, "_pace", AsyncMock()),
    ):
        content = MessageTask(
            task_type="content", window_id="@1", parts=["Bash(ls)"], thread_id=42
        )
        await message_queue._process_content_task(bot, 5, content)
        await message_queue._process_status_update_task(bot, 5, status)
        bot.send_message.assert_awaited_once()  # the content, no status after it

        message_queue.end_turn(5, 42)  # poller saw Claude idle
        await message_queue._process_status_update_task(bot, 5, status)
        assert bot.send_message.await_count == 2


async def test_status_is_off_unless_enabled():
    with (
        patch.object(message_queue.config, "show_status", False),
        patch.object(message_queue, "get_or_create_queue") as queue,
    ):
        await message_queue.enqueue_status_update(AsyncMock(), 5, "@1", "Zesting…", 42)
    queue.assert_not_called()


async def test_user_message_starts_a_new_turn():
    message_queue._turn_has_output.add((5, 42))
    with patch.object(message_queue, "get_or_create_queue"):
        await message_queue.enqueue_status_update(AsyncMock(), 5, "@1", None, 42)
    assert (5, 42) not in message_queue._turn_has_output


@pytest.mark.usefixtures("_clean_status_state")
async def test_tool_result_edit_leaves_the_status_message_in_place():
    bot = AsyncMock()
    message_queue._remember_status((5, 0), 55, "@1", "Working…")
    message_queue._tool_msg_ids[("tool-1", 5, 0)] = 1234
    task = _tool_task("tool_result", "tool-1", thread_id=None)
    with (
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(message_queue, "_pace", AsyncMock()),
    ):
        await message_queue._process_content_task(bot, 5, task)

    assert bot.edit_message_text.await_args.kwargs["message_id"] == 1234
    bot.delete_message.assert_not_awaited()  # status is still the last message
    assert message_queue._status_msg_info[(5, 0)][0] == 55


@pytest.mark.usefixtures("_clean_status_state")
async def test_attachments_after_tool_result_edit_do_not_land_below_status():
    bot = AsyncMock()
    message_queue._remember_status((5, 0), 55, "@1", "Working…")
    message_queue._tool_msg_ids[("tool-1", 5, 0)] = 1234
    task = _tool_task("tool_result", "tool-1", thread_id=None)
    task.image_data = [("image/png", b"png")]
    with (
        patch.object(
            message_queue.session_manager, "resolve_chat_id", return_value=777
        ),
        patch.object(message_queue, "_pace", AsyncMock()),
        patch.object(message_queue, "send_photo", AsyncMock()) as photo,
    ):
        await message_queue._process_content_task(bot, 5, task)

    bot.delete_message.assert_awaited_once_with(chat_id=777, message_id=55)
    photo.assert_awaited_once()


# --- Deleted topics: queued tasks for them are dropped, not sent ---


@pytest.mark.usefixtures("_clean_status_state")
async def test_worker_drops_tasks_for_a_deleted_topic():
    bot = AsyncMock()
    held, delivered = _ticket()
    message_sender._gone_topics.add((777, 42))
    send = AsyncMock()
    resolve, fallback, status, sleep = _content_patches(send)
    with resolve, fallback, status, sleep:
        queue = message_queue.get_or_create_queue(bot, 4545)
        try:
            queue.put_nowait(
                MessageTask(
                    task_type="content",
                    window_id="@1",
                    thread_id=42,
                    parts=["x"],
                    tickets=[held],
                )
            )
            await asyncio.wait_for(queue.join(), 1)
        finally:
            message_queue._queue_workers.pop(4545).cancel()
            message_queue._message_queues.pop(4545, None)
            message_queue._queue_locks.pop(4545, None)

    send.assert_not_awaited()
    assert delivered == ["delivered"]  # the monitor offset moves on


# --- Lag: a combined message is as late as its earliest-read part ---


async def test_merged_and_absorbed_tasks_keep_the_earliest_read_time():
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    first = MessageTask(task_type="content", window_id="@1", parts=["a"])
    queue.put_nowait(
        MessageTask(task_type="content", window_id="@1", parts=["b"], read_at=100.0)
    )
    merged, _ = await message_queue._merge_content_tasks(queue, first, asyncio.Lock())
    assert merged.read_at == 100.0  # unknown time doesn't win

    tool_use = _tool_task("tool_use", "A")
    tool_use.read_at = 200.0
    result = _tool_task("tool_result", "A")
    result.read_at = 205.0
    queue.put_nowait(result)
    sent, _ = await message_queue._absorb_queued_tool_result(
        queue, tool_use, asyncio.Lock()
    )
    assert sent.read_at == 200.0


# --- Merged tasks share messages: every part costs a request ---


def test_merged_parts_join_into_one_message_while_they_fit():
    parts = ["∴ Thinking…\n(thinking)"]
    message_queue._append_parts(parts, ["∴ Thinking…\nsecond", "tail"])
    assert parts == ["∴ Thinking…\n(thinking)\n\n∴ Thinking…\nsecond\n\ntail"]


def test_merged_parts_stay_apart_when_escaping_would_overflow():
    # 3000 raw chars of MarkdownV2-escaped punctuation convert to > 4096
    heavy = "a.b-c(d)" * 375
    parts = ["short"]
    message_queue._append_parts(parts, [heavy])
    assert parts == ["short", heavy]
