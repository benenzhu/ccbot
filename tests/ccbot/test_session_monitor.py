"""Unit tests for SessionMonitor JSONL reading and offset handling."""

import json
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from ccbot.monitor_state import TrackedSession
from ccbot.session_monitor import SessionInfo, SessionMonitor


class TestReadNewLinesOffsetRecovery:
    """Tests for _read_new_lines offset corruption recovery."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a SessionMonitor with temp state file."""
        return SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "monitor_state.json",
        )

    @pytest.mark.asyncio
    async def test_mid_line_offset_recovery(self, monitor, tmp_path, make_jsonl_entry):
        """Recover from corrupted offset pointing mid-line."""
        # Create JSONL file with two valid lines
        jsonl_file = tmp_path / "session.jsonl"
        entry1 = make_jsonl_entry(msg_type="assistant", content="first message")
        entry2 = make_jsonl_entry(msg_type="assistant", content="second message")
        jsonl_file.write_text(
            json.dumps(entry1) + "\n" + json.dumps(entry2) + "\n",
            encoding="utf-8",
        )

        # Calculate offset pointing into the middle of line 1
        line1_bytes = len(json.dumps(entry1).encode("utf-8")) // 2
        session = TrackedSession(
            session_id="test-session",
            file_path=str(jsonl_file),
            last_byte_offset=line1_bytes,  # Mid-line (corrupted)
        )

        # Read should recover and return empty (offset moved to next line)
        result = await monitor._read_new_lines(session, jsonl_file)

        # Should return empty list (recovery skips to next line, no new content yet)
        assert result == []

        # Offset should now point to start of line 2
        line1_full = len(json.dumps(entry1).encode("utf-8")) + 1  # +1 for newline
        assert session.last_byte_offset == line1_full

    @pytest.mark.asyncio
    async def test_valid_offset_reads_normally(
        self, monitor, tmp_path, make_jsonl_entry
    ):
        """Normal reading when offset points to line start."""
        jsonl_file = tmp_path / "session.jsonl"
        entry1 = make_jsonl_entry(msg_type="assistant", content="first")
        entry2 = make_jsonl_entry(msg_type="assistant", content="second")
        jsonl_file.write_text(
            json.dumps(entry1) + "\n" + json.dumps(entry2) + "\n",
            encoding="utf-8",
        )

        # Offset at 0 should read both lines
        session = TrackedSession(
            session_id="test-session",
            file_path=str(jsonl_file),
            last_byte_offset=0,
        )

        result = await monitor._read_new_lines(session, jsonl_file)

        assert len(result) == 2
        assert session.last_byte_offset == jsonl_file.stat().st_size

    @pytest.mark.asyncio
    async def test_truncation_detection(self, monitor, tmp_path, make_jsonl_entry):
        """Detect file truncation and reset offset."""
        jsonl_file = tmp_path / "session.jsonl"
        entry = make_jsonl_entry(msg_type="assistant", content="content")
        jsonl_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        # Set offset beyond file size (simulates truncation)
        session = TrackedSession(
            session_id="test-session",
            file_path=str(jsonl_file),
            last_byte_offset=9999,  # Beyond file size
        )

        result = await monitor._read_new_lines(session, jsonl_file)

        # Should reset offset to 0 and read the line
        assert session.last_byte_offset == jsonl_file.stat().st_size
        assert len(result) == 1

    @pytest.mark.parametrize(
        "bad_line",
        [
            b'{"message":"unfinished',
            b'{"gitBra{"type":"assistant"}',
            b'{"message":"\xff"}',
            b"[]",
            b"null",
        ],
        ids=["truncated-json", "spliced-records", "invalid-utf8", "array", "null"],
    )
    async def test_malformed_record_does_not_block_later_messages(
        self, monitor, tmp_path, make_jsonl_entry, caplog, bad_line
    ):
        """A bad record with a newline is skipped once, preserving later messages."""
        path = tmp_path / "session.jsonl"
        first = make_jsonl_entry(content="before corruption")
        second = make_jsonl_entry(content="恢复后的消息")
        prefix = (json.dumps(first) + "\n").encode()
        path.write_bytes(
            prefix
            + bad_line
            + b"\n"
            + (json.dumps(second, ensure_ascii=False) + "\n").encode()
        )
        session = TrackedSession(session_id="test-session", file_path=str(path))

        assert await monitor._read_new_lines(session, path) == [first, second]
        assert session.last_byte_offset == path.stat().st_size
        assert await monitor._read_new_lines(session, path) == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert f"byte offset {len(prefix)}" in warnings[0].getMessage()

    @pytest.mark.parametrize("split_utf8", [False, True], ids=["json", "utf8"])
    async def test_partial_write_is_retried_without_losing_messages(
        self, monitor, tmp_path, make_jsonl_entry, caplog, split_utf8
    ):
        """Read complete records while retaining a partial JSON/UTF-8 tail."""
        path = tmp_path / "session.jsonl"
        first = make_jsonl_entry(content="complete message")
        second = make_jsonl_entry(content="你好")
        prefix = (json.dumps(first) + "\n").encode()
        tail = (json.dumps(second, ensure_ascii=False) + "\n").encode()
        split = tail.index("你".encode()) + 1 if split_utf8 else len(tail) // 2
        path.write_bytes(prefix + tail[:split])
        session = TrackedSession(session_id="test-session", file_path=str(path))

        assert await monitor._read_new_lines(session, path) == [first]
        assert session.last_byte_offset == len(prefix)
        assert await monitor._read_new_lines(session, path) == []
        assert session.last_byte_offset == len(prefix)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

        with path.open("ab") as f:
            f.write(tail[split:])
        assert await monitor._read_new_lines(session, path) == [second]
        assert session.last_byte_offset == path.stat().st_size
        assert await monitor._read_new_lines(session, path) == []

    async def test_whitespace_and_empty_objects_do_not_block_records(
        self, monitor, tmp_path, make_jsonl_entry, caplog
    ):
        """Line boundaries allow blank lines, empty objects and indentation."""
        path = tmp_path / "session.jsonl"
        prefix = b'{"type":"system"}\n'
        entry = make_jsonl_entry(content="next message")
        path.write_bytes(
            prefix + b"\n \t\r\n{}\n  " + json.dumps(entry).encode() + b"\n"
        )
        session = TrackedSession(
            session_id="test-session",
            file_path=str(path),
            last_byte_offset=len(prefix),
        )

        assert await monitor._read_new_lines(session, path) == [entry]
        assert session.last_byte_offset == path.stat().st_size
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.parametrize("split_utf8", [False, True], ids=["nested-object", "utf8"])
    async def test_mid_line_offset_uses_byte_boundaries(
        self, monitor, tmp_path, make_jsonl_entry, split_utf8
    ):
        """Neither a nested '{' nor a UTF-8 continuation is a record boundary."""
        path = tmp_path / "session.jsonl"
        first = make_jsonl_entry(content="你好")
        second = make_jsonl_entry(content="next message")
        prefix = (json.dumps(first, ensure_ascii=False) + "\n").encode()
        offset = (
            prefix.index("你".encode()) + 1
            if split_utf8
            else prefix.index(b'{"content"')
        )
        path.write_bytes(prefix + (json.dumps(second) + "\n").encode())
        session = TrackedSession(
            session_id="test-session", file_path=str(path), last_byte_offset=offset
        )

        assert await monitor._read_new_lines(session, path) == []
        assert session.last_byte_offset == len(prefix)
        assert await monitor._read_new_lines(session, path) == [second]

    async def test_mid_line_recovery_waits_for_a_newline(
        self, monitor, tmp_path, make_jsonl_entry
    ):
        """An offset inside an unfinished record stays put until its line ends."""
        path = tmp_path / "session.jsonl"
        first = (json.dumps(make_jsonl_entry(content="partial")) + "\n").encode()
        second = make_jsonl_entry(content="next message")
        split = len(first) // 2
        path.write_bytes(first[:split])
        session = TrackedSession(
            session_id="test-session", file_path=str(path), last_byte_offset=10
        )

        assert await monitor._read_new_lines(session, path) == []
        assert session.last_byte_offset == 10
        with path.open("ab") as f:
            f.write(first[split:] + (json.dumps(second) + "\n").encode())
        assert await monitor._read_new_lines(session, path) == []
        assert session.last_byte_offset == len(first)
        assert await monitor._read_new_lines(session, path) == [second]

    async def test_partial_write_does_not_rewrite_state_each_poll(
        self, monitor, tmp_path, make_jsonl_entry, monkeypatch
    ):
        """Waiting for disk writes must not cause repeated fsyncs of unchanged state."""
        path = tmp_path / "session.jsonl"
        entry = make_jsonl_entry(content="complete after append")
        line = (json.dumps(entry) + "\n").encode()
        split = len(line) // 2
        path.write_bytes(line[:split])
        session = TrackedSession(session_id="test-session", file_path=str(path))
        monitor.state.update_session(session)
        monitor.state.save()
        save = Mock(wraps=monitor.state.save)
        monkeypatch.setattr(monitor.state, "save", save)
        monkeypatch.setattr(
            monitor,
            "scan_projects",
            AsyncMock(return_value=[SessionInfo(session.session_id, path)]),
        )

        assert await monitor.check_for_updates({session.session_id}) == []
        assert await monitor.check_for_updates({session.session_id}) == []
        save.assert_not_called()

        with path.open("ab") as f:
            f.write(line[split:])
        messages = await monitor.check_for_updates({session.session_id})
        assert len(messages) == 1
        assert messages[0].text == "complete after append"
        assert session.last_byte_offset == path.stat().st_size
        save.assert_called_once()
        persisted = json.loads(monitor.state.state_file.read_text())
        assert persisted["tracked_sessions"][session.session_id][
            "last_byte_offset"
        ] == len(line)
