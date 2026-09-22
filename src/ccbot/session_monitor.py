"""Session monitoring service — watches JSONL files for new messages.

Runs an async polling loop that:
  1. Loads the current session_map to know which sessions to watch.
  2. Detects session_map changes (new/changed/deleted windows) and cleans up.
  3. Reads new JSONL lines from each session file using byte-offset tracking.
  4. Parses entries via TranscriptParser and emits NewMessage objects to a callback.

Delivery is at-least-once: every NewMessage carries a DeliveryTicket that pins
the persisted offset at its JSONL line until the message has been sent.
History replays are compacted (config.replay_compact): thinking and tool
messages are dropped, each run of them replaced by one divider line.

Optimizations: mtime cache skips unchanged files; byte offset avoids re-reading.

Key classes: SessionMonitor, NewMessage, SessionInfo.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

import aiofiles

from .config import config
from .monitor_state import DeliveryTicket, MonitorState, TrackedSession
from .tmux_manager import tmux_manager
from .transcript_parser import TranscriptParser
from .utils import read_cwd_from_jsonl

logger = logging.getLogger(__name__)

# Compact replay: content types dropped, and the line that stands in for them
REPLAY_DROPPED_TYPES = frozenset({"thinking", "tool_use", "tool_result"})
REPLAY_DIVIDER = "┄┄┄┄┄┄┄┄┄┄"


@dataclass
class SessionInfo:
    """Information about a Claude Code session."""

    session_id: str
    file_path: Path


@dataclass
class NewMessage:
    """A new message detected by the monitor."""

    session_id: str
    text: str
    is_complete: bool  # True when stop_reason is set (final message)
    content_type: str = "text"  # "text", "thinking", "tool_use", "tool_result"
    tool_use_id: str | None = None
    role: str = "assistant"  # "user" or "assistant"
    tool_name: str | None = None  # For tool_use messages, the tool name
    image_data: list[tuple[str, bytes]] | None = None  # From tool_result images
    is_replay: bool = False  # True when replaying past history (e.g. --resume)
    # Pins the persisted offset until this message is delivered. Whoever
    # queues the message takes a hold(); the monitor releases its own
    # reference once the callback returns.
    ticket: DeliveryTicket | None = None


class SessionMonitor:
    """Monitors Claude Code sessions for new assistant messages.

    Uses simple async polling with aiofiles for non-blocking I/O.
    Emits both intermediate and complete assistant messages.
    """

    def __init__(
        self,
        projects_path: Path | None = None,
        poll_interval: float | None = None,
        state_file: Path | None = None,
    ):
        self.projects_path = (
            projects_path if projects_path is not None else config.claude_projects_path
        )
        self.poll_interval = (
            poll_interval if poll_interval is not None else config.monitor_poll_interval
        )

        self.state = MonitorState(state_file=state_file or config.monitor_state_file)
        self.state.load()

        self._running = False
        self._task: asyncio.Task | None = None
        self._message_callback: Callable[[NewMessage], Awaitable[None]] | None = None
        # Per-session pending tool_use state carried across poll cycles
        self._pending_tools: dict[str, dict[str, Any]] = {}  # session_id -> pending
        # Track last known session_map for detecting changes
        # Keys may be window_id (@12) or window_name (old format) during transition
        self._last_session_map: dict[str, str] = {}  # window_key -> session_id
        # In-memory mtime cache for quick file change detection (not persisted)
        self._file_mtimes: dict[str, float] = {}  # session_id -> last_seen_mtime
        # Replaying sessions currently inside a run of dropped blocks (the
        # divider for that run has already been emitted).
        self._replay_gap_open: set[str] = set()
        # Forks copy messages with their original UUIDs but rewrite the JSONL,
        # so a source byte offset cannot be used to skip inherited history.
        self._fork_history_uuids: dict[str, set[str]] = {}

    async def skip_fork_history(self, session_id: str, source_path: str) -> None:
        """Remember inherited message UUIDs before starting a fork without replay."""
        uuids: set[str] = set()
        async with aiofiles.open(source_path, "r", encoding="utf-8") as f:
            async for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and isinstance(entry.get("uuid"), str):
                    uuids.add(entry["uuid"])
        self._fork_history_uuids[session_id] = uuids

    def mark_replay(self, session_id: str) -> None:
        """Flag a tracked session as replaying history.

        Messages read from it are flagged is_replay (compact, no TTS) until
        the offset first reaches EOF. Persisted with the session, so it
        survives a restart in the middle of the replay.
        """
        tracked = self.state.get_session(session_id)
        if tracked is None:
            logger.warning("mark_replay: session %s is not tracked", session_id)
            return
        tracked.replaying = True
        self.state.update_session(tracked)

    def set_message_callback(
        self, callback: Callable[[NewMessage], Awaitable[None]]
    ) -> None:
        self._message_callback = callback

    async def _get_active_cwds(self) -> set[str]:
        """Get normalized cwds of all active tmux windows."""
        cwds = set()
        windows = await tmux_manager.list_windows()
        for w in windows:
            try:
                cwds.add(str(Path(w.cwd).resolve()))
            except (OSError, ValueError):
                cwds.add(w.cwd)
        return cwds

    async def scan_projects(self) -> list[SessionInfo]:
        """Scan projects that have active tmux windows."""
        active_cwds = await self._get_active_cwds()
        if not active_cwds:
            return []

        sessions = []

        if not self.projects_path.exists():
            return sessions

        for project_dir in self.projects_path.iterdir():
            if not project_dir.is_dir():
                continue

            index_file = project_dir / "sessions-index.json"
            original_path = ""
            indexed_ids: set[str] = set()

            if index_file.exists():
                try:
                    async with aiofiles.open(index_file, "r") as f:
                        content = await f.read()
                    index_data = json.loads(content)
                    entries = index_data.get("entries", [])
                    original_path = index_data.get("originalPath", "")

                    for entry in entries:
                        session_id = entry.get("sessionId", "")
                        full_path = entry.get("fullPath", "")
                        project_path = entry.get("projectPath", original_path)

                        if not session_id or not full_path:
                            continue

                        try:
                            norm_pp = str(Path(project_path).resolve())
                        except (OSError, ValueError):
                            norm_pp = project_path
                        if norm_pp not in active_cwds:
                            continue

                        indexed_ids.add(session_id)
                        file_path = Path(full_path)
                        if file_path.exists():
                            sessions.append(
                                SessionInfo(
                                    session_id=session_id,
                                    file_path=file_path,
                                )
                            )

                except (json.JSONDecodeError, OSError) as e:
                    logger.debug(f"Error reading index {index_file}: {e}")

            # Pick up un-indexed .jsonl files
            try:
                for jsonl_file in project_dir.glob("*.jsonl"):
                    session_id = jsonl_file.stem
                    if session_id in indexed_ids:
                        continue

                    # Determine project_path for this file
                    file_project_path = original_path
                    if not file_project_path:
                        file_project_path = await asyncio.to_thread(
                            read_cwd_from_jsonl, jsonl_file
                        )
                    if not file_project_path:
                        dir_name = project_dir.name
                        if dir_name.startswith("-"):
                            file_project_path = dir_name.replace("-", "/")

                    try:
                        norm_fp = str(Path(file_project_path).resolve())
                    except (OSError, ValueError):
                        norm_fp = file_project_path

                    if norm_fp not in active_cwds:
                        continue

                    sessions.append(
                        SessionInfo(
                            session_id=session_id,
                            file_path=jsonl_file,
                        )
                    )
            except OSError as e:
                logger.debug(f"Error scanning jsonl files in {project_dir}: {e}")

        return sessions

    async def _read_new_lines(
        self, session: TrackedSession, file_path: Path
    ) -> list[dict]:
        """Read new lines from a session file (entries only, see below)."""
        return [
            data
            for _, data in await self._read_new_lines_with_offsets(session, file_path)
        ]

    async def _read_new_lines_with_offsets(
        self, session: TrackedSession, file_path: Path
    ) -> list[tuple[int, dict]]:
        """Read new lines from a session file using byte offset for efficiency.

        Returns (line_start_offset, entry) pairs so each message can pin the
        persisted offset at its own line until it is delivered.

        Detects file truncation (e.g. after /clear) and resets offset.
        Recovers from corrupted offsets (mid-line) by scanning to next line.
        Retries incomplete writes, but skips malformed newline-terminated records.
        """
        new_entries: list[tuple[int, dict]] = []
        try:
            # Read bytes so an incomplete UTF-8 character at EOF cannot prevent
            # decoding earlier, complete records in the same buffered read.
            async with aiofiles.open(file_path, "rb") as f:
                # Get file size to detect truncation
                await f.seek(0, 2)  # Seek to end
                file_size = await f.tell()

                # Detect file truncation: if offset is beyond file size, reset
                if session.last_byte_offset > file_size:
                    logger.info(
                        "File truncated for session %s "
                        "(offset %d > size %d). Resetting.",
                        session.session_id,
                        session.last_byte_offset,
                        file_size,
                    )
                    session.last_byte_offset = 0

                # A byte offset is at a line boundary only if the preceding
                # byte is a newline. Looking for '{' can mistake a nested
                # object for a record, or reject a valid indented record.
                if session.last_byte_offset > 0:
                    await f.seek(session.last_byte_offset - 1)
                    if await f.read(1) != b"\n":
                        remainder = await f.readline()
                        if remainder.endswith(b"\n"):
                            logger.warning(
                                "Corrupted offset %d in session %s (mid-line), "
                                "scanning to next line",
                                session.last_byte_offset,
                                session.session_id,
                            )
                            session.last_byte_offset = await f.tell()
                        return []
                await f.seek(session.last_byte_offset)

                # Only an unterminated invalid line may still be in flight.
                # Retrying a malformed record that already has a newline would
                # block every subsequent message in this session forever.
                safe_offset = session.last_byte_offset
                async for line in f:
                    try:
                        data = TranscriptParser.parse_line(line.decode("utf-8"))
                    except UnicodeDecodeError:
                        data = None
                    if isinstance(data, dict):
                        if data:
                            new_entries.append((safe_offset, data))
                    elif line.strip():
                        if not line.endswith(b"\n"):
                            logger.debug(
                                "Partial JSONL line in session %s at byte offset %d, "
                                "will retry next cycle",
                                session.session_id,
                                safe_offset,
                            )
                            break
                        logger.warning(
                            "Skipping malformed JSONL line in session %s "
                            "at byte offset %d",
                            session.session_id,
                            safe_offset,
                        )
                    safe_offset = await f.tell()

                session.last_byte_offset = safe_offset

        except OSError as e:
            logger.error("Error reading session file %s: %s", file_path, e)
        return new_entries

    async def check_for_updates(self, active_session_ids: set[str]) -> list[NewMessage]:
        """Check all sessions for new assistant messages.

        Reads from last byte offset. Emits both intermediate
        (stop_reason=null) and complete messages.

        Args:
            active_session_ids: Set of session IDs currently in session_map
        """
        new_messages = []

        # Scan projects to get available session files
        sessions = await self.scan_projects()

        # Only process sessions that are in session_map
        for session_info in sessions:
            if session_info.session_id not in active_session_ids:
                continue
            try:
                tracked = self.state.get_session(session_info.session_id)

                if tracked is None:
                    # For new sessions, initialize offset to end of file
                    # to avoid re-processing old messages
                    try:
                        file_size = session_info.file_path.stat().st_size
                        current_mtime = session_info.file_path.stat().st_mtime
                    except OSError:
                        file_size = 0
                        current_mtime = 0.0
                    tracked = TrackedSession(
                        session_id=session_info.session_id,
                        file_path=str(session_info.file_path),
                        last_byte_offset=file_size,
                    )
                    self.state.update_session(tracked)
                    self._file_mtimes[session_info.session_id] = current_mtime
                    logger.info(f"Started tracking session: {session_info.session_id}")
                    continue

                # Check mtime + file size to see if file has changed
                try:
                    st = session_info.file_path.stat()
                    current_mtime = st.st_mtime
                    current_size = st.st_size
                except OSError:
                    continue

                last_mtime = self._file_mtimes.get(session_info.session_id, 0.0)
                if (
                    current_mtime <= last_mtime
                    and current_size <= tracked.last_byte_offset
                ):
                    # File hasn't changed, skip reading
                    continue

                # Replay state is captured before the read: everything in this
                # batch is old history, so it must not be spoken aloud.
                was_replaying = tracked.replaying

                # File changed, read new content from last offset
                previous_offset = tracked.last_byte_offset
                new_lines = await self._read_new_lines_with_offsets(
                    tracked, session_info.file_path
                )
                self._file_mtimes[session_info.session_id] = current_mtime

                inherited = self._fork_history_uuids.get(session_info.session_id)
                if inherited is not None:
                    new_lines = [
                        (offset, entry)
                        for offset, entry in new_lines
                        if entry.get("uuid") not in inherited
                    ]
                    # Keep the filter across partial writes of the copied
                    # transcript, until the first new conversation message.
                    if any(
                        entry.get("type") in ("user", "assistant") and entry.get("uuid")
                        for _, entry in new_lines
                    ):
                        self._fork_history_uuids.pop(session_info.session_id, None)

                line_offsets = [offset for offset, _ in new_lines]
                new_entries = [entry for _, entry in new_lines]

                if new_entries:
                    logger.debug(
                        f"Read {len(new_entries)} new entries for "
                        f"session {session_info.session_id}"
                    )

                # Parse new entries using the shared logic, carrying over pending tools
                carry = self._pending_tools.get(session_info.session_id, {})
                parsed_entries, remaining = TranscriptParser.parse_entries(
                    new_entries,
                    pending_tools=carry,
                )
                if remaining:
                    self._pending_tools[session_info.session_id] = remaining
                else:
                    self._pending_tools.pop(session_info.session_id, None)

                for entry in parsed_entries:
                    if not entry.text and not entry.image_data:
                        continue
                    # Skip user messages unless show_user_messages is enabled
                    if entry.role == "user" and not config.show_user_messages:
                        continue

                    line_offset = (
                        line_offsets[entry.source_index]
                        if 0 <= entry.source_index < len(line_offsets)
                        else previous_offset
                    )
                    # Also true for history re-read after a restart
                    is_replay = was_replaying or tracked.is_replay(line_offset)

                    text = entry.text
                    content_type = entry.content_type
                    dropped = (
                        is_replay
                        and config.replay_compact
                        and content_type in REPLAY_DROPPED_TYPES
                    )
                    if dropped:
                        # One divider stands in for the whole run
                        if session_info.session_id in self._replay_gap_open:
                            continue
                        self._replay_gap_open.add(session_info.session_id)
                        text, content_type = REPLAY_DIVIDER, "text"
                    else:
                        self._replay_gap_open.discard(session_info.session_id)

                    # The ticket pins the saved offset at this message's line
                    # until it is delivered (opened before the save below).
                    new_messages.append(
                        NewMessage(
                            session_id=session_info.session_id,
                            text=text,
                            is_complete=True,
                            content_type=content_type,
                            tool_use_id=None if dropped else entry.tool_use_id,
                            role="assistant" if dropped else entry.role,
                            tool_name=None if dropped else entry.tool_name,
                            image_data=None if dropped else entry.image_data,
                            is_replay=is_replay,
                            ticket=self.state.open_ticket(tracked, line_offset),
                        )
                    )

                # Replay ends once we've caught up with the file; a long
                # history may take several cycles (partial lines, growth).
                if was_replaying and tracked.last_byte_offset >= current_size:
                    tracked.replaying = False
                    tracked.replay_until = tracked.last_byte_offset
                    self.state.update_session(tracked)
                    self._replay_gap_open.discard(session_info.session_id)

                if tracked.last_byte_offset != previous_offset:
                    self.state.update_session(tracked)

            except OSError as e:
                logger.debug(f"Error processing session {session_info.session_id}: {e}")

        self.state.save_if_dirty()
        return new_messages

    async def _load_current_session_map(self) -> dict[str, str]:
        """Load current session_map and return window_key -> session_id mapping.

        Keys in session_map are formatted as "tmux_session:window_id"
        (e.g. "ccbot:@12"). Old-format keys ("ccbot:window_name") are also
        accepted so that sessions running before a code upgrade continue
        to be monitored until the hook re-fires with new format.
        Only entries matching our tmux_session_name are processed.
        """
        window_to_session: dict[str, str] = {}
        if config.session_map_file.exists():
            try:
                async with aiofiles.open(config.session_map_file, "r") as f:
                    content = await f.read()
                session_map = json.loads(content)
                prefix = f"{config.tmux_session_name}:"
                for key, info in session_map.items():
                    # Only process entries for our tmux session
                    if not key.startswith(prefix):
                        continue
                    window_key = key[len(prefix) :]
                    session_id = info.get("session_id", "")
                    if session_id:
                        window_to_session[window_key] = session_id
            except (json.JSONDecodeError, OSError):
                pass
        return window_to_session

    async def _cleanup_all_stale_sessions(self) -> None:
        """Clean up all tracked sessions not in current session_map (used on startup)."""
        current_map = await self._load_current_session_map()
        active_session_ids = set(current_map.values())

        stale_sessions = []
        for session_id in self.state.tracked_sessions.keys():
            if session_id not in active_session_ids:
                stale_sessions.append(session_id)

        if stale_sessions:
            logger.info(
                f"[Startup cleanup] Removing {len(stale_sessions)} stale sessions"
            )
            for session_id in stale_sessions:
                self.state.remove_session(session_id)
                self._file_mtimes.pop(session_id, None)
                self._fork_history_uuids.pop(session_id, None)
            self.state.save_if_dirty()

    async def _detect_and_cleanup_changes(self) -> dict[str, str]:
        """Detect session_map changes and cleanup replaced/removed sessions.

        Returns current session_map for further processing.
        """
        current_map = await self._load_current_session_map()

        sessions_to_remove: set[str] = set()

        # Check for window session changes (window exists in both, but session_id changed)
        for window_id, old_session_id in self._last_session_map.items():
            new_session_id = current_map.get(window_id)
            if new_session_id and new_session_id != old_session_id:
                logger.info(
                    "Window '%s' session changed: %s -> %s",
                    window_id,
                    old_session_id,
                    new_session_id,
                )
                sessions_to_remove.add(old_session_id)

        # Check for deleted windows (window in old map but not in current)
        old_windows = set(self._last_session_map.keys())
        current_windows = set(current_map.keys())
        deleted_windows = old_windows - current_windows

        for window_id in deleted_windows:
            old_session_id = self._last_session_map[window_id]
            logger.info(
                "Window '%s' deleted, removing session %s",
                window_id,
                old_session_id,
            )
            sessions_to_remove.add(old_session_id)

        # Perform cleanup
        if sessions_to_remove:
            for session_id in sessions_to_remove:
                self.state.remove_session(session_id)
                self._file_mtimes.pop(session_id, None)
                self._fork_history_uuids.pop(session_id, None)
            self.state.save_if_dirty()

        # Update last known map
        self._last_session_map = current_map

        return current_map

    async def _monitor_loop(self) -> None:
        """Background loop for checking session updates.

        Uses simple async polling with aiofiles for non-blocking I/O.
        """
        logger.info("Session monitor started, polling every %ss", self.poll_interval)

        # Deferred import to avoid circular dependency (cached once)
        from .session import session_manager

        # Clean up all stale sessions on startup
        await self._cleanup_all_stale_sessions()
        # Initialize last known session_map
        self._last_session_map = await self._load_current_session_map()

        while self._running:
            try:
                # Load hook-based session map updates
                await session_manager.load_session_map()

                # Detect session_map changes and cleanup replaced/removed sessions
                current_map = await self._detect_and_cleanup_changes()
                active_session_ids = set(current_map.values())

                # Check for new messages (all I/O is async)
                new_messages = await self.check_for_updates(active_session_ids)

                for msg in new_messages:
                    status = "complete" if msg.is_complete else "streaming"
                    preview = msg.text[:80] + ("..." if len(msg.text) > 80 else "")
                    logger.info("[%s] session=%s: %s", status, msg.session_id, preview)
                    try:
                        if self._message_callback:
                            await self._message_callback(msg)
                    except Exception as e:
                        logger.error(f"Message callback error: {e}")
                    finally:
                        # Queued tasks hold their own references; a message
                        # nobody queued counts as delivered right here.
                        if msg.ticket:
                            msg.ticket.release()

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            await asyncio.sleep(self.poll_interval)

        logger.info("Session monitor stopped")

    def start(self) -> None:
        if self._running:
            logger.warning("Monitor already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self.state.save()
        logger.info("Session monitor stopped and state saved")
