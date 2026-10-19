"""Session monitoring service — watches JSONL files for new messages.

Runs an async polling loop that:
  1. Loads the current session_map to know which sessions to watch.
  2. Detects session_map changes (new/changed/deleted windows) and cleans up.
  3. Reads new JSONL lines from each session file using byte-offset tracking.
  4. Parses entries via TranscriptParser and emits NewMessage objects to a callback.

Delivery is at-least-once: every NewMessage carries a DeliveryTicket that pins
the persisted offset at its JSONL line until the message has been sent.
Session names: Claude's /rename writes a custom-title line; a title that
changed is handed to the rename callback (which renames window and topic).
A /fork starts out with its source's name, which renames nothing.

History replays are compacted (config.replay_compact): thinking and tool
messages are dropped, each run of them marked by a divider line at the top of
the next message kept.

Subagents (config.subagent_topics): each watched session's subagents write
their own transcripts (<session>/subagents/agent-<id>.jsonl, the task in
agent-<id>.meta.json). A subagent started after ccbot began watching the
session is read from its first line; an older one from where it is, so it
shows up again once it writes more (a follow-up from the main session is
appended to the same transcript). Its messages carry agent_id (the bot sends
them to the subagent's own topic). Its task and messages from the main
session become notifications, not 👤 echoes. A fork subagent (meta.json
isFork) begins with a copy of its parent's transcript; that copy is skipped.

Optimizations: mtime cache skips unchanged files; byte offset avoids re-reading.

Key classes: SessionMonitor, NewMessage, SessionInfo.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from .config import config
from .monitor_state import DeliveryTicket, MonitorState, TrackedSession
from .tmux_manager import tmux_manager
from .transcript_parser import TranscriptParser
from .utils import read_cwd_from_jsonl

logger = logging.getLogger(__name__)

# A session's subagent transcripts: <session>/subagents/agent-<id>.jsonl
SUBAGENTS_DIR = "subagents"
AGENT_KEY_PREFIX = "agent-"  # also the tracked-session key: "agent-<id>"
# Seconds a new subagent waits for its meta.json (which says whether it is a
# fork) before it is read as an ordinary subagent
AGENT_META_WAIT = 10.0

# Compact replay: content types dropped, and the line that marks where they were
REPLAY_DROPPED_TYPES = frozenset(
    {"narration", "notification", "thinking", "tool_use", "tool_result"}
)
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
    content_type: str = (
        "text"  # "text", "narration", "thinking", "tool_use", "tool_result"
    )
    tool_use_id: str | None = None
    role: str = "assistant"  # "user" or "assistant"
    tool_name: str | None = None  # For tool_use messages, the tool name
    image_data: list[tuple[str, bytes]] | None = None  # From tool_result images
    is_replay: bool = False  # True when replaying past history (e.g. --resume)
    read_at: float | None = None  # epoch seconds the monitor read the JSONL entry
    # Pins the persisted offset until this message is delivered. Whoever
    # queues the message takes a hold(); the monitor releases its own
    # reference once the callback returns.
    ticket: DeliveryTicket | None = None
    # A subagent's message (session_id is then the parent session's): which
    # subagent, the task Claude gave it (3-5 words), its transcript, and
    # whether this is the reply that ended its work
    agent_id: str | None = None
    agent_description: str = ""
    agent_file: str = ""
    agent_finished: bool = False


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
        self._rename_callback: Callable[[str, str], Awaitable[None]] | None = None
        # Last session title (/rename) seen per session, and changes not yet
        # handed to the rename callback
        self._titles: dict[str, str] = {}
        self._renames: list[tuple[str, str]] = []
        # Per-session pending tool_use state carried across poll cycles
        self._pending_tools: dict[str, dict[str, Any]] = {}  # session_id -> pending
        # Track last known session_map for detecting changes
        # Keys may be window_id (@12) or window_name (old format) during transition
        self._last_session_map: dict[str, str] = {}  # window_key -> session_id
        # In-memory mtime cache for quick file change detection (not persisted)
        self._file_mtimes: dict[str, float] = {}  # session_id -> last_seen_mtime
        # Replaying sessions currently inside a run of dropped blocks (the
        # divider for that run has already been emitted).
        # session_id -> line offset where a dropped replay run began
        self._replay_gap_open: dict[str, int] = {}
        # Forks copy messages with their original UUIDs but rewrite the JSONL,
        # so a source byte offset cannot be used to skip inherited history.
        self._fork_history_uuids: dict[str, set[str]] = {}
        # Each /fork's source transcript, re-read once the fork's shows up
        self._fork_sources: dict[str, str] = {}
        # Fork subagents whose next 👤-less user message is their task
        self._fork_task_pending: set[str] = set()
        # Each subagent's task description: agent key -> description
        self._agent_descriptions: dict[str, str] = {}
        # New subagents waiting for their meta.json: agent key -> first seen
        self._agent_first_seen: dict[str, float] = {}

    @staticmethod
    def _title_of(entry: dict[str, Any]) -> str | None:
        """The session name (/rename) a custom-title line sets, if it is one."""
        title = entry.get("customTitle")
        if entry.get("type") != "custom-title" or not isinstance(title, str):
            return None
        return title.strip() or None

    @classmethod
    async def _read_history(cls, path: str | Path) -> tuple[set[str], str | None]:
        """The message UUIDs in a transcript, and its last session name."""
        uuids: set[str] = set()
        name = None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            async for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if isinstance(entry.get("uuid"), str):
                    uuids.add(entry["uuid"])
                name = cls._title_of(entry) or name
        return uuids, name

    @classmethod
    async def _read_uuids(cls, path: str | Path) -> set[str]:
        """The message UUIDs in a transcript."""
        uuids, _ = await cls._read_history(path)
        return uuids

    async def skip_fork_history(self, session_id: str, source_path: str) -> None:
        """Remember inherited message UUIDs before starting a fork without replay.

        The fork also inherits its source's name, which Claude keeps writing
        into the fork's transcript until /rename: it is noted as the fork's
        title, so it renames nothing.
        """
        uuids, name = await self._read_history(source_path)
        self._fork_history_uuids[session_id] = uuids
        self._fork_sources[session_id] = source_path
        if name:
            self._titles[session_id] = name

    async def _refresh_fork_history(self, session_id: str) -> None:
        """Add what a fork's source wrote after skip_fork_history.

        Claude copies the source only once it has started (half a minute after
        the launch, on a long session), and the source keeps writing meanwhile.
        """
        source = self._fork_sources.get(session_id)
        inherited = self._fork_history_uuids.get(session_id)
        if source is None or inherited is None:
            return
        try:
            uuids, name = await self._read_history(source)
        except OSError as e:
            logger.warning("Fork %s: cannot re-read %s: %s", session_id, source, e)
            return
        inherited |= uuids
        if name:
            self._titles[session_id] = name

    def forget_fork_history(self, key: str) -> None:
        """Stop filtering a fork's inherited history."""
        self._fork_history_uuids.pop(key, None)
        self._fork_sources.pop(key, None)

    def _drop_inherited(
        self, key: str, lines: list[tuple[int, dict]]
    ) -> list[tuple[int, dict]]:
        """Drop the history a fork copied from its source (_fork_history_uuids)."""
        inherited = self._fork_history_uuids.get(key)
        if inherited is None:
            return lines
        lines = [(offset, e) for offset, e in lines if e.get("uuid") not in inherited]
        # Keep the filter across partial writes of the copied transcript,
        # until the first new conversation message.
        if any(
            e.get("type") in ("user", "assistant") and e.get("uuid") for _, e in lines
        ):
            self.forget_fork_history(key)
        return lines

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

    def set_rename_callback(
        self, callback: Callable[[str, str], Awaitable[None]]
    ) -> None:
        """Called with (session_id, title) when a session's /rename title changes."""
        self._rename_callback = callback

    def _note_title(self, session_id: str, entries: list[dict[str, Any]]) -> None:
        """Queue a rename when the session's title (Claude's /rename) changes.

        Claude Code re-appends the current title line every so often, so only a
        title different from the last one seen counts. Forks carry the source
        session's title lines, which name that session, not this one (older
        Claude Code kept their sessionId; see also skip_fork_history).
        """
        title = None
        for entry in entries:
            if entry.get("sessionId", session_id) == session_id:
                title = self._title_of(entry) or title
        if title and self._titles.get(session_id) != title:
            self._titles[session_id] = title
            self._renames.append((session_id, title))

    def take_renames(self) -> list[tuple[str, str]]:
        """Title changes noted since the last call, as (session_id, title)."""
        renames, self._renames = self._renames, []
        return renames

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
                read_at = time.time()
                self._file_mtimes[session_info.session_id] = current_mtime

                await self._refresh_fork_history(session_info.session_id)
                new_lines = self._drop_inherited(session_info.session_id, new_lines)

                line_offsets = [offset for offset, _ in new_lines]
                new_entries = [entry for _, entry in new_lines]
                self._note_title(session_info.session_id, new_entries)

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
                    if entry.content_type == "thinking" and not config.show_thinking:
                        continue

                    line_offset = (
                        line_offsets[entry.source_index]
                        if 0 <= entry.source_index < len(line_offsets)
                        else previous_offset
                    )
                    # Also true for history re-read after a restart
                    is_replay = was_replaying or tracked.is_replay(line_offset)

                    if (
                        is_replay
                        and config.replay_compact
                        and entry.content_type in REPLAY_DROPPED_TYPES
                    ):
                        # The run gets a divider on the next message kept
                        self._replay_gap_open.setdefault(
                            session_info.session_id, line_offset
                        )
                        continue

                    text = entry.text
                    ticket_offset = line_offset
                    gap_offset = self._replay_gap_open.pop(
                        session_info.session_id, None
                    )
                    if gap_offset is not None and is_replay:
                        text = f"{REPLAY_DIVIDER}\n{text}"
                        # Re-read from the run's start, so a restart before
                        # delivery brings the divider back too
                        ticket_offset = gap_offset

                    # The ticket pins the saved offset at this message's line
                    # until it is delivered (opened before the save below).
                    new_messages.append(
                        NewMessage(
                            session_id=session_info.session_id,
                            text=text,
                            is_complete=True,
                            content_type=entry.content_type,
                            tool_use_id=entry.tool_use_id,
                            role=entry.role,
                            tool_name=entry.tool_name,
                            image_data=entry.image_data,
                            is_replay=is_replay,
                            read_at=read_at,
                            ticket=self.state.open_ticket(tracked, ticket_offset),
                        )
                    )

                # Replay ends once we've caught up with the file; a long
                # history may take several cycles (partial lines, growth).
                if was_replaying and tracked.last_byte_offset >= current_size:
                    tracked.replaying = False
                    tracked.replay_until = tracked.last_byte_offset
                    self.state.update_session(tracked)
                    # A run at the very end of the history has nothing to mark
                    self._replay_gap_open.pop(session_info.session_id, None)

                if tracked.last_byte_offset != previous_offset:
                    self.state.update_session(tracked)

            except OSError as e:
                logger.debug(f"Error processing session {session_info.session_id}: {e}")

        if config.subagent_topics:
            seen: set[str] = set()
            for session_info in sessions:
                parent = self.state.get_session(session_info.session_id)
                if (
                    session_info.session_id in active_session_ids
                    and session_info.session_id not in seen
                    and parent is not None
                ):
                    seen.add(session_info.session_id)
                    new_messages.extend(
                        await self._check_subagents(parent, session_info.file_path)
                    )

        self.state.save_if_dirty()
        return new_messages

    @staticmethod
    def _first_timestamp(path: Path, skip: set[str] | None = None) -> float | None:
        """When a transcript began (epoch seconds); None if not written yet.

        Entries whose uuid is in skip (the history a fork inherited) don't count.
        """
        try:
            with path.open("rb") as f:
                looked = 0
                while looked < 5:  # a fork's first line has no timestamp
                    line = f.readline()
                    if not line.endswith(b"\n"):
                        return None
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        entry = None
                    if not isinstance(entry, dict):
                        entry = {}
                    elif skip and entry.get("uuid") in skip:
                        continue
                    looked += 1
                    stamp = entry.get("timestamp")
                    if isinstance(stamp, str):
                        try:
                            return datetime.fromisoformat(stamp).timestamp()
                        except ValueError:
                            continue
        except OSError:
            pass
        return None

    @staticmethod
    def _agent_meta(path: Path) -> dict[str, Any] | None:
        """A subagent's agent-<id>.meta.json; None if not written yet."""
        try:
            meta = json.loads(path.with_suffix(".meta.json").read_text())
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) else {}

    def _agent_description(self, path: Path) -> str:
        """The task Claude gave the subagent (agent-<id>.meta.json)."""
        key = path.stem
        if key not in self._agent_descriptions:
            meta = self._agent_meta(path)
            if meta is None:
                return ""  # not written yet: ask again next time
            description = meta.get("description")
            self._agent_descriptions[key] = (
                description.strip() if isinstance(description, str) else ""
            )
        return self._agent_descriptions[key]

    async def _fork_inherited(self, path: Path, parent_file: Path) -> set[str] | None:
        """UUIDs of the history a fork subagent inherited; empty for others.

        A fork's transcript begins with a copy of its parent's (the subagent
        in meta.json parentAgentId, else the session) under the original
        UUIDs. None while meta.json, which says whether it is a fork, is not
        written yet (for AGENT_META_WAIT at most).
        """
        key = path.stem
        meta = self._agent_meta(path)
        if meta is None:
            first_seen = self._agent_first_seen.setdefault(key, time.monotonic())
            if time.monotonic() - first_seen < AGENT_META_WAIT:
                return None
            meta = {}
        self._agent_first_seen.pop(key, None)
        if not meta.get("isFork"):
            return set()
        parent_agent = meta.get("parentAgentId")
        source = (
            path.with_name(f"{AGENT_KEY_PREFIX}{parent_agent}.jsonl")
            if isinstance(parent_agent, str) and parent_agent
            else parent_file
        )
        try:
            return await self._read_uuids(source)
        except OSError:
            return set()

    @staticmethod
    def _agent_entry_shown(entry: dict[str, Any]) -> bool:
        """False for what Claude Code writes into a subagent's transcript for
        the model only (compaction summaries, skill text); messages from the
        main session (origin coordinator) are shown."""
        if entry.get("isCompactSummary"):
            return False
        if entry.get("isMeta"):
            origin = entry.get("origin")
            return isinstance(origin, dict) and origin.get("kind") == "coordinator"
        return True

    async def _check_subagents(
        self, parent: TrackedSession, parent_file: Path
    ) -> list[NewMessage]:
        """New messages from the subagents a session started."""
        agent_dir = parent_file.with_suffix("") / SUBAGENTS_DIR
        try:
            paths = sorted(agent_dir.glob(f"{AGENT_KEY_PREFIX}*.jsonl"))
        except OSError:
            return []
        messages: list[NewMessage] = []
        for path in paths:
            key = path.stem  # agent-<id>
            tracked = self.state.get_session(key)
            if tracked is None:
                inherited = self._fork_history_uuids.get(key)
                if inherited is None:
                    inherited = await self._fork_inherited(path, parent_file)
                    if inherited is None:
                        continue  # its meta.json is not written yet
                    if inherited:
                        self._fork_history_uuids[key] = inherited
                started = self._first_timestamp(path, skip=inherited)
                if started is None:
                    continue  # nothing (of its own) written yet
                tracked = TrackedSession(
                    session_id=key,
                    file_path=str(path),
                    parent_session_id=parent.session_id,
                )
                if started < parent.subagents_since:
                    # Started before the session was watched: only what it
                    # writes from now on (e.g. after a follow-up)
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    tracked.last_byte_offset = st.st_size
                    self._file_mtimes[key] = st.st_mtime
                    self._fork_history_uuids.pop(key, None)
                else:
                    if inherited:
                        self._fork_task_pending.add(key)
                    logger.info(
                        "Started tracking subagent %s of session %s",
                        key,
                        parent.session_id,
                    )
                self.state.update_session(tracked)
            try:
                messages.extend(await self._read_subagent(tracked, path))
            except OSError as e:
                logger.debug(f"Error processing subagent {key}: {e}")
        return messages

    async def _read_subagent(
        self, tracked: TrackedSession, path: Path
    ) -> list[NewMessage]:
        """New messages in one subagent's transcript."""
        key = tracked.session_id
        st = path.stat()
        if (
            st.st_mtime <= self._file_mtimes.get(key, 0.0)
            and st.st_size <= tracked.last_byte_offset
        ):
            return []
        previous_offset = tracked.last_byte_offset
        new_lines = await self._read_new_lines_with_offsets(tracked, path)
        lines = [
            (offset, entry)
            for offset, entry in self._drop_inherited(key, new_lines)
            if self._agent_entry_shown(entry)
        ]
        read_at = time.time()
        self._file_mtimes[key] = st.st_mtime
        entries = [entry for _, entry in lines]

        parsed_entries, remaining = TranscriptParser.parse_entries(
            entries, pending_tools=self._pending_tools.get(key, {})
        )
        if remaining:
            self._pending_tools[key] = remaining
        else:
            self._pending_tools.pop(key, None)

        agent_id = key.removeprefix(AGENT_KEY_PREFIX)
        messages: list[NewMessage] = []
        for entry in parsed_entries:
            if not entry.text and not entry.image_data:
                continue
            if entry.content_type == "thinking" and not config.show_thinking:
                continue
            if 0 <= entry.source_index < len(lines):
                line_offset, source = lines[entry.source_index]
            else:
                line_offset, source = previous_offset, {}
            text, content_type, role = entry.text, entry.content_type, entry.role
            if role == "user":
                # Its task (the first message, or a fork's first after the
                # history it inherited) or a message from the main session:
                # not the user's words
                is_task = (
                    source.get("parentUuid") is None or key in self._fork_task_pending
                )
                self._fork_task_pending.discard(key)
                text = TranscriptParser.format_subagent_input(text, is_task=is_task)
                content_type, role = "notification", "assistant"
            message = source.get("message")
            finished = (
                source.get("type") == "assistant"
                and isinstance(message, dict)
                and message.get("stop_reason") == "end_turn"
            )
            messages.append(
                NewMessage(
                    session_id=tracked.parent_session_id,
                    text=text,
                    is_complete=True,
                    content_type=content_type,
                    tool_use_id=entry.tool_use_id,
                    role=role,
                    tool_name=entry.tool_name,
                    image_data=entry.image_data,
                    read_at=read_at,
                    ticket=self.state.open_ticket(tracked, line_offset),
                    agent_id=agent_id,
                    agent_description=self._agent_description(path),
                    agent_file=str(path),
                    agent_finished=finished,
                )
            )
        if tracked.last_byte_offset != previous_offset:
            self.state.update_session(tracked)
        return messages

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

        # A subagent goes with the session that started it
        stale_sessions = {
            session_id
            for session_id, tracked in self.state.tracked_sessions.items()
            if (tracked.parent_session_id or session_id) not in active_session_ids
        }

        if stale_sessions:
            logger.info(
                f"[Startup cleanup] Removing {len(stale_sessions)} stale sessions"
            )
            self._forget_sessions(stale_sessions)

    def _forget_sessions(self, session_ids: set[str]) -> None:
        """Stop tracking these sessions and the subagents they started."""
        for session_id, tracked in list(self.state.tracked_sessions.items()):
            if session_id in session_ids or tracked.parent_session_id in session_ids:
                self.state.remove_session(session_id)
                self._file_mtimes.pop(session_id, None)
                self.forget_fork_history(session_id)
                self._fork_task_pending.discard(session_id)
                self._pending_tools.pop(session_id, None)
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
            self._forget_sessions(sessions_to_remove)

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

                for session_id, title in self.take_renames():
                    try:
                        if self._rename_callback:
                            await self._rename_callback(session_id, title)
                    except Exception as e:
                        logger.error(f"Rename callback error: {e}")

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
