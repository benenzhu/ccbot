"""Monitor state persistence — tracks byte offsets for each session.

Persists TrackedSession records (session_id, file_path, last_byte_offset)
to ~/.ccbot/monitor_state.json so the session monitor can resume
incremental reading after restarts without re-sending old messages.

Delivery is at-least-once: the saved offset never moves past a message that
has not reached Telegram yet. Each message read from a transcript holds a
DeliveryTicket until it is sent, so a restart re-reads (and re-sends)
whatever was still queued instead of losing it.

Key classes: MonitorState, TrackedSession, DeliveryTicket.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DeliveryTicket:
    """Ref-counted receipt for one transcript message.

    The monitor opens a ticket per message and every queued task carrying
    part of that message holds a reference. When the last holder releases,
    the message counts as delivered and stops holding back the saved offset.
    """

    def __init__(self, on_delivered: Callable[[], None]) -> None:
        self._refs = 1  # the opener's own reference
        self._on_delivered = on_delivered

    def hold(self) -> "DeliveryTicket":
        """Take another reference; the holder must release() it."""
        self._refs += 1
        return self

    def release(self) -> None:
        """Drop one reference; the last release marks the message delivered."""
        self._refs -= 1
        if self._refs == 0:
            self._on_delivered()


@dataclass
class TrackedSession:
    """State for a tracked Claude Code session."""

    session_id: str
    file_path: str  # Path to .jsonl file
    last_byte_offset: int = 0  # Byte offset for incremental reading
    # Messages read but not yet delivered: ticket id -> byte offset of the
    # JSONL line they came from. In-memory only; see resume_offset.
    undelivered: dict[int, int] = field(default_factory=dict, compare=False, repr=False)
    # History replay (resume with replay). `replaying` holds until the read
    # head first catches up with the file; `replay_until` then records where
    # the history ended. Both are persisted: a restart re-reads undelivered
    # history, which must still count as replay (compact, no TTS).
    replaying: bool = False
    replay_until: int = 0

    def is_replay(self, line_offset: int) -> bool:
        """True if the JSONL line at `line_offset` is replayed history."""
        return self.replaying or line_offset < self.replay_until

    @property
    def resume_offset(self) -> int:
        """Offset a restart must resume from so nothing undelivered is skipped."""
        return min([self.last_byte_offset, *self.undelivered.values()])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization.

        Persists resume_offset rather than the read head, so a restart
        re-reads messages that were still waiting to be sent.
        """
        return {
            "session_id": self.session_id,
            "file_path": self.file_path,
            "last_byte_offset": self.resume_offset,
            "replaying": self.replaying,
            "replay_until": self.replay_until,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackedSession":
        """Create from dict."""
        return cls(
            session_id=data.get("session_id", ""),
            file_path=data.get("file_path", ""),
            last_byte_offset=data.get("last_byte_offset", 0),
            replaying=data.get("replaying", False),
            replay_until=data.get("replay_until", 0),
        )


@dataclass
class MonitorState:
    """Persistent state for the session monitor.

    Stores tracking information for all monitored sessions
    to prevent duplicate notifications after restarts.
    """

    state_file: Path
    tracked_sessions: dict[str, TrackedSession] = field(default_factory=dict)
    _dirty: bool = field(default=False, repr=False)
    _ticket_seq: int = field(default=0, repr=False)

    def load(self) -> None:
        """Load state from file."""
        if not self.state_file.exists():
            logger.debug(f"State file does not exist: {self.state_file}")
            return

        try:
            data = json.loads(self.state_file.read_text())
            sessions = data.get("tracked_sessions", {})
            self.tracked_sessions = {
                k: TrackedSession.from_dict(v) for k, v in sessions.items()
            }
            logger.info(
                f"Loaded {len(self.tracked_sessions)} tracked sessions from state"
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load state file: {e}")
            self.tracked_sessions = {}

    def save(self) -> None:
        """Save state to file atomically."""
        from .utils import atomic_write_json

        data = {
            "tracked_sessions": {
                k: v.to_dict() for k, v in self.tracked_sessions.items()
            }
        }

        try:
            atomic_write_json(self.state_file, data)
            self._dirty = False
            logger.debug(
                "Saved %d tracked sessions to state", len(self.tracked_sessions)
            )
        except OSError as e:
            logger.error("Failed to save state file: %s", e)

    def get_session(self, session_id: str) -> TrackedSession | None:
        """Get tracked session by ID."""
        return self.tracked_sessions.get(session_id)

    def update_session(self, session: TrackedSession) -> None:
        """Update or add a tracked session."""
        self.tracked_sessions[session.session_id] = session
        self._dirty = True

    def remove_session(self, session_id: str) -> None:
        """Remove a tracked session."""
        if session_id in self.tracked_sessions:
            del self.tracked_sessions[session_id]
            self._dirty = True

    def open_ticket(self, session: TrackedSession, offset: int) -> DeliveryTicket:
        """Hold the saved offset of a session at `offset` until delivery.

        `offset` is where the message's JSONL line starts. The ticket is tied
        to this TrackedSession object, so re-seeding a session (which replaces
        the object) drops old holds instead of mixing them with the new ones.
        """
        self._ticket_seq += 1
        ticket_id = self._ticket_seq
        session.undelivered[ticket_id] = offset

        def _delivered() -> None:
            if session.undelivered.pop(ticket_id, None) is not None:
                self._dirty = True

        return DeliveryTicket(_delivered)

    def save_if_dirty(self) -> None:
        """Save state only if it has been modified."""
        if self._dirty:
            self.save()
