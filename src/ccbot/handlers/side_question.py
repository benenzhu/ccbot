"""Side questions (/btw) relayed through Claude Code's own TUI panel.

Claude Code answers /btw in an overlay panel and keeps the exchange only in
process memory, so the JSONL monitor never sees it. This module types the
/btw into the tmux window, watches the pane until the panel settles, presses
`c` so the TUI copies the full answer (as Markdown) into a tmux paste buffer
via OSC 52, reads that buffer, and closes the panel with Escape. Using the
real /btw keeps the main conversation's prompt cache, unlike a separate
`claude -p` run.

The panel needs room: in a narrow window its hint bar wraps over lines, in a
short one a long answer pushes the hint bar off screen. A window smaller than
BTW_MIN_WIDTH x BTW_MIN_HEIGHT is enlarged while the side question runs, then
put back; the parser also reads a wrapped hint bar.

Key pieces: parse_btw_panel (pane text → PanelState), run_side_question.
"""

import asyncio
import logging
import re
import textwrap
import time
from dataclasses import dataclass

from ..config import config
from ..session import session_manager
from ..tmux_manager import tmux_manager

logger = logging.getLogger(__name__)

# Panel anatomy (observed on Claude Code 2.1.263, 2.1.302):
#     /btw <earlier question>            ← dim history lines, optional
#     /btw <current question>            ← may be truncated with …
#       ✢ Answering…                     ← pending
#       <answer, indented 6 spaces>      ← settled, cut to the panel's height
#     ⇧←/→ to browse · c to copy · f to fork · x to clear history · Esc to close
# The hint bar (last) wraps over lines when the window is narrow, and is not
# drawn when the window is too short for it below a long answer.
_RE_QUESTION_LINE = re.compile(r"^\s*/btw\s")
_RE_HINT_BAR = re.compile(r"Esc to close")
_RE_SETTLED_HINT = re.compile(r"\bc to copy\b|Copied to clipboard")
_RE_PENDING = re.compile(r"Answering…")
_RE_ERROR = re.compile(r"Failed to get response|No response received")
_HINT_MAX_LINES = 4  # lines a wrapped hint bar may take

# A smaller window is enlarged to this while a side question runs (the hint
# bar, ~80 columns with history, fits on one line and below a long answer)
BTW_MIN_WIDTH = 100
BTW_MIN_HEIGHT = 30

# Windows with a side question in flight; a second /btw would type into the
# open panel instead of starting a new question.
_active_windows: set[str] = set()


@dataclass
class PanelState:
    """What the /btw panel currently shows, if it is on screen at all."""

    present: bool = False
    settled: bool = False  # answer rendered, hint bar offers copy
    error: str | None = None
    answer: str = ""  # text scraped from the panel body (screen-wrapped)


def _find_hint_bar(lines: list[str]) -> tuple[int, str] | None:
    """(first line, text) of the hint bar, the lowest block of lines that
    reads "Esc to close" once its wrapped lines are joined."""
    end = len(lines) - 1
    while end >= 0:
        if not lines[end].strip():
            end -= 1
            continue
        start = end
        while start > 0 and lines[start - 1].strip():
            start -= 1
        start = max(start, end - _HINT_MAX_LINES + 1)
        text = " ".join(line.strip() for line in lines[start : end + 1])
        if _RE_HINT_BAR.search(text):
            return start, text
        end = start - 1
    return None


def parse_btw_panel(pane_text: str) -> PanelState:
    """Locate the /btw panel in captured pane text and classify its state."""
    lines = pane_text.split("\n")
    found = _find_hint_bar(lines)
    if found is None:
        return PanelState()
    hint_idx, hint = found
    q_idx = next(
        (i for i in range(hint_idx - 1, -1, -1) if _RE_QUESTION_LINE.match(lines[i])),
        None,
    )
    if q_idx is None:
        return PanelState()

    body = textwrap.dedent(
        "\n".join(line.rstrip() for line in lines[q_idx + 1 : hint_idx])
    ).strip("\n")

    err = _RE_ERROR.search(body)
    if err:
        return PanelState(present=True, error=err.group(0), answer=body)
    if _RE_PENDING.search(body) and not _RE_SETTLED_HINT.search(hint):
        return PanelState(present=True)
    return PanelState(
        present=True, settled=bool(_RE_SETTLED_HINT.search(hint)), answer=body
    )


async def _tmux(*args: str) -> str | None:
    """Run a tmux CLI command, returning stdout or None on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "tmux",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
    except OSError as e:
        logger.error("tmux %s failed to start: %s", args[0], e)
        return None
    if proc.returncode != 0:
        logger.warning("tmux %s failed: %s", " ".join(args), err.decode().strip())
        return None
    return out.decode("utf-8", "replace")


async def _newest_buffer_name() -> str | None:
    out = await _tmux("list-buffers", "-F", "#{buffer_name}")
    if not out:
        return None
    return out.split("\n", 1)[0].strip() or None


async def _copy_answer_via_buffer(window_id: str) -> str:
    """Press `c` in the panel and read the paste buffer the TUI creates.

    The TUI copies with OSC 52; tmux turns that into a new paste buffer even
    with `set-clipboard external`. Returns "" if no buffer appears.
    """
    before = await _newest_buffer_name()
    if not await tmux_manager.send_keys(window_id, "c", enter=False):
        return ""
    for _ in range(6):
        await asyncio.sleep(0.4)
        name = await _newest_buffer_name()
        if name and name != before:
            text = await _tmux("show-buffer", "-b", name)
            await _tmux("delete-buffer", "-b", name)
            return (text or "").strip()
    return ""


async def _close_panel(window_id: str) -> None:
    await tmux_manager.send_keys(window_id, "\x1b", enter=False)


async def _make_room(window_id: str) -> tuple[int, int] | None:
    """Enlarge a window smaller than the panel needs; its old size, or None."""
    out = await _tmux(
        "display-message", "-p", "-t", window_id, "#{window_width} #{window_height}"
    )
    try:
        width, height = (int(n) for n in (out or "").split())
    except ValueError:
        return None
    if width >= BTW_MIN_WIDTH and height >= BTW_MIN_HEIGHT:
        return None
    resized = await _tmux(
        "resize-window",
        "-t",
        window_id,
        "-x",
        str(max(width, BTW_MIN_WIDTH)),
        "-y",
        str(max(height, BTW_MIN_HEIGHT)),
    )
    if resized is None:
        return None
    logger.info("Enlarged %s from %dx%d for a side question", window_id, width, height)
    await asyncio.sleep(0.5)  # let the TUI redraw at the new size
    return width, height


async def _restore_size(window_id: str, size: tuple[int, int]) -> None:
    width, height = size
    await _tmux("resize-window", "-t", window_id, "-x", str(width), "-y", str(height))
    # resize-window pinned the size (window-size manual): follow clients again
    await _tmux("set-option", "-w", "-u", "-t", window_id, "window-size")


async def run_side_question(
    window_id: str,
    question: str,
    timeout: float | None = None,
    poll_interval: float = 1.0,
) -> tuple[bool, str]:
    """Ask `question` via /btw in `window_id` and return (ok, answer_or_error)."""
    if window_id in _active_windows:
        return False, "a side question is already in progress for this window"
    timeout = config.btw_timeout if timeout is None else timeout
    _active_windows.add(window_id)
    old_size = None
    try:
        old_size = await _make_room(window_id)
        return await _run(window_id, question, timeout, poll_interval)
    finally:
        if old_size is not None:
            await _restore_size(window_id, old_size)
        _active_windows.discard(window_id)


async def _run(
    window_id: str, question: str, timeout: float, poll_interval: float
) -> tuple[bool, str]:
    ok, msg = await session_manager.send_to_window(window_id, f"/btw {question}")
    if not ok:
        return False, msg

    deadline = time.monotonic() + timeout
    seen_panel = False
    state = PanelState()
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_interval)
        pane = await tmux_manager.capture_pane(window_id)
        if not pane:
            continue
        state = parse_btw_panel(pane)
        if not state.present:
            # Not up yet, or the panel stepped aside for the main turn.
            continue
        seen_panel = True
        if state.error:
            await _close_panel(window_id)
            return False, state.error
        if state.settled:
            break
    else:
        why = "panel never appeared" if not seen_panel else "still answering"
        logger.warning("Side question timed out in %s (%s)", window_id, why)
        return False, f"timed out after {timeout:.0f}s ({why})"

    answer = await _copy_answer_via_buffer(window_id) or state.answer
    await _close_panel(window_id)
    if not answer.strip():
        return False, "empty answer"
    logger.info("Side question answered in %s (len=%d)", window_id, len(answer))
    return True, answer.strip()
