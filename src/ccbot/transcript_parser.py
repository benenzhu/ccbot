"""JSONL transcript parser for Claude Code session files.

Parses Claude Code session JSONL files and extracts structured messages.
Handles: text, thinking, tool_use, tool_result, local_command, and user messages.
Tool pairing: tool_use blocks in assistant messages are matched with
tool_result blocks in subsequent user messages via tool_use_id.

Shared by both session.py (history) and session_monitor.py (real-time).
Format reference: https://github.com/desis123/claude-code-viewer

Key classes: TranscriptParser (static methods), ParsedEntry, ParsedMessage, PendingToolInfo.
"""

import base64
import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# What Claude Code puts in front of a fork subagent's directive
_RE_FORK_BOILERPLATE = re.compile(
    r"<fork-boilerplate>.*?</fork-boilerplate>\s*", re.DOTALL
)


@dataclass
class ParsedMessage:
    """Parsed message from a transcript."""

    message_type: str  # "user", "assistant", "tool_use", "tool_result", etc.
    text: str  # Extracted text content
    tool_name: str | None = None  # For tool_use messages


@dataclass
class ParsedEntry:
    """A single parsed message entry ready for display."""

    role: str  # "user" | "assistant"
    text: str  # Already formatted text
    # "text" | "narration" | "notification" | "thinking" | "tool_use"
    # | "tool_result" | "local_command"
    content_type: str
    tool_use_id: str | None = None
    timestamp: str | None = None  # ISO timestamp from JSONL
    tool_name: str | None = (
        None  # For tool_use entries, the tool name (e.g. "AskUserQuestion")
    )
    image_data: list[tuple[str, bytes]] | None = (
        None  # For tool_result entries with images: (media_type, raw_bytes)
    )
    # Index into the `entries` list passed to parse_entries that produced
    # this entry (-1 if none). Lets the monitor map a message back to the
    # byte offset of its JSONL line.
    source_index: int = field(default=-1, compare=False)


@dataclass
class PendingToolInfo:
    """Information about a pending tool_use waiting for its tool_result."""

    summary: str  # Formatted tool summary (e.g. "**Read**(file.py)")
    tool_name: str  # Tool name (e.g. "Read", "Edit")
    input_data: Any = None  # Tool input parameters (for Edit to generate diff)
    full_input: str = ""  # The input in full, when the summary line cut it short


class TranscriptParser:
    """Parser for Claude Code JSONL session files.

    Expected JSONL entry structure:
    - type: "user" | "assistant" | "summary" | "file-history-snapshot" | ...
    - message.content: list of blocks (text, tool_use, tool_result, thinking)
    - sessionId, cwd, timestamp, uuid: metadata fields

    Tool pairing model: tool_use blocks appear in assistant messages,
    matching tool_result blocks appear in the next user message (keyed by tool_use_id).
    """

    # Magic string constants
    _NO_CONTENT_PLACEHOLDER = "(no content)"
    _INTERRUPTED_TEXT = "[Request interrupted by user for tool use]"
    # A tool's summary line fits one phone line; the input in full goes first
    # in the collapsed quote (up to _MAX_FULL_INPUT_LENGTH)
    _MAX_SUMMARY_LENGTH = 40
    _MAX_FULL_INPUT_LENGTH = 1000

    @staticmethod
    def parse_line(line: str) -> dict | None:
        """Parse a single JSONL line.

        Args:
            line: A single line from the JSONL file

        Returns:
            Parsed dict or None if line is empty/invalid
        """
        line = line.strip()
        if not line:
            return None

        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def get_message_type(data: dict) -> str | None:
        """Get the message type from parsed data.

        Returns:
            Message type: "user", "assistant", "file-history-snapshot", etc.
        """
        return data.get("type")

    @staticmethod
    def is_user_message(data: dict) -> bool:
        """Check if this is a user message."""
        return data.get("type") == "user"

    @staticmethod
    def extract_text_only(content_list: list[Any]) -> str:
        """Extract only text content from structured content.

        This is used for Telegram notifications where we only want
        the actual text response, not tool calls or thinking.

        Args:
            content_list: List of content blocks

        Returns:
            Combined text content only
        """
        if not isinstance(content_list, list):
            if isinstance(content_list, str):
                return content_list
            return ""

        texts = []
        for item in content_list:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    texts.append(text)

        return "\n".join(texts)

    _RE_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

    _RE_COMMAND_NAME = re.compile(r"<command-name>(.*?)</command-name>")
    _RE_LOCAL_STDOUT = re.compile(
        r"<local-command-stdout>(.*?)</local-command-stdout>", re.DOTALL
    )
    _RE_SYSTEM_TAGS = re.compile(
        r"<(bash-input|bash-stdout|bash-stderr|local-command-caveat|system-reminder)"
    )

    @staticmethod
    def _format_edit_diff(old_string: str, new_string: str) -> str:
        """Generate a compact unified diff between old_string and new_string."""
        old_lines = old_string.splitlines(keepends=True)
        new_lines = new_string.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
        # Skip the --- / +++ header lines
        result_lines: list[str] = []
        for line in diff:
            if line.startswith(("---", "+++")):
                continue
            # Strip trailing newline for clean display
            result_lines.append(line.rstrip("\n"))
        return "\n".join(result_lines)

    _RE_XML_FIELD = re.compile(r"<([\w-]+)>(.*?)</\1>", re.DOTALL)
    _RE_PEER_MESSAGE = re.compile(r"<([\w-]+-message)\b([^>]*)>(.*?)</\1>", re.DOTALL)
    _RE_XML_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
    # Notification fields not worth showing (ids, the summary already shown)
    _NOTIFICATION_HIDDEN_FIELDS = frozenset({"summary", "task-id", "tool-use-id"})

    @classmethod
    def _format_notification(cls, data: dict, text: str) -> str:
        """A user entry Claude Code wrote itself, as one line + collapsed details.

        Background task notices ("<task-notification>…") and messages from
        other sessions or subagents ("Another Claude session sent a message:")
        arrive as user entries (origin.kind task-notification / peer). They
        are not the user's words; "" for anything else.
        """
        origin = data.get("origin")
        kind = origin.get("kind") if isinstance(origin, dict) else None
        stripped = text.strip()
        if kind == "task-notification" or stripped.startswith("<task-notification>"):
            inner = stripped.removeprefix("<task-notification>").removesuffix(
                "</task-notification>"
            )
            fields = {
                m.group(1): m.group(2).strip()
                for m in cls._RE_XML_FIELD.finditer(inner)
            }
            head = "🔔 " + (fields.get("summary") or "Background task update")
            details = "\n".join(
                f"{key}: {value}"
                for key, value in fields.items()
                if key not in cls._NOTIFICATION_HIDDEN_FIELDS and value
            )
        elif kind == "peer" or stripped.startswith("Another Claude session sent"):
            m = cls._RE_PEER_MESSAGE.search(stripped)
            attrs = dict(cls._RE_XML_ATTR.findall(m.group(2))) if m else {}
            sender = attrs.get("from-name") or attrs.get("from", "")
            head = f"📨 Message from {sender}" if sender else "📨 Message"
            details = m.group(3).strip() if m else stripped
        else:
            return ""
        if not details:
            return head
        return f"{head}\n{cls._format_expandable_quote(details)}"

    @classmethod
    def format_subagent_input(cls, text: str, *, is_task: bool) -> str:
        """What a subagent was told, as one line + the text collapsed.

        Its task (its transcript's first message) and later messages from the
        session that started it are not the user's words, so they get no 👤.
        A fork's task loses the boilerplate Claude Code wraps it in.
        """
        if is_task:
            text = _RE_FORK_BOILERPLATE.sub("", text)
        head = "📋 Task" if is_task else "📨 Message from the main session"
        return f"{head}\n{cls._format_expandable_quote(text)}"

    @staticmethod
    def _is_narration(block: dict) -> bool:
        """True for a thinking block that is a progress update for the user.

        Recent models keep their reasoning hidden (thinking text empty) and
        send short updates ("job 13102 started on node 06, waiting for …")
        as thinking blocks whose signature is tagged "narration"; Claude Code
        shows them as ordinary text.
        """
        signature = block.get("signature")
        if not isinstance(signature, str) or not signature:
            return False
        try:
            raw = base64.b64decode(signature + "=" * (-len(signature) % 4))
        except ValueError:
            return False
        return b"narration" in raw

    @staticmethod
    def _queued_user_prompt(data: dict) -> str:
        """Text of a message the user sent while Claude was mid-turn.

        Claude Code queues such a message and records it as a queued_command
        attachment, never as a user entry. Task notifications and other
        non-human queued prompts give "".
        """
        att = data.get("attachment")
        if not isinstance(att, dict) or att.get("type") != "queued_command":
            return ""
        origin = att.get("origin")
        if att.get("commandMode") != "prompt" or not isinstance(origin, dict):
            return ""
        if origin.get("kind") != "human":
            return ""
        prompt = att.get("prompt")
        if isinstance(prompt, list):
            prompt = "\n".join(
                b.get("text", "")
                for b in prompt
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return prompt.strip() if isinstance(prompt, str) else ""

    @staticmethod
    def _bash_edit_diff(data: dict) -> tuple[list[str], str]:
        """Files a Bash command changed, from Claude Code's bashEditDiff.

        Returns ("⎿  Updated <path> (+a -b)" lines, the hunks as a diff);
        both empty when the command changed no file.
        """
        tool_result = data.get("toolUseResult")
        edits = (
            tool_result.get("bashEditDiff") if isinstance(tool_result, dict) else None
        )
        if not isinstance(edits, dict):
            return [], ""
        cwd = data.get("cwd") or ""
        files = [f for f in edits.get("files") or [] if isinstance(f, dict)]
        stats: list[str] = []
        diff: list[str] = []
        for f in files:
            path = f.get("filePath", "")
            if cwd and path.startswith(cwd.rstrip("/") + "/"):
                path = path[len(cwd.rstrip("/")) + 1 :]
            lines: list[str] = []
            for h in f.get("hunks") or []:
                lines.append(
                    f"@@ -{h.get('oldStart')},{h.get('oldLines')} "
                    f"+{h.get('newStart')},{h.get('newLines')} @@"
                )
                lines.extend(h.get("lines") or [])
            added = sum(1 for line in lines if line.startswith("+"))
            removed = sum(1 for line in lines if line.startswith("-"))
            stats.append(f"  ⎿  Updated {path} (+{added} -{removed})")
            if len(files) > 1:
                diff.append(path)
            diff.extend(lines)
        more = edits.get("moreFiles") or 0
        if more:
            stats.append(f"  ⎿  … {more} more files changed")
        return stats, "\n".join(diff)

    @classmethod
    def format_tool_use_summary(cls, name: str, input_data: dict | Any) -> str:
        """Format a tool_use block into a one-line summary like "**Read**(file.py)".

        A longer input is cut short; tool_full_input has the rest for the
        collapsed quote.
        """
        summary = cls._tool_summary_value(name, input_data)
        if not summary:
            return f"**{name}**"
        line = " ".join(summary.split())
        if len(line) > cls._MAX_SUMMARY_LENGTH:
            line = line[: cls._MAX_SUMMARY_LENGTH] + "…"
        return f"**{name}**({line})"

    @classmethod
    def tool_full_input(cls, name: str, input_data: dict | Any) -> str:
        """The tool's input in full when its summary line cut it short, else ""."""
        summary = cls._tool_summary_value(name, input_data)
        if len(" ".join(summary.split())) <= cls._MAX_SUMMARY_LENGTH:
            return ""
        if len(summary) > cls._MAX_FULL_INPUT_LENGTH:
            return summary[: cls._MAX_FULL_INPUT_LENGTH] + "…"
        return summary

    # Lines that report a tool's result under its summary line
    _STATS_PREFIXES = ("  ⎿  ", "⏹ ")
    _RE_BOLD_TOOL_NAME = re.compile(r"^\*\*([^*]+)\*\*")

    # Blank lines atop a tool call's quote: they fill its collapsed preview
    # (the client shows about 3 lines), so the call is an empty block until
    # expanded. U+2800 (braille blank) is not whitespace, so Telegram keeps it;
    # real blank lines at the start of a message would be trimmed.
    _TOOL_FOLD_PAD = "\u2800\n" * 3

    @classmethod
    def _fold_tool_message(cls, text: str) -> str:
        """A tool message as one collapsed quote, blank until expanded.

        Under _TOOL_FOLD_PAD: "Bash(cmd…) -- Output 70 lines", then the
        output, then the input in full. In Telegram the whole call is one
        small empty block, easy to skip. A quote shows its text verbatim, so
        the tool name loses its bold.
        """
        start, end = cls.EXPANDABLE_QUOTE_START, cls.EXPANDABLE_QUOTE_END
        head, _, rest = text.partition("\n")
        head = cls._RE_BOLD_TOOL_NAME.sub(r"\1", head)
        body = rest.replace(start, "").replace(end, "").strip("\n")
        inner = f"{head}\n{body}" if body else head
        return f"{start}{cls._TOOL_FOLD_PAD}{inner}{end}"

    @classmethod
    def _inline_stats(cls, text: str) -> str:
        """Join the stats lines under a tool's summary line onto it.

        "**Bash**(cmd…)\n  ⎿  Output 70 lines" → "**Bash**(cmd…) -- Output 70 lines"
        """
        head, *rest = text.split("\n")
        while rest and rest[0].startswith(cls._STATS_PREFIXES):
            line = rest.pop(0)
            for prefix in cls._STATS_PREFIXES:
                line = line.removeprefix(prefix)
            head += f" -- {line.strip()}"
        return "\n".join([head, *rest])

    @classmethod
    def _with_full_input(cls, text: str, full_input: str) -> str:
        """Add a tool's full input at the end of the message's collapsed quote.

        At the end: a collapsed quote previews its first lines, and the
        output says more there than the command the summary line starts.
        """
        if not full_input:
            return text
        end = cls.EXPANDABLE_QUOTE_END
        if end in text:
            head, _, tail = text.rpartition(end)
            return f"{head}\n\n{full_input}{end}{tail}"
        return f"{text}\n{cls._format_expandable_quote(full_input)}"

    @staticmethod
    def _tool_summary_value(name: str, input_data: dict | Any) -> str:
        """The input value a tool's summary shows (a path, command, query…)."""
        if not isinstance(input_data, dict):
            return ""

        # Pick a meaningful short summary based on tool name
        summary = ""
        if name in ("Read", "Glob"):
            summary = input_data.get("file_path") or input_data.get("pattern", "")
        elif name == "Write":
            summary = input_data.get("file_path", "")
        elif name in ("Edit", "NotebookEdit"):
            summary = input_data.get("file_path") or input_data.get("notebook_path", "")
            # Note: Edit/Update diff and stats are generated in tool_result stage,
            # not here. We just show the tool name and file path.
        elif name == "Bash":
            summary = input_data.get("command", "")
        elif name == "Grep":
            summary = input_data.get("pattern", "")
        elif name == "Task":
            summary = input_data.get("description", "")
        elif name == "WebFetch":
            summary = input_data.get("url", "")
        elif name == "WebSearch":
            summary = input_data.get("query", "")
        elif name == "TodoWrite":
            todos = input_data.get("todos", [])
            if isinstance(todos, list):
                summary = f"{len(todos)} item(s)"
        elif name == "TodoRead":
            summary = ""
        elif name == "AskUserQuestion":
            questions = input_data.get("questions", [])
            if isinstance(questions, list) and questions:
                q = questions[0]
                if isinstance(q, dict):
                    summary = q.get("question", "")
        elif name == "ExitPlanMode":
            summary = ""
        elif name == "Skill":
            summary = input_data.get("skill", "")
        else:
            # Generic: show first string value
            for v in input_data.values():
                if isinstance(v, str) and v:
                    summary = v
                    break

        return summary if isinstance(summary, str) else ""

    @staticmethod
    def extract_tool_result_text(content: list | Any) -> str:
        """Extract text from a tool_result content block."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    if t:
                        parts.append(t)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return ""

    @staticmethod
    def extract_tool_result_images(
        content: list | Any,
    ) -> list[tuple[str, bytes]] | None:
        """Extract base64-encoded images from a tool_result content block.

        Returns list of (media_type, raw_bytes) tuples, or None if no images found.
        """
        if not isinstance(content, list):
            return None
        images: list[tuple[str, bytes]] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            source = item.get("source")
            if not isinstance(source, dict) or source.get("type") != "base64":
                continue
            media_type = source.get("media_type", "image/png")
            data_str = source.get("data", "")
            if not data_str:
                continue
            try:
                raw_bytes = base64.b64decode(data_str)
                images.append((media_type, raw_bytes))
            except Exception:
                logger.debug("Failed to decode base64 image in tool_result")
        return images if images else None

    @classmethod
    def parse_message(cls, data: dict) -> ParsedMessage | None:
        """Parse a message entry from the JSONL data.

        Args:
            data: Parsed JSON dict from a JSONL line

        Returns:
            ParsedMessage or None if not a parseable message
        """
        msg_type = cls.get_message_type(data)

        if msg_type not in ("user", "assistant"):
            return None

        message = data.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content", "")

        if isinstance(content, list):
            text = cls.extract_text_only(content)
        else:
            text = str(content) if content else ""
        text = cls._RE_ANSI_ESCAPE.sub("", text)

        # Detect local command responses in user messages.
        # These are rendered as bot replies: "❯ /cmd\n  ⎿  output"
        if msg_type == "user" and text:
            stdout_match = cls._RE_LOCAL_STDOUT.search(text)
            if stdout_match:
                stdout = stdout_match.group(1).strip()
                cmd_match = cls._RE_COMMAND_NAME.search(text)
                cmd = cmd_match.group(1) if cmd_match else None
                return ParsedMessage(
                    message_type="local_command",
                    text=stdout,
                    tool_name=cmd,  # reuse field for command name
                )
            # Pure command invocation (no stdout) — carry command name
            cmd_match = cls._RE_COMMAND_NAME.search(text)
            if cmd_match:
                return ParsedMessage(
                    message_type="local_command_invoke",
                    text="",
                    tool_name=cmd_match.group(1),
                )

        return ParsedMessage(
            message_type=msg_type,
            text=text,
        )

    @staticmethod
    def get_timestamp(data: dict) -> str | None:
        """Extract timestamp from message data."""
        return data.get("timestamp")

    EXPANDABLE_QUOTE_START = "\x02EXPQUOTE_START\x02"
    EXPANDABLE_QUOTE_END = "\x02EXPQUOTE_END\x02"

    @classmethod
    def _format_expandable_quote(cls, text: str) -> str:
        """Format text as a Telegram expandable blockquote.

        Wraps text with sentinel markers. The actual MarkdownV2 formatting
        (> prefix, || suffix, escaping) is done in convert_markdown() after
        telegramify processes the surrounding content.
        """
        return f"{cls.EXPANDABLE_QUOTE_START}{text}{cls.EXPANDABLE_QUOTE_END}"

    @classmethod
    def _format_tool_result_text(cls, text: str, tool_name: str | None = None) -> str:
        """Format tool result text with statistics summary.

        Shows relevant statistics for each tool type, with expandable quote for full content.

        No truncation here — per project principles, truncation is handled
        only at the send layer (split_message / _render_expandable_quote).
        """
        if not text:
            return ""

        line_count = text.count("\n") + 1 if text else 0

        # Tool-specific statistics
        if tool_name == "Read":
            # Read: show line count instead of full content
            return f"  ⎿  Read {line_count} lines"

        elif tool_name == "Write":
            # Write: show lines written
            stats = f"  ⎿  Wrote {line_count} lines"
            return stats

        elif tool_name == "Bash":
            # Bash: show output line count
            if line_count > 0:
                stats = f"  ⎿  Output {line_count} lines"
                return stats + "\n" + cls._format_expandable_quote(text)
            return cls._format_expandable_quote(text)

        elif tool_name == "Grep":
            # Grep: show match count (count non-empty lines)
            matches = len([line for line in text.split("\n") if line.strip()])
            stats = f"  ⎿  Found {matches} matches"
            return stats + "\n" + cls._format_expandable_quote(text)

        elif tool_name == "Glob":
            # Glob: show file count
            files = len([line for line in text.split("\n") if line.strip()])
            stats = f"  ⎿  Found {files} files"
            return stats + "\n" + cls._format_expandable_quote(text)

        elif tool_name == "Task":
            # Task: show output length
            if line_count > 0:
                stats = f"  ⎿  Agent output {line_count} lines"
                return stats + "\n" + cls._format_expandable_quote(text)
            return cls._format_expandable_quote(text)

        elif tool_name == "WebFetch":
            # WebFetch: show content length
            char_count = len(text)
            stats = f"  ⎿  Fetched {char_count} characters"
            return stats + "\n" + cls._format_expandable_quote(text)

        elif tool_name == "WebSearch":
            # WebSearch: show results count (estimate by sections)
            results = text.count("\n\n") + 1 if text else 0
            stats = f"  ⎿  {results} search results"
            return stats + "\n" + cls._format_expandable_quote(text)

        # Default: expandable quote without stats
        return cls._format_expandable_quote(text)

    @classmethod
    def parse_entries(
        cls,
        entries: list[dict],
        pending_tools: dict[str, PendingToolInfo] | None = None,
    ) -> tuple[list[ParsedEntry], dict[str, PendingToolInfo]]:
        """Parse a list of JSONL entries into a flat list of display-ready messages.

        This is the shared core logic used by both get_recent_messages (history)
        and check_for_updates (monitor).

        Args:
            entries: List of parsed JSONL dicts (already filtered through parse_line)
            pending_tools: Optional carry-over pending tool_use state from a
                previous call (tool_use_id -> formatted summary). Used by the
                monitor to handle tool_use and tool_result arriving in separate
                poll cycles.

        Returns:
            Tuple of (parsed entries, remaining pending_tools state)
        """
        result: list[ParsedEntry] = []
        last_cmd_name: str | None = None
        # Pending tool_use blocks keyed by id
        _carry_over = pending_tools is not None
        if pending_tools is None:
            pending_tools = {}
        else:
            pending_tools = dict(pending_tools)  # don't mutate caller's dict

        tagged = 0  # result[:tagged] already carry their source_index
        for index, data in enumerate(entries):
            # Whatever the previous iteration appended came from its entry
            for entry in result[tagged:]:
                entry.source_index = index - 1
            tagged = len(result)

            msg_type = cls.get_message_type(data)
            if msg_type == "attachment":
                prompt = cls._queued_user_prompt(data)
                if prompt:
                    result.append(
                        ParsedEntry(
                            role="user",
                            text=prompt,
                            content_type="text",
                            timestamp=cls.get_timestamp(data),
                        )
                    )
                continue
            if msg_type not in ("user", "assistant"):
                continue

            # Extract timestamp for this entry
            entry_timestamp = cls.get_timestamp(data)

            message = data.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content", "")
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}] if content else []

            parsed = cls.parse_message(data)

            # Handle local command messages first
            if parsed:
                if parsed.message_type == "local_command_invoke":
                    last_cmd_name = parsed.tool_name
                    continue
                if parsed.message_type == "local_command":
                    cmd = parsed.tool_name or last_cmd_name or ""
                    text = parsed.text
                    if cmd:
                        if "\n" in text:
                            formatted = f"❯ `{cmd}`\n```\n{text}\n```"
                        else:
                            formatted = f"❯ `{cmd}`\n`{text}`"
                    else:
                        if "\n" in text:
                            formatted = f"```\n{text}\n```"
                        else:
                            formatted = f"`{text}`"
                    result.append(
                        ParsedEntry(
                            role="assistant",
                            text=formatted,
                            content_type="local_command",
                            timestamp=entry_timestamp,
                        )
                    )
                    last_cmd_name = None
                    continue
            last_cmd_name = None

            if msg_type == "assistant":
                # Process content blocks
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")

                    if btype == "text":
                        t = block.get("text", "").strip()
                        if t and t != cls._NO_CONTENT_PLACEHOLDER:
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=t,
                                    content_type="text",
                                    timestamp=entry_timestamp,
                                )
                            )

                    elif btype == "tool_use":
                        tool_id = block.get("id", "")
                        name = block.get("name", "unknown")
                        inp = block.get("input", {})
                        summary = cls.format_tool_use_summary(name, inp)
                        full_input = cls.tool_full_input(name, inp)

                        # ExitPlanMode: emit plan content as text before tool_use entry
                        if name == "ExitPlanMode" and isinstance(inp, dict):
                            plan = inp.get("plan", "")
                            if plan:
                                result.append(
                                    ParsedEntry(
                                        role="assistant",
                                        text=plan,
                                        content_type="text",
                                        timestamp=entry_timestamp,
                                    )
                                )
                        if tool_id:
                            # Store tool info for later tool_result formatting
                            # Edit tool needs input_data to generate diff in tool_result stage
                            input_data = (
                                inp if name in ("Edit", "NotebookEdit") else None
                            )
                            pending_tools[tool_id] = PendingToolInfo(
                                summary=summary,
                                tool_name=name,
                                input_data=input_data,
                                full_input=full_input,
                            )
                            # Also emit tool_use entry with tool_name for immediate handling
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=cls._fold_tool_message(
                                        cls._with_full_input(summary, full_input)
                                    ),
                                    content_type="tool_use",
                                    tool_use_id=tool_id,
                                    timestamp=entry_timestamp,
                                    tool_name=name,
                                )
                            )
                        else:
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=cls._fold_tool_message(
                                        cls._with_full_input(summary, full_input)
                                    ),
                                    content_type="tool_use",
                                    tool_use_id=tool_id or None,
                                    timestamp=entry_timestamp,
                                    tool_name=name,
                                )
                            )

                    elif btype == "thinking":
                        # Thinking stored without its text says nothing
                        # worth a message, so only real thinking is shown.
                        # Narration is a progress update written for the
                        # user that arrives as a thinking block: plain text.
                        thinking_text = block.get("thinking", "")
                        if thinking_text and cls._is_narration(block):
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=thinking_text.strip(),
                                    content_type="narration",
                                    timestamp=entry_timestamp,
                                )
                            )
                        elif thinking_text:
                            quoted = cls._format_expandable_quote(thinking_text)
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=quoted,
                                    content_type="thinking",
                                    timestamp=entry_timestamp,
                                )
                            )

            elif msg_type == "user":
                # Check for tool_result blocks and merge with pending tools
                user_text_parts: list[str] = []

                for block in content:
                    if not isinstance(block, dict):
                        if isinstance(block, str) and block.strip():
                            user_text_parts.append(block.strip())
                        continue
                    btype = block.get("type", "")

                    if btype == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")
                        result_text = cls.extract_tool_result_text(result_content)
                        result_images = cls.extract_tool_result_images(result_content)
                        is_error = block.get("is_error", False)
                        is_interrupted = result_text == cls._INTERRUPTED_TEXT
                        tool_info = pending_tools.pop(tool_use_id, None)
                        _tuid = tool_use_id or None
                        bash_stats, bash_diff = cls._bash_edit_diff(data)

                        # Extract tool info from PendingToolInfo object
                        if tool_info is None:
                            tool_summary = None
                            tool_name = None
                            tool_input_data = None
                            full_input = ""
                        else:
                            tool_summary = tool_info.summary
                            tool_name = tool_info.tool_name
                            tool_input_data = tool_info.input_data
                            full_input = tool_info.full_input

                        if is_interrupted:
                            # Show interruption inline with tool summary
                            entry_text = tool_summary or ""
                            if entry_text:
                                entry_text += "\n⏹ Interrupted"
                            else:
                                entry_text = "⏹ Interrupted"
                            entry_text = cls._fold_tool_message(
                                cls._with_full_input(
                                    cls._inline_stats(entry_text), full_input
                                )
                            )
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=entry_text,
                                    content_type="tool_result",
                                    tool_use_id=_tuid,
                                    timestamp=entry_timestamp,
                                )
                            )
                        elif is_error:
                            # Show error in stats line
                            if tool_summary:
                                entry_text = tool_summary
                            else:
                                entry_text = "**Error**"
                            # Add error message in stats format
                            if result_text:
                                # Take first line of error as summary
                                error_summary = result_text.split("\n")[0]
                                if len(error_summary) > 100:
                                    error_summary = error_summary[:100] + "…"
                                entry_text += f"\n  ⎿  Error: {error_summary}"
                                # If multi-line error, add expandable quote
                                if "\n" in result_text:
                                    entry_text += "\n" + cls._format_expandable_quote(
                                        result_text
                                    )
                            else:
                                entry_text += "\n  ⎿  Error"
                            if bash_stats:
                                entry_text += "\n" + "\n".join(bash_stats)
                                if bash_diff:
                                    entry_text += "\n" + cls._format_expandable_quote(
                                        bash_diff
                                    )
                            entry_text = cls._fold_tool_message(
                                cls._with_full_input(
                                    cls._inline_stats(entry_text), full_input
                                )
                            )
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=entry_text,
                                    content_type="tool_result",
                                    tool_use_id=_tuid,
                                    timestamp=entry_timestamp,
                                    image_data=result_images,
                                )
                            )
                        elif tool_summary:
                            entry_text = tool_summary
                            # For Edit tool, generate diff stats and expandable quote
                            if tool_name == "Edit" and tool_input_data and result_text:
                                old_s = tool_input_data.get("old_string", "")
                                new_s = tool_input_data.get("new_string", "")
                                if old_s and new_s:
                                    diff_text = cls._format_edit_diff(old_s, new_s)
                                    if diff_text:
                                        added = sum(
                                            1
                                            for line in diff_text.split("\n")
                                            if line.startswith("+")
                                            and not line.startswith("+++")
                                        )
                                        removed = sum(
                                            1
                                            for line in diff_text.split("\n")
                                            if line.startswith("-")
                                            and not line.startswith("---")
                                        )
                                        stats = f"  ⎿  Added {added} lines, removed {removed} lines"
                                        entry_text += (
                                            "\n"
                                            + stats
                                            + "\n"
                                            + cls._format_expandable_quote(diff_text)
                                        )
                            # Bash that changed files: their diff first (the
                            # quote is cut to fit a message), then the output
                            elif bash_stats:
                                stats = list(bash_stats)
                                if result_text:
                                    lines = result_text.count("\n") + 1
                                    stats.insert(0, f"  ⎿  Output {lines} lines")
                                entry_text += "\n" + "\n".join(stats)
                                quoted = "\n".join(
                                    t for t in (bash_diff, result_text) if t
                                )
                                if quoted:
                                    entry_text += "\n" + cls._format_expandable_quote(
                                        quoted
                                    )
                            # For other tools, append formatted result text
                            elif (
                                result_text
                                and cls.EXPANDABLE_QUOTE_START not in tool_summary
                            ):
                                entry_text += "\n" + cls._format_tool_result_text(
                                    result_text, tool_name
                                )
                            entry_text = cls._fold_tool_message(
                                cls._with_full_input(
                                    cls._inline_stats(entry_text), full_input
                                )
                            )
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=entry_text,
                                    content_type="tool_result",
                                    tool_use_id=_tuid,
                                    timestamp=entry_timestamp,
                                    image_data=result_images,
                                )
                            )
                        elif result_text or result_images:
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=cls._format_tool_result_text(
                                        result_text, tool_name
                                    )
                                    if result_text
                                    else (tool_summary or ""),
                                    content_type="tool_result",
                                    tool_use_id=_tuid,
                                    timestamp=entry_timestamp,
                                    image_data=result_images,
                                )
                            )

                    elif btype == "text":
                        t = block.get("text", "").strip()
                        if t and not cls._RE_SYSTEM_TAGS.search(t):
                            user_text_parts.append(t)

                # Add user text if present (skip if message was only tool_results)
                notice = (
                    cls._format_notification(data, "\n".join(user_text_parts))
                    if user_text_parts
                    else ""
                )
                if notice:
                    result.append(
                        ParsedEntry(
                            role="assistant",
                            text=notice,
                            content_type="notification",
                            timestamp=entry_timestamp,
                        )
                    )
                elif user_text_parts:
                    combined = "\n".join(user_text_parts)
                    # Skip if it looks like local command XML
                    if not cls._RE_LOCAL_STDOUT.search(
                        combined
                    ) and not cls._RE_COMMAND_NAME.search(combined):
                        result.append(
                            ParsedEntry(
                                role="user",
                                text=combined,
                                content_type="text",
                                timestamp=entry_timestamp,
                            )
                        )

        for entry in result[tagged:]:
            entry.source_index = len(entries) - 1

        # Flush remaining pending tools at end.
        # In carry-over mode (monitor), keep them pending for the next call
        # without emitting entries. In one-shot mode (history), emit them.
        remaining_pending = dict(pending_tools)
        if not _carry_over:
            for tool_id, tool_info in pending_tools.items():
                result.append(
                    ParsedEntry(
                        role="assistant",
                        text=tool_info.summary,
                        content_type="tool_use",
                        tool_use_id=tool_id,
                    )
                )

        # Strip whitespace
        for entry in result:
            entry.text = entry.text.strip()

        return result, remaining_pending
