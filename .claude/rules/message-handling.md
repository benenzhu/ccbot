# Message Handling

## Message Queue Architecture

Per-user message queues + worker pattern for all send tasks:
- Messages are sent in receive order (FIFO)
- Status messages always follow content messages
- Multi-user concurrent processing without interference

**Message merging**: The worker automatically merges consecutive mergeable content messages on dequeue:
- Content messages for the same window can be merged (including text, thinking); merged parts are joined into one Telegram message while the converted MarkdownV2 fits 4096 chars (each part is its own send, so joining is what saves requests)
- tool_use breaks the merge chain and is sent separately (message ID recorded for later editing)
- tool_result breaks the merge chain and is edited into the tool_use message (preventing order confusion)
- Backlog: when a tool_use's tool_result is already queued, the tool_use is not sent — the result goes out once in its place (same final message, one request instead of send + edit)
- Busy window: while the bot-wide window has `STATUS_RESERVE` (5) or fewer requests free (`_window_busy`), a topic's queued tool calls (each a tool_use with its tool_result queued, in a row) go out together, up to `TOOL_BATCH_MAX` (5) per message (`_batch_queued_tool_calls`). The collapsed preview's first line has a dot per call (`...` for three) in place of the first padding line; below the padding, the numbered summary lines, then each call's output and input, cut so the message fits 4096 chars. The topic's next non-tool message, or a tool still running, ends the batch
- Merging stops when combined length exceeds 3800 characters (to avoid pagination)

**Tables**: markdown tables in Claude's prose go out as native Telegram tables (one message each, at their position). Tool calls and thinking keep theirs inline — inside the collapsed quote they show verbatim, not as cards — since a message per table isn't worth its request there.

## What Gets Sent

- Claude's replies, tool calls (`CCBOT_SHOW_TOOL_CALLS`, on) and the user's own messages as `👤 …` (`CCBOT_SHOW_USER_MESSAGES`, on). A message the user sends while Claude is mid-turn is queued by Claude Code and recorded as a `queued_command` attachment, not a user entry; it is echoed all the same.
- Thinking blocks (`CCBOT_SHOW_THINKING`) and the status message below (`CCBOT_SHOW_STATUS`) are off by default: the `👤` echo already shows that a turn started.
- A tool call is one collapsed quote (`_fold_tool_message`) that looks empty until expanded, so it is easy to skip: it opens with 3 blank lines (`_TOOL_FOLD_PAD`, U+2800 — Telegram trims real blank lines at a message's start), which fill the collapsed preview (the client shows a collapsed quote's first ~3 lines; the Bot API can't change that). Below them: the summary with the result's stats joined on (`Bash(cmd…) -- Output 70 lines`, the input cut to 40 chars), then the output, then the input in full (up to 1000 chars). A quote is verbatim, so the tool name isn't bold.
- Notices Claude Code writes as user entries (`origin.kind` task-notification / peer: background task finished, a message from another session or subagent) are not echoed as `👤`: they become a `notification` — one line (`🔔 <summary>` / `📨 Message from <name>`) with the details collapsed.
- A subagent's transcript goes to its own topic (see topic-architecture.md, Subagent topics), with the same formatting; it shares the bot-wide rate window with every other topic, so many busy subagents slow everything down.
- Narration is always sent, as plain text: recent models keep their reasoning hidden (thinking text empty) and write short progress updates for the user as thinking blocks whose signature is tagged `narration` (`TranscriptParser._is_narration`); Claude Code shows them as ordinary text. Compact history replay drops them like tool calls.

## Status Message Handling (`CCBOT_SHOW_STATUS=true`)

**Conversion**: The status message is edited into the first content message, reducing message count:
- When a status message exists, the first content message updates it via edit
- Subsequent content messages are sent as new messages

**Polling**: Background task polls terminal status for all active windows at 5-second intervals. The status line is the spinner line just above the chrome separator (indented `⎿ Tip` / todo sub-lines under it are skipped; the separator above the prompt carries the session name after `/rename`). The line a finished turn leaves there (`Brewed for 3m 14s · done …`) is not a status.

**Once per turn, never edited**: the status message only announces that Claude started working. It goes out before Claude's first output of a turn, shows only the spinner text (`Zesting…`, without the timer/token details, which would go stale), is not updated, and turns into that first output (conversion above). Once a tool call or reply is on screen Claude is evidently working (`_turn_has_output`), so no further status is sent until the poller sees Claude idle (`end_turn`) or the user writes again.

**Status yields to content**: a status send goes out only while more than `STATUS_RESERVE` (5) slots are free in the bot-wide window (`_status_slot_free`), so a burst of Claude's output still finds slots. No "typing" chat action is sent. A tool_result edit leaves the status message in place (it is still the last message); images sent after such an edit clear it first.

## Rate Limiting

- `TelegramRateLimiter` (`rate_limit.py`) on the Application, in place of python-telegram-bot's `AIORateLimiter` (the library's `BaseRateLimiter` plug-in point): every request that targets a chat waits for a slot in sliding windows of 15 requests per 30s and 20 per 60s, bot-wide (every topic lives in one private chat). A burst of 15 goes out at once; a sustained flow is held to 20 a minute (15/30s alone, flat out for 4 minutes, earned a 429), and never waits much over 30s, so a busy topic still shows something every minute
- On top of it the queue paces each chat (`_pace`: ~1 message/s after a burst of 3), so a full window doesn't go out in one go
- On 429 ccbot stops: the limiter refuses every later request, calls `Application.stop_running()`, and `main` exits with the reason (ban length, when it lifts), first posting it to `CCBOT_ALERT_WEBHOOK` (a WeCom group-bot webhook, `alert.py`) if set. Undelivered messages are re-sent after the restart (offsets only advance on delivery)
- On restart, the window starts filled as if the bot had been sending at full rate (Telegram's server-side counter outlives a restart): one request per 3s until the 60s window rolls over
- Status polling interval: 5 seconds (skips enqueue when queue is non-empty)

## Performance Optimizations

**mtime cache**: The monitoring loop maintains an in-memory file mtime cache, skipping reads for unchanged files.

**Byte offset incremental reads**: Each tracked session records `last_byte_offset`, reading only new content. File truncation (offset > file_size) is detected and offset is auto-reset.

## No Message Truncation

Historical messages (tool_result text, user/assistant messages) are always kept in full — no character-level truncation at the parsing layer. Tool_use summary lines are the exception by design: one short line, with the input itself further down the collapsed quote. So are tool calls batched under a busy window: their output is cut to fit one message. Long text is handled exclusively at the send layer: `split_message` splits by Telegram's 4096-character limit; real-time messages get `[1/N]` text suffixes, history pages get inline keyboard navigation.
