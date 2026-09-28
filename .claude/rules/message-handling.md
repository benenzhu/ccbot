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
- Merging stops when combined length exceeds 3800 characters (to avoid pagination)

## Status Message Handling

**Conversion**: The status message is edited into the first content message, reducing message count:
- When a status message exists, the first content message updates it via edit
- Subsequent content messages are sent as new messages

**Polling**: Background task polls terminal status for all active windows at 5-second intervals, so status edits use only a small share of the send quota.

**Status yields to content**: a status send/edit goes out only when the bot-wide limiter is idle (`_status_slot_free`), so it never makes Claude's output wait. No "typing" chat action is sent. A tool_result edit leaves the status message in place (it is still the last message); tables/images sent after such an edit clear it first.

**Deduplication**: Identical status text skips the edit. A status line whose only change is its counters (elapsed time, tokens) is refreshed at most every 30s (`STATUS_COUNTER_REFRESH`); any other change shows on the next poll.

## Rate Limiting

- `AIORateLimiter(overall_max_rate=1, overall_time_period=2.2, max_retries=5)` on the Application: one request per 2.2s bot-wide (every topic lives in one private chat, which AIORateLimiter otherwise leaves unlimited)
- On 429, AIORateLimiter pauses all concurrent requests (`_retry_after_event`) and retries after the ban
- On restart, the global bucket is pre-filled (`_level=max_rate`) to avoid burst against Telegram's persisted server-side counter
- Status polling interval: 5 seconds (skips enqueue when queue is non-empty)

## Performance Optimizations

**mtime cache**: The monitoring loop maintains an in-memory file mtime cache, skipping reads for unchanged files.

**Byte offset incremental reads**: Each tracked session records `last_byte_offset`, reading only new content. File truncation (offset > file_size) is detected and offset is auto-reset.

## No Message Truncation

Historical messages (tool_use summaries, tool_result text, user/assistant messages) are always kept in full — no character-level truncation at the parsing layer. Long text is handled exclusively at the send layer: `split_message` splits by Telegram's 4096-character limit; real-time messages get `[1/N]` text suffixes, history pages get inline keyboard navigation.
