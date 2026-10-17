# Topic-Only Architecture

The bot operates exclusively in Telegram Forum (topics) mode. There is **no** `active_sessions` mapping, **no** `/list` command, **no** General topic routing, and **no** backward-compatibility logic for older non-topic modes. Every code path assumes named topics.

## 1 Topic = 1 Window = 1 Session

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Topic ID   │ ───▶ │ Window ID   │ ───▶ │ Session ID  │
│  (Telegram) │      │ (tmux @id)  │      │  (Claude)   │
└─────────────┘      └─────────────┘      └─────────────┘
     thread_bindings      session_map.json
     (state.json)         (written by hook)
```

Window IDs (e.g. `@0`, `@12`) are guaranteed unique within a tmux server session. Window names are stored separately as display names (`window_display_names` map).

## Mapping 1: Topic → Window ID (thread_bindings)

```python
# session.py: SessionManager
thread_bindings: dict[int, dict[int, str]]  # user_id → {thread_id → window_id}
window_display_names: dict[str, str]        # window_id → window_name (for display)
```

- Storage: memory + `state.json`
- Written when: user creates a new session via the directory browser in a topic
- Purpose: route user messages to the correct tmux window

## Mapping 2: Window ID → Session (session_map.json)

```python
# session_map.json (key format: "tmux_session:window_id")
{
  "ccbot:@0": {"session_id": "uuid-xxx", "cwd": "/path/to/project", "window_name": "project"},
  "ccbot:@5": {"session_id": "uuid-yyy", "cwd": "/path/to/project", "window_name": "project-2"}
}
```

- Storage: `session_map.json`
- Written when: Claude Code's `SessionStart` hook fires
- Property: one window maps to one session; session_id changes after `/clear`
- Purpose: SessionMonitor uses this mapping to decide which sessions to watch

## Message Flows

**Outbound** (user → Claude):
```
User sends "hello" in topic (thread_id=42)
  → thread_bindings[user_id][42] → "@0"
  → send_to_window("@0", "hello")   # resolves via find_window_by_id
```

**Inbound** (Claude → user):
```
SessionMonitor reads new message (session_id = "uuid-xxx")
  → Iterate thread_bindings, find (user, thread) whose window_id maps to this session
  → Deliver message to user in the correct topic (thread_id)
```

**New topic flow**: First message in an unbound topic → directory browser → select directory → session picker (if existing sessions found) or create window → bind topic → forward pending message.

**Resume session flow**: When selecting a directory with existing Claude sessions, a session picker UI is shown. Choosing a session runs `claude --resume <session_id>`. Note: `--resume` makes the hook report a new session_id but messages continue writing to the original JSONL file; the bot overrides window_state to track the original session_id.

**Fork command**: `/fork <name>` in a bound topic (a task may follow on the next lines) creates a new topic, starts `claude --resume <sid> --fork-session` in a new tmux window at the same cwd (new session id; inherited history is not re-sent), binds it, and names topic and window `<name>`. It then sends `/rename <name>` to the fork, so the Naming rule below keeps all three in step, followed by the task prefixed with `FORK_TASK_PROMPT` ("you are a fork, leave the original work to the original window"). If the launch fails, the new topic is deleted.

**Topic lifecycle**: Closing/deleting a topic auto-kills the associated tmux window and unbinds the thread. Closing arrives as an update; deletion does not, so it is detected when a send to the topic fails with "message thread not found" (`is_topic_gone`) and cleaned up on the next status poll — there is no periodic probe. Stale bindings (window deleted externally) are cleaned up by the status polling loop.

**Naming**: Claude's session name is the source of truth. `/rename <name>` (sent in the topic — it is forwarded to Claude — or typed in the terminal) makes Claude write a `custom-title` JSONL line; the monitor hands a *changed* title to `handle_session_renamed`, which renames the tmux window, its display name and the topic (Claude re-appends the same title often, so unchanged titles cost nothing). Renaming the topic in Telegram renames the tmux window (not Claude's name). The session picker shows the custom title. The `window_name` in session_map is the name at SessionStart, so it only names a window that has no display name yet; at startup, bound windows' display names follow their live tmux names (which every rename sets).

**Subagent topics** (`CCBOT_SUBAGENT_TOPICS`, on): the exception to 1 topic = 1 window. Each subagent a bound session starts gets a topic of its own, `sub-<parent window>-<task>` (the task is the 3-5 word `description` Claude gave it, from `agent-<id>.meta.json`), created on its first message in the parent topic's chat (`subagent_topics.py`). It shows the subagent's transcript like a main topic: tool calls included; its task (`📋 Task`) and later messages from the main session (`📨`) as collapsed notifications, not `👤`. Queue tasks carry the pseudo window id `agent:<id>`. The topic is read-only: anything sent there is answered with a pointer to the parent topic. It is created 🟢 (no extra rename) and turns ✅ once the subagent's reply ended its turn (`stop_reason: end_turn`; or its transcript has been quiet for an hour: stopped). The bot never deletes these topics. A follow-up to a subagent (the main session messages it again) is appended to the same transcript, so the same topic carries on. A subagent started before ccbot began watching the session (`subagents_since`) is read from where its transcript was when first seen: it gets a topic only once it writes again, starting there. Agent → topic is persisted in `subagent_topics.json`, the read offsets in `monitor_state.json` under `agent-<id>`, tied to the parent session.

**Status light**: the topic name carries a light in front of the window name — 🟢 while Claude works (spinner up), 🟡 while Claude is idle but background shells/agents listed in its bottom bar still run, ✅ for 24h after the work finished (`DONE_FOR`), none otherwise (`topic_light.py`, driven by the 5s status poll). It is the bot's: every rename the bot makes keeps it (`topic_name`), a rename in Telegram has it stripped before it reaches the window (`strip_light`). A change costs one `editForumTopic`, so a state must hold first (lighting up one poll, finishing 30s) and lights wait while the bot-wide window is short of room. Each lit topic's state, the name it was given and since when are persisted in `topic_lights.json` (✅ keeps its 24h across restarts); a lit topic whose window name changed is renamed again right away.

## Session Lifecycle

**Startup revival**: A topic whose window no longer exists at startup (tmux server died with all windows) but whose window_state knows its session and cwd is not unbound: `resolve_stale_ids()` returns it as a `DeadBinding`, and `_revive_topics` resumes the session in a new window (`claude --resume`, offset at EOF so nothing is replayed), names it as before and re-binds the topic. Topics without a known session, or whose transcript is gone, fall back to the directory browser.

**Startup cleanup**: On bot startup, all tracked sessions not present in session_map are cleaned up, preventing monitoring of closed sessions.

**Runtime change detection**: Each polling cycle checks for session_map changes:
- Window's session_id changed (e.g., after `/clear`) → clean up old session
- Window deleted → clean up corresponding session
