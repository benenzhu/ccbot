#!/usr/bin/env python3
"""Convert a Claude Code transcript (JSONL) into readable Markdown.

Run with no arguments to pick a session from an interactive list; pass a
path or a session-id prefix to convert one directly.

Transcripts are trees, not flat logs: re-sending an edited prompt forks a
new branch off the same parentUuid. Only the branch ending at the newest
leaf is emitted, so abandoned drafts never show up in the output.

Tool calls and their results are dropped entirely. When that leaves two
assistant replies adjacent, they stay separate blocks — Claude often
speaks, runs a tool, then speaks again, and those are distinct remarks.

Thinking is only emitted with --thinking, and recent Claude versions no
longer persist it in plaintext, so that flag is usually a no-op.

Usage:
  scripts/transcript_to_md.py                     # pick from a list
  scripts/transcript_to_md.py ce242b07            # by session-id prefix
  scripts/transcript_to_md.py path/to.jsonl -o out.md
  scripts/transcript_to_md.py --thinking          # include thinking
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECTS_DIR = Path.home() / ".claude" / "projects"


@dataclass
class Session:
    """A transcript file plus the metadata shown in the picker."""

    path: Path
    project: str
    size: int
    mtime: float
    turns: int
    preview: str

    @property
    def session_id(self) -> str:
        return self.path.stem


def _decode_project(dir_name: str) -> str:
    """Turn a flattened project dir name back into a path-ish label."""
    return dir_name.replace("-", "/") if dir_name.startswith("-") else dir_name


def _iter_entries(path: Path):
    """Yield parsed JSONL objects, skipping malformed lines."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _block_text(content, want: str) -> str:
    """Join text from content blocks of the given type ("text"/"thinking")."""
    if isinstance(content, str):
        return content if want == "text" else ""
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != want:
            continue
        # Thinking blocks carry their prose under "thinking", not "text"
        parts.append(block.get("text") or block.get("thinking") or "")
    return "\n".join(p for p in parts if p)


def _is_visible_user_text(entry: dict, text: str) -> bool:
    """True for prose the human actually typed.

    Filters out meta entries and the synthetic <...> payloads Claude Code
    injects (command output, hook context, system reminders).
    """
    if entry.get("isMeta"):
        return False
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("<")


def main_branch(entries: list[dict]) -> list[dict]:
    """Return only the entries on the branch ending at the newest leaf.

    Walks parentUuid links back from the latest-timestamped leaf, so an
    edited-and-resent prompt drops its abandoned sibling branch.
    """
    nodes = {e["uuid"]: e for e in entries if "uuid" in e}
    if not nodes:
        return entries
    parents = {e.get("parentUuid") for e in nodes.values()}
    leaves = [u for u in nodes if u not in parents]
    if not leaves:
        return entries

    newest = max(leaves, key=lambda u: nodes[u].get("timestamp") or "")
    chain: list[dict] = []
    seen: set[str] = set()
    cursor: str | None = newest
    while cursor and cursor in nodes and cursor not in seen:
        seen.add(cursor)
        chain.append(nodes[cursor])
        cursor = nodes[cursor].get("parentUuid")
    chain.reverse()
    return chain


def scan_sessions() -> list[Session]:
    """Collect every transcript under ~/.claude/projects, newest first."""
    sessions: list[Session] = []
    if not PROJECTS_DIR.is_dir():
        return sessions

    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for path in project_dir.glob("*.jsonl"):
            try:
                stat = path.stat()
            except OSError:
                continue

            turns, preview = 0, ""
            for entry in _iter_entries(path):
                if entry.get("type") != "user":
                    continue
                text = _block_text((entry.get("message") or {}).get("content"), "text")
                if not _is_visible_user_text(entry, text):
                    continue
                turns += 1
                if not preview:
                    preview = " ".join(text.split())[:60]

            sessions.append(
                Session(
                    path=path,
                    project=_decode_project(project_dir.name),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    turns=turns,
                    preview=preview,
                )
            )

    sessions.sort(key=lambda s: s.mtime, reverse=True)
    return sessions


def _human_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f}MB"
    return f"{size / 1024:.0f}KB"


def choose_session(sessions: list[Session]) -> Session | None:
    """Print a numbered list and read the user's pick from stdin."""
    print(f"\n{PROJECTS_DIR} 下的会话（按最近修改排序）:\n")
    width = len(str(len(sessions)))
    for i, s in enumerate(sessions, 1):
        when = datetime.fromtimestamp(s.mtime).astimezone().strftime("%m-%d %H:%M")
        print(
            f"  {i:>{width}}. {_human_size(s.size):>6}  {when}  "
            f"{s.turns:>3}轮  {s.project}"
        )
        if s.preview:
            print(f"  {'':>{width}}  {s.preview}")
    print()

    try:
        raw = input(f"选择要转换的会话 [1-{len(sessions)}, 回车取消]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not raw:
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= len(sessions):
        print(f"无效的选择: {raw}", file=sys.stderr)
        return None
    return sessions[int(raw) - 1]


def resolve_target(token: str) -> Path | None:
    """Resolve a CLI argument that is either a path or a session-id prefix."""
    path = Path(token)
    if path.is_file():
        return path
    matches = [s for s in scan_sessions() if s.session_id.startswith(token)]
    if not matches:
        print(f"找不到匹配的会话: {token}", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"'{token}' 匹配到多个会话:", file=sys.stderr)
        for s in matches:
            print(f"  {s.session_id}  {s.project}", file=sys.stderr)
        return None
    return matches[0].path


def to_markdown(path: Path, include_thinking: bool = False) -> str:
    """Render one transcript's main branch as Markdown."""
    entries = list(_iter_entries(path))
    chain = main_branch(entries)

    started = next((e.get("timestamp") for e in chain if e.get("timestamp")), "")
    cwd = next((e.get("cwd") for e in chain if e.get("cwd")), "")

    lines = [f"# 对话记录 · {path.stem[:8]}", ""]
    if cwd:
        lines.append(f"- **目录**: `{cwd}`")
    if started:
        lines.append(f"- **开始**: {started[:19].replace('T', ' ')}")
    lines += [f"- **来源**: `{path}`", "", "---", ""]

    # An assistant turn often contains several separate replies, split by
    # tool calls that we drop ("先说一句 → 查文件 → 再说一句"). Those are
    # distinct utterances, not one paragraph, so each keeps its own block;
    # only the *heading* is skipped on repeats, marking continuations with
    # a rule instead.
    speaker: str | None = None

    def say(role: str, heading: str, body: str) -> None:
        nonlocal speaker
        if speaker != role:
            lines.extend([heading, ""])
        else:
            lines.extend(["·　·　·", ""])
        speaker = role
        lines.extend([body, ""])

    for entry in chain:
        kind = entry.get("type")
        content = (entry.get("message") or {}).get("content")

        if kind == "user":
            text = _block_text(content, "text")
            if _is_visible_user_text(entry, text):
                say("user", "## 👤 用户", text.strip())

        elif kind == "assistant":
            if include_thinking:
                thinking = _block_text(content, "thinking").strip()
                if thinking:
                    say(
                        "assistant",
                        "## 🤖 Claude",
                        "<details><summary>💭 思考</summary>\n\n"
                        f"{thinking}\n\n</details>",
                    )
            text = _block_text(content, "text").strip()
            if text:
                say("assistant", "## 🤖 Claude", text)

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 Claude Code 的 JSONL transcript 转成 Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="jsonl 路径或 session-id 前缀；省略则交互式选择",
    )
    parser.add_argument("-o", "--output", help="输出文件（默认打印到 stdout）")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="保留 thinking 段落（新版 Claude 不再明文保存，多半为空）",
    )
    args = parser.parse_args()

    if args.target:
        path = resolve_target(args.target)
    else:
        sessions = scan_sessions()
        if not sessions:
            print(f"{PROJECTS_DIR} 下没有找到任何会话", file=sys.stderr)
            return 1
        chosen = choose_session(sessions)
        path = chosen.path if chosen else None

    if path is None:
        return 1

    markdown = to_markdown(path, include_thinking=args.thinking)

    if args.output:
        out = Path(args.output)
        out.write_text(markdown, encoding="utf-8")
        print(f"已写入 {out}  ({len(markdown)} 字符)", file=sys.stderr)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
