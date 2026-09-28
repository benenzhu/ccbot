#!/usr/bin/env python3
"""Telegram API traffic report from the ccbot log — where the send quota goes.

Reads the one-line-per-request entries written by ccbot.request_log
("api sendMessage purpose=... msg=... wait=... text=...") and prints, for the
bot's latest run by default:
  - a summary: request rate vs. the 27/min the bot-wide limit allows, time
    spent waiting for a slot, the bot's lag (read from the JSONL → sent), counts
    by API method and by purpose
  - hints at waste: failed requests, status churn, messages touched many times
  - the full timeline, oldest first, one request per line
With -f it instead follows the log like `tail -f`, printing requests live.

Every output line is cut to the terminal width (Chinese counts as two
columns), found from stdout, stderr or /dev/tty — so `| view -` fits too.

Usage:
  python scripts/api_log.py | view -
  python scripts/api_log.py --since 30m          # last 30 minutes of the log
  python scripts/api_log.py --all --enqueue      # whole log, plus queued content
  python scripts/api_log.py -f                   # live, last 20 requests first
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

CAPACITY_PER_MIN = 60 / 2.2  # AIORateLimiter(overall_max_rate=1, period=2.2)
BURST_GAP = timedelta(seconds=30)  # blank line in the timeline after a pause
FOLLOW_POLL_SECONDS = 0.5

RE_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - (\S+) - \w+ - (.*)$")
RE_ENQUEUE = re.compile(r"Enqueue (content|table|voice): user=\d+, window_id=(@\d+)")
RE_CONTENT_TYPE = re.compile(r"content_type=(\w+)")
RE_MDV2_ESCAPE = re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")


@dataclass
class Request:
    time: datetime
    method: str
    purpose: str = "-"
    thread: str = "-"
    msg: str = "-"
    wait: float = 0.0
    took: float = 0.0
    lag: float | None = None  # seconds from the bot reading it to sent
    text: str | None = None
    error: str | None = None


@dataclass
class Enqueued:
    time: datetime
    kind: str
    window: str


@dataclass
class BotStart:
    time: datetime


Event = Request | Enqueued | BotStart


# --- Parsing ---


def _parse_fields(rest: str) -> dict[str, str]:
    """Parse `key=value` pairs; text/error values are JSON strings."""
    fields: dict[str, str] = {}
    decoder = json.JSONDecoder()
    i = 0
    while i < len(rest):
        eq = rest.find("=", i)
        if eq < 0:
            break
        key = rest[i:eq]
        if rest.startswith('"', eq + 1):
            value, end = decoder.raw_decode(rest, eq + 1)
        else:
            end = rest.find(" ", eq + 1)
            end = len(rest) if end < 0 else end
            value = rest[eq + 1 : end]
        fields[key] = value
        i = end + 1
    return fields


def parse_event(line: str) -> Event | None:
    """One log line → the event it records, or None if it isn't one."""
    m = RE_LINE.match(line.rstrip("\n"))
    if not m:
        return None  # continuation of a multi-line message
    stamp, name, msg = m.groups()
    # The log writes local wall-clock time; naive on purpose
    when = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007
    if name == "ccbot.request_log" and msg.startswith("api "):
        method, _, rest = msg[4:].partition(" ")
        f = _parse_fields(rest)
        return Request(
            time=when,
            method=method,
            purpose=f.get("purpose", "-"),
            thread=f.get("thread", "-"),
            msg=f.get("msg", "-"),
            wait=float(f.get("wait", 0)),
            took=float(f.get("took", 0)),
            lag=float(f["lag"]) if "lag" in f else None,
            text=f.get("text"),
            error=f.get("error"),
        )
    if name == "ccbot.main" and msg.startswith("Starting Telegram bot"):
        return BotStart(when)
    if name == "ccbot.handlers.message_queue" and msg.startswith("Enqueue"):
        em = RE_ENQUEUE.search(msg)
        if em:
            ct = RE_CONTENT_TYPE.search(msg)
            return Enqueued(when, ct.group(1) if ct else em.group(1), em.group(2))
    return None


def _parse_since(value: str) -> timedelta:
    m = re.fullmatch(r"(\d+)([smhd])", value)
    if not m:
        raise argparse.ArgumentTypeError("use e.g. 90s, 30m, 2h, 1d")
    units = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    return timedelta(**{units[m.group(2)]: int(m.group(1))})


# --- Output that fits the terminal ---


def _char_width(ch: str) -> int:
    if unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0  # combining marks, zero-width joiners, variation selectors
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def fit(line: str, cols: int) -> str:
    """Cut line to cols display columns (0: don't cut), marking the cut with …."""
    if cols <= 0 or len(line) * 2 <= cols:
        return line
    if sum(map(_char_width, line)) <= cols:
        return line
    used = 0
    for i, ch in enumerate(line):
        used += _char_width(ch)
        if used > cols - 1:  # keep one column for the …
            return line[:i] + "…"
    return line


def terminal_columns() -> int:
    """Width of the terminal this runs in, even when stdout is a pipe; 0 if none."""
    for fd in (1, 2, 0):
        try:
            return os.get_terminal_size(fd).columns
        except OSError:
            continue
    try:
        with open("/dev/tty") as tty:
            return os.get_terminal_size(tty.fileno()).columns
    except OSError:
        return 0


class Output:
    """Prints lines cut to a fixed width, or to the current terminal width."""

    def __init__(self, cols: int | None) -> None:
        self._cols = cols  # None: ask the terminal per line, so resizes apply

    def __call__(self, line: str = "") -> None:
        cols = self._cols if self._cols is not None else terminal_columns()
        print(fit(line, cols))


def _readable(text: str | None) -> str:
    if text is None:
        return ""
    text = RE_MDV2_ESCAPE.sub(r"\1", text).replace("\n", " ⏎ ")
    # Tool output can carry tabs, carriage returns, ANSI escapes
    return "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in text)


# --- Report ---


def _pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:3.0f}%" if whole else "  -"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def print_summary(out: Output, reqs: list[Request], enq: list[Enqueued]) -> None:
    span_min = max((reqs[-1].time - reqs[0].time).total_seconds() / 60, 1 / 60)
    per_minute = Counter(r.time.strftime("%H:%M") for r in reqs)
    busiest, busiest_n = per_minute.most_common(1)[0]
    errors = [r for r in reqs if r.error]
    waits = [r.wait for r in reqs]

    out("== Summary ==")
    out(
        f"requests: {len(reqs)}  ({len(reqs) / span_min:.1f}/min over "
        f"{span_min:.1f} min; the limit allows {CAPACITY_PER_MIN:.1f}/min)"
    )
    out(f"busiest minute: {busiest} ({busiest_n} requests)")
    out(
        f"wait for a slot: p50 {_quantile(waits, 0.5):.1f}s  "
        f"p90 {_quantile(waits, 0.9):.1f}s  max {max(waits):.1f}s"
    )
    lags = [r.lag for r in reqs if r.lag is not None]
    if lags:
        out(
            f"bot lag (read from the JSONL → sent, {len(lags)} sends): "
            f"p50 {_quantile(lags, 0.5):.1f}s  p90 {_quantile(lags, 0.9):.1f}s  "
            f"max {max(lags):.1f}s"
        )
    out(f"failed: {len(errors)}")
    if enq:
        kinds = Counter(e.kind for e in enq)
        out(
            f"queued by Claude in the same span: {len(enq)} "
            f"({', '.join(f'{k} {n}' for k, n in kinds.most_common())})"
        )
    out()

    for title, key in (("by method", "method"), ("by purpose", "purpose")):
        counts = Counter(getattr(r, key) for r in reqs)
        out(f"{title}:")
        for name, n in counts.most_common():
            out(f"  {name:<22} {n:6}  {_pct(n, len(reqs))}")
        out()


def print_waste(out: Output, reqs: list[Request]) -> None:
    out("== Possibly redundant ==")
    errors = Counter(f"{r.method}: {r.error}" for r in reqs if r.error)
    out(f"failed requests (each still used a slot): {sum(errors.values())}")
    for what, n in errors.most_common(8):
        out(f"  {n:5}  {what}")

    status = [r for r in reqs if r.purpose == "status"]
    out(f"status requests: {len(status)}  ({_pct(len(status), len(reqs))} of all)")
    actions = sum(1 for r in reqs if r.method == "sendChatAction")
    if actions:
        out(f"sendChatAction (typing): {actions}")

    # Messages sent and deleted again shortly after (status churn)
    sent_at: dict[str, datetime] = {}
    short_lived = 0
    for r in reqs:
        if r.msg == "-" or r.error:
            continue
        if r.method.startswith("send"):
            sent_at[r.msg] = r.time
        elif (
            r.method == "deleteMessage"
            and r.msg in sent_at
            and r.time - sent_at[r.msg] <= timedelta(seconds=60)
        ):
            short_lived += 1
    out(f"messages deleted within 60s of being sent: {short_lived}")

    touched: dict[str, list[Request]] = defaultdict(list)
    for r in reqs:
        if r.msg != "-":
            touched[r.msg].append(r)
    busy = sorted(
        ((m, rs) for m, rs in touched.items() if len(rs) >= 3),
        key=lambda item: -len(item[1]),
    )
    out(f"messages hit by 3+ requests: {len(busy)}")
    for msg, rs in busy[:10]:
        steps = " → ".join(
            r.method.removesuffix("Text").removesuffix("Message") for r in rs
        )
        purposes = ",".join(sorted({r.purpose for r in rs}))
        out(f"  #{msg} ×{len(rs)} [{purposes}] {steps}")
    out()


class Timeline:
    """Formats events one line each, remembering what earlier lines established.

    Edits and deletes carry no topic, so it comes from the send that made the
    message; × counts the requests that hit the same message so far.
    """

    HEADER = (
        f"{'time':<8} {'wait':>4} {'lag':>5} {'method':<10} {'purpose':<11} "
        f"{'thread':<6} {'msg':<6} {'×':>2} text / error"
    )

    def __init__(self) -> None:
        self._last: datetime | None = None
        self._touches: Counter[str] = Counter()
        self._thread_of: dict[str, str] = {}

    def lines(self, ev: Event) -> list[str]:
        """Lines for one event (a blank one first after a pause)."""
        lines: list[str] = []
        if self._last is not None and ev.time - self._last > BURST_GAP:
            lines.append("")
        self._last = ev.time
        stamp = ev.time.strftime("%H:%M:%S")
        if isinstance(ev, BotStart):
            lines.append(f"{stamp}  ───── bot started ─────")
        elif isinstance(ev, Enqueued):
            lines.append(f"{stamp}            + queued {ev.kind} ({ev.window})")
        else:
            touch = self._touch(ev)
            detail = f"ERROR {ev.error}  " if ev.error else ""
            detail += _readable(ev.text)
            lines.append(
                f"{stamp} {ev.wait:4.1f} {_lag(ev.lag):>5} "
                f"{_short_method(ev.method):<10} "
                f"{ev.purpose:<11} {ev.thread:<6} {ev.msg:<6} {touch:>2} {detail}"
            )
        return lines

    def _touch(self, r: Request) -> int:
        if r.msg == "-":
            return 1
        self._touches[r.msg] += 1
        if r.thread != "-":
            self._thread_of[r.msg] = r.thread
        else:
            r.thread = self._thread_of.get(r.msg, "-")
        return self._touches[r.msg]


# Timeline column names for methods whose name doesn't shorten on its own
_SHORT_METHODS = {
    "answerCallbackQuery": "answerCb",
    "editMessageReplyMarkup": "editMarkup",
    "sendChatAction": "chatAction",
    "sendMediaGroup": "sendMedia",
}


def _short_method(method: str) -> str:
    """sendMessage → send, editMessageText → edit, deleteMessage → delete."""
    short = _SHORT_METHODS.get(method)
    return short or method.removesuffix("Text").removesuffix("Message")


def _lag(lag: float | None) -> str:
    if lag is None:
        return "-"
    return f"{lag:.1f}" if lag < 1000 else f"{lag / 60:.0f}m"


def _in_timeline(ev: Event, with_enqueue: bool) -> bool:
    return with_enqueue or not isinstance(ev, Enqueued)


# --- Modes ---


def report(out: Output, events: list[Event], args: argparse.Namespace) -> int:
    reqs = [ev for ev in events if isinstance(ev, Request)]
    if not reqs:
        print(
            f"No API request lines in {args.log}.\n"
            "They are written by ccbot.request_log — restart the bot on the "
            "current code first (or use -f to wait for them).",
            file=sys.stderr,
        )
        return 1

    last = reqs[-1].time
    runs = [ev.time for ev in events if isinstance(ev, BotStart) and ev.time <= last]
    if args.since:
        start = last - args.since
        scope_desc = f"last {args.since}"
    elif args.all or not runs:
        start = reqs[0].time
        scope_desc = "whole log"
    else:
        start = runs[-1]
        scope_desc = f"latest run (bot started {start:%Y-%m-%d %H:%M:%S})"
    scoped = [ev for ev in events if ev.time >= start]
    reqs = [ev for ev in scoped if isinstance(ev, Request)]
    if not reqs:
        print(f"No API requests in scope ({scope_desc}).", file=sys.stderr)
        return 1

    out(f"ccbot API requests — {args.log}")
    out(f"scope: {scope_desc}; {reqs[0].time} → {reqs[-1].time}")
    out()
    print_summary(out, reqs, [ev for ev in scoped if isinstance(ev, Enqueued)])
    print_waste(out, reqs)
    if not args.no_timeline:
        out("== Timeline (oldest first) ==")
        out(Timeline.HEADER)
        timeline = Timeline()
        for ev in scoped:
            if _in_timeline(ev, args.enqueue):
                for line in timeline.lines(ev):
                    out(line)
    return 0


def follow(
    out: Output,
    f: TextIO,
    partial: str,
    events: list[Event],
    args: argparse.Namespace,
) -> None:
    """Print the last requests of the latest run, then new ones as they come.

    partial is the unterminated last line already read from f.
    """
    starts = [i for i, ev in enumerate(events) if isinstance(ev, BotStart)]
    run = events[starts[-1] if starts else 0 :]
    run = [ev for ev in run if _in_timeline(ev, args.enqueue)]
    req_idx = [i for i, ev in enumerate(run) if isinstance(ev, Request)]
    if args.lines <= 0:
        shown_from = len(run)
    elif args.lines >= len(req_idx):
        shown_from = 0
    else:
        shown_from = req_idx[-args.lines]

    print(f"following {args.log} — Ctrl-C to stop", file=sys.stderr)
    out(Timeline.HEADER)
    timeline = Timeline()
    for i, ev in enumerate(run):
        lines = timeline.lines(ev)  # older events still set × and thread
        if i >= shown_from:
            for line in lines:
                out(line)

    while True:
        chunk = f.read()
        if not chunk:
            try:
                rotated = args.log.stat().st_size < f.tell()
            except FileNotFoundError:
                rotated = False
            if rotated:  # truncated or replaced: read the new file from the top
                f.close()
                f = args.log.open(encoding="utf-8", errors="replace")
                partial = ""
            time.sleep(FOLLOW_POLL_SECONDS)
            continue
        *complete, partial = (partial + chunk).split("\n")
        for line in complete:
            ev = parse_event(line)
            if ev is not None and _in_timeline(ev, args.enqueue):
                for text in timeline.lines(ev):
                    out(text)


def main() -> int:
    default_log = (
        Path(os.environ.get("CCBOT_DIR", Path.home() / ".ccbot")) / "ccbot.log"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=default_log, help="log file")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--since", type=_parse_since, help="e.g. 30m: window before the last entry"
    )
    scope.add_argument(
        "--all", action="store_true", help="whole log, not just the latest run"
    )
    parser.add_argument(
        "--enqueue", action="store_true", help="also show content queued by Claude"
    )
    parser.add_argument("--no-timeline", action="store_true", help="summary only")
    parser.add_argument(
        "-f", "--follow", action="store_true", help="live timeline, like tail -f"
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=20,
        help="with -f: past requests shown first (default 20)",
    )
    parser.add_argument(
        "--cut",
        type=int,
        default=None,
        help="cut lines to this many columns (default: terminal width, 0: never)",
    )
    args = parser.parse_args()

    f = args.log.open(encoding="utf-8", errors="replace")
    complete, _, partial = f.read().rpartition("\n")
    events = [ev for ev in map(parse_event, complete.split("\n")) if ev]
    if not args.follow:
        f.close()
        cols = args.cut if args.cut is not None else terminal_columns()
        return report(Output(cols), events, args)
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    follow(Output(args.cut), f, partial, events, args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BrokenPipeError, KeyboardInterrupt):  # piped into head, or Ctrl-C
        sys.exit(0)
