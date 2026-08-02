#!/usr/bin/env bash
set -euo pipefail

# One-click installer for the Claude Code -> DingTalk notification hook.
#
# Usage:
#   bash install_claude_dingtalk_hook.sh            # install
#   bash install_claude_dingtalk_hook.sh --test     # install + send a test message
#
# Optional overrides (env):
#   CLAUDE_HOME=/path/to/.claude
#   DINGTALK_WEBHOOK_URL=https://oapi.dingtalk.com/robot/send?access_token=...
#   DINGTALK_SECRET=SECxxxxxxxx

CLAUDE_HOME="${CLAUDE_HOME:-$HOME/.claude}"
HOOK_DIR="$CLAUDE_HOME/hooks"
HOOK_PY="$HOOK_DIR/claude_dingtalk_notify.py"
SETTINGS_JSON="$CLAUDE_HOME/settings.json"
WEBHOOK_FILE="$HOOK_DIR/dingtalk_webhook_url"
SECRET_FILE="$HOOK_DIR/dingtalk_secret"

DEFAULT_WEBHOOK_URL="https://oapi.dingtalk.com/robot/send?access_token=deaac2701205882bea1621d4e8a0de8e2c188dd4dd6637279165267ab26fb03d"
DEFAULT_SECRET="SEC85dc00f27dd41610bf74e8282206faf031484cf013e57867a62774f3d7b87ea9"

WEBHOOK_URL="${DINGTALK_WEBHOOK_URL:-$DEFAULT_WEBHOOK_URL}"
SECRET="${DINGTALK_SECRET:-$DEFAULT_SECRET}"

PYTHON_BIN="$(command -v python3 || echo /usr/bin/python3)"

DO_TEST=0
if [[ "${1:-}" == "--test" ]]; then
  DO_TEST=1
fi

mkdir -p "$HOOK_DIR"

echo "==> Writing hook script: $HOOK_PY"
cat > "$HOOK_PY" <<'PYEOF'
#!/usr/bin/env python3
"""Claude Code -> DingTalk robot notification hook.

Reads the Claude hook JSON event on stdin and posts a text message to a
DingTalk robot webhook.
"""
import base64
import datetime as _dt
import hashlib
import hmac
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo


HOME = Path.home()
HOOK_DIR = Path(__file__).resolve().parent
WEBHOOK_FILE = HOOK_DIR / "dingtalk_webhook_url"
SECRET_FILE = HOOK_DIR / "dingtalk_secret"
LOG_FILE = HOOK_DIR / "dingtalk_claude_notify.log"
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
MAX_MESSAGE_BYTES = 1900


def log(message: str) -> None:
    try:
        stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {message}\n")
    except Exception:
        pass


def read_event() -> dict:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        log(f"failed to parse hook stdin: {exc}")
        return {}


def read_value(env_name: str, file_path: Path) -> str:
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def signed_url(webhook_url: str, secret: str) -> str:
    if not secret:
        return webhook_url
    timestamp = str(int(time.time() * 1000))
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), string_to_sign, hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(digest).decode("utf-8"))
    separator = "&" if "?" in webhook_url else "?"
    return f"{webhook_url}{separator}timestamp={timestamp}&sign={sign}"


def run_text(command: list, cwd: str) -> str:
    import subprocess

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def git_summary(cwd: str) -> list:
    root = run_text(["git", "rev-parse", "--show-toplevel"], cwd)
    if not root:
        return ["repo: not a git repo"]
    branch = run_text(["git", "branch", "--show-current"], cwd) or "detached"
    commit = run_text(["git", "rev-parse", "--short", "HEAD"], cwd) or "unknown"
    dirty = "dirty" if run_text(["git", "status", "--porcelain"], cwd) else "clean"
    return [f"repo: {root}", f"git: {branch} @ {commit} ({dirty})"]


def trim(value: str, limit: int) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def limit_bytes(value: str, max_bytes: int) -> str:
    raw = value.encode("utf-8")
    if len(raw) <= max_bytes:
        return value
    suffix = "\n... truncated"
    budget = max_bytes - len(suffix.encode("utf-8"))
    return raw[: max(0, budget)].decode("utf-8", errors="ignore").rstrip() + suffix


def transcript_summary(transcript_path) -> dict:
    if not transcript_path:
        return {}
    path = Path(transcript_path)
    if not path.is_file():
        return {}

    summary: dict = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        log(f"transcript read failed: {exc}")
        return {}

    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        item_type = item.get("type")
        message = item.get("message") or {}
        if item_type == "user":
            content = message.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                )
            text = text.strip()
            # Skip tool_result-only / hook-injected entries.
            if text and not text.startswith("<"):
                summary["latest_user"] = text
        elif item_type == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                text = "\n".join(
                    c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                ).strip()
                if text:
                    summary["last_assistant"] = text
            if message.get("model"):
                summary["model"] = message["model"]
    return summary


def build_message(event: dict) -> str:
    cwd = event.get("cwd") or os.getcwd()
    hook_event = event.get("hook_event_name") or "unknown"
    permission_mode = event.get("permission_mode") or "unknown"
    session_id = str(event.get("session_id") or "unknown")
    transcript_path = event.get("transcript_path")

    ts = transcript_summary(transcript_path)
    model = event.get("model") or ts.get("model") or "unknown"

    latest_user = ts.get("latest_user")
    last = event.get("last_assistant_message") or ts.get("last_assistant")
    notif = event.get("message")  # Notification events carry a message
    headline = trim(
        str(latest_user or notif or last or f"{hook_event} finished").replace("\r\n", "\n"),
        120,
    )
    finished_at = _dt.datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S CST")

    lines = [
        f"claude:{headline}",
        f"time: {finished_at} (Asia/Shanghai)",
        f"event: {hook_event}",
        f"model: {model}",
        f"permission: {permission_mode}",
        *([f"notice: {trim(str(notif), 120)}"] if notif else []),
        f"host: {socket.gethostname()}",
        f"cwd: {cwd}",
        *git_summary(cwd),
        f"session: {session_id[:18]}",
    ]

    if transcript_path:
        lines.append(f"transcript: {transcript_path}")

    if last:
        lines.extend(["", "final:", trim(str(last).replace("\r\n", "\n"), 650)])

    return limit_bytes("\n".join(lines), MAX_MESSAGE_BYTES)


def send_text(content: str) -> None:
    webhook_url = read_value("DINGTALK_WEBHOOK_URL", WEBHOOK_FILE)
    if not webhook_url:
        log("missing DingTalk webhook URL")
        return
    secret = read_value("DINGTALK_SECRET", SECRET_FILE)
    body = json.dumps(
        {"msgtype": "text", "text": {"content": content}},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        signed_url(webhook_url, secret),
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        response_body = resp.read().decode("utf-8", errors="replace")
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {response_body}")
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            parsed = {}
        if parsed.get("errcode", 0) != 0:
            raise RuntimeError(f"dingtalk response: {response_body}")
IGNORED_NOTIFICATION_TYPES = {"idle_prompt"}
def should_skip(event: dict) -> bool:
    if (event.get("hook_event_name") or "") != "Notification":
        return False
    notif_type = event.get("notification_type") or ""
    log(f"notification_type={notif_type!r} message={str(event.get('message'))[:80]!r}")
    return notif_type in IGNORED_NOTIFICATION_TYPES

def main() -> int:
    event = read_event()
    if should_skip(event):
        return 0
    try:
        send_text(build_message(event))
    except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
        log(f"notification failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF
chmod +x "$HOOK_PY"

echo "==> Writing credential files"
printf '%s' "$WEBHOOK_URL" > "$WEBHOOK_FILE"
printf '%s' "$SECRET" > "$SECRET_FILE"
chmod 600 "$WEBHOOK_FILE" "$SECRET_FILE"

echo "==> Merging hooks into $SETTINGS_JSON"
CLAUDE_SETTINGS="$SETTINGS_JSON" HOOK_CMD="$PYTHON_BIN $HOOK_PY" "$PYTHON_BIN" - <<'MERGEEOF'
import json
import os
from pathlib import Path

settings_path = Path(os.environ["CLAUDE_SETTINGS"])
hook_cmd = os.environ["HOOK_CMD"]

data = {}
if settings_path.is_file():
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: {settings_path} is not valid JSON; refusing to overwrite.")

hooks = data.setdefault("hooks", {})

hook_entry = {
    "type": "command",
    "command": hook_cmd,
    "timeout": 15,
    "statusMessage": "Sending DingTalk notification",
}


def already_present(groups):
    for group in groups:
        for h in group.get("hooks", []):
            if h.get("command") == hook_cmd:
                return True
    return False


for event in ("Notification", "Stop"):
    groups = hooks.setdefault(event, [])
    if already_present(groups):
        print(f"    {event}: already installed, skipping")
        continue
    if groups and isinstance(groups[0], dict):
        groups[0].setdefault("hooks", []).append(hook_entry)
    else:
        groups.append({"hooks": [hook_entry]})
    print(f"    {event}: added")

settings_path.parent.mkdir(parents=True, exist_ok=True)
settings_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"    wrote {settings_path}")
MERGEEOF

if [[ "$DO_TEST" == "1" ]]; then
  echo "==> Sending test notification"
  printf '%s' '{"hook_event_name":"Notification","message":"install test from '"$(hostname)"'","cwd":"'"$PWD"'"}' | "$PYTHON_BIN" "$HOOK_PY"
  echo "    (check your DingTalk group; see $HOOK_DIR/dingtalk_claude_notify.log on failure)"
fi

echo "==> Done. Hook installed for Notification + Stop events."
