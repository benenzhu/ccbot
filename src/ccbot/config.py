"""Application configuration — reads env vars and exposes a singleton.

Loads platform selection (CCBOT_PLATFORM), credentials, tmux/Claude paths,
and monitoring intervals from environment variables (with .env support).
.env loading priority: local .env (cwd) > $CCBOT_DIR/.env (default ~/.ccbot).
The module-level `config` instance is imported by nearly every other module.

Supported platforms:
  - telegram (default): requires TELEGRAM_BOT_TOKEN, ALLOWED_USERS
  - feishu: requires FEISHU_APP_ID, FEISHU_APP_SECRET, ALLOWED_USERS

Key class: Config (singleton instantiated as `config`).
"""

import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from .utils import ccbot_dir

logger = logging.getLogger(__name__)

# Env vars that must not leak to child processes (e.g. Claude Code via tmux)
SENSITIVE_ENV_VARS = {
    "TELEGRAM_BOT_TOKEN",
    "ALLOWED_USERS",
    "OPENAI_API_KEY",
    "FEISHU_APP_SECRET",
    "FEISHU_ENCRYPT_KEY",
    "FEISHU_VERIFICATION_TOKEN",
}

PlatformType = Literal["telegram", "feishu"]


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.config_dir = ccbot_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load .env: local (cwd) takes priority over config_dir
        # load_dotenv default override=False means first-loaded wins
        local_env = Path(".env")
        global_env = self.config_dir / ".env"
        if local_env.is_file():
            load_dotenv(local_env)
            logger.debug("Loaded env from %s", local_env.resolve())
        if global_env.is_file():
            load_dotenv(global_env)
            logger.debug("Loaded env from %s", global_env)

        # Platform selection: "telegram" (default) or "feishu"
        platform_str = os.getenv("CCBOT_PLATFORM", "telegram").lower()
        if platform_str not in ("telegram", "feishu"):
            raise ValueError(
                f"CCBOT_PLATFORM must be 'telegram' or 'feishu', got '{platform_str}'"
            )
        self.platform: PlatformType = platform_str  # type: ignore[assignment]

        # --- Platform-specific credentials ---
        if self.platform == "telegram":
            self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN") or ""
            if not self.telegram_bot_token:
                raise ValueError(
                    "TELEGRAM_BOT_TOKEN environment variable is required "
                    "when CCBOT_PLATFORM=telegram"
                )
        else:
            self.telegram_bot_token = ""

        if self.platform == "feishu":
            self.feishu_app_id: str = os.getenv("FEISHU_APP_ID") or ""
            self.feishu_app_secret: str = os.getenv("FEISHU_APP_SECRET") or ""
            if not self.feishu_app_id or not self.feishu_app_secret:
                raise ValueError(
                    "FEISHU_APP_ID and FEISHU_APP_SECRET are required "
                    "when CCBOT_PLATFORM=feishu"
                )
            self.feishu_encrypt_key: str = os.getenv("FEISHU_ENCRYPT_KEY") or ""
            self.feishu_verification_token: str = (
                os.getenv("FEISHU_VERIFICATION_TOKEN") or ""
            )
            # "feishu" for feishu.cn, "lark" for larksuite.com
            self.feishu_domain: str = os.getenv("FEISHU_DOMAIN", "feishu")
        else:
            self.feishu_app_id = ""
            self.feishu_app_secret = ""
            self.feishu_encrypt_key = ""
            self.feishu_verification_token = ""
            self.feishu_domain = "feishu"

        # --- Allowed users (shared across platforms) ---
        allowed_users_str = os.getenv("ALLOWED_USERS", "")
        if not allowed_users_str:
            raise ValueError("ALLOWED_USERS environment variable is required")
        self.allowed_users: set[str] = {
            uid.strip() for uid in allowed_users_str.split(",") if uid.strip()
        }

        # Tmux session name and window naming
        self.tmux_session_name = os.getenv("TMUX_SESSION_NAME", "ccbot")
        self.tmux_main_window_name = "__main__"

        # Claude command to run in new windows
        self.claude_command = os.getenv("CLAUDE_COMMAND", "claude")

        # All state files live under config_dir
        self.state_file = self.config_dir / "state.json"
        self.session_map_file = self.config_dir / "session_map.json"
        self.monitor_state_file = self.config_dir / "monitor_state.json"

        # Claude Code session monitoring configuration
        # Support custom projects path for Claude variants (e.g., cc-mirror, zai)
        # Priority: CCBOT_CLAUDE_PROJECTS_PATH > CLAUDE_CONFIG_DIR/projects > default
        custom_projects_path = os.getenv("CCBOT_CLAUDE_PROJECTS_PATH")
        claude_config_dir = os.getenv("CLAUDE_CONFIG_DIR")

        if custom_projects_path:
            self.claude_projects_path = Path(custom_projects_path)
        elif claude_config_dir:
            self.claude_projects_path = Path(claude_config_dir) / "projects"
        else:
            self.claude_projects_path = Path.home() / ".claude" / "projects"

        self.monitor_poll_interval = float(os.getenv("MONITOR_POLL_INTERVAL", "2.0"))

        # Display user messages in history and real-time notifications
        self.show_user_messages = (
            os.getenv("CCBOT_SHOW_USER_MESSAGES", "true").lower() != "false"
        )

        # Show tool call notifications (tool_use/tool_result)
        self.show_tool_calls = (
            os.getenv("CCBOT_SHOW_TOOL_CALLS", "true").lower() != "false"
        )

        # Show hidden (dot) directories in directory browser
        self.show_hidden_dirs = (
            os.getenv("CCBOT_SHOW_HIDDEN_DIRS", "").lower() == "true"
        )

        # OpenAI API for voice message transcription (optional)
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url: str = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )

        # Scrub sensitive vars from os.environ so child processes never inherit them.
        for var in SENSITIVE_ENV_VARS:
            os.environ.pop(var, None)

        logger.debug(
            "Config initialized: platform=%s, dir=%s, allowed_users=%d, "
            "tmux_session=%s, claude_projects_path=%s",
            self.platform,
            self.config_dir,
            len(self.allowed_users),
            self.tmux_session_name,
            self.claude_projects_path,
        )

    def is_user_allowed(self, user_id: int | str) -> bool:
        """Check if a user is in the allowed list.

        Accepts both int (Telegram user_id) and str (Feishu open_id).
        """
        return str(user_id) in self.allowed_users


config = Config()
