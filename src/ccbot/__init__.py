"""CCBot - Bot for managing Claude Code sessions via tmux.

Supports Telegram and Feishu (Lark) backends. Set CCBOT_PLATFORM env var
to choose the backend (default: telegram).

Package entry point. Exports the version string only; all functional
modules are imported lazily by main.py to keep startup fast.
"""

__version__ = "0.1.0"
