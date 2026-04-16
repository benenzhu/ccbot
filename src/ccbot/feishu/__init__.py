"""Feishu (Lark) backend for CCBot.

Provides a Feishu bot that bridges Feishu group chats/topics to Claude Code
sessions via tmux, mirroring the Telegram backend's functionality.

Uses the lark-oapi SDK with WebSocket long-connection (no public IP needed).

Modules:
  - bot: Main Feishu bot lifecycle — event dispatch, message routing
  - sender: Message sending helpers with markdown formatting
  - state: Chat/thread state management for Feishu conversations
"""
