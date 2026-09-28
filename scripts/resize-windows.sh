#!/usr/bin/env bash
# Resize every window in a tmux session to a fixed size.
# Usage: resize-windows.sh [session] [width] [height]   (default: ccbot 200 60)
set -euo pipefail

SESSION="${1:-ccbot}"
WIDTH="${2:-200}"
HEIGHT="${3:-60}"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Error: tmux session '$SESSION' does not exist"
    exit 1
fi

tmux list-windows -t "$SESSION" -F '#{window_id} #{window_index} #{window_name}' |
    while read -r wid idx name; do
        tmux resize-window -t "$wid" -x "$WIDTH" -y "$HEIGHT"
        echo "resized ${SESSION}:${idx} (${wid}) ${name} -> ${WIDTH}x${HEIGHT}"
    done
