#!/usr/bin/env bash
# Auto-responder for Claude Code interactive sessions.
#
# Monitors the terminal for Claude's input prompt and auto-responds
# after a timeout, UNLESS the user types something first.
#
# This script is meant to run in the background alongside Claude.
# It uses expect to detect when Claude is waiting for input.
#
# Environment:
#   AUTO_RESPOND_TIMEOUT - seconds to wait before auto-responding (default: 30)
#   AUTO_RESPOND_MESSAGE - the auto-response message

TIMEOUT="${AUTO_RESPOND_TIMEOUT:-30}"
MESSAGE="${AUTO_RESPOND_MESSAGE:-I would like you to figure this out and make the best choice based on my previous description. Be conservative and do not break anything, but make sure the most likely options I requested are completed as possible.}"

echo "[auto-responder] Running with ${TIMEOUT}s timeout."
echo "[auto-responder] To override: attach to container and type before timeout expires."

# TODO: This is a placeholder. The actual implementation needs to detect
# Claude's input prompt pattern and send responses via the PTY.
# For now, the --append-system-prompt approach handles most cases by
# telling Claude not to ask questions. This script will be enhanced
# in a later commit once we can test the actual Claude prompt patterns.

# Keep alive so the PID remains valid for cleanup
while true; do
    sleep 60
done
