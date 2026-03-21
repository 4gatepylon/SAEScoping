#!/usr/bin/env bash
set -euo pipefail

# Configure git
git config --global user.name "${GIT_USER_NAME:-sandbox-agent}"
git config --global user.email "${GIT_USER_EMAIL:-sandbox@agent.local}"
git config --global --add safe.directory /workspace

# Install project Python deps if pyproject.toml exists
if [ -f /workspace/pyproject.toml ]; then
    echo "[entrypoint] Installing Python project dependencies..."
    pip install --no-cache-dir -e /workspace 2>&1 | tail -5
    echo "[entrypoint] Python deps installed."
fi

# Set up git post-commit hook for auto-push
if [ "${AUTO_PUSH:-1}" = "1" ] && [ -d /workspace/.git ]; then
    mkdir -p /workspace/.git/hooks
    cat > /workspace/.git/hooks/post-commit << 'HOOK'
#!/usr/bin/env bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "[auto-push] Pushing commit to origin/$BRANCH..."
git push origin "$BRANCH" 2>&1 || echo "[auto-push] Push failed (will retry on next commit)"
HOOK
    chmod +x /workspace/.git/hooks/post-commit
    echo "[entrypoint] Auto-push hook installed."
fi

# Build Claude command
CLAUDE_CMD="claude --dangerously-skip-permissions"

if [ -n "${CLAUDE_MODEL:-}" ]; then
    CLAUDE_CMD="$CLAUDE_CMD --model $CLAUDE_MODEL"
fi

if [ -n "${MAX_BUDGET:-}" ]; then
    CLAUDE_CMD="$CLAUDE_CMD --max-budget-usd $MAX_BUDGET"
fi

if [ -n "${CLAUDE_SYSTEM_PROMPT:-}" ]; then
    CLAUDE_CMD="$CLAUDE_CMD --append-system-prompt \"$CLAUDE_SYSTEM_PROMPT\""
fi

# If a prompt is provided, pass it as the initial message
# Claude runs interactively so the user can attach and intervene
if [ -n "${CLAUDE_PROMPT:-}" ]; then
    echo "[entrypoint] Starting Claude Code (interactive) with initial prompt..."
    echo "[entrypoint] Auto-responder will handle follow-up questions."
    echo "[entrypoint] Attach to this container to interact directly."
    echo ""

    # Start auto-responder in background
    AUTO_RESPOND_TIMEOUT="${AUTO_RESPOND_TIMEOUT:-30}" \
        /usr/local/bin/auto_responder.sh &
    AUTO_RESPONDER_PID=$!

    # Run Claude interactively with the prompt
    eval "$CLAUDE_CMD" "$CLAUDE_PROMPT"

    # Clean up auto-responder
    kill "$AUTO_RESPONDER_PID" 2>/dev/null || true
else
    echo "[entrypoint] Starting Claude Code (interactive, no initial prompt)..."
    echo "[entrypoint] Attach to this container to interact."
    echo ""
    eval "$CLAUDE_CMD"
fi
