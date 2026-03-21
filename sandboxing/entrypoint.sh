#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Setting up sandbox environment..."

# Configure git
git config --global user.name "${GIT_USER_NAME:-sandbox-agent}"
git config --global user.email "${GIT_USER_EMAIL:-sandbox@agent.local}"
git config --global --add safe.directory /workspace

# SSH: accept known hosts automatically (GitHub, etc.)
mkdir -p /home/sandbox/.ssh 2>/dev/null || true
ssh-keyscan github.com >> /home/sandbox/.ssh/known_hosts 2>/dev/null || true

# Install project Python deps if pyproject.toml exists
if [ -f /workspace/pyproject.toml ]; then
    echo "[entrypoint] Installing Python project dependencies..."
    pip install --no-cache-dir -e /workspace 2>&1 | tail -5
    echo "[entrypoint] Python deps installed."
fi

# Set up git auto-push: post-commit hook + initial branch push
if [ "${AUTO_PUSH:-1}" = "1" ] && [ -d /workspace/.git ]; then
    BRANCH=$(git -C /workspace rev-parse --abbrev-ref HEAD)

    # Push initial branch so remote tracking is set up
    echo "[entrypoint] Pushing initial branch $BRANCH to origin..."
    git -C /workspace push -u origin "$BRANCH" 2>&1 || echo "[entrypoint] Initial push failed (will retry on commit)"

    # Hook: push after every commit
    mkdir -p /workspace/.git/hooks
    cat > /workspace/.git/hooks/post-commit << 'HOOK'
#!/usr/bin/env bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "[auto-push] Pushing to origin/$BRANCH..."
git push origin "$BRANCH" 2>&1 || echo "[auto-push] Push failed (will retry on next commit)"
HOOK
    chmod +x /workspace/.git/hooks/post-commit
    echo "[entrypoint] Auto-push hook installed."
fi

# Build Claude command as an array (safe quoting)
CLAUDE_ARGS=(claude --dangerously-skip-permissions)

if [ -n "${CLAUDE_MODEL:-}" ]; then
    CLAUDE_ARGS+=(--model "$CLAUDE_MODEL")
fi

if [ -n "${MAX_BUDGET:-}" ]; then
    CLAUDE_ARGS+=(--max-budget-usd "$MAX_BUDGET")
fi

# System prompt: tell Claude to be autonomous and not ask questions
AUTONOMY_PROMPT="You are running autonomously in a sandbox. Do NOT ask the user questions. If you need to make a decision, make the most conservative choice that still accomplishes the task. Do not break existing functionality. Commit your work frequently with descriptive messages."
CLAUDE_ARGS+=(--append-system-prompt "$AUTONOMY_PROMPT")

# Launch Claude
if [ -n "${CLAUDE_PROMPT:-}" ]; then
    echo "[entrypoint] Starting Claude Code with initial prompt..."
    echo "[entrypoint] To interact: docker exec -it <container> bash"
    echo ""
    exec "${CLAUDE_ARGS[@]}" "$CLAUDE_PROMPT"
else
    echo "[entrypoint] Starting Claude Code (no initial prompt)..."
    echo ""
    exec "${CLAUDE_ARGS[@]}"
fi
