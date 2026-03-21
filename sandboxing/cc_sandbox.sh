#!/usr/bin/env bash
# cc-sandbox: Run Claude Code in an isolated Docker container with GPU access.
#
# Usage:
#   ./cc_sandbox.sh [OPTIONS]
#
# Options:
#   --cuda-visible-devices DEVS   GPU indices (e.g. "0" or "0,1"). Default: none.
#   --prompt TEXT                  Task instructions as a string.
#   --prompt-file PATH            Task instructions from a file.
#   --model MODEL                 Claude model. Default: opus.
#   --max-budget USD              Max spend in USD. Default: 5.
#   --branch NAME                 Git branch name. Default: sandbox/<timestamp>.
#   --no-push                     Skip auto-push on each commit.
#   --name NAME                   Session name for identification.
#   --screen                      Launch in a detached screen session.
#   --help                        Show this help message.
#
# If neither --prompt nor --prompt-file is given, reads from stdin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Defaults ---
CUDA_DEVICES=""
PROMPT=""
PROMPT_FILE=""
MODEL="opus"
MAX_BUDGET="5"
BRANCH=""
AUTO_PUSH="1"
SESSION_NAME=""
USE_SCREEN="0"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda-visible-devices) CUDA_DEVICES="$2"; shift 2 ;;
        --prompt)               PROMPT="$2"; shift 2 ;;
        --prompt-file)          PROMPT_FILE="$2"; shift 2 ;;
        --model)                MODEL="$2"; shift 2 ;;
        --max-budget)           MAX_BUDGET="$2"; shift 2 ;;
        --branch)               BRANCH="$2"; shift 2 ;;
        --no-push)              AUTO_PUSH="0"; shift ;;
        --name)                 SESSION_NAME="$2"; shift 2 ;;
        --screen)               USE_SCREEN="1"; shift ;;
        --help)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Resolve prompt ---
if [[ -n "$PROMPT_FILE" ]]; then
    PROMPT="$(cat "$PROMPT_FILE")"
elif [[ -z "$PROMPT" ]]; then
    echo "Enter task instructions (Ctrl+D when done):"
    PROMPT="$(cat)"
fi

if [[ -z "$PROMPT" ]]; then
    echo "Error: No prompt provided."
    exit 1
fi

# --- Resolve branch and session name ---
if [[ -z "$BRANCH" ]]; then
    BRANCH="sandbox/$(date +%Y%m%d-%H%M%S)-$(head -c4 /dev/urandom | xxd -p)"
fi

if [[ -z "$SESSION_NAME" ]]; then
    # Derive from branch: sandbox/20260321-123456-abc -> 20260321-123456-abc
    SESSION_NAME="$(basename "$BRANCH")"
fi

# --- Clone repo ---
CLONE_DIR="/tmp/cc-sandbox-clones/$BRANCH"
REMOTE_URL="$(git -C "$REPO_ROOT" remote get-url origin)"
echo "[cc-sandbox] Cloning $REMOTE_URL into $CLONE_DIR ..."
mkdir -p "$(dirname "$CLONE_DIR")"
git clone "$REMOTE_URL" "$CLONE_DIR"
cd "$CLONE_DIR"
git checkout -b "$BRANCH"
echo "[cc-sandbox] On branch: $BRANCH"

# --- Load .env ---
ENV_FILE="$REPO_ROOT/.env"
ENV_ARGS=()
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        ENV_ARGS+=(--env "$key=$value")
    done < "$ENV_FILE"
fi

# --- Build docker run command ---
DOCKER_ARGS=(
    docker run
    --rm
    -it
    --runtime=nvidia
    --name "cc-sandbox-$SESSION_NAME"
)

# GPU passthrough
if [[ -n "$CUDA_DEVICES" ]]; then
    DOCKER_ARGS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES")
    DOCKER_ARGS+=(--env "NVIDIA_VISIBLE_DEVICES=$CUDA_DEVICES")
else
    DOCKER_ARGS+=(--env "NVIDIA_VISIBLE_DEVICES=none")
fi

# Mounts
DOCKER_ARGS+=(
    -v "$CLONE_DIR:/workspace"
    -v "$HOME/.claude:/home/sandbox/.claude:ro"
    -v "$HOME/.ssh:/home/sandbox/.ssh:ro"
)

# Environment
DOCKER_ARGS+=(
    "${ENV_ARGS[@]}"
    --env "CLAUDE_MODEL=$MODEL"
    --env "MAX_BUDGET=$MAX_BUDGET"
    --env "AUTO_PUSH=$AUTO_PUSH"
    --env "GIT_USER_NAME=$(git config user.name 2>/dev/null || echo sandbox-agent)"
    --env "GIT_USER_EMAIL=$(git config user.email 2>/dev/null || echo sandbox@agent.local)"
    --env "CLAUDE_PROMPT=$PROMPT"
)

# Image
DOCKER_ARGS+=(cc-sandbox:latest)

echo "[cc-sandbox] Launching container..."
echo "[cc-sandbox] GPU: ${CUDA_DEVICES:-none}"
echo "[cc-sandbox] Model: $MODEL | Budget: \$$MAX_BUDGET"
echo "[cc-sandbox] Branch: $BRANCH"
echo "[cc-sandbox] Clone: $CLONE_DIR"

if [[ "$USE_SCREEN" == "1" ]]; then
    SCREEN_NAME="cc-$SESSION_NAME"
    echo "[cc-sandbox] Screen session: $SCREEN_NAME"
    echo "[cc-sandbox] Attach with: screen -r $SCREEN_NAME"
    echo ""
    # Launch in detached screen. The docker command runs inside screen.
    # We write the docker command to a temp script so screen can execute it.
    LAUNCH_SCRIPT="/tmp/cc-sandbox-launch-$$.sh"
    printf '%q ' "${DOCKER_ARGS[@]}" > "$LAUNCH_SCRIPT"
    chmod +x "$LAUNCH_SCRIPT"
    screen -dmS "$SCREEN_NAME" bash -c "$(cat "$LAUNCH_SCRIPT"); echo '[cc-sandbox] Container exited. Press enter to close.'; read"
    rm -f "$LAUNCH_SCRIPT"
    echo "[cc-sandbox] Running in background. Use: screen -r $SCREEN_NAME"
else
    echo ""
    exec "${DOCKER_ARGS[@]}"
fi
