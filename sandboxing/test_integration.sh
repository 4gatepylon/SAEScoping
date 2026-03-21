#!/usr/bin/env bash
# Smoke test for cc-sandbox. Launches a container with a trivial task
# and verifies basic functionality.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="sandbox/test-$(date +%s)"
CLONE_DIR="/tmp/cc-sandbox-clones/$BRANCH"
CONTAINER_NAME="cc-sandbox-test-$$"
PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
cleanup() {
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    rm -rf "$CLONE_DIR"
}
trap cleanup EXIT

echo "cc-sandbox integration test"
echo "==========================="
echo ""

# --- Test 1: Image exists ---
echo "Test 1: Docker image exists"
if docker image inspect cc-sandbox:latest >/dev/null 2>&1; then
    pass "Image cc-sandbox:latest found"
else
    fail "Image not found. Run ./build.sh first."
    echo "Cannot continue without image."
    exit 1
fi

# --- Test 2: Container starts and Claude is available ---
echo "Test 2: Container starts, Claude Code is installed"
# Run a quick command in the container (not the full entrypoint)
CLAUDE_VERSION=$(docker run --rm --entrypoint claude cc-sandbox:latest --version 2>&1 || true)
if [[ "$CLAUDE_VERSION" == *"Claude Code"* ]]; then
    pass "Claude Code available: $CLAUDE_VERSION"
else
    fail "Claude Code not found in container (got: $CLAUDE_VERSION)"
fi

# --- Test 3: GPU passthrough works ---
echo "Test 3: GPU passthrough"
GPU_OUT=$(docker run --rm --runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=0 \
    --entrypoint nvidia-smi cc-sandbox:latest --query-gpu=name --format=csv,noheader 2>&1 || true)
if [[ "$GPU_OUT" == *"H100"* ]] || [[ "$GPU_OUT" == *"GPU"* ]] || [[ "$GPU_OUT" == *"NVIDIA"* ]]; then
    pass "GPU visible: $GPU_OUT"
else
    fail "GPU not visible (got: $GPU_OUT)"
fi

# --- Test 4: Python + torch available ---
echo "Test 4: Python and torch"
TORCH_OUT=$(docker run --rm --entrypoint bash cc-sandbox:latest \
    -c "source /home/sandbox/venv/bin/activate && python3 -c 'import torch; print(torch.__version__)'" 2>&1 || true)
if [[ "$TORCH_OUT" == *"2.7"* ]]; then
    pass "torch $TORCH_OUT"
else
    fail "torch not available (got: $TORCH_OUT)"
fi

# --- Test 5: Git clone works ---
echo "Test 5: Git clone"
REMOTE_URL="$(git -C "$SCRIPT_DIR/.." remote get-url origin)"
if git clone "$REMOTE_URL" "$CLONE_DIR" 2>&1 | tail -1; then
    pass "Clone to $CLONE_DIR"
else
    fail "Clone failed"
fi

# --- Test 6: Branch creation ---
echo "Test 6: Branch creation"
if cd "$CLONE_DIR" && git checkout -b "$BRANCH" 2>&1; then
    pass "Branch $BRANCH created"
else
    fail "Branch creation failed"
fi

# --- Summary ---
echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] && echo "All tests passed." || echo "Some tests failed."
exit $FAIL
