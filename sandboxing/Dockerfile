# Use CUDA base if your work needs GPU libs; otherwise use ubuntu:24.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Node.js (required for Claude Code) + basic tools
RUN apt-get update && apt-get install -y \
    curl git ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN set -o pipefail && curl -fsSL https://claude.ai/install.sh | bash

# TODO(Claude) change this to do the following:
# 1. Copy . to a temporary location (either passed by user or will be defined)
# 2. Make sure sandboxing folder is not copied in (so you can rm -rf it in the target location for example)
# 3. Run pip install -e . for this in docker (because)
# 4. Document whether this will start an interactive shell or not (should this be run in a tmux screen?)
# 5. 
# (feel free to make a script for this)
# OLD PLEASE REFACTOR:
# # Create expected mount points with clear ownership
# RUN mkdir -p /input /output
# WORKDIR /workspace

# # Copy in project files at build time (or use a bind mount instead — see below)
# # COPY ./myproject /workspace

# ENTRYPOINT ["claude"]
# CMD ["--dangerously-skip-permissions"]