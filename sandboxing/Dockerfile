# Based on the official Claude Code devcontainer:
#   https://github.com/anthropics/claude-code/blob/main/.devcontainer/Dockerfile
#
# Changes from upstream:
#   - Base image: nvidia/cuda instead of node:20 (for GPU/CUDA support)
#   - Added: Node.js installed manually (since we don't use the node base image)
#   - Added: Python 3, pip, venv, and pre-installed ML dependencies (torch, etc.)
#   - Added: screen, expect (for auto-responder and session management)
#   - Removed: iptables/ipset/firewall (we allow full network access)
#   - User: "sandbox" instead of "node"
# ---- END CHANGES FROM UPSTREAM ----

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG TZ=America/New_York
ENV TZ="$TZ"
ARG CLAUDE_CODE_VERSION=latest
ENV DEBIAN_FRONTEND=noninteractive

# Core system packages (from upstream + our additions marked with # ADDED)
RUN apt-get update && apt-get install -y --no-install-recommends \
    less \
    git \
    procps \
    sudo \
    fzf \
    zsh \
    man-db \
    unzip \
    gnupg2 \
    gh \
    jq \
    nano \
    vim \
    curl \      # ADDED
    wget \
    openssh-client \  # ADDED
    screen \    # ADDED (session management)
    expect \    # ADDED (auto-responder)
    python3 \   # ADDED (ML workloads)
    python3-pip \     # ADDED
    python3-venv \    # ADDED
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ADDED: Node.js 22 (upstream uses node:20 base image; we need to install manually)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# git-delta for nice diffs (from official devcontainer)
ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Create sandbox user (like official devcontainer's 'node' user)
RUN useradd -m -s /bin/zsh sandbox \
    && mkdir -p /workspace /home/sandbox/.claude /commandhistory \
    && chown -R sandbox:sandbox /workspace /home/sandbox/.claude /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown sandbox /commandhistory/.bash_history

# Allow sandbox user passwordless sudo (needed for some operations)
RUN echo "sandbox ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sandbox \
    && chmod 0440 /etc/sudoers.d/sandbox

# npm global config
RUN mkdir -p /usr/local/share/npm-global \
    && chown -R sandbox:sandbox /usr/local/share/npm-global

USER sandbox

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano
ENV DEVCONTAINER=true

# ZSH setup (from official devcontainer)
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
    -p git \
    -p fzf \
    -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    -x

# Claude Code
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# ADDED: Python venv for project deps
RUN python3 -m venv /home/sandbox/venv
ENV PATH="/home/sandbox/venv/bin:$PATH"

# ADDED: Pre-install heavy Python deps (so container startup is fast)
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    numpy==1.26.4 \
    transformers==4.56.1

# Copy scripts
USER root
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY auto_responder.sh /usr/local/bin/auto_responder.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/auto_responder.sh
USER sandbox

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
