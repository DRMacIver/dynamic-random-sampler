FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Add GitHub CLI repository
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null

# Install base dependencies (including build essentials for Rust and Python dev)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gh \
    jq \
    rsync \
    build-essential \
    pkg-config \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Try to install profiling tools (optional - may not be available on all architectures)
# These are kernel-specific and may not work in containers anyway
RUN apt-get update && apt-get install -y linux-perf linux-tools-generic 2>/dev/null \
    || echo "Note: linux-perf not available on this architecture, skipping" \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js LTS and Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code @beads/bd

# Install just (command runner)
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Install Rust toolchain for vscode user
USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/home/vscode/.cargo/bin:$PATH"

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/vscode/.local/bin:$PATH"

# Install maturin for building Rust+Python packages
RUN /home/vscode/.cargo/bin/cargo install maturin

USER root
