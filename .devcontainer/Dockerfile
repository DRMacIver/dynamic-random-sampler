FROM ubuntu:24.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Create dev user with sudo access
RUN useradd -m -s /bin/bash dev \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && rm -rf /var/lib/apt/lists/*

# Add GitHub CLI repository
RUN apt-get update \
    && apt-get install -y curl gpg \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && rm -rf /var/lib/apt/lists/*

# Install base dependencies including build tools for Rust compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    gh \
    jq \
    rsync \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install Node.js LTS (needed for beads and Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @beads/bd @anthropic-ai/claude-code

# Install just (command runner) from prebuilt binaries
RUN ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "arm64" ]; then ARCH="aarch64"; elif [ "$ARCH" = "amd64" ]; then ARCH="x86_64"; fi \
    && curl -fsSL "https://github.com/casey/just/releases/download/1.38.0/just-1.38.0-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar -xz -C /usr/local/bin just

# Install uv (Python package manager) and pre-commit
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv tool install pre-commit

# Install uv and Rust for dev user
USER dev
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/dev/.local/bin:/home/dev/.cargo/bin:$PATH"
USER root
