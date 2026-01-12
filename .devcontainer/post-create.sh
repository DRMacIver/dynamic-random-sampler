#!/bin/bash
set -e

# Copy Claude config into container
if [ -d /mnt/host-claude ]; then
    mkdir -p ~/.claude
    rsync -av /mnt/host-claude/ ~/.claude/
fi

# Copy SSH keys with correct permissions
if [ -d /mnt/host-ssh ]; then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    rsync -av /mnt/host-ssh/ ~/.ssh/
    chmod 600 ~/.ssh/id_* 2>/dev/null || true
    chmod 644 ~/.ssh/*.pub 2>/dev/null || true
fi

# Update Claude Code and beads to latest (installed in Dockerfile, this ensures latest version)
sudo npm install -g @anthropic-ai/claude-code @beads/bd || true

# Set up uv to use a venv in $HOME to avoid conflicts with host .venv
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/$(basename "$PWD")"
echo 'export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/$(basename "$PWD")"' >> ~/.bashrc

# Ensure cargo is in PATH for this script
export PATH="$HOME/.cargo/bin:$PATH"

# Install profiling tools (samply for CPU profiling)
echo "Installing samply profiler..."
cargo install --locked samply 2>/dev/null || true

# Install Python project dependencies
if [ -f pyproject.toml ]; then
    uv sync
fi

# Build Rust extension if Cargo.toml exists
if [ -f Cargo.toml ]; then
    echo "Building Rust extension with maturin..."
    maturin develop || true
fi

# Initialize beads if not already done
if [ ! -d .beads ]; then
    bd init || true
fi

echo "Development environment ready!"
