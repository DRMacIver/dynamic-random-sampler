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

# Update Claude Code and beads to latest
sudo npm install -g @anthropic-ai/claude-code @beads/bd || true

# Configure beads
if [ ! -d .beads ]; then
    # New project - initialize beads
    bd init || true
else
    # Cloned project - ensure beads is properly configured
    # Fix repo fingerprint if needed (common after cloning)
    if bd doctor 2>&1 | grep -q "Repo Fingerprint.*different repository"; then
        echo "y" | bd migrate --update-repo-id || true
    fi
fi

# Always ensure git hooks are installed
bd hooks install 2>/dev/null || true

# Install pre-commit hooks (includes beads interaction check)
if [ -f .pre-commit-config.yaml ] && [ -f .githooks/check-beads-interaction.sh ]; then
    chmod +x .githooks/check-beads-interaction.sh
    pre-commit install 2>/dev/null || true
fi

# Install Rust toolchain if not present (needed for PyO3/maturin projects)
# This runs after volume mount, so installs into the persistent home directory
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi

echo "Development environment ready!"

# Install Python project dependencies
# Note: .venv is a volume mount, isolated from host
if [ -f pyproject.toml ]; then
    uv sync --group dev
fi
