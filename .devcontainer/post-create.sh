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

# Install Python project dependencies
# Note: .venv is a volume mount, isolated from host
if [ -f pyproject.toml ]; then
    uv sync
fi

# Initialize beads if not already done
if [ ! -d .beads ]; then
    bd init || true
fi

echo "Development environment ready!"
