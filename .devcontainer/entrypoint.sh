#!/bin/bash
# Container entrypoint - runs setup if needed, then executes command
set -e

SETUP_MARKER="$HOME/.container-setup-done"

# Set up environment
export PATH="/home/vscode/.local/bin:$PATH"

# Set Claude config directory
export CLAUDE_CONFIG_DIR="$HOME/.claude"

# Copy Claude credentials from host if not present
CLAUDE_DIR_HOME="$HOME/.claude"
CLAUDE_KEYCHAIN_HOST="/mnt/credentials/claude-keychain.json"
CLAUDE_COPY_MARKER="$HOME/.claude-credentials-copied"

if [ ! -f "$CLAUDE_COPY_MARKER" ]; then
    echo "Setting up Claude credentials..."

    # Copy OAuth tokens from macOS Keychain extract (if available)
    if [ -f "$CLAUDE_KEYCHAIN_HOST" ] && [ -s "$CLAUDE_KEYCHAIN_HOST" ]; then
        mkdir -p "$CLAUDE_DIR_HOME"
        cp "$CLAUDE_KEYCHAIN_HOST" "$CLAUDE_DIR_HOME/.credentials.json"
        chmod 600 "$CLAUDE_DIR_HOME/.credentials.json"
        echo "  Copied OAuth credentials from Keychain"
    fi

    touch "$CLAUDE_COPY_MARKER"
fi

# Run setup if not done yet (or if post-create.sh changed)
POST_CREATE="/workspaces/dynamic-random-sampler/.devcontainer/post-create.sh"
POST_CREATE_HASH=$(sha256sum "$POST_CREATE" 2>/dev/null | cut -d' ' -f1 || echo "none")
CACHED_HASH=""
if [ -f "$SETUP_MARKER" ]; then
    CACHED_HASH=$(cat "$SETUP_MARKER")
fi

if [ "$POST_CREATE_HASH" != "$CACHED_HASH" ]; then
    echo "Running container setup..."
    cd /workspaces/dynamic-random-sampler
    bash "$POST_CREATE"

    # Mark setup as done
    echo "$POST_CREATE_HASH" > "$SETUP_MARKER"
    echo "Container setup complete!"
fi

# Execute the provided command
exec "$@"
