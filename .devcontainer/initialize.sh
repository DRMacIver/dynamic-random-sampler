#!/bin/bash
# Runs on the HOST before the container starts
# Generates credentials that get mounted into the container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CREDS_DIR="$PROJECT_DIR/.devcontainer/.credentials"
SSH_DIR="$PROJECT_DIR/.devcontainer/.ssh"

# Create directories
mkdir -p "$CREDS_DIR"
mkdir -p "$SSH_DIR"

# Ensure scripts are executable
chmod +x "$SCRIPT_DIR/entrypoint.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/post-create.sh" 2>/dev/null || true

echo "Host initialization complete."
