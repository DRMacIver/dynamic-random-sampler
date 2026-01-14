#!/bin/bash
# Container entrypoint - runs setup if needed, then executes command
set -e

SETUP_MARKER="$HOME/.container-setup-done"

# Set up environment
export PATH="/home/vscode/.local/bin:$PATH"

# Set Claude config directory
export CLAUDE_CONFIG_DIR="$HOME/.claude"

# Copy Claude credentials from host if not present in container
# Claude Code needs:
# 1. OAuth tokens in ~/.claude/.credentials.json
# 2. Config with oauthAccount in ~/.claude/.claude.json (identifies logged-in user)
CLAUDE_DIR_HOME="$HOME/.claude"
CLAUDE_KEYCHAIN_HOST="/mnt/credentials/claude-keychain.json"
CLAUDE_CONFIG_HOST="/mnt/credentials/claude-config.json"
CLAUDE_COPY_MARKER="$HOME/.claude-credentials-copied"

# Check if we need to copy credentials
# We re-copy if: marker missing, credentials missing, OR config missing
NEED_CREDS_COPY="no"
if [ ! -f "$CLAUDE_COPY_MARKER" ]; then
    NEED_CREDS_COPY="yes"
elif [ ! -f "$CLAUDE_DIR_HOME/.credentials.json" ]; then
    NEED_CREDS_COPY="yes"
    rm -f "$CLAUDE_COPY_MARKER"
elif [ ! -f "$CLAUDE_DIR_HOME/.claude.json" ]; then
    NEED_CREDS_COPY="yes"
    rm -f "$CLAUDE_COPY_MARKER"
fi

if [ "$NEED_CREDS_COPY" = "yes" ]; then
    echo "Setting up Claude credentials..."
    mkdir -p "$CLAUDE_DIR_HOME"
    CREDS_OK="yes"

    # Copy OAuth tokens
    if [ -f "$CLAUDE_KEYCHAIN_HOST" ] && [ -s "$CLAUDE_KEYCHAIN_HOST" ]; then
        cp "$CLAUDE_KEYCHAIN_HOST" "$CLAUDE_DIR_HOME/.credentials.json"
        chmod 600 "$CLAUDE_DIR_HOME/.credentials.json"
        echo "  OAuth tokens: copied ($(wc -c < "$CLAUDE_DIR_HOME/.credentials.json") bytes)"
    else
        echo "  WARNING: OAuth tokens not found at $CLAUDE_KEYCHAIN_HOST"
        CREDS_OK="no"
    fi

    # Copy config file (contains oauthAccount)
    if [ -f "$CLAUDE_CONFIG_HOST" ] && [ -s "$CLAUDE_CONFIG_HOST" ]; then
        cp "$CLAUDE_CONFIG_HOST" "$CLAUDE_DIR_HOME/.claude.json"
        chmod 600 "$CLAUDE_DIR_HOME/.claude.json"
        echo "  Config file: copied ($(wc -c < "$CLAUDE_DIR_HOME/.claude.json") bytes)"
    else
        echo "  WARNING: Config file not found at $CLAUDE_CONFIG_HOST"
        CREDS_OK="no"
    fi

    if [ "$CREDS_OK" = "yes" ]; then
        touch "$CLAUDE_COPY_MARKER"
        echo "  Claude credentials setup complete"
    else
        echo "  WARNING: Incomplete credentials - Claude may ask to log in"
        echo "  Run 'claude' on macOS host and log in first"
    fi
fi

# Set up GitHub token from scoped credentials (if available)
# This uses the GitHub App installation token, NOT user credentials
# IMPORTANT: We use a credential helper script that reads fresh from the file
# each time, so token refresh on the host is picked up automatically.
GH_TOKEN_FILE="/mnt/credentials/github_token.json"
if [ -f "$GH_TOKEN_FILE" ] && [ -s "$GH_TOKEN_FILE" ]; then
    # Create a credential helper script that reads token fresh each time
    mkdir -p ~/.local/bin
    cat > ~/.local/bin/git-credential-github-token << 'CREDENTIAL_HELPER'
#!/bin/bash
# Git credential helper that reads from the mounted token file
# This ensures we always use the latest token after host-side refresh
TOKEN_FILE="/mnt/credentials/github_token.json"
if [ "$1" = "get" ]; then
    if [ -f "$TOKEN_FILE" ] && [ -s "$TOKEN_FILE" ]; then
        TOKEN=$(jq -r '.token' "$TOKEN_FILE" 2>/dev/null)
        if [ -n "$TOKEN" ] && [ "$TOKEN" != "null" ]; then
            echo "protocol=https"
            echo "host=github.com"
            echo "username=x-access-token"
            echo "password=$TOKEN"
        fi
    fi
fi
CREDENTIAL_HELPER
    chmod +x ~/.local/bin/git-credential-github-token

    # Configure git to use our credential helper
    git config --global credential.helper ""
    git config --global credential.https://github.com.helper "~/.local/bin/git-credential-github-token"

    # For gh CLI, create a wrapper script that shadows /usr/bin/gh
    # Named 'gh' so it's found first in PATH (~/.local/bin is already first)
    # This works in both interactive and non-interactive shells (unlike aliases)
    cat > ~/.local/bin/gh << 'GH_WRAPPER'
#!/bin/bash
# Wrapper that sets GH_TOKEN fresh from file before running gh
TOKEN_FILE="/mnt/credentials/github_token.json"
if [ -f "$TOKEN_FILE" ] && [ -s "$TOKEN_FILE" ]; then
    if ! TOKEN=$(jq -r '.token' "$TOKEN_FILE" 2>&1); then
        echo "WARNING: Failed to parse $TOKEN_FILE: $TOKEN" >&2
        TOKEN=""
    fi
    if [ -n "$TOKEN" ] && [ "$TOKEN" != "null" ]; then
        export GH_TOKEN="$TOKEN"
    else
        echo "WARNING: No valid token found in $TOKEN_FILE" >&2
    fi
elif [ ! -f "$TOKEN_FILE" ]; then
    echo "WARNING: Token file not found: $TOKEN_FILE" >&2
    echo "Run 'just develop' from the host to generate a GitHub token" >&2
fi
exec /usr/bin/gh "$@"
GH_WRAPPER
    chmod +x ~/.local/bin/gh

    echo "GitHub: scoped token configured (reads fresh from file)"
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
