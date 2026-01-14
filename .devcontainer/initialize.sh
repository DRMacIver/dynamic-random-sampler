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

# Set up SSH deploy key for git push (using host's gh CLI with full permissions)
if ! command -v gh &> /dev/null; then
    echo "SSH: gh CLI not found, skipping deploy key setup"
elif ! gh auth status &> /dev/null; then
    echo "SSH: gh not authenticated, skipping deploy key setup"
else
    SSH_KEY="$SSH_DIR/devcontainer"

    # Generate key if not exists
    if [ ! -f "$SSH_KEY" ]; then
        ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "devcontainer-$(basename "$PROJECT_DIR")" > /dev/null
        echo "SSH: generated new key pair"
    else
        echo "SSH: using existing key pair"
    fi

    # Get repo info from git remote
    REMOTE_URL=$(git -C "$PROJECT_DIR" remote get-url origin 2>/dev/null || echo "")
    if [[ "$REMOTE_URL" =~ github.com[:/]([^/]+)/([^/]+?)(\.git)?$ ]]; then
        OWNER="${BASH_REMATCH[1]}"
        REPO="${BASH_REMATCH[2]}"
        echo "SSH: setting up deploy key for $OWNER/$REPO"

        KEY_TITLE="Devcontainer (auto-generated)"
        PUB_KEY=$(cat "$SSH_KEY.pub")

        # Check existing deploy keys
        EXISTING=$(gh api "repos/$OWNER/$REPO/keys" 2>/dev/null || echo "[]")

        # Find our key by content (first two parts of public key)
        OUR_KEY_CONTENT=$(echo "$PUB_KEY" | awk '{print $1" "$2}')
        FOUND_KEY=$(echo "$EXISTING" | jq -r --arg key "$OUR_KEY_CONTENT" '.[] | select(.key == $key)')

        if [ -n "$FOUND_KEY" ]; then
            KEY_ID=$(echo "$FOUND_KEY" | jq -r '.id')
            IS_READ_ONLY=$(echo "$FOUND_KEY" | jq -r '.read_only')

            if [ "$IS_READ_ONLY" = "true" ]; then
                # Delete read-only key and re-add with write access
                gh api -X DELETE "repos/$OWNER/$REPO/keys/$KEY_ID" > /dev/null
                gh api "repos/$OWNER/$REPO/keys" -f title="$KEY_TITLE" -f key="$PUB_KEY" -F read_only=false > /dev/null
                echo "SSH: replaced deploy key (now with write access)"
            else
                echo "SSH: deploy key already has write access"
            fi
        else
            # Add new key with write access
            if gh api "repos/$OWNER/$REPO/keys" -f title="$KEY_TITLE" -f key="$PUB_KEY" -F read_only=false > /dev/null 2>&1; then
                echo "SSH: added deploy key with write access"
            else
                echo "SSH: could not add deploy key (may already exist)"
            fi
        fi
    else
        echo "SSH: could not parse remote URL: $REMOTE_URL"
    fi
fi

echo "Host initialization complete."
