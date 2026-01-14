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

# Set up GitHub credentials (if available)
# Uses:
# - SSH deploy keys for git push (permanent, per-project)
# - GitHub App token for API calls like gh CLI (expires but refreshed)
GH_TOKEN_FILE="/mnt/credentials/github_token.json"
if [ -f "$GH_TOKEN_FILE" ] && [ -s "$GH_TOKEN_FILE" ]; then
    mkdir -p ~/.local/bin

    # For gh CLI, create a wrapper script that shadows /usr/bin/gh
    # This reads the token fresh each time (handles refresh on host)
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

    # Set up SSH deploy keys for git push
    # These are permanent (don't expire like tokens) and scoped to specific repos
    # Use uv to run with httpx dependency
    uv run --quiet --with httpx python3 << 'SETUP_SSH_KEYS'
import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

TOKEN_FILE = Path("/mnt/credentials/github_token.json")
SSH_PERSISTENT_DIR = Path("/mnt/ssh-keys")
SSH_DIR = Path.home() / ".ssh"
PROJECT_DIR = Path("/workspaces/dynamic-random-sampler")

if not TOKEN_FILE.exists():
    print("No GitHub token file, skipping SSH key setup")
    sys.exit(0)

try:
    creds = json.loads(TOKEN_FILE.read_text())
except json.JSONDecodeError as e:
    print(f"Invalid JSON in token file: {e}")
    sys.exit(0)

token = creds.get("token")
repos = creds.get("repos", [])
if not token or not repos:
    print("No token or repos in credentials")
    sys.exit(0)

# Ensure directories exist
SSH_DIR.mkdir(parents=True, exist_ok=True)
SSH_DIR.chmod(0o700)
SSH_PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)

private_key = SSH_DIR / "devcontainer"
public_key = SSH_DIR / "devcontainer.pub"
persistent_private = SSH_PERSISTENT_DIR / "devcontainer"
persistent_public = SSH_PERSISTENT_DIR / "devcontainer.pub"

# Check for existing keys in persistent storage first
if persistent_private.exists() and persistent_public.exists():
    shutil.copy(persistent_private, private_key)
    shutil.copy(persistent_public, public_key)
    private_key.chmod(0o600)
    public_key.chmod(0o644)
    print("SSH: copied keys from persistent storage")
elif not private_key.exists():
    # Generate new key pair
    result = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(private_key), "-N", "",
         "-C", f"devcontainer-{'-'.join(repos)}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"SSH keygen failed: {result.stderr}")
        sys.exit(0)
    print("SSH: generated new key pair")
    # Save to persistent storage
    shutil.copy(private_key, persistent_private)
    shutil.copy(public_key, persistent_public)
    persistent_private.chmod(0o600)
    persistent_public.chmod(0o644)
    print("SSH: saved to persistent storage")

# Get the owner from git remote
owner = "DRMacIver"
try:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True, check=True,
        cwd=PROJECT_DIR
    )
    remote_url = result.stdout.strip()
    if "github.com" in remote_url:
        if remote_url.startswith("git@"):
            owner = remote_url.split(":")[-1].split("/")[0]
        else:
            owner = remote_url.split("/")[-2]
except (subprocess.CalledProcessError, IndexError):
    pass

# Register deploy key on each repo
pub_key_content = public_key.read_text().strip()
for repo in repos:
    url = f"https://api.github.com/repos/{owner}/{repo}/keys"
    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "title": "Devcontainer (auto-generated)",
                "key": pub_key_content,
                "read_only": False,
            },
            timeout=30,
        )
        if response.status_code == 201:
            print(f"SSH: added deploy key to {repo}")
        elif response.status_code == 422:
            print(f"SSH: deploy key already exists on {repo}")
        else:
            print(f"SSH: could not add key to {repo}: {response.status_code}")
    except Exception as e:
        print(f"SSH: error adding key to {repo}: {e}")

# Write SSH config
ssh_config = SSH_DIR / "config"
ssh_config_content = (
    "Host github.com\n"
    "    HostName github.com\n"
    "    User git\n"
    "    IdentityFile " + str(private_key) + "\n"
    "    IdentitiesOnly yes\n"
)
ssh_config.write_text(ssh_config_content)
ssh_config.chmod(0o600)
print("SSH: configured for github.com")

# Convert HTTPS remote to SSH (if applicable)
try:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True, check=True,
        cwd=PROJECT_DIR
    )
    remote_url = result.stdout.strip()
    # Check if it's an HTTPS GitHub URL
    if remote_url.startswith("https://github.com/"):
        # Convert to SSH: https://github.com/owner/repo.git -> git@github.com:owner/repo.git
        path = remote_url.replace("https://github.com/", "")
        if not path.endswith(".git"):
            path += ".git"
        ssh_url = f"git@github.com:{path}"
        subprocess.run(
            ["git", "remote", "set-url", "origin", ssh_url],
            check=True, capture_output=True,
            cwd=PROJECT_DIR
        )
        print(f"SSH: converted remote to {ssh_url}")
except subprocess.CalledProcessError:
    pass  # No remote or git not available
SETUP_SSH_KEYS

    # Add github.com to known hosts
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true

    echo "GitHub: credentials configured (SSH for push, token for API)"
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
