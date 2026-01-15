# List available commands
# Docker image name
# Install dependencies
# Check next release version
# Serve documentation locally

default:
    @just --list

DOCKER_IMAGE := "dynamic-random-sampler-dev"

_docker-build:
    #!/usr/bin/env bash
    set -e
    HASH=$(cat .devcontainer/Dockerfile | sha256sum | cut -d' ' -f1)
    SENTINEL=".devcontainer/.docker-build-hash"
    CACHED_HASH=""
    if [ -f "$SENTINEL" ]; then
        CACHED_HASH=$(cat "$SENTINEL")
    fi
    if [ "$HASH" != "$CACHED_HASH" ]; then
        echo "Dockerfile changed, rebuilding image..."
        docker build -t {{DOCKER_IMAGE}} -f .devcontainer/Dockerfile .
        echo "$HASH" > "$SENTINEL"
    fi

bench:
    cargo criterion

build:
    uv run maturin develop

build-release:
    uv run maturin develop --release

check: lint test-cov

clean:
    rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .ruff_cache/ .cache/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

coverage:
    cargo +nightly llvm-cov --no-report
    cargo +nightly llvm-cov report --fail-under-functions 100 --ignore-filename-regex "(lib.rs|debug.rs)"
    cargo +nightly llvm-cov report --show-missing-lines 2>&1 | python3 scripts/check_coverage.py

develop *ARGS:
    #!/usr/bin/env bash
    set -e

    # Build image if needed
    just _docker-build

    # Run host initialization
    bash .devcontainer/initialize.sh

    # Generate GitHub token if GitHub App is configured
    bash .devcontainer/scripts/generate-github-token.sh

    # Extract Claude credentials from macOS
    # Claude Code needs two things:
    # 1. OAuth tokens from Keychain -> .credentials.json
    # 2. Config file with oauthAccount -> .claude.json (tells Claude who is logged in)
    CLAUDE_CREDS_DIR="$(pwd)/.devcontainer/.credentials"
    mkdir -p "$CLAUDE_CREDS_DIR"

    if command -v security &> /dev/null; then
        echo "Extracting Claude credentials from macOS..."

        # Extract OAuth tokens from Keychain
        CLAUDE_KEYCHAIN_FILE="$CLAUDE_CREDS_DIR/claude-keychain.json"
        security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null > "$CLAUDE_KEYCHAIN_FILE" || true
        if [ -s "$CLAUDE_KEYCHAIN_FILE" ]; then
            echo "  OAuth tokens: $(wc -c < "$CLAUDE_KEYCHAIN_FILE") bytes"
        else
            echo "  WARNING: No OAuth tokens in Keychain"
            echo "  Run 'claude' on macOS and log in first"
            rm -f "$CLAUDE_KEYCHAIN_FILE"
        fi

        # Copy Claude config file (contains oauthAccount which identifies logged-in user)
        CLAUDE_CONFIG_FILE="$CLAUDE_CREDS_DIR/claude-config.json"
        if [ -f "$HOME/.claude/.claude.json" ]; then
            cp "$HOME/.claude/.claude.json" "$CLAUDE_CONFIG_FILE"
            echo "  Config file: copied from ~/.claude/.claude.json"
        elif [ -f "$HOME/.claude.json" ]; then
            cp "$HOME/.claude.json" "$CLAUDE_CONFIG_FILE"
            echo "  Config file: copied from ~/.claude.json"
        else
            echo "  WARNING: No Claude config file found"
            echo "  Run 'claude' on macOS and complete login first"
        fi
    else
        echo "Note: Not running on macOS, skipping credential extraction"
    fi

    # Detect terminal background color mode
    THEME="light-ansi"
    if [ -n "$COLORFGBG" ]; then
        BG=$(echo "$COLORFGBG" | cut -d';' -f2)
        if [ "$BG" -lt 7 ] 2>/dev/null; then
            THEME="dark-ansi"
        fi
    elif [ "$TERM_BACKGROUND" = "dark" ]; then
        THEME="dark-ansi"
    fi

    # Determine command to run
    if [ -z "{{ARGS}}" ]; then
        SETTINGS="{\"theme\":\"$THEME\"}"
        DOCKER_CMD="claude --dangerously-skip-permissions --settings '$SETTINGS'"
    else
        DOCKER_CMD="{{ARGS}}"
    fi

    # Run container with all necessary mounts
    docker run -it --rm \
        -v "$(pwd):/workspaces/dynamic-random-sampler" \
        -v "$(pwd)/.devcontainer/.credentials:/mnt/credentials:ro" \
        -v "$(pwd)/.devcontainer/.ssh:/mnt/ssh-keys" \
        -v "dynamic-random-sampler-home:/home/dev" \
        -v "dynamic-random-sampler-.cache:/workspaces/dynamic-random-sampler/.cache" \
        -e ANTHROPIC_API_KEY= \
        -w /workspaces/dynamic-random-sampler \
        --user dev \
        --entrypoint /workspaces/dynamic-random-sampler/.devcontainer/entrypoint.sh \
        {{DOCKER_IMAGE}} \
        bash -c "$DOCKER_CMD"

docs-build:
    uv run mkdocs build

docs-serve:
    uv run mkdocs serve

format:
    uv run ruff format .
    uv run ruff check --fix .

format-py:
    uv run ruff format .
    uv run ruff check --fix .

format-rust:
    cargo fmt

install:
    uv sync --group dev

install-rust-tools:
    rustup component add clippy rustfmt llvm-tools-preview
    cargo install cargo-llvm-cov

lint:
    uv run ruff check .
    uv run basedpyright
    uv run python scripts/extra_lints.py

lint-py:
    uv run ruff check .
    uv run basedpyright
    uv run python scripts/extra_lints.py

lint-rust:
    cargo clippy --all-targets --all-features -- -D warnings
    cargo fmt --check

release:
    #!/usr/bin/env bash
    set -euo pipefail
    # Update version
    uv run python scripts/release.py
    # Update lock file
    uv lock
    # Get the version that was set
    VERSION=$(uv run python scripts/release.py --version-only)
    # Commit version update
    git add pyproject.toml uv.lock src/
    git commit -m "Release $VERSION"
    # Create and push tag
    git tag "v$VERSION"
    git push origin main "v$VERSION"
    echo "Released v$VERSION - future pushes to main will auto-release"

release-preview:
    uv run python scripts/release.py --dry-run

release-version:
    uv run python scripts/release.py --version-only

sync-from-template:
    #!/usr/bin/env bash
    set -e

    if [ ! -f ".template-version" ]; then
        echo "No .template-version file found. This project may not support template sync."
        exit 1
    fi

    CURRENT_COMMIT=$(jq -r '.commit' .template-version)
    TEMPLATE_TYPE=$(jq -r '.template' .template-version)

    echo "Checking for template updates..."
    LATEST_COMMIT=$(gh api repos/DRMacIver/new-project-template/commits/main --jq '.sha')

    if [ "$CURRENT_COMMIT" = "$LATEST_COMMIT" ]; then
        echo "Already up to date with template."
        exit 0
    fi

    echo ""
    echo "Template update available:"
    echo "  Current: ${CURRENT_COMMIT:0:8}"
    echo "  Latest:  ${LATEST_COMMIT:0:8}"
    echo "  Template: $TEMPLATE_TYPE"
    echo ""

    SYNC_PROMPT="Sync this $TEMPLATE_TYPE project from the latest template. Current commit: $CURRENT_COMMIT, Latest: $LATEST_COMMIT. Update infrastructure files (Dockerfile, .claude/*, justfile), preserve project customizations."

    echo "Attempting automatic sync with Claude Code..."
    echo "(This may take a few minutes. Claude will clone the template repo and update files.)"
    echo ""
    if claude -p "$SYNC_PROMPT" --max-budget-usd 5 --verbose; then
        echo ""
        echo "Sync complete! Run 'just check' to verify."
    else
        echo ""
        echo "Automatic sync needs assistance. Launching interactive mode..."
        claude --append-system-prompt "$SYNC_PROMPT"
    fi

test *ARGS:
    uv run pytest {{ARGS}}

test-all:
    uv run pytest

test-cov:
    uv run pytest --cov --cov-report=term-missing --cov-fail-under=100

test-slow:
    uv run pytest -m slow

test-v:
    uv run pytest -v
