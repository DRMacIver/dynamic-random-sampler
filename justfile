# List available commands
default:
    @just --list

# Start devcontainer and run claude (or custom command if args provided)
develop *ARGS:
    #!/usr/bin/env bash
    set -e
    devcontainer up --workspace-folder .
    if [ -z "{{ARGS}}" ]; then
        # Detect terminal background color mode
        # Check COLORFGBG (format: "fg;bg" - bg >= 7 usually means light background)
        # Check common env vars, default to light mode
        THEME="light-ansi"
        if [ -n "$COLORFGBG" ]; then
            BG=$(echo "$COLORFGBG" | cut -d';' -f2)
            if [ "$BG" -lt 7 ] 2>/dev/null; then
                THEME="dark-ansi"
            fi
        elif [ "$TERM_BACKGROUND" = "dark" ]; then
            THEME="dark-ansi"
        fi
        devcontainer exec --workspace-folder . claude --dangerously-skip-permissions --settings "{\"theme\":\"$THEME\"}"
    else
        devcontainer exec --workspace-folder . {{ARGS}}
    fi

# Install dependencies (Rust + Python)
install:
    uv sync --group dev
    uv run maturin develop

# Build Rust extension
build:
    uv run maturin develop

# Build Rust extension in release mode
build-release:
    uv run maturin develop --release

# Run all tests (Python tests cover both Rust and Python)
test:
    uv run pytest

# Run Python tests with verbose output
test-v:
    uv run pytest -v

# Run tests with coverage
test-cov:
    uv run pytest --cov --cov-report=term-missing

# Run all linters
lint: lint-rust lint-py

# Run Rust linters
lint-rust:
    cargo clippy --all-targets --all-features -- -D warnings
    cargo fmt --check

# Run Python linters
lint-py:
    uv run ruff check .
    uv run basedpyright
    uv run python scripts/extra_lints.py

# Format all code
format: format-rust format-py

# Format Rust code
format-rust:
    cargo fmt

# Format Python code
format-py:
    uv run ruff format .
    uv run ruff check --fix .

# Run all checks
check: lint test

# Clean build artifacts
clean:
    rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .ruff_cache/ .cache/
    rm -rf target/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Install Rust development tools
install-rust-tools:
    rustup component add clippy rustfmt llvm-tools-preview
    cargo install cargo-llvm-cov

# Sync project from latest template
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
