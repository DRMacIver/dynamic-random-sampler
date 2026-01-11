# List available commands
default:
    @just --list

# Start devcontainer and run claude (or custom command if args provided)
develop *ARGS:
    #!/usr/bin/env bash
    set -e
    devcontainer up --workspace-folder .
    if [ -z "{{ARGS}}" ]; then
        devcontainer exec --workspace-folder . claude --dangerously-skip-permissions
    else
        devcontainer exec --workspace-folder . {{ARGS}}
    fi

# Install dependencies (Rust + Python)
install:
    uv sync --group dev
    maturin develop

# Build Rust extension
build:
    maturin develop

# Build Rust extension in release mode
build-release:
    maturin develop --release

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
    cargo clippy -- -D warnings
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
    LATEST_COMMIT=$(curl -s https://api.github.com/repos/DRMacIver/new-project-template/commits/main | jq -r '.sha')
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
    SYNC_PROMPT="Sync this $TEMPLATE_TYPE project from the latest template. Current commit: $CURRENT_COMMIT, Latest: $LATEST_COMMIT. Update infrastructure files, preserve project customizations."
    echo "Attempting automatic sync..."
    if claude -p "$SYNC_PROMPT" --max-budget-usd 5 2>/dev/null; then
        echo ""
        echo "Sync complete! Run 'just check' to verify."
    else
        echo ""
        echo "Automatic sync needs assistance. Launching interactive mode..."
        claude --append-system-prompt "$SYNC_PROMPT"
    fi
