#!/bin/bash
# Run a Python script with the project's virtual environment
set -e

# Use uv if available AND pyproject.toml exists (uv run needs a project)
if command -v uv &> /dev/null && [ -f pyproject.toml ]; then
    uv run python "$@"
elif [ -d .venv ]; then
    .venv/bin/python "$@"
else
    python3 "$@"
fi
