#!/bin/bash
# Run a Python script with the project's virtual environment
set -e

if command -v uv &> /dev/null; then
    uv run python "$@"
elif [ -d .venv ]; then
    .venv/bin/python "$@"
else
    python3 "$@"
fi
