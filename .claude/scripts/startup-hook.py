#!/usr/bin/env python3
"""Hook that runs at Claude Code startup."""

import os
import sys
from pathlib import Path


def main() -> int:
    # Check for build failures from previous session
    failure_marker = Path(".build-failure")
    if failure_marker.exists():
        print("WARNING: Previous build failed. Run quality checks.")
        failure_marker.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
