#!/usr/bin/env python3
"""Hook to detect when autonomous mode should stop."""

import sys
from pathlib import Path


def main() -> int:
    session_file = Path(".claude/autonomous-session.local.md")

    if not session_file.exists():
        return 0  # No session, nothing to check

    # Read session state
    content = session_file.read_text()

    # Check for staleness marker
    if "STALE: true" in content:
        print("Autonomous mode stopped due to staleness")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
