#!/usr/bin/env python3
"""
Autonomous mode stop hook.

This hook runs when Claude attempts to stop/exit. It implements the autonomous
development loop by:
1. Checking if autonomous mode is active
2. Evaluating stopping conditions
3. Blocking exit if work should continue
4. Providing guidance on what to do next
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

# Magic string that allows stopping when human input is required
HUMAN_INPUT_REQUIRED = (
    "I have completed all work that I can and require human input to proceed."
)

# Number of iterations without issue changes before stopping
STALENESS_THRESHOLD = 5

SESSION_FILE = Path(".claude/autonomous-session.local.md")


def eprint(*args: object) -> None:
    """Print to stderr."""
    print(*args, file=sys.stderr)


def check_stdin_for_bypass() -> bool:
    """Check if stdin contains the bypass string."""
    try:
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read()
            if HUMAN_INPUT_REQUIRED in stdin_content:
                return True
    except Exception:
        pass
    return False


def run_command(cmd: list[str]) -> tuple[int, str]:
    """Run a command and return (exit_code, output)."""
    import subprocess

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


def parse_session_file() -> dict[str, Any] | None:
    """Parse the autonomous session configuration file."""
    if not SESSION_FILE.exists():
        return None

    content = SESSION_FILE.read_text()
    if not content.startswith("---"):
        return None

    # Extract YAML frontmatter
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        import yaml

        result: object = yaml.safe_load(parts[1])
        if isinstance(result, dict):
            return cast(dict[str, Any], result)
        return None
    except Exception:
        # Fallback to basic parsing
        config: dict[str, Any] = {}
        for line in parts[1].strip().split("\n"):
            if ":" in line:
                key, value_str = line.split(":", 1)
                key = key.strip()
                value_str = value_str.strip()
                value: Any = int(value_str) if value_str.isdigit() else value_str
                config[key] = value
        return config


def update_session_file(config: dict[str, Any]) -> None:
    """Update the session file with new config."""
    import yaml

    content = f"""---
{yaml.dump(config, default_flow_style=False)}---

# Autonomous Session Log

This file tracks the autonomous development session.
"""
    SESSION_FILE.write_text(content)


def get_issue_ids(output: str) -> set[str]:
    """Extract issue IDs from bd output."""
    ids: set[str] = set()
    for line in output.splitlines():
        # Look for issue ID patterns (e.g., "project-123")
        parts = line.split()
        for part in parts:
            if "-" in part and any(c.isdigit() for c in part):
                # Likely an issue ID
                ids.add(part.split()[0] if " " in part else part)
                break
    return ids


def get_current_issues() -> tuple[set[str], set[str]]:
    """Get current open and in-progress issues."""
    _, open_output = run_command(["bd", "list", "--status=open"])
    _, in_progress_output = run_command(["bd", "list", "--status=in_progress"])

    open_ids = get_issue_ids(open_output)
    in_progress_ids = get_issue_ids(in_progress_output)

    return open_ids, in_progress_ids


def run_quality_check() -> bool:
    """Run the quality check script."""
    script = Path(".claude/scripts/quality-check.sh")
    if not script.exists():
        # Try justfile
        exit_code, _ = run_command(["just", "check"])
        return exit_code == 0

    exit_code, _ = run_command(["bash", str(script)])
    return exit_code == 0


def check_uncommitted_changes() -> tuple[bool, str]:
    """Check for uncommitted git changes.

    Returns (has_changes, description) where has_changes is True if there
    are uncommitted changes that should block exit.
    """
    # Check for unstaged changes
    exit_code, diff_output = run_command(["git", "diff", "--stat"])
    if exit_code != 0:
        # Not a git repo or git error, allow exit
        return False, ""

    # Check for staged changes
    _, staged_output = run_command(["git", "diff", "--cached", "--stat"])

    # Check for untracked files (but not ignored ones)
    untracked_cmd = ["git", "ls-files", "--others", "--exclude-standard"]
    _, untracked_output = run_command(untracked_cmd)

    changes: list[str] = []
    if diff_output.strip():
        changes.append("unstaged changes")
    if staged_output.strip():
        changes.append("staged changes")
    if untracked_output.strip():
        changes.append("untracked files")

    if changes:
        return True, ", ".join(changes)
    return False, ""


def main() -> int:
    # Check for bypass string first
    if check_stdin_for_bypass():
        eprint("Human input required acknowledged. Allowing stop.")
        return 0

    # Always check for uncommitted changes (even outside autonomous mode)
    has_changes, change_desc = check_uncommitted_changes()
    if has_changes:
        eprint("# Uncommitted Changes Detected")
        eprint()
        eprint(f"Cannot exit with {change_desc}.")
        eprint()
        eprint("Before stopping, please:")
        eprint()
        eprint("1. Run `just check` to verify quality gates pass")
        eprint("2. Stage your changes: `git add <files>`")
        eprint("3. Commit with a descriptive message: `git commit -m '...'`")
        eprint("4. Push to remote: `git push`")
        eprint()
        eprint("Work is incomplete until `git push` succeeds.")
        eprint()
        return 2  # Block exit

    # Check if autonomous mode is active
    config = parse_session_file()
    if config is None:
        # Not in autonomous mode, allow normal exit
        return 0

    iteration: int = int(config.get("iteration", 0)) + 1
    last_change_iteration: int = int(config.get("last_issue_change_iteration", 0))
    snapshot_list: list[str] = config.get("issue_snapshot", [])
    previous_snapshot: set[str] = set(snapshot_list)

    # Get current issue state
    open_ids, in_progress_ids = get_current_issues()
    current_snapshot = open_ids | in_progress_ids
    total_outstanding = len(current_snapshot)

    # Check if issues changed
    if current_snapshot != previous_snapshot:
        last_change_iteration = iteration
        eprint(f"Issue state changed at iteration {iteration}")

    # Update session file
    config["iteration"] = iteration
    config["last_issue_change_iteration"] = last_change_iteration
    config["issue_snapshot"] = list(current_snapshot)
    update_session_file(config)

    # Check staleness
    iterations_since_change = iteration - last_change_iteration
    if iterations_since_change >= STALENESS_THRESHOLD:
        eprint("# Staleness Detected")
        eprint()
        eprint(f"No issue changes for {iterations_since_change} iterations.")
        eprint("Autonomous mode is stopping due to lack of progress.")
        eprint()
        eprint("This could mean:")
        eprint("- The remaining work requires human decisions")
        eprint("- There's a blocker that needs manual intervention")
        eprint("- The loop is stuck in an unproductive pattern")
        eprint()
        eprint("Run `/autonomous-mode` to start a new session with fresh goals.")
        return 0  # Allow exit

    # Check if all work is done
    if total_outstanding == 0:
        eprint("# Checking Completion")
        eprint()
        eprint("No outstanding issues. Running quality gates...")
        eprint()

        if run_quality_check():
            eprint("All quality gates passed!")
            eprint("No open issues remain.")
            eprint()
            eprint("## Options")
            eprint()
            eprint("1. Run `/ideate` to generate new work items")
            eprint("2. Exit if the session goal is complete")
            eprint()
            eprint("If your session goal is achieved, you may exit.")
            eprint("Otherwise, run `/ideate` to continue working.")
            return 2  # Block exit, prompt for ideation
        else:
            eprint("Quality gates failed. Fix issues before completing.")
            return 2  # Block exit

    # Work remains
    eprint("# Autonomous Mode Active")
    eprint()
    eprint(f"**Iteration {iteration}** | Outstanding issues: {total_outstanding}")
    eprint(f"Iterations since last issue change: {iterations_since_change}")
    eprint()
    eprint("## Current State")
    eprint(f"- Open issues: {len(open_ids)}")
    eprint(f"- In progress: {len(in_progress_ids)}")
    eprint()
    eprint("## Action Required")
    eprint()
    eprint("Continue working on outstanding issues:")
    eprint()
    eprint("1. Run `bd ready` to see available work")
    eprint("2. Pick an issue and work on it")
    eprint("3. Run quality checks after completing work")
    eprint("4. Close completed issues with `bd close <id>`")
    eprint()

    if iterations_since_change > 2:
        eprint(f"**Warning**: No issue changes for {iterations_since_change} loops.")
        eprint(f"Staleness threshold: {STALENESS_THRESHOLD}")
        eprint()

    eprint("---")
    eprint()
    eprint("If you cannot proceed without human input, include this exact string:")
    eprint()
    eprint(f'  "{HUMAN_INPUT_REQUIRED}"')
    eprint()

    return 2  # Block exit


if __name__ == "__main__":
    sys.exit(main())
