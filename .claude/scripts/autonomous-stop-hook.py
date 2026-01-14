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

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import yaml

# Magic string that allows stopping when human input is required
HUMAN_INPUT_REQUIRED = (
    "I have completed all work that I can and require human input to proceed."
)

# Number of iterations without issue changes before stopping
STALENESS_THRESHOLD = 5

SESSION_FILE = Path(".claude/autonomous-session.local.md")
BUILD_FAILING_MARKER = Path(".claude/build-already-failing.local")


def cleanup_session_file() -> None:
    """Delete the session file on successful exit."""
    try:
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
    except OSError:
        pass  # Ignore errors during cleanup


def eprint(*args: object) -> None:
    """Print to stderr."""
    print(*args, file=sys.stderr)


def get_last_assistant_output(transcript_path: str) -> str:
    """Read the last assistant message from the transcript file."""
    import json as json_module

    try:
        transcript = Path(transcript_path)
        if not transcript.exists():
            return ""

        last_output = ""
        with transcript.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry: dict[str, Any] = json_module.loads(line)
                    # Look for assistant messages
                    if entry.get("type") == "assistant":
                        message = cast(dict[str, Any], entry.get("message", {}))
                        content = cast(list[Any], message.get("content", []))
                        # Get text from content blocks
                        for block in content:
                            if isinstance(block, dict):
                                block_dict = cast(dict[str, Any], block)
                                if block_dict.get("type") == "text":
                                    text = block_dict.get("text", "")
                                    if isinstance(text, str):
                                        last_output = text
                            elif isinstance(block, str):
                                last_output = block
                except json_module.JSONDecodeError:
                    continue
        return last_output
    except Exception:
        return ""


def get_open_issues_count() -> int:
    """Get the count of open beads issues."""
    import json as json_module

    exit_code, output = run_command(["bd", "list", "--status=open", "--format=json"])
    if exit_code != 0:
        return 0

    try:
        issues = json_module.loads(output)
        if isinstance(issues, list):
            return len(cast(list[Any], issues))
    except (json_module.JSONDecodeError, TypeError):
        pass

    # Fallback: count lines from bd ready
    exit_code, output = run_command(["bd", "ready"])
    if exit_code == 0 and output.strip():
        # Count lines that look like issue entries (start with number and bracket)
        count = 0
        for line in output.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "[" in line:
                count += 1
        return count
    return 0


def check_for_bypass() -> tuple[bool, str | None]:
    """Check if Claude's last output contains the bypass string.

    Returns (allowed, rejection_reason) where allowed is True if bypass is
    permitted, and rejection_reason explains why bypass was rejected.
    """
    import json as json_module

    try:
        if sys.stdin.isatty():
            return False, None

        # Read hook input JSON from stdin
        stdin_content = sys.stdin.read()
        if not stdin_content.strip():
            return False, None

        hook_input = json_module.loads(stdin_content)
        transcript_path = hook_input.get("transcript_path", "")

        if not transcript_path:
            return False, None

        # Get Claude's last output from the transcript
        last_output = get_last_assistant_output(transcript_path)

        if HUMAN_INPUT_REQUIRED in last_output:
            # Check if there are remaining issues
            open_count = get_open_issues_count()
            if open_count > 0:
                return False, f"There are {open_count} open issue(s) remaining"
            return True, None

    except Exception:
        pass
    return False, None


def run_command(cmd: list[str]) -> tuple[int, str]:
    """Run a command and return (exit_code, output)."""
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


def check_github_build_status() -> tuple[bool, str | None]:
    """Check GitHub Actions build status for the current branch.

    Returns (should_block, message) where should_block is True if builds
    failed or are still running and we should wait.
    """
    import json

    # Get current branch
    exit_code, branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if exit_code != 0 or not branch.strip():
        return False, None  # Can't determine branch, don't block

    branch = branch.strip()

    # Check if gh is available and authenticated
    exit_code, _ = run_command(["gh", "auth", "status"])
    if exit_code != 0:
        # gh not authenticated, skip this check silently
        return False, None

    # Get workflow runs for current branch
    exit_code, output = run_command([
        "gh", "run", "list",
        "--branch", branch,
        "--limit", "5",
        "--json", "status,conclusion,name,headSha",
    ])

    if exit_code != 0:
        return False, None  # API error, don't block

    try:
        runs: list[dict[str, str]] = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return False, None

    if not runs:
        return False, None  # No runs found

    # Get current commit to check if runs are for our code
    exit_code, current_sha = run_command(["git", "rev-parse", "HEAD"])
    current_sha = current_sha.strip() if exit_code == 0 else ""

    # Check for in-progress or failed runs
    in_progress: list[str] = []
    failed: list[str] = []

    for run in runs:
        status = run.get("status", "")
        conclusion = run.get("conclusion", "")
        name = run.get("name", "unknown")
        run_sha = run.get("headSha", "")

        # Only consider runs for the current commit
        if current_sha and run_sha and not run_sha.startswith(current_sha[:7]):
            continue

        if status in ("in_progress", "queued", "waiting", "pending"):
            in_progress.append(name)
        elif conclusion in ("failure", "cancelled", "timed_out"):
            failed.append(name)

    if failed:
        msg = f"GitHub Actions failed: {', '.join(failed)}"
        return True, msg

    if in_progress:
        msg = f"GitHub Actions running: {', '.join(in_progress)}"
        return True, msg

    return False, None


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


def get_diff_added_lines() -> list[tuple[str, int, str]]:
    """Get added lines from git diff.

    Returns list of (filename, line_number, line_content) tuples.
    """
    # Get unified diff with line numbers
    _, diff = run_command(["git", "diff", "-U0"])
    _, staged_diff = run_command(["git", "diff", "--cached", "-U0"])

    added_lines: list[tuple[str, int, str]] = []
    current_file = ""
    line_num = 0

    for diff_content in [diff, staged_diff]:
        for line in diff_content.splitlines():
            if line.startswith("diff --git"):
                # Extract filename from diff header
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3][2:]  # Remove 'b/' prefix
            elif line.startswith("@@"):
                # Parse hunk header: @@ -old,count +new,count @@
                match = re.search(r"\+([0-9]+)", line)
                if match:
                    line_num = int(match.group(1))
            elif line.startswith("+") and not line.startswith("+++"):
                # This is an added line
                added_lines.append((current_file, line_num, line[1:]))
                line_num += 1
            elif not line.startswith("-"):
                line_num += 1

    return added_lines


def check_error_suppression() -> list[str]:
    """Check for error suppression patterns in added lines.

    Returns list of warning messages if suppression patterns found.
    Only flags patterns that appear in actual end-of-line comments.
    """
    # Regex patterns for detecting error suppression directives
    suppression_regexes = [
        # Python suppression directives
        (re.compile(r"#\s*type:\s*ignore"), "type: ignore comment"),
        (re.compile(r"#\s*noqa"), "noqa comment"),
        (re.compile(r"#\s*pylint:\s*disable"), "pylint disable comment"),
        # TypeScript/JavaScript suppression directives
        (re.compile(r"//\s*@ts-ignore"), "TypeScript @ts-ignore comment"),
        (re.compile(r"//\s*@ts-expect-error"), "TypeScript @ts-expect-error comment"),
        # ESLint suppression directives
        (re.compile(r"(/\*|//)\s*eslint-disable"), "ESLint disable comment"),
    ]

    added_lines = get_diff_added_lines()
    violations: list[str] = []

    for filename, line_num, content in added_lines:
        # Skip lines that are inside string literals (basic heuristic)
        # If line has more quotes before # than after, # is likely in string
        for regex, description in suppression_regexes:
            if regex.search(content):
                violations.append(f"  {filename}:{line_num}: {description}")
                break  # One violation per line is enough

    return violations


def check_empty_except() -> list[str]:
    """Check for empty or pass-only except blocks in added lines.

    Returns list of warning messages if empty except blocks found.
    """
    added_lines = get_diff_added_lines()
    violations: list[str] = []

    for filename, line_num, content in added_lines:
        stripped = content.strip()
        # Check for bare except or except with just pass/...
        if "except" in stripped:
            # Look for patterns like "except: pass" or "except Exception: ..."
            if stripped.endswith(": pass") or stripped.endswith(": ..."):
                violations.append(f"  {filename}:{line_num}: empty except block")
            elif stripped == "except:" or stripped == "except Exception:":
                # Check next line would need context, flag as suspicious
                violations.append(f"  {filename}:{line_num}: bare except clause")

    return violations


def check_todo_without_issue() -> list[str]:
    """Check for work item markers in comments without issue references.

    Returns list of warning messages (not blocking).
    Only flags markers in actual comments (after comment start characters).
    """
    # Regex to find work markers in comments without issue references
    todo_in_comment = re.compile(
        r"(#|//)\s*(TODO|FIXME|HACK|XXX)\b(?!.*#[a-zA-Z]+-[a-zA-Z0-9]+)",
        re.IGNORECASE,
    )

    added_lines = get_diff_added_lines()
    warnings: list[str] = []

    for filename, line_num, content in added_lines:
        match = todo_in_comment.search(content)
        if match:
            marker = match.group(2).upper()
            warnings.append(
                f"  {filename}:{line_num}: {marker} without issue reference"
            )

    return warnings


# Token patterns that indicate hardcoded secrets
# Patterns are split/constructed to avoid self-detection
SECRET_PATTERNS = [
    # GitHub tokens (installation, personal, oauth, user, refresh)
    (re.compile(r"gh[spo]_[A-Za-z0-9]{36,}"), "GitHub token"),
    (re.compile(r"ghu_[A-Za-z0-9]{36,}"), "GitHub user token"),
    (re.compile(r"ghr_[A-Za-z0-9]{36,}"), "GitHub refresh token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), "GitHub fine-grained PAT"),
    # AWS credentials
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
]


def check_hardcoded_secrets() -> list[str]:
    """Check for hardcoded secrets/tokens in added lines.

    Returns list of violation messages if secrets found.
    This is a BLOCKING check - secrets should never be committed.
    """
    added_lines = get_diff_added_lines()
    violations: list[str] = []

    for filename, line_num, content in added_lines:
        # Skip files in .credentials directory
        if ".credentials" in filename:
            continue

        for regex, description in SECRET_PATTERNS:
            if regex.search(content):
                violations.append(f"  {filename}:{line_num}: {description}")
                break  # One violation per line

    return violations


# Large file size threshold (500KB - reasonable for source code)
LARGE_FILE_THRESHOLD_BYTES = 500 * 1024


def check_large_files() -> list[str]:
    """Check for large files being committed.

    Returns list of warning messages for files over the threshold.
    """
    warnings: list[str] = []

    # Check staged files
    _, staged = run_command(["git", "diff", "--cached", "--name-only"])

    # Check untracked files
    _, untracked = run_command(
        ["git", "ls-files", "--others", "--exclude-standard"]
    )

    files_to_check: list[str] = []
    if staged.strip():
        files_to_check.extend(staged.strip().split("\n"))
    if untracked.strip():
        files_to_check.extend(untracked.strip().split("\n"))

    for filepath in files_to_check:
        if not filepath:
            continue
        try:
            path = Path(filepath)
            if path.exists() and path.is_file():
                size = path.stat().st_size
                if size > LARGE_FILE_THRESHOLD_BYTES:
                    size_kb = size / 1024
                    size_mb = size / (1024 * 1024)
                    if size_mb >= 1:
                        size_str = f"{size_mb:.1f}MB"
                    else:
                        size_str = f"{size_kb:.0f}KB"
                    warnings.append(f"  {filepath}: {size_str}")
        except (OSError, PermissionError):
            pass

    return warnings


def main() -> int:
    # Bypass the stop hook during repo critique
    if os.environ.get("CLAUDE_REPO_CRITIQUE"):
        return 0

    # Check for bypass string in Claude's last output
    bypass_allowed, rejection_reason = check_for_bypass()
    if bypass_allowed:
        eprint("Human input required acknowledged. Allowing stop.")
        cleanup_session_file()
        return 0
    elif rejection_reason:
        # Bypass phrase was used but rejected
        eprint("# Exit Phrase Rejected")
        eprint()
        eprint(f"{rejection_reason}.")
        eprint()
        eprint("Please work on the remaining issues before exiting.")
        eprint("Run `bd ready` to see available work.")
        eprint()
        return 2

    # Always check for uncommitted changes (even outside autonomous mode)
    has_changes, change_desc = check_uncommitted_changes()
    if has_changes:
        # Check for code quality issues in the diff
        suppression_violations = check_error_suppression()
        empty_except_violations = check_empty_except()
        secret_violations = check_hardcoded_secrets()
        todo_warnings = check_todo_without_issue()
        large_file_warnings = check_large_files()

        # Run quality checks to ensure they pass
        eprint("# Running Quality Checks...")
        eprint()
        quality_passed = run_quality_check()
        eprint()

        eprint("# Uncommitted Changes Detected")
        eprint()
        eprint(f"Cannot exit with {change_desc}.")
        eprint()

        if not quality_passed:
            eprint("## Quality Checks Failed")
            eprint()
            eprint("Quality gates did not pass. Fix issues before committing.")
            eprint("Run `just check` to see details.")
            eprint()

        # Show blocking violations first
        if suppression_violations:
            eprint("## Error Suppression Detected")
            eprint()
            eprint("The following error suppressions were added:")
            eprint()
            for v in suppression_violations:
                eprint(v)
            eprint()
            eprint("Fix the underlying issues instead of suppressing errors.")
            eprint()

        if empty_except_violations:
            eprint("## Empty Exception Handlers Detected")
            eprint()
            eprint("The following empty except blocks were added:")
            eprint()
            for v in empty_except_violations:
                eprint(v)
            eprint()
            eprint("Handle exceptions properly or re-raise them.")
            eprint()

        if secret_violations:
            eprint("## SECURITY: Hardcoded Secrets Detected")
            eprint()
            eprint("The following secrets/tokens were found in staged changes:")
            eprint()
            for v in secret_violations:
                eprint(v)
            eprint()
            eprint("NEVER commit secrets. Use environment variables instead.")
            eprint("If this was accidental, the secret may need to be rotated.")
            eprint()

        # Show non-blocking warnings
        if todo_warnings:
            eprint("## Untracked Work Items")
            eprint()
            eprint("Consider linking these items to beads issues:")
            eprint()
            for w in todo_warnings:
                eprint(w)
            eprint()
            eprint("Format: # ITEM(#project-abc): description".replace("ITEM", "TODO"))
            eprint()

        if large_file_warnings:
            eprint("## Large Files Detected")
            eprint()
            eprint("The following files exceed 500KB:")
            eprint()
            for w in large_file_warnings:
                eprint(w)
            eprint()
            eprint("Consider using Git LFS or adding to .gitignore if appropriate.")
            eprint()

        # Check for files that might need gitignoring
        untracked_cmd = ["git", "ls-files", "--others", "--exclude-standard"]
        _, untracked = run_command(untracked_cmd)
        if untracked.strip():
            suspicious_patterns = [
                ".env", ".local", ".secret", "credentials",
                "__pycache__", ".pyc", "node_modules", ".DS_Store",
                ".coverage", "htmlcov", ".pytest_cache", ".mypy_cache",
            ]
            suspicious_files: list[str] = []
            for f in untracked.strip().split("\n"):
                for pattern in suspicious_patterns:
                    if pattern in f.lower():
                        suspicious_files.append(f)
                        break

            if suspicious_files:
                eprint("## Review Untracked Files")
                eprint()
                eprint("These untracked files may need to be added to .gitignore:")
                eprint()
                for f in suspicious_files[:10]:  # Limit to first 10
                    eprint(f"  {f}")
                if len(suspicious_files) > 10:
                    eprint(f"  ... and {len(suspicious_files) - 10} more")
                eprint()
                eprint("Run `git status` to review all untracked files.")
                eprint()

        eprint("Before stopping, please:")
        eprint()
        eprint("1. Run `git status` to check for files that should be gitignored")
        eprint("2. Run `just check` to verify quality gates pass")
        eprint("3. For substantial implementations, run `/goal-verify`")
        eprint("4. Stage your changes: `git add <files>`")
        eprint("5. Commit with a descriptive message: `git commit -m '...'`")
        eprint("6. Push to remote: `git push`")
        eprint()
        eprint("Work is incomplete until `git push` succeeds.")
        eprint()
        return 2  # Block exit

    # Check GitHub build status (runs after all other checks pass)
    # Skip if build was already failing at session start
    if BUILD_FAILING_MARKER.exists():
        build_should_block = False
        build_message = None
    else:
        build_should_block, build_message = check_github_build_status()
    if build_should_block and build_message:
        eprint("# GitHub Actions Check")
        eprint()
        eprint(build_message)
        eprint()
        if "running" in build_message.lower():
            eprint("Waiting for builds to complete...")
            eprint("Run `gh run list` to check status.")
        else:
            eprint("Fix the failing builds before stopping.")
            eprint("Run `gh run view` to see details.")
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
        cleanup_session_file()
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
            eprint("2. Say the exit phrase to end the session")
            eprint()
            eprint("To exit, include this exact phrase in your response:")
            eprint()
            eprint(f'  "{HUMAN_INPUT_REQUIRED}"')
            eprint()
            return 2  # Block exit, require explicit choice
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
    eprint("4. For substantial work, run `/goal-verify`")
    eprint("5. Close completed issues with `bd close <id>`")
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
