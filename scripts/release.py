#!/usr/bin/env python3
"""Release script for dynamic-random-sampler.

Updates version to calver (YY.M.D.N), where N is the release number for the day.
This script is called by the release workflow.

Usage:
    python scripts/release.py              # Update version only (no tag)
    python scripts/release.py --dry-run    # Preview changes
    python scripts/release.py --version-only  # Just print next version
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_current_pyproject_version() -> str | None:
    """Get the current version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    content = pyproject_path.read_text()
    pattern = r'^version = "([^"]+)"'
    match = re.search(pattern, content, flags=re.MULTILINE)
    return match.group(1) if match else None


def parse_version_release_number(version: str, base_version: str) -> int | None:
    """Parse the release number from a version string.

    Returns the release number if the version matches base_version.N format,
    or 0 if it matches base_version exactly (old versioning scheme).
    Returns None if the version doesn't match the base_version date.
    """
    # Check for new format: base_version.N
    match = re.match(rf"^{re.escape(base_version)}\.(\d+)$", version)
    if match:
        return int(match.group(1))

    # Check for old format: base_version (without .N suffix)
    if version == base_version:
        return 0

    return None


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.stdout:
            print(result.stdout)
        if check:
            sys.exit(result.returncode)

    return result


def get_calver() -> str:
    """Generate calver version string in YY.M.D.N format.

    N is the release number for the day (0 for first, 1 for second, etc.).
    """
    now = datetime.now()
    year = now.strftime("%y")
    month = str(now.month)  # No leading zero
    day = str(now.day)  # No leading zero
    base_version = f"{year}.{month}.{day}"

    release_numbers: list[int] = []

    # Check pyproject.toml for current version
    current_version = get_current_pyproject_version()
    if current_version:
        pyproject_release = parse_version_release_number(current_version, base_version)
        if pyproject_release is not None:
            release_numbers.append(pyproject_release)

    # Find existing tags for today (format v{base}.N)
    result = run_command(["git", "tag", "-l", f"v{base_version}.*"], check=False)
    if result.returncode == 0 and result.stdout.strip():
        for tag in result.stdout.strip().split("\n"):
            match = re.match(rf"^v{re.escape(base_version)}\.(\d+)$", tag)
            if match:
                release_numbers.append(int(match.group(1)))

    # Also check for old format tag (v{base} without .N)
    result = run_command(["git", "tag", "-l", f"v{base_version}"], check=False)
    if result.returncode == 0 and result.stdout.strip() == f"v{base_version}":
        release_numbers.append(0)

    if not release_numbers:
        # No releases today yet, start at 0
        return f"{base_version}.0"

    # Next release number is max + 1
    next_release = max(release_numbers) + 1
    return f"{base_version}.{next_release}"


def update_version(new_version: str) -> None:
    """Update version in pyproject.toml and __init__.py."""
    # Update pyproject.toml
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Safety check: verify this is the right pyproject.toml
    name_pattern = r'^name = "([^"]+)"'
    name_match = re.search(name_pattern, content, flags=re.MULTILINE)
    if not name_match or name_match.group(1) != "dynamic-random-sampler":
        print(
            "Error: pyproject.toml does not belong to dynamic-random-sampler",
            file=sys.stderr,
        )
        sys.exit(1)

    content = re.sub(
        r'^version = "[^"]+"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    pyproject_path.write_text(content)

    # Update __init__.py if it exists
    init_path = (
        Path(__file__).parent.parent / "src" / "dynamic_random_sampler" / "__init__.py"
    )
    if init_path.exists():
        init_content = init_path.read_text()
        init_content = re.sub(
            r'^__version__ = "[^"]+"',
            f'__version__ = "{new_version}"',
            init_content,
            flags=re.MULTILINE,
        )
        init_path.write_text(init_content)


def main() -> None:
    """Main release script."""
    version_only = "--version-only" in sys.argv
    dry_run = "--dry-run" in sys.argv

    # Generate calver version
    new_version = get_calver()

    if version_only:
        print(new_version)
        return

    if dry_run:
        print(f"Would update to version: {new_version}")
        return

    # Update version files
    update_version(new_version)
    print(f"Updated to version {new_version}")


if __name__ == "__main__":
    main()
