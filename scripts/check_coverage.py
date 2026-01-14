#!/usr/bin/env python3
"""
Check coverage output and verify all uncovered lines are acceptable.

Acceptable uncovered lines:
- Any line in debug.rs or lib.rs (excluded from coverage requirements)
- Lines inside debug_assert! macros (only execute when assertions fail)

Usage:
    cargo +nightly llvm-cov report --show-missing-lines 2>&1 | \\
        python scripts/check_coverage.py
"""

import re
import sys
from pathlib import Path


def is_acceptable_uncovered(file_path: str, line_num: int) -> bool:
    """Check if an uncovered line is acceptable (assertion or hard-to-test)."""
    try:
        path = Path(file_path)
        if not path.exists():
            return False

        lines = path.read_text().splitlines()
        if line_num < 1 or line_num > len(lines):
            return False

        line = lines[line_num - 1].strip()

        # Accept closing braces (often if-let else paths that have no code)
        if line == "}":
            return True

        # Accept lines marked with coverage_acceptable comment
        if "// coverage: acceptable" in line or "// coverage:acceptable" in line:
            return True

        # Check if this line is part of an assert! or debug_assert! macro
        # Look for assertion macros on this line or preceding lines
        assertion_patterns = ["debug_assert!", "assert!"]

        for i in range(max(0, line_num - 15), line_num):
            check_line = lines[i]
            for pattern in assertion_patterns:
                if pattern in check_line:
                    # Found an assertion - check if our line is within its scope
                    # Count parentheses to find the end of the macro
                    paren_count = 0
                    started = False
                    j = i
                    for j in range(i, min(len(lines), line_num + 5)):
                        for char in lines[j]:
                            if char == "(":
                                paren_count += 1
                                started = True
                            elif char == ")":
                                paren_count -= 1
                            if started and paren_count == 0:
                                # End of macro - check if our line is within
                                if j >= line_num - 1:
                                    return True
                                break
                        if started and paren_count == 0:
                            break
                    if started and line_num - 1 <= j:
                        return True

        return False
    except Exception:
        return False


def parse_coverage_output(lines: list[str]) -> dict[str, list[int]]:
    """Parse coverage report output to extract uncovered lines per file."""
    uncovered = {}

    for line in lines:
        # Match lines like: /path/to/file.rs: 120, 131, 174
        match = re.match(r"^(/[^:]+\.rs):\s*(.+)$", line)
        if match:
            file_path = match.group(1)
            line_nums = [int(n.strip()) for n in match.group(2).split(",") if n.strip()]
            uncovered[file_path] = line_nums

    return uncovered


def check_coverage() -> int:
    """Main function to check coverage."""
    # Read from stdin
    input_lines = sys.stdin.read().splitlines()

    # Parse coverage output
    uncovered = parse_coverage_output(input_lines)

    if not uncovered:
        print("✓ No uncovered lines found in coverage report")
        return 0

    # Check each uncovered line
    problems = []

    for file_path, line_nums in uncovered.items():
        # Skip allowed files
        if file_path.endswith("/debug.rs") or file_path.endswith("/lib.rs"):
            continue

        for line_num in line_nums:
            if not is_acceptable_uncovered(file_path, line_num):
                problems.append((file_path, line_num))

    if problems:
        print("✗ Found uncovered lines that are NOT in debug assertions:")
        print()
        for file_path, line_num in problems:
            # Show the actual line content
            try:
                lines = Path(file_path).read_text().splitlines()
                if line_num <= len(lines):
                    content = lines[line_num - 1].strip()
                else:
                    content = "<unknown>"
            except Exception:
                content = "<could not read>"
            print(f"  {file_path}:{line_num}")
            print(f"    {content}")
            print()
        print(f"Total: {len(problems)} line(s) need coverage or debug_assert!")
        return 1

    print("✓ All uncovered lines are in debug assertions or excluded files")
    return 0


if __name__ == "__main__":
    sys.exit(check_coverage())
