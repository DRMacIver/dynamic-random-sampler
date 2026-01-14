#!/usr/bin/env python3
"""Custom linting rules for code quality.

Rules:
1. No class-based tests in test files (use module-level functions)
2. No imports inside functions
3. No mutable default arguments
4. No print() statements in source code (use logging)
5. No TODO/FIXME comments without issue references

Usage: python scripts/extra_lints.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider


@dataclass
class LintError:
    file: Path
    line: int
    column: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.rule}: {self.message}"


class LintVisitor(cst.CSTVisitor):
    """CST visitor that checks for lint violations."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, file: Path) -> None:
        self.file = file
        self.errors: list[LintError] = []
        self._is_test_file = file.name.startswith("test_")
        self._function_depth = 0

    def _get_position(self, node: cst.CSTNode) -> tuple[int, int]:
        """Get line and column for a node."""
        try:
            pos = self.get_metadata(PositionProvider, node)
            if isinstance(pos, CodeRange):
                return pos.start.line, pos.start.column
        except KeyError:
            pass
        return 1, 0

    def _add_error(self, node: cst.CSTNode, rule: str, message: str) -> None:
        line, col = self._get_position(node)
        self.errors.append(LintError(self.file, line, col, rule, message))

    def _is_hypothesis_stateful_class(self, node: cst.ClassDef) -> bool:
        """Check if class inherits from RuleBasedStateMachine or *.TestCase."""
        for base in node.bases:
            if isinstance(base.value, cst.Name):
                if base.value.value == "RuleBasedStateMachine":
                    return True
            elif isinstance(base.value, cst.Attribute):
                # Allow RuleBasedStateMachine or *.TestCase (Hypothesis pattern)
                attr_name = base.value.attr.value
                if attr_name in ("RuleBasedStateMachine", "TestCase"):
                    return True
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Rule 1: No class-based tests (except Hypothesis stateful tests)
        is_test_class = (
            self._is_test_file
            and node.name.value.startswith("Test")
            and not self._is_hypothesis_stateful_class(node)
        )
        if is_test_class:
            self._add_error(
                node,
                "no-class-tests",
                f"Class-based test '{node.name.value}' found. "
                "Use module-level test functions.",
            )
        return True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._function_depth += 1

        # Rule 3: No mutable default arguments
        for param in node.params.params:
            if param.default is not None and self._is_mutable_default(param.default):
                self._add_error(
                    param.default,
                    "mutable-default",
                    "Mutable default argument. "
                    "Use None and initialize in function body.",
                )

        for param in node.params.kwonly_params:
            if param.default is not None and self._is_mutable_default(param.default):
                self._add_error(
                    param.default,
                    "mutable-default",
                    "Mutable default argument. "
                    "Use None and initialize in function body.",
                )

        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._function_depth -= 1

    def _is_mutable_default(self, node: cst.BaseExpression) -> bool:
        """Check if a default value is a mutable type."""
        if isinstance(node, (cst.List, cst.Dict, cst.Set)):
            return True
        if isinstance(node, cst.Call):
            func = node.func
            if isinstance(func, cst.Name) and func.value in ("list", "dict", "set"):
                return True
        return False

    def visit_Import(self, node: cst.Import) -> bool:
        # Rule 2: No imports inside functions (except in test files)
        if self._function_depth > 0 and not self._is_test_file:
            self._add_error(
                node,
                "import-in-function",
                "Import inside function. Move to module level.",
            )
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        # Rule 2: No imports inside functions (except in test files)
        if self._function_depth > 0 and not self._is_test_file:
            self._add_error(
                node,
                "import-in-function",
                "Import inside function. Move to module level.",
            )
        return True

    def visit_Call(self, node: cst.Call) -> bool:
        # Rule 4: No print() in source code (not test files)
        if not self._is_test_file:
            func = node.func
            if isinstance(func, cst.Name) and func.value == "print":
                self._add_error(
                    node,
                    "no-print",
                    "Use logging instead of print() in source code.",
                )
        return True


def check_todo_comments(file: Path, source: str) -> list[LintError]:
    """Rule 5: Check for TODO/FIXME without issue references."""
    errors: list[LintError] = []
    todo_pattern = re.compile(r"#\s*(TODO|FIXME)(?!:\s*\w+-\d+)", re.IGNORECASE)

    for i, line in enumerate(source.splitlines(), 1):
        match = todo_pattern.search(line)
        if match:
            errors.append(
                LintError(
                    file,
                    i,
                    match.start(),
                    "todo-needs-issue",
                    f"{match.group(1)} comment should reference an issue "
                    "(e.g., TODO: PROJ-123).",
                )
            )
    return errors


def lint_file(path: Path) -> list[LintError]:
    """Lint a single file and return any errors."""
    try:
        source = path.read_text()
        tree = cst.parse_module(source)
        wrapper = MetadataWrapper(tree)
        visitor = LintVisitor(path)
        wrapper.visit(visitor)
        errors = visitor.errors

        # Check TODO comments
        errors.extend(check_todo_comments(path, source))

        return errors
    except cst.ParserSyntaxError as e:
        line = e.raw_line if e.raw_line else 1
        col = e.raw_column if e.raw_column else 0
        return [LintError(path, line, col, "syntax-error", e.message)]


def main() -> int:
    """Run linting on all Python files in src and tests."""
    errors: list[LintError] = []

    for directory in ["src", "tests"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            file_errors = lint_file(py_file)
            errors.extend(file_errors)

    if errors:
        for error in sorted(errors, key=lambda e: (str(e.file), e.line, e.column)):
            print(error)
        print(f"\nFound {len(errors)} custom lint error(s)")
        return 1

    print("All custom lint checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
