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

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LintError:
    file: Path
    line: int
    column: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.rule}: {self.message}"


class LintVisitor(ast.NodeVisitor):
    """AST visitor that checks for lint violations."""

    def __init__(self, file: Path, source: str) -> None:
        self.file = file
        self.source = source
        self.source_lines = source.splitlines()
        self.errors: list[LintError] = []
        self._is_test_file = file.name.startswith("test_")
        self._in_function = False
        self._function_depth = 0

    def _add_error(self, node: ast.AST, rule: str, message: str) -> None:
        lineno = getattr(node, "lineno", 0)
        col_offset = getattr(node, "col_offset", 0)
        self.errors.append(LintError(self.file, lineno, col_offset, rule, message))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Rule 1: No class-based tests (except Hypothesis stateful tests)
        if self._is_test_file and node.name.startswith("Test"):
            # Allow Hypothesis stateful test classes (inherit from *.TestCase)
            is_hypothesis_stateful = any(
                isinstance(base, ast.Attribute) and base.attr == "TestCase"
                for base in node.bases
            )
            if not is_hypothesis_stateful:
                msg = f"Class-based test '{node.name}' found. Use functions."
                self._add_error(node, "no-class-tests", msg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_depth += 1

        # Rule 3: No mutable default arguments
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None and self._is_mutable_default(default):
                msg = "Mutable default argument. Use None instead."
                self._add_error(default, "mutable-default", msg)

        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_depth += 1
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None and self._is_mutable_default(default):
                msg = "Mutable default argument. Use None instead."
                self._add_error(default, "mutable-default", msg)
        self.generic_visit(node)
        self._function_depth -= 1

    def _is_mutable_default(self, node: ast.expr) -> bool:
        """Check if a default value is a mutable type."""
        if isinstance(node, (ast.List, ast.Dict, ast.Set)):
            return True
        is_mutable_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("list", "dict", "set")
        )
        return is_mutable_call

    def visit_Import(self, node: ast.Import) -> None:
        # Rule 2: No imports inside functions (except in test files)
        if self._function_depth > 0 and not self._is_test_file:
            self._add_error(
                node,
                "import-in-function",
                "Import inside function. Move to module level.",
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Rule 2: No imports inside functions (except in test files)
        if self._function_depth > 0 and not self._is_test_file:
            self._add_error(
                node,
                "import-in-function",
                "Import inside function. Move to module level.",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Rule 4: No print() in source code (not test files)
        is_print = isinstance(node.func, ast.Name) and node.func.id == "print"
        if not self._is_test_file and is_print:
            self._add_error(
                node,
                "no-print",
                "Use logging instead of print() in source code.",
            )
        self.generic_visit(node)


def check_todo_comments(file: Path, source: str) -> list[LintError]:
    """Rule 5: Check for TODO/FIXME without issue references."""
    errors: list[LintError] = []
    todo_pattern = re.compile(r"#\s*(TODO|FIXME)(?!:\s*\w+-\d+)", re.IGNORECASE)

    for i, line in enumerate(source.splitlines(), 1):
        match = todo_pattern.search(line)
        if match:
            msg = f"{match.group(1)} needs issue reference (e.g., TODO: PROJ-123)."
            errors.append(LintError(file, i, match.start(), "todo-needs-issue", msg))
    return errors


def lint_file(path: Path) -> list[LintError]:
    """Lint a single file and return any errors."""
    try:
        source = path.read_text()
        tree = ast.parse(source)
        visitor = LintVisitor(path, source)
        visitor.visit(tree)
        errors = visitor.errors

        # Check TODO comments
        errors.extend(check_todo_comments(path, source))

        return errors
    except SyntaxError as e:
        return [LintError(path, e.lineno or 0, e.offset or 0, "syntax-error", str(e))]


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
