---
description: Run all quality gates for the project
---

# Quality Check

Run all quality gates to verify the project is in a good state.

## Checks Performed

1. **Type Checking** - basedpyright must pass
2. **Linting** - ruff check must pass
3. **Tests** - pytest must pass
4. **Coverage** - 100% test coverage required
5. **Code Simplification** - Review recent changes for complexity

## How to Run

Execute the quality check script:

```bash
.claude/scripts/quality-check.sh
```

Or run individual checks:

```bash
just lint    # Runs ruff + basedpyright
just test    # Runs pytest
```

## Interpreting Results

- **PASSED**: All checks green, ready for next task or completion
- **PASSED WITH WARNINGS**: Generally OK but could be improved
- **FAILED**: Must fix issues before proceeding

## Code Simplification Review

After mechanical checks pass, review recently modified files for simplification opportunities.
Focus on files changed in the current session or commit - don't review the entire codebase.

### What to Look For

- **Nested ternaries**: Convert to if/else or match statements for clarity
- **Overly complex conditionals**: Break into helper functions or simplify logic
- **Unnecessary abstractions**: If a helper is used once, inline it
- **Premature optimization**: Prioritize readability over micro-optimizations
- **Duplicate code**: Extract common patterns (but only if used 3+ times)
- **Deep nesting**: Flatten with early returns or guard clauses

### Simplification Principles

1. **Preserve exact functionality** - simplification must not change behavior
2. **Choose clarity over brevity** - longer but clearer code is better
3. **Maintain helpful abstractions** - don't over-simplify to the point of obscurity
4. **Verify against CLAUDE.md patterns** - ensure code follows project conventions

### When to Suggest vs. Apply

- **Suggest** simplifications that change structure or could affect behavior
- **Apply directly** trivial cleanups (unused imports, dead code removal)
- **Always ask** before making significant refactoring changes

## When Quality Fails

1. Read the output to identify which check(s) failed
2. Fix the issues one at a time
3. Re-run quality check until all gates pass
4. Only then proceed to the next task

## Common Issues

### Type Errors
- Ensure all functions have proper type annotations
- Check for implicit `any` types
- Verify interface implementations match

### Lint Errors
- Many can be auto-fixed with `--fix` flags
- For remaining issues, follow linter suggestions
- Don't suppress errors without good reason

### Test Failures
- Read the failure message carefully
- Check if the test or the implementation is wrong
- Ensure test isolation (no shared state issues)
