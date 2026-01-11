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
