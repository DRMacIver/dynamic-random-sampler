---
description: Perform a thorough self-review of recent changes
---

# Self-Review

Before committing, perform a rigorous self-review. You are the sole author of this code - every bug is your bug.

## Review Checklist

### Code Quality
- [ ] Is this sloppy? Did I take any shortcuts?
- [ ] Would I flag this in someone else's PR?
- [ ] Is every code path tested?
- [ ] Are there any TODO comments that should be addressed now?
- [ ] Did I add any suppressions (# noqa, # type: ignore, etc.)?

### Architecture
- [ ] Does this follow existing patterns in the codebase?
- [ ] Are there any import cycles introduced?
- [ ] Is there unnecessary complexity that could be simplified?
- [ ] Did I avoid premature abstractions?

### Testing
- [ ] Do all tests pass?
- [ ] Is coverage at 100%?
- [ ] Are edge cases covered?
- [ ] Are tests testing behavior, not implementation?

### Security
- [ ] No hardcoded secrets or credentials?
- [ ] Input validation at system boundaries?
- [ ] No SQL injection, XSS, or command injection risks?
- [ ] Proper error handling without leaking information?

### Style
- [ ] Consistent naming conventions?
- [ ] Functions have type annotations?
- [ ] No commented-out code?
- [ ] Clear, self-documenting code (comments only where necessary)?

## How to Use

1. Review the diff: `git diff HEAD`
2. Go through each checklist item
3. Fix any issues found
4. Run quality checks: `/quality-check`
5. Only then proceed to commit

## Common Issues to Watch For

1. **Forgotten debug code**: Print statements, console.log, temporary test data
2. **Incomplete error handling**: Catching exceptions without proper handling
3. **Missing edge cases**: Empty inputs, boundary conditions, error states
4. **Security issues**: User input flowing to dangerous operations
5. **Performance**: Unnecessary loops, missing indexes, N+1 queries

## Remember

- If you find issues, fix them now - don't leave them for later
- A suppression is a code smell - refactor instead of hiding
- Test coverage is a minimum, not a goal - aim for quality tests
