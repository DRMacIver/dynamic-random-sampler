---
description: Safe code restructuring with test-first approach
---

# Refactor

This command guides safe code restructuring by ensuring adequate test coverage before making changes.

## Why Test-First Refactoring?

Refactoring without sufficient tests is dangerous:
- Subtle behavior changes go unnoticed
- Edge cases get broken
- Original intent gets lost

This process ensures changes preserve existing behavior.

## The Refactoring Process

### Phase 1: Understand What You're Refactoring

Before writing any code:

1. **Identify the target** - Which functions/classes/modules are being refactored?
2. **Document current behavior** - What does this code actually do?
3. **List the contracts** - What invariants must be preserved?
4. **Note the callers** - Who depends on this code?

```bash
# Find all usages of the code
grep -rn "function_name" --include="*.py"
```

### Phase 2: Write Tests First (Critical!)

**Do not proceed without this step.**

Write tests that capture the current behavior:

```python
def test_current_behavior_basic():
    """Capture basic behavior before refactoring."""
    result = function_to_refactor(standard_input)
    assert result == expected_output

def test_current_behavior_edge_cases():
    """Capture edge case behavior before refactoring."""
    # Empty input
    assert function_to_refactor([]) == []

    # Single element
    assert function_to_refactor([1]) == [1]

    # Error cases
    with pytest.raises(ValueError):
        function_to_refactor(None)

def test_current_behavior_corner_cases():
    """Test cases that might break during refactoring."""
    # Large inputs
    large_input = list(range(10000))
    result = function_to_refactor(large_input)
    assert len(result) == 10000

    # Unicode/special characters
    assert function_to_refactor("café") == "expected"
```

### Phase 3: Verify Coverage

Before refactoring, ensure tests cover the code:

```bash
# Run tests with coverage on just the target file
pytest --cov=path/to/module --cov-report=term-missing tests/

# Look for uncovered lines in the refactoring target
```

**Coverage checklist:**
- [ ] All public functions have tests
- [ ] Error paths are tested
- [ ] Edge cases are covered
- [ ] Main code paths are exercised

### Phase 4: Make Incremental Changes

**Never refactor everything at once.**

1. **Make one small change**
2. **Run tests immediately**
3. **Commit if tests pass**
4. **Repeat**

```bash
# After each small change
just test  # Or pytest path/to/tests

# If tests pass, commit
git add -p  # Add just this change
git commit -m "Refactor: [small specific change]"
```

### Phase 5: Verify Original Intent

After refactoring, run `/goal-verify` to ensure the code still accomplishes its purpose.

## Rollback Strategy

If tests fail:

```bash
# See what changed
git diff

# If the change was small, fix it
# If it's complex, rollback
git checkout -- path/to/file.py

# Or rollback to last good commit
git reset --hard HEAD~1
```

## Refactoring Patterns

### Safe Pattern: Extract Function

1. Write tests for the code to extract
2. Copy code to new function
3. Run tests (should still pass)
4. Replace original with function call
5. Run tests again

### Safe Pattern: Rename

1. Use IDE/tool for rename (safer than manual)
2. Run tests
3. Check for string references that weren't caught

```bash
# Find string references that might need updating
grep -rn "old_name" --include="*.py" --include="*.md"
```

### Safe Pattern: Change Signature

1. Add new signature alongside old
2. Deprecate old signature
3. Update callers one by one (with tests)
4. Remove old signature

### Dangerous Patterns to Avoid

**Pattern: Big Bang Refactor**
Changing everything at once. If tests fail, you don't know which change broke them.

**Pattern: Optimistic Refactor**
"This is obviously equivalent" - it often isn't.

**Pattern: Test-While-Refactoring**
Writing tests while making changes means tests might encode bugs.

## Checklist Before Starting

- [ ] I understand what the code does
- [ ] I have identified all callers
- [ ] I have written tests capturing current behavior
- [ ] Tests pass with current code
- [ ] Coverage is adequate for the refactoring target
- [ ] I have a rollback plan

## Checklist After Finishing

- [ ] All original tests still pass
- [ ] New tests pass
- [ ] /goal-verify confirms original intent preserved
- [ ] Code review completed (or /self-review run)
- [ ] Changes committed in logical chunks

## When to Stop

Stop and reassess if:
- Tests keep failing in unexpected ways
- You find yourself changing tests to make them pass
- The refactoring scope keeps growing
- You can't explain why a test fails

These are signs the refactoring needs to be broken into smaller pieces or the original behavior wasn't fully understood.
