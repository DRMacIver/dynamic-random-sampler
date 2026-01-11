---
description: Quality checkpoint - lint, test, review, and commit
---

# Checkpoint

A checkpoint is a quality gate that ensures code is ready to commit. Use this after completing a logical unit of work.

## Checkpoint Process

1. **Lint** - Run all linters and fix any issues
2. **Test** - Run all tests and ensure 100% coverage
3. **Self-Review** - Review changes using the self-review checklist
4. **Commit** - Create a focused, descriptive commit

## How to Run

```bash
# Step 1: Run quality checks
just check

# Step 2: If any issues, fix them first
# ... make fixes ...
just check  # Re-run until clean

# Step 3: Review changes
git diff HEAD

# Step 4: Commit with a good message
git add <specific-files>
git commit -m "..."
```

## Commit Guidelines

- Small, logically self-contained commits
- Each commit should pass all tests
- Use conventional commit format if appropriate:
  - `feat:` New feature
  - `fix:` Bug fix
  - `refactor:` Code change that neither fixes nor adds
  - `test:` Adding or updating tests
  - `docs:` Documentation only
  - `chore:` Maintenance tasks

## When to Checkpoint

- After completing a feature or fix
- After significant refactoring
- Before switching to a different task
- At natural stopping points

## What to Avoid

- Giant commits with multiple unrelated changes
- Commits that break tests
- "WIP" commits on the main branch
- Commits without running quality checks
