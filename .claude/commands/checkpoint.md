---
description: Quality checkpoint - lint, test, review, and commit
---

# Checkpoint

A checkpoint is a quality gate that ensures code is ready to commit. Use this after completing a logical unit of work.

## Pre-Flight Check: Project Configuration

Before proceeding, verify the project is properly configured:

```bash
cat .claude/project-config.json | jq '.'
```

If you see default values (project_type is null, all characteristics are false), run `/project-setup` first to configure the project. This ensures commands like `/goal-verify` can provide appropriate advice.

**Check for defaults:**
```python
import json
config = json.loads(open('.claude/project-config.json').read())
is_default = (
    config.get('project_type') is None and
    not any(config.get('characteristics', {}).values())
)
if is_default:
    print("Project config is at defaults - run /project-setup first")
```

## Checkpoint Process

1. **Lint** - Run all linters and fix any issues
2. **Test** - Run all tests and ensure 100% coverage
3. **Gitignore Check** - Review untracked files for anything that should be ignored
4. **Self-Review** - Review changes using the self-review checklist
5. **Goal Verify** - For substantial implementations, run `/goal-verify` to ensure the work accomplishes its stated purpose
6. **Commit** - Create a focused, descriptive commit

## How to Run

```bash
# Step 0: Check project config (if not yet configured)
cat .claude/project-config.json | jq '.project_type'
# If null, run /project-setup first

# Step 1: Run quality checks
just check

# Step 2: If any issues, fix them first
# ... make fixes ...
just check  # Re-run until clean

# Step 3: Check for files that should be gitignored
git status
# Look for generated files, caches, secrets, etc.
# Add patterns to .gitignore as needed

# Step 4: Review changes
git diff HEAD

# Step 5: For substantial implementations, verify goal completion
# /goal-verify

# Step 6: Commit with a good message
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
