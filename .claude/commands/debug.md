---
description: Troubleshoot project issues and environment problems
---

# Debug

This command helps diagnose common issues when the project isn't working correctly.

## Quick Diagnostics

Run these commands to gather diagnostic information:

### 1. Environment Check

```bash
python --version && uv --version && just --version
```

**Expected**: Tool versions should be recent and compatible with the project.

### 2. Dependency Installation

```bash
just install
```

**Expected**: Should complete without errors. If it fails:
- Check network connectivity
- Verify you have write permissions
- Look for version conflicts in error messages

### 3. Quality Gates Status

```bash
just check
```

**Expected**: Should pass. If it fails:
- Note which specific check failed (lint, test, type check)
- Look at the error output for specific files/issues

### 4. Git Status

```bash
git status
git remote -v
git fetch --dry-run 2>&1
```

**Expected**:
- `git status` shows clean working tree or expected changes
- `git remote -v` shows valid remote URLs
- `git fetch --dry-run` completes without errors (connectivity)

### 5. Beads Status

```bash
bd ready
bd list --status=open
bd sync 2>&1 | head -20
```

**Expected**:
- `bd ready` shows available issues
- `bd sync` completes without errors

## Common Issues and Solutions

### Issue: `just: command not found`

The `just` task runner isn't installed.

```bash
# Install via cargo (if Rust is available)
cargo install just

# Or install via Homebrew
brew install just

# Or via other package managers
# See: https://github.com/casey/just#installation
```

### Issue: `uv: command not found`

The `uv` Python package manager isn't installed.

```bash
# Install via pip
pip install uv

# Or install standalone
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Issue: Dependencies fail to install

Try cleaning and reinstalling:

```bash
# For Python projects
rm -rf .venv
just install

# For Node projects (if applicable)
rm -rf node_modules package-lock.json
npm install
```

### Issue: Tests failing but code looks correct

Check if tests are stale:

```bash
# Clear any cached bytecode/artifacts
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Re-run tests with verbose output
just test -v
```

### Issue: Type checker errors that don't match code

Type stubs might be outdated:

```bash
# Reinstall dependencies
rm -rf .venv && just install

# Check if types package needs updating
uv pip list | grep types-
```

### Issue: Git operations failing

Check SSH and authentication:

```bash
# Test GitHub connectivity
ssh -T git@github.com

# Check credential helper
git config --get credential.helper

# View git remote URL type
git remote -v
```

### Issue: Beads not syncing

```bash
# Initialize beads if missing
bd init

# Force re-sync
bd sync --force

# Check for lock file issues
ls -la .beads/
```

### Issue: Autonomous mode stuck

If autonomous mode isn't making progress:

1. Check the session file:
```bash
cat .claude/autonomous-session.local.md
```

2. Look at the iteration count and last issue change
3. If staleness detected, the loop should stop automatically
4. Use `/cancel-autonomous` to stop if needed

## Environment Verification Checklist

Run through this checklist when setting up or debugging:

- [ ] Python version matches pyproject.toml requirement
- [ ] `uv` is installed and working
- [ ] `just` is installed and working
- [ ] `just install` succeeds
- [ ] `just check` passes
- [ ] Git remote is accessible
- [ ] Beads is initialized (`bd list` works)

## Reporting Issues

If debugging doesn't resolve the issue, gather this information:

```bash
# System info
uname -a
python --version && uv --version && just --version

# Project state
git log --oneline -5
git status
ls -la

# Error output
just check 2>&1 | tail -50
```

Share this output when reporting issues.
