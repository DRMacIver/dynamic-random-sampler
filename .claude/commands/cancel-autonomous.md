---
description: Cancel the active autonomous mode session
---

# Cancel Autonomous Mode

This command cancels the active autonomous development session.

## What This Does

1. Removes the autonomous session configuration file
2. Disables the autonomous stop hook
3. Returns Claude to normal interactive mode

## How to Cancel

Run this command or manually:

```bash
rm -f .claude/autonomous-session.local.md
```

## When to Cancel

Use this when:
- You want to take manual control
- The autonomous session is stuck in an unproductive loop
- You need to change the session goals
- You want to exit without completing all work

## After Canceling

You can start a new autonomous session at any time with `/autonomous-mode`.
