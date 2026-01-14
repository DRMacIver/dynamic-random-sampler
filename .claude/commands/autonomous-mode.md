---
description: Enter autonomous development mode with iterative task completion
---

# Autonomous Development Mode

You are entering **autonomous development mode**. This mode allows you to work iteratively on tasks with minimal human intervention, using beads for issue tracking and automatic progress detection.

## Setup Phase

Before the loop begins, you MUST gather information from the user:

1. **Understand the Goal**: What should be accomplished in this autonomous session?
2. **Define Success Criteria**: How will you know when work is complete?
3. **Identify Constraints**: Are there areas of the codebase to avoid? Time limits? Scope boundaries?
4. **Quality Requirements**: What quality gates must pass? (tests, lint, type checking)

Use the AskUserQuestion tool to clarify any ambiguities. Write the session configuration to `.claude/autonomous-session.local.md` with:

```yaml
---
goal: "<user's goal>"
success_criteria:
  - "<criterion 1>"
  - "<criterion 2>"
constraints:
  - "<constraint 1>"
quality_gates:
  - "<gate 1: e.g., just test>"
  - "<gate 2: e.g., just lint>"
started_at: "<ISO timestamp>"
iteration: 0
last_issue_change_iteration: 0
issue_snapshot: []
---

# Session Log

(Session log will be maintained below during the session)
```

## Loop Behavior

Once setup is complete, the autonomous loop will:

1. **Check for available work**: Run `bd ready` to find issues to work on
2. **Work on issues**: Pick the highest priority available issue and work on it
3. **Run quality gates**: After completing work, run quality checks
4. **Close completed issues**: Use `bd close <id>` when work is done
5. **Ideate if empty**: If no issues exist, run `/ideate` to generate new work
6. **Repeat**: Continue until stopping conditions are met

## Session Tracking

During the session, maintain a log in `.claude/autonomous-session.local.md`:

### After Each Issue

When you complete or update an issue, log it:

```markdown
## Issue: <issue-id>
- **Status**: completed/blocked/skipped
- **Work Done**: Brief description
- **Commits**: List commit hashes if any
- **Quality Gates**: pass/fail
```

### After Each Quality Gate Run

Log the result:

```markdown
### Quality Gate Run (iteration N)
- **Result**: pass/fail
- **Issues**: Any failures noted
```

### Increment Counters

Update the YAML frontmatter:
- `iteration`: Increment after each work unit
- `last_issue_change_iteration`: Update when an issue is created or closed

## Stopping Conditions

The loop will stop when ANY of these conditions are met:

1. **All issues closed**: No open or in-progress issues remain AND quality gates pass
2. **Staleness detected**: No issues created or closed for 5 consecutive iterations
3. **Human input required**: A task requires decisions you cannot make autonomously
4. **Error state**: Quality gates consistently fail with no clear fix

## Working in Autonomous Mode

While in autonomous mode:

- **Be thorough**: Don't cut corners. Fix issues properly.
- **Commit often**: Make small, logical commits after each piece of work
- **Track everything**: Add discovered issues to beads immediately
- **Run quality checks**: Ensure quality gates pass before moving on
- **Verify goals**: For substantial work, run `/goal-verify` before closing
- **Document blockers**: If stuck, document why and move to the next issue

## Exiting Autonomous Mode

To exit properly, you must either:

1. Complete all work (no open issues, quality gates pass)
2. Reach a staleness threshold (no progress for 5 iterations)
3. Output the bypass phrase when human input is genuinely required:

   "I have completed all work that I can and require human input to proceed."

This phrase allows the stop hook to let you exit. Only use it when truly blocked.

## Session Summary (Generate on Exit)

Before exiting, generate a session summary by appending to `.claude/autonomous-session.local.md`:

```markdown
---

# Session Summary

**Duration**: <start time> to <end time>
**Exit Reason**: all work complete / staleness / human input required / error

## Issues Worked On
| Issue ID | Status | Outcome |
|----------|--------|---------|
| xxx-123  | closed | Description of what was done |
| xxx-456  | open   | Blocked because... |

## Commits Made
```bash
git log --oneline <first-commit>..HEAD
```

## Quality Gate History
- Total runs: N
- Passes: N
- Failures: N

## Metrics
- **Iterations**: N
- **Issues Closed**: N
- **Issues Created**: N
- **Staleness Events**: N (iterations without progress)

## Notes
Any observations, patterns, or suggestions for future sessions.
```

Run this command to get commits made during session:
```bash
git log --oneline --since="<started_at timestamp>"
```

## Starting the Loop

After writing the configuration file:

1. Run `bd list --status=open` to see current issues
2. If no issues exist, run `/ideate` first
3. Begin working on the highest priority available issue
4. The stop hook will prevent premature exits and guide you to continue
