---
description: Generate new features and improvements for the project
---

# Idea Generation Mode

You are entering **idea generation mode**. Your goal is to analyze the current state of the project and generate new features, improvements, and enhancements to work on.

## Process

1. **Analyze Current State**
   - Review the existing codebase structure
   - Check what features are implemented
   - Look at existing beads issues for context
   - Review recent commits for project direction

2. **Generate Ideas** across these categories:

   ### Features
   - What functionality is missing?
   - What would make the tool more useful?
   - What do similar projects offer that this doesn't?

   ### Quality Improvements
   - Are there areas with poor test coverage?
   - Code that needs refactoring?
   - Documentation gaps?

   ### Technical Debt
   - Are there TODOs or FIXMEs in the code?
   - Outdated dependencies?
   - Performance bottlenecks?

   ### Developer Experience
   - Can the development workflow be improved?
   - Are error messages helpful?
   - Is the code easy to understand?

3. **Create Issues**
   For each viable idea, create a beads issue with:
   - Clear title describing the work
   - Type: feature, task, bug, or enhancement
   - Priority: 0-4 (be realistic - not everything is P0)
   - Dependencies if applicable

   ```bash
   bd create --title="..." --type=feature --priority=N
   ```

4. **Prioritize**
   - Consider what builds on existing work
   - What provides the most value?
   - What's achievable in reasonable scope?
   - Balance quick wins with meaningful improvements

## Output

After ideation:
1. Run `bd list --status=open` to show the new backlog
2. Provide a summary of what was added
3. Recommend which issues to tackle first

## Guidelines

- Keep ideas in scope for the project's purpose
- Be specific in issue titles and descriptions
- Consider the effort-to-value ratio
- Don't create duplicate issues
- Group related work with dependencies when appropriate
