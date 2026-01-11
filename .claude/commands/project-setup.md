---
description: Initial project setup conversation
---

# Project Setup

Welcome! Let's set up your new python project. I'll help you configure it based on your needs.

## Information Gathering

Please tell me about your project:

1. **What does this project do?**
   - What problem does it solve?
   - Who is it for?

2. **Key features or capabilities**
   - What are the main things it needs to do?
   - Any specific technologies or APIs it will use?

3. **Development priorities**
   - Performance, simplicity, extensibility?
   - Any specific constraints?

## Setup Process

Once I understand your project, I will:

1. **Update CLAUDE.md** with your project description and architecture notes
2. **Review template files** and suggest any customizations
3. **Run initial setup** (`just install`)
4. **Verify everything works** (`just check`)
5. **Make initial commit** with your customizations

## Template Information

This project was created from the `python` template, which includes:

- Python 3.12+ with uv for package management
- pytest for testing with 100% coverage requirement
- ruff for linting and formatting
- basedpyright for type checking







- Claude Code commands for autonomous development
- beads for issue tracking
- Quality gates enforcing 100% test coverage

## Commands Available

After setup, you'll have these commands:

```bash
just develop   # Start devcontainer (auto-rebuilds if changed)
just shell     # Open shell in devcontainer
just install   # Install dependencies
just test      # Run tests
just lint      # Run linters
just format    # Format code
just check     # Run all checks (lint + test)
just sync-from-template  # Sync updates from template
```

## Development Approach

This project follows **Test-Driven Development (TDD)**:
1. Write tests BEFORE implementation
2. When creating beads issues, create test issues first
3. Make implementation issues depend on their test issues
4. Run `bd ready` to see what to work on - tests should be at the top

## Let's Get Started

Tell me about your project, and I'll help you set it up!
