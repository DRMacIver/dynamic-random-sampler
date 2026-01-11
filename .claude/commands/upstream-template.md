---
description: Push template improvements back to new-project-template
---

# Upstream Template Improvements

You've identified improvements that should go back to the template repository. This command guides you through the process of upstreaming those changes.

## When to Upstream

Upstream changes when you've made improvements that would benefit ALL projects created from this template, such as:

- Bug fixes in generated files (Dockerfile, scripts, configurations)
- New quality checks or linting rules
- Improved Claude commands or autonomous mode behavior
- Better defaults in configuration files

**Don't upstream** project-specific changes like:
- Custom CLAUDE.md descriptions
- Project-specific dependencies
- Application code

## Process

### 1. Clone/Update Template Repository

The template repo should be checked out at `.template-repo/` (gitignored):

```bash
if [ -d ".template-repo" ]; then
    cd .template-repo && git pull origin main && cd ..
else
    git clone git@github.com:DRMacIver/new-project-template.git .template-repo
fi
```

### 2. Identify Template Files to Change

Template files are generated from Python functions in:
- `src/new_drmaciver_project/templates/base.py` - CLAUDE.md, .gitignore, .editorconfig
- `src/new_drmaciver_project/templates/devcontainer.py` - Dockerfile, devcontainer.json, post-create.sh
- `src/new_drmaciver_project/templates/claude.py` - .claude/commands/*, scripts, settings
- `src/new_drmaciver_project/templates/python.py` - pyproject.toml, justfile (Python)
- `src/new_drmaciver_project/templates/typescript.py` - package.json, tsconfig (TypeScript)
- `src/new_drmaciver_project/templates/rust.py` - Cargo.toml, justfile (Rust)
- `src/new_drmaciver_project/templates/cpp.py` - CMakeLists.txt, justfile (C++)
- `src/new_drmaciver_project/templates/hybrid.py` - Hybrid Python+Rust/C++ projects

### 3. Make Changes

Edit the appropriate template file(s) in the template repo. Remember:
- Changes affect ALL templates of that type
- Test that changes work for the template type
- Follow the existing code style

### 4. Test Changes

```bash
cd .template-repo
just check  # Run lints and tests
```

### 5. Commit and Create PR

```bash
cd .template-repo
git checkout -b improve-templates
git add .
git commit -m "Improve template: <description>"
git push -u origin improve-templates
gh pr create --title "Improve template: <description>" --body "..."
```

### 6. Update This Project

After the PR is merged, run:

```bash
just sync-from-template
```

This will pull in your improvements along with any other template updates.

## Tips

- Keep changes minimal and focused
- One PR per logical change
- Update tests if adding new functionality
- Consider backwards compatibility
