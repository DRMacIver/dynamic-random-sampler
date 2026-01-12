---
description: Review and update project characteristics
---

# Project Scope

Review and update the project's configuration to ensure commands like `/goal-verify` provide appropriate advice.

## Current Configuration

First, display the current project configuration:

```bash
cat .claude/project-config.json | jq '.'
```

## Understanding the Settings

### Project Type

The `project_type` field describes what kind of software this is:

| Type | Description | Typical Characteristics |
|------|-------------|------------------------|
| `cli` | Command-line tool | User-facing, argument parsing, output formatting |
| `api` | Web API / service | HTTP endpoints, request handling, authentication |
| `library` | Reusable code | API design, documentation, versioning |
| `application` | Standalone app | UI, user experience, state management |

### Characteristics

These flags affect which advice and checks are relevant:

| Characteristic | When to Enable | Commands Affected |
|---------------|----------------|-------------------|
| `algorithm_heavy` | Implementing specific algorithms or data structures | `/goal-verify` shows algorithm verification guidance |
| `performance_critical` | Has specific performance requirements | `/goal-verify` shows performance testing patterns |
| `has_native_bindings` | Python + Rust/C++ hybrid | `/goal-verify` shows FFI integration checks |
| `api_service` | Exposes HTTP/RPC endpoints | (future: API testing guidance) |
| `cli_tool` | Command-line interface | (future: CLI testing guidance) |

### Custom Advice

The `custom_advice` array stores project-specific guidance that should be shown during quality checks.

## Updating Configuration

To update the project configuration, use the AskUserQuestion tool to gather:

1. **Project Type**: What type of project is this? (cli, api, library, application)
2. **Characteristics**: Which of the following apply? (multi-select)
   - Algorithm/data-structure heavy
   - Performance critical
   - Has native bindings (Python + Rust/C++)
   - API service
   - CLI tool

Then update `.claude/project-config.json`:

```python
import json
from pathlib import Path

config_path = Path(".claude/project-config.json")
config = json.loads(config_path.read_text())

# Update based on answers
config["project_type"] = "cli"  # or "api", "library", "application"
config["characteristics"]["algorithm_heavy"] = True  # if applicable
config["characteristics"]["performance_critical"] = True  # if applicable
# ... etc

config_path.write_text(json.dumps(config, indent=2) + "\n")
```

## Auto-Detection Hints

Look for patterns that suggest characteristics should be updated:

### Algorithm Heavy
- Files with "algorithm", "data_structure", "heap", "tree", "graph" in names
- Imports from `heapq`, `collections`, `sortedcontainers`
- Classes with names like `*Tree`, `*Heap`, `*Queue`, `*Graph`

### Performance Critical
- Benchmark files or performance tests
- Profiling configuration
- Comments mentioning "O(n)", "complexity", "latency", "throughput"

### Native Bindings
- Presence of `Cargo.toml` alongside Python code
- `pyo3`, `maturin`, `cffi`, `ctypes` imports
- `CMakeLists.txt` with Python bindings setup

### API Service
- FastAPI, Flask, Django, Express imports
- Files named `routes.py`, `endpoints.py`, `handlers.py`
- OpenAPI/Swagger configuration

### CLI Tool
- Click, argparse, typer, commander imports
- `if __name__ == "__main__":` with argument parsing
- `bin` or `scripts` directories

## When to Run /scope

Run this command when:

1. **Starting work on an existing project** - ensure config reflects current state
2. **After adding new capabilities** - e.g., adding performance tests means `performance_critical` might be relevant
3. **Before /goal-verify** - to ensure you get appropriate verification guidance
4. **If /checkpoint suggests it** - when default config is detected

## Template: python

This project uses the `python` template, which influences available characteristics and defaults.
