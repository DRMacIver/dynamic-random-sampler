# dynamic-random-sampler

Implementation of the data structure from "Dynamic Generation of Discrete Random Variates"
by Matias, Vitter, and Ni (1993/2003).

## Project Overview

A high-performance weighted random sampler that supports dynamic weight updates. Given N
elements with weights w₁, w₂, ..., wₙ, generates random variate j with probability wⱼ/Σwᵢ.

**Key features:**
- O(log* N) expected time for sampling (practically constant for realistic N)
- O(log* N) amortized expected time for weight updates
- O(N) space
- Weights stored in log space (f64) to handle wide dynamic ranges

**Algorithm summary:**
Elements are partitioned into ranges based on powers of 2 of their weights. These ranges
form a forest of trees with O(log* N) height. Sampling walks down from a root using the
rejection method; updates propagate changes up the tree with lazy evaluation.

## Language & Tools

- Rust (core implementation)
- Python 3.12+ (bindings via PyO3/maturin)
- uv for Python package management
- cargo for Rust builds
- pytest + hypothesis for property-based testing
- ruff for Python linting/formatting
- basedpyright for Python type checking

## Development Commands

```bash
just install    # Install dependencies (Rust + Python)
just build      # Build Rust extension
just test       # Run all tests (Rust + Python)
just test-rust  # Run Rust tests only
just test-py    # Run Python tests only
just lint       # Run all linters
just format     # Format all code
just check      # Run all checks
```

## Quality Standards

- **TDD approach**: Write tests BEFORE implementation. Create beads issues for tests first,
  then make implementation issues depend on them. This ensures tests drive the design.
- Rust code: unit tests for internal logic
- Python code: property-based tests with Hypothesis for correctness
- No linter suppressions without clear justification
- Fix problems properly rather than suppressing errors
- Make small, logically self-contained commits

## Architecture

### Core Data Structure (Rust)

The implementation follows the "First Algorithm" from Section 2 of the paper:

1. **LogWeight**: Weights stored as f64 in log₂ space. A weight w is stored as log₂(w).
   Range index j = ⌊log_weight⌋ + 1, so range Rⱼ covers [2^(j-1), 2^j).

2. **Range**: Contains elements whose weights fall in [2^(j-1), 2^j). Stores:
   - Vector of (element_index, log_weight) pairs
   - Total weight (in log space, using log-sum-exp)

3. **Level**: Collection of ranges at a given tree level. Ranges with ≥2 children
   propagate to the next level; ranges with 1 child become roots.

4. **SamplerList**: Main structure containing:
   - Element log-weights array
   - Forest of trees (levels 1 to L, where L ≤ log* N + 1)
   - Level tables storing root ranges at each level

### Key Operations

- **sample()**: Pick level table → pick root range → walk down using rejection method
- **update(index, new_weight)**: Update element, propagate range changes up tree

### Python Bindings (PyO3)

Thin wrapper exposing:
- `SamplerList(weights: list[float])` - constructor
- `sample() -> int` - generate random index
- `update(index: int, weight: float)` - update weight
- `__len__() -> int` - number of elements
- `weight(index: int) -> float` - get current weight

## Implementation Status

**Current state**: Project infrastructure is set up. A placeholder Rust implementation exists
with basic PyO3 bindings. The core algorithm from the paper is NOT yet implemented - the
current `sample()` uses a simple linear scan instead of the O(log* N) tree-based algorithm.

**Next steps** (tracked in beads, run `bd ready` to see available work):
1. Implement Range data structure (Section 2)
2. Implement Level data structure
3. Implement tree building algorithm (Section 2.1)
4. Implement O(log* N) sampling (Section 2.2)
5. Implement weight updates (Section 2.3)
6. Add comprehensive tests
7. Optional: Add Section 4 optimizations for O(log* N) amortized updates

**Reference**: The paper PDF is at `dynamic generation of random variates.pdf` in the project root.

## Template Management

This project was created from the `new-project-template`. To make changes that should apply
to all projects from this template:

1. Clone the template repo: `git clone git@github.com:DRMacIver/new-project-template.git .template-repo`
2. Make changes in `.template-repo/src/new_drmaciver_project/templates/`
3. Test with `cd .template-repo && just check`
4. Commit and push directly to main (no PR required)
5. Run `just sync-from-template` to pull updates into this project

The `.template-repo/` directory is gitignored. Use `/upstream-template` for guided help.

## Session Completion

Work is incomplete until `git push` succeeds. Before ending a session:
1. Run `just check` and fix any issues
2. Commit changes with descriptive messages
3. Push to remote
