# Dynamic Random Sampler

A high-performance weighted random sampler with dynamic weight updates, implementing
the data structure from "Dynamic Generation of Discrete Random Variates" by
Matias, Vitter, and Ni (1993/2003).

## Features

- **O(log* N) sampling**: Expected constant time for all practical N (up to 2^65536)
- **O(log* N) updates**: Amortized expected time for weight changes
- **Dynamic operations**: Insert, delete, and update weights without rebuilding
- **Numerically stable**: Handles weights spanning 10^-300 to 10^300
- **Python bindings**: Easy-to-use Python API via PyO3

## Installation

### Python

```bash
pip install dynamic-random-sampler
```

### From source

```bash
git clone https://github.com/DRMacIver/dynamic-random-sampler.git
cd dynamic-random-sampler
just install  # Install dependencies
just build    # Build the Rust extension
```

## Quick Start

### Python

```python
from dynamic_random_sampler import DynamicSampler

# Create sampler with weights
sampler = DynamicSampler([1.0, 2.0, 3.0])

# Sample an index (returns 0, 1, or 2 with probabilities 1/6, 2/6, 3/6)
index = sampler.sample()

# Update a weight
sampler.update(0, 10.0)  # Now index 0 has weight 10

# Insert a new element
new_index = sampler.insert(5.0)  # Returns 3

# Delete an element (soft delete - index remains stable)
sampler.delete(1)

# Check deletion status
is_deleted = sampler.is_deleted(1)  # True

# Get active (non-deleted) count
count = sampler.active_count()  # 3
```

### Statistical Testing

The sampler includes built-in chi-squared testing:

```python
sampler = DynamicSampler([1.0, 2.0, 3.0])
result = sampler.test_distribution(num_samples=10000)
print(f"Chi-squared: {result.chi_squared}")
print(f"P-value: {result.p_value}")
print(f"Passes at alpha=0.05: {result.passes(0.05)}")
```

## Performance

The algorithm achieves sub-logarithmic time complexity through a tree structure
where elements are partitioned by weight ranges. Key optimizations include:

- **O(1) random child access**: Dual Vec+HashMap storage for rejection sampling
- **Gumbel-max trick**: Log-space sampling without normalization
- **Weight caching**: Avoid redundant log-sum-exp computations
- **Lazy propagation**: Small weight changes don't propagate through tree

### Benchmarks

| Operation | Size | Time |
|-----------|------|------|
| single_sample (uniform) | 1000 | ~198ns |
| single_sample (power_law) | 1000 | ~370ns |
| batch_1000 | 1000 | ~199us |
| construction | 1000 | ~135us |
| update (same range) | 1000 | ~1.8us |
| update (cross range) | 1000 | ~750ns |
| insert | 1000 | ~5.2us |
| delete | 1000 | ~4.1us |

(Measured on development machine)

## Algorithm Overview

Given N elements with weights w_1, ..., w_N, samples index j with probability
w_j / sum(w_i).

### Key Ideas

1. **Range partitioning**: Elements grouped by weight range R_j = [2^(j-1), 2^j)
2. **Tree structure**: Non-root ranges (degree >= 2) become elements at next level
3. **Rejection sampling**: Within each range, accept with probability w/2^j >= 1/2
4. **Log-space arithmetic**: Weights stored as log2(w) for numerical stability

The tree height is bounded by O(log* N), the iterated logarithm, which is <= 5
for all practical N.

See [docs/algorithm.md](docs/algorithm.md) for detailed documentation with
mathematical notation.

## Development

### Requirements

- Rust 1.75+
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [just](https://github.com/casey/just) (command runner)

### Commands

```bash
just install    # Install all dependencies
just build      # Build the Rust extension
just test       # Run fast tests
just test-slow  # Run slow tests (statistical validation)
just test-all   # Run all tests
just lint       # Run all linters
just format     # Format all code
just check      # Run full quality check
cargo bench     # Run benchmarks
```

### Project Structure

```
src/
  core/           # Pure Rust implementation
    mod.rs        # Module definitions, log-sum-exp
    sampler.rs    # Sampling algorithm with Gumbel-max
    range.rs      # Range data structure (O(1) random access)
    level.rs      # Level data structure
    tree.rs       # Immutable tree for sampling
    update.rs     # MutableTree with insert/delete/update
    config.rs     # Section 4 optimization configuration
    stats.rs      # Chi-squared testing
  lib.rs          # PyO3 bindings
tests/
  test_distributions.rs  # Rust statistical tests
  test_hypothesis.py     # Property-based Python tests
benches/
  sampling.rs     # Criterion benchmarks
docs/
  algorithm.md    # Detailed algorithm documentation
```

## References

Matias, Y., Vitter, J. S., & Ni, W. (2003). "Dynamic generation of discrete
random variates." *Theory of Computing Systems*, 36(4), 329-358.

Original conference version: SODA 1993.

## License

MIT
