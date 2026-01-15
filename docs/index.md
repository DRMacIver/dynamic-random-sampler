# Dynamic Random Sampler

A high-performance weighted random sampler with dynamic weight updates, implemented in Rust with Python bindings.

## Overview

Dynamic Random Sampler implements the data structure from ["Dynamic Generation of Discrete Random Variates"](https://link.springer.com/article/10.1007/s00224-003-1078-6) by Matias, Vitter, and Ni (1993/2003). Given N elements with weights w₁, w₂, ..., wₙ, it generates random index j with probability wⱼ/Σwᵢ.

### Performance Characteristics

| Operation | Time Complexity |
|-----------|-----------------|
| Construction | O(N) |
| Sample | O(log* N) expected |
| Update weight | O(log* N) amortized expected |
| Insert | O(log* N) amortized expected |
| Delete | O(log* N) amortized expected |

**Space complexity:** O(N)

Where log* N is the [iterated logarithm](https://en.wikipedia.org/wiki/Iterated_logarithm) - the number of times you must apply log₂ to N before reaching ≤1. For all practical values of N (up to 2^65536), log* N ≤ 5, making operations effectively constant time.

## Key Features

- **O(log* N) sampling** - Practically constant time for any realistic dataset size
- **Dynamic weight updates** - Change weights without rebuilding the entire structure
- **Stable indices** - Indices never shift; soft-delete via zero weight preserves index validity
- **Wide weight ranges** - Weights stored internally in log space for numerical stability (supports 10⁻³⁰⁰ to 10³⁰⁰)
- **Reproducible sampling** - Seed the RNG for deterministic results
- **Pythonic API** - `SamplerList` behaves like a list, `SamplerDict` behaves like a dict
- **Statistical validation** - Built-in chi-squared goodness-of-fit testing

## Installation

```bash
pip install dynamic-random-sampler
```

## Quick Start

### List-based Sampling

```python
from dynamic_random_sampler import SamplerList

# Create a sampler with weights
sampler = SamplerList([1.0, 2.0, 3.0, 4.0])

# Sample returns index 3 most often (weight 4.0)
for _ in range(10):
    idx = sampler.sample()
    print(f"Sampled index: {idx}")

# Update weights dynamically
sampler[0] = 10.0  # Index 0 now has highest weight

# Exclude an element from sampling (soft delete)
sampler[1] = 0  # Index 1 excluded but still valid
```

### Dictionary-based Sampling

```python
from dynamic_random_sampler import SamplerDict

# Create a sampler with string keys
sampler = SamplerDict()
sampler["common"] = 10.0
sampler["rare"] = 1.0
sampler["very_rare"] = 0.1

# Sample returns "common" most frequently
result = sampler.sample()
print(f"Sampled: {result}")
```

## Use Cases

- **A/B testing** - Sample variants with configurable probabilities
- **Load balancing** - Route requests proportionally to server capacity
- **Game mechanics** - Weighted loot drops, enemy spawning
- **Simulation** - Monte Carlo methods with dynamic probability updates
- **NLP** - Weighted vocabulary sampling for text generation
- **Recommendation systems** - Sample items with relevance-weighted probabilities

## Documentation

- [Getting Started](getting-started.md) - Installation and basic usage examples
- [API Reference](api.md) - Complete API documentation
- [Algorithm Details](algorithm.md) - How the data structure works

## License

This project is licensed under the MIT License.
