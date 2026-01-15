# Getting Started

This guide covers installation and basic usage of dynamic-random-sampler.

## Prerequisites

- Python 3.12 or higher
- pip or uv package manager

## Installation

### Using pip

```bash
pip install dynamic-random-sampler
```

### Using uv

```bash
uv add dynamic-random-sampler
```

### From Source

```bash
git clone https://github.com/DRMacIver/dynamic-random-sampler.git
cd dynamic-random-sampler
uv sync
```

## Basic Usage

### SamplerList - List-like Interface

`SamplerList` is ideal when you have a fixed set of elements identified by index.

```python
from dynamic_random_sampler import SamplerList

# Create a sampler from a list of weights
# Index 0 has weight 1.0, index 1 has weight 2.0, etc.
sampler = SamplerList([1.0, 2.0, 3.0, 4.0])

# Sample returns an index with probability proportional to its weight
# Index 3 (weight 4.0) is returned 40% of the time (4/10)
idx = sampler.sample()
print(f"Sampled index: {idx}")

# Get the current weight at an index
print(f"Weight at index 0: {sampler[0]}")  # 1.0

# Get length
print(f"Number of elements: {len(sampler)}")  # 4
```

### SamplerDict - Dictionary-like Interface

`SamplerDict` is ideal when elements are identified by string keys.

```python
from dynamic_random_sampler import SamplerDict

# Create an empty sampler and add items
sampler = SamplerDict()
sampler["apple"] = 5.0
sampler["banana"] = 3.0
sampler["cherry"] = 2.0

# Sample returns a key with probability proportional to its weight
key = sampler.sample()
print(f"Sampled: {key}")  # "apple" most likely (50% of the time)

# Get weight for a key
print(f"Apple weight: {sampler['apple']}")  # 5.0

# Check if key exists
print(f"Has apple: {'apple' in sampler}")  # True
```

## Dynamic Weight Updates

A key feature is the ability to efficiently update weights without rebuilding the structure.

### Updating Weights in SamplerList

```python
from dynamic_random_sampler import SamplerList

sampler = SamplerList([1.0, 1.0, 1.0, 1.0])

# Initially all indices have equal probability (25% each)
# Update index 0 to dominate
sampler[0] = 100.0

# Now index 0 is sampled ~97% of the time (100/103)
```

### Soft Deletion (Exclusion)

Setting a weight to 0 excludes the element from sampling while keeping its index valid.

```python
from dynamic_random_sampler import SamplerList

sampler = SamplerList([1.0, 2.0, 3.0])

# Exclude index 1 from sampling
sampler[1] = 0

# Index 1 will never be sampled, but remains in the list
# Indices 0 and 2 now split the probability (1/4 and 3/4)
print(len(sampler))  # Still 3

# Re-include by setting a positive weight
sampler[1] = 5.0  # Index 1 is back
```

### Updating Weights in SamplerDict

```python
from dynamic_random_sampler import SamplerDict

sampler = SamplerDict()
sampler["item_a"] = 10.0
sampler["item_b"] = 10.0

# Both items have equal probability
# Boost item_a
sampler["item_a"] = 100.0

# Delete item_b entirely
del sampler["item_b"]
```

## Reproducible Sampling

For deterministic results, provide a seed when creating the sampler or call `seed()`.

```python
from dynamic_random_sampler import SamplerList

# Seed at construction time
sampler = SamplerList([1.0, 2.0, 3.0], seed=42)

# These samples will always be the same
results = [sampler.sample() for _ in range(5)]
print(results)  # Deterministic sequence

# Re-seed for a different sequence
sampler.seed(123)
results = [sampler.sample() for _ in range(5)]
print(results)  # Different deterministic sequence
```

## Appending and Removing Elements

`SamplerList` supports growing and shrinking at the end (stable indices).

```python
from dynamic_random_sampler import SamplerList

sampler = SamplerList([1.0, 2.0])
print(len(sampler))  # 2

# Append new elements
sampler.append(3.0)
print(len(sampler))  # 3

# Extend with multiple elements
sampler.extend([4.0, 5.0])
print(len(sampler))  # 5

# Pop removes from the end
weight = sampler.pop()
print(f"Popped: {weight}")  # 5.0
print(len(sampler))  # 4
```

## Slicing and Iteration

`SamplerList` supports Python list idioms.

```python
from dynamic_random_sampler import SamplerList

sampler = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])

# Negative indexing
print(sampler[-1])  # 5.0 (last element)

# Slicing
print(sampler[1:3])  # [2.0, 3.0]
print(sampler[::2])  # [1.0, 3.0, 5.0] (every other)

# Slice assignment
sampler[1:3] = [20.0, 30.0]
print(list(sampler))  # [1.0, 20.0, 30.0, 4.0, 5.0]

# Iteration
for weight in sampler:
    print(weight)
```

## Statistical Validation

Test that sampling matches the expected distribution using the built-in chi-squared test.

```python
from dynamic_random_sampler import SamplerList

sampler = SamplerList([1.0, 2.0, 3.0, 4.0])

# Run chi-squared goodness-of-fit test
result = sampler.test_distribution(num_samples=10000, seed=42)

print(f"Chi-squared: {result.chi_squared:.4f}")
print(f"Degrees of freedom: {result.degrees_of_freedom}")
print(f"P-value: {result.p_value:.4f}")

# Check if distribution passes at 5% significance level
if result.passes(0.05):
    print("Distribution is correct!")
else:
    print("Distribution may be biased")
```

## Performance Tips

1. **Batch operations**: If updating many weights, updates are individually O(log* N) but the constant factors matter. For bulk initialization, construct with all weights at once rather than appending one by one.

2. **Avoid unnecessary reads**: Reading weights via `sampler[i]` is O(1), but if you need all weights, iterate once rather than making individual accesses.

3. **Use seeds for testing**: Always seed your sampler in tests for reproducibility.

## Next Steps

- See the [API Reference](api.md) for complete method documentation
- Read the [Algorithm Details](algorithm.md) to understand how the data structure works
