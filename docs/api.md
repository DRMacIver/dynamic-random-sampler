# API Reference

Complete API documentation for dynamic-random-sampler.

## SamplerList

A dynamic weighted random sampler that behaves like a Python list.

Implements the data structure from "Dynamic Generation of Discrete Random Variates" by Matias, Vitter, and Ni (1993/2003). Uses stable indices - indices never shift. Elements can only be added at the end (`append`) or removed from the end (`pop`). Setting weight to 0 excludes an element from sampling but keeps its index valid.

### Constructor

```python
SamplerList(weights: list[float], seed: int | None = None)
```

Create a new sampler from a list of weights.

**Parameters:**

- `weights` - List of positive weights. Must not be empty.
- `seed` - Optional seed for the random number generator. If `None`, uses system entropy.

**Raises:**

- `ValueError` - If weights is empty or contains non-positive values
- `ValueError` - If any weight is infinite or NaN

**Example:**

```python
from dynamic_random_sampler import SamplerList

# Basic construction
sampler = SamplerList([1.0, 2.0, 3.0, 4.0])

# With seed for reproducibility
sampler = SamplerList([1.0, 2.0, 3.0], seed=42)
```

---

### Core Methods

#### sample

```python
sample() -> int
```

Sample a random index according to the weight distribution.

Returns an index `j` with probability `w_j / sum(w_i)`. Uses O(log* N) expected time. Elements with weight 0 are excluded from sampling.

**Returns:** The sampled index

**Raises:**

- `ValueError` - If the sampler is empty
- `ValueError` - If all elements have weight 0

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
idx = sampler.sample()  # Returns 0, 1, or 2
```

---

#### seed

```python
seed(seed: int) -> None
```

Reseed the internal random number generator.

**Parameters:**

- `seed` - New seed value for the RNG

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
sampler.seed(12345)
# Sampling is now deterministic
```

---

### Indexing and Slicing

#### \_\_getitem\_\_

```python
__getitem__(index: int) -> float
__getitem__(slice: slice) -> list[float]
```

Get the weight at the given index or slice.

Supports negative indices and slices like Python lists.

**Parameters:**

- `index` - Integer index (can be negative)
- `slice` - Python slice object

**Returns:** Weight value(s)

**Raises:**

- `IndexError` - If index is out of bounds

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])

sampler[0]      # 1.0
sampler[-1]     # 5.0 (last element)
sampler[1:3]    # [2.0, 3.0]
sampler[::2]    # [1.0, 3.0, 5.0]
```

---

#### \_\_setitem\_\_

```python
__setitem__(index: int, weight: float) -> None
__setitem__(slice: slice, weights: list[float]) -> None
```

Set the weight at the given index or slice.

Setting weight to 0 excludes the element from sampling but keeps it in the list (indices stay stable). For slices, the value must be an iterable of the same length as the slice.

**Parameters:**

- `index` - Integer index (can be negative)
- `weight` - New weight value (non-negative)
- `slice` - Python slice object
- `weights` - List of new weight values

**Raises:**

- `ValueError` - If weight is negative, infinite, or NaN
- `IndexError` - If index is out of bounds
- `ValueError` - If slice and weights have different lengths

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])

sampler[0] = 10.0        # Update single weight
sampler[1] = 0           # Exclude from sampling
sampler[0:2] = [5.0, 6.0]  # Update multiple weights
```

---

### List Operations

#### \_\_len\_\_

```python
__len__() -> int
```

Return the number of elements.

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
len(sampler)  # 3
```

---

#### \_\_iter\_\_

```python
__iter__() -> Iterator[float]
```

Return an iterator over all weights.

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
for weight in sampler:
    print(weight)
```

---

#### \_\_contains\_\_

```python
__contains__(weight: float) -> bool
```

Check if a weight value exists among elements.

Uses approximate comparison (tolerance 1e-10) due to floating-point representation.

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
2.0 in sampler  # True
5.0 in sampler  # False
```

---

#### append

```python
append(weight: float) -> None
```

Append a weight to the end.

**Parameters:**

- `weight` - Positive weight value

**Raises:**

- `ValueError` - If weight is non-positive, infinite, or NaN

**Example:**

```python
sampler = SamplerList([1.0, 2.0])
sampler.append(3.0)
len(sampler)  # 3
```

---

#### extend

```python
extend(weights: list[float]) -> None
```

Extend the sampler with multiple weights.

**Parameters:**

- `weights` - List of positive weight values

**Raises:**

- `ValueError` - If any weight is non-positive, infinite, or NaN

**Example:**

```python
sampler = SamplerList([1.0])
sampler.extend([2.0, 3.0, 4.0])
len(sampler)  # 4
```

---

#### pop

```python
pop() -> float
```

Remove and return the last weight.

**Returns:** The removed weight value

**Raises:**

- `IndexError` - If the sampler is empty

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
weight = sampler.pop()  # 3.0
len(sampler)  # 2
```

---

#### clear

```python
clear() -> None
```

Remove all elements.

After calling `clear()`, the sampler will be empty (len = 0).

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
sampler.clear()
len(sampler)  # 0
```

---

#### index

```python
index(weight: float) -> int
```

Find the first index of an element with the given weight.

Uses approximate comparison (tolerance 1e-10).

**Parameters:**

- `weight` - Weight value to search for

**Returns:** Index of the first matching element

**Raises:**

- `ValueError` - If no element with this weight exists

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0, 2.0])
sampler.index(2.0)  # 1
```

---

#### count

```python
count(weight: float) -> int
```

Count the number of elements with the given weight.

Uses approximate comparison (tolerance 1e-10).

**Parameters:**

- `weight` - Weight value to count

**Returns:** Number of elements with this weight

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0, 2.0])
sampler.count(2.0)  # 2
```

---

### RNG State Methods

#### getstate

```python
getstate() -> bytes
```

Get the current state of the random number generator.

!!! warning "Not Implemented"
    State persistence is not yet fully implemented. Currently returns an empty bytes object. For reproducibility, use construction-time seeding with the `seed` parameter.

**Returns:** A bytes object (currently empty placeholder)

---

#### setstate

```python
setstate(state: bytes) -> None
```

Set the state of the random number generator.

!!! warning "Not Implemented"
    State persistence is not yet fully implemented.

**Raises:**

- `NotImplementedError` - Always (not yet implemented)

---

### Statistical Testing

#### test_distribution

```python
test_distribution(num_samples: int = 10000, seed: int | None = None) -> ChiSquaredResult
```

Run a chi-squared goodness-of-fit test on this sampler.

Takes `num_samples` samples and tests whether the observed distribution matches the expected distribution based on weights.

**Parameters:**

- `num_samples` - Number of samples to take (default: 10000)
- `seed` - Optional random seed for reproducibility

**Returns:** A `ChiSquaredResult` containing the test statistics

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0, 4.0])
result = sampler.test_distribution(num_samples=10000, seed=42)
print(f"P-value: {result.p_value}")
if result.passes(0.05):
    print("Distribution is correct")
```

---

## ChiSquaredResult

Result of a chi-squared goodness-of-fit test.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `chi_squared` | `float` | The chi-squared statistic |
| `degrees_of_freedom` | `int` | Degrees of freedom (number of categories - 1) |
| `p_value` | `float` | The p-value (probability of observing this or more extreme result) |
| `num_samples` | `int` | Number of samples taken |
| `excluded_count` | `int` | Number of indices excluded from chi-squared (expected < threshold) |
| `unexpected_samples` | `int` | Number of unexpected samples in excluded indices |

### Methods

#### passes

```python
passes(alpha: float) -> bool
```

Returns `True` if the test passes at the given significance level.

A test "passes" if the p-value is greater than alpha, meaning we cannot reject the null hypothesis that the observed distribution matches expected.

**Parameters:**

- `alpha` - Significance level (commonly 0.05 or 0.01)

**Returns:** `True` if p_value > alpha

**Example:**

```python
result = sampler.test_distribution()
if result.passes(0.05):
    print("Distribution correct at 5% significance")
```

---

## SamplerDict

A dictionary-like type with weighted random sampling.

Keys are strings. Values are non-negative floats representing weights. The `sample()` method returns a random key with probability proportional to its weight.

### Constructor

```python
SamplerDict(seed: int | None = None)
```

Create a new empty `SamplerDict`.

**Parameters:**

- `seed` - Optional seed for the random number generator

**Example:**

```python
from dynamic_random_sampler import SamplerDict

# Basic construction
sampler = SamplerDict()

# With seed for reproducibility
sampler = SamplerDict(seed=42)
```

---

### Core Methods

#### sample

```python
sample() -> str
```

Sample a random key according to the weight distribution.

Returns a key with probability proportional to its weight. Keys with weight 0 are excluded from sampling.

**Returns:** The sampled key

**Raises:**

- `ValueError` - If the dictionary is empty
- `ValueError` - If all weights are 0

**Example:**

```python
sampler = SamplerDict()
sampler["a"] = 1.0
sampler["b"] = 2.0
key = sampler.sample()  # Returns "a" or "b"
```

---

#### seed

```python
seed(seed: int) -> None
```

Reseed the internal random number generator.

**Parameters:**

- `seed` - New seed value for the RNG

---

### Dictionary Methods

#### \_\_getitem\_\_

```python
__getitem__(key: str) -> float
```

Get the weight for a key.

**Raises:**

- `KeyError` - If the key is not present

---

#### \_\_setitem\_\_

```python
__setitem__(key: str, weight: float) -> None
```

Set the weight for a key.

If the key already exists, updates its weight. If the key is new, inserts it. Setting weight to 0 keeps the key present but excludes it from sampling.

**Raises:**

- `ValueError` - If weight is negative, infinite, or NaN

---

#### \_\_delitem\_\_

```python
__delitem__(key: str) -> None
```

Delete a key from the dictionary.

Uses swap-remove internally: the last key is moved to the deleted position for efficiency.

**Raises:**

- `KeyError` - If the key is not present

---

#### \_\_contains\_\_

```python
__contains__(key: str) -> bool
```

Check if a key exists in the dictionary.

---

#### \_\_len\_\_

```python
__len__() -> int
```

Return the number of keys.

---

#### \_\_iter\_\_

```python
__iter__() -> Iterator[str]
```

Return an iterator over keys.

---

#### keys

```python
keys() -> list[str]
```

Return a list of all keys.

---

#### values

```python
values() -> list[float]
```

Return a list of all weights (values).

---

#### items

```python
items() -> list[tuple[str, float]]
```

Return a list of (key, weight) tuples.

---

#### get

```python
get(key: str, default: float | None = None) -> float | None
```

Get the weight for a key, or a default value if not present.

**Parameters:**

- `key` - The key to look up
- `default` - Value to return if key is not present (default: None)

---

#### pop

```python
pop(key: str) -> float
```

Remove and return the weight for a key.

**Raises:**

- `KeyError` - If the key is not present

---

#### update

```python
update(other: dict[str, float]) -> None
```

Update the dictionary with key-weight pairs from another dict.

**Raises:**

- `ValueError` - If any weight is invalid

---

#### clear

```python
clear() -> None
```

Remove all keys from the dictionary.

---

#### setdefault

```python
setdefault(key: str, default: float) -> float
```

Set a key's weight if not already present.

Returns the weight for the key (new or existing).

**Raises:**

- `ValueError` - If the weight is invalid

---

## Exceptions

The library raises standard Python exceptions:

| Exception | When |
|-----------|------|
| `ValueError` | Invalid weight (negative, infinite, NaN), empty weights list, cannot sample |
| `IndexError` | Index out of bounds, pop from empty list |
| `KeyError` | Key not found in SamplerDict |
| `TypeError` | Invalid index type |
| `NotImplementedError` | `setstate()` called (not yet implemented) |
