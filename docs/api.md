# API Reference

Complete API documentation for dynamic-random-sampler.

## SamplerList

A dynamic weighted random sampler that behaves like a Python list.

Implements the data structure from "Dynamic Generation of Discrete Random Variates" by Matias, Vitter, and Ni (1993/2003). Supports most list operations including indexing, slicing, append, extend, pop, insert, remove, reverse, sort, and clear. Setting weight to 0 excludes an element from sampling but keeps it in the list.

### Constructor

```python
SamplerList(weights: list[float] | None = None, *, seed: int | None = None)
```

Create a new sampler from an optional list of weights.

**Parameters:**

- `weights` - Optional list of positive weights. If `None` or empty, creates an empty sampler.
- `seed` - Optional seed for the random number generator (keyword-only). If `None`, uses system entropy.

**Raises:**

- `ValueError` - If weights contains non-positive values
- `ValueError` - If any weight is infinite or NaN

**Example:**

```python
from dynamic_random_sampler import SamplerList

# Create empty sampler
sampler = SamplerList()
sampler.append(1.0)

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

Setting weight to 0 excludes the element from sampling but keeps it in the list. For slices with step 1, the list will be resized if the replacement has a different length. For extended slices (step != 1), the replacement must have the same length as the slice.

**Parameters:**

- `index` - Integer index (can be negative)
- `weight` - New weight value (non-negative)
- `slice` - Python slice object
- `weights` - List of new weight values

**Raises:**

- `ValueError` - If weight is negative, infinite, or NaN
- `IndexError` - If index is out of bounds
- `ValueError` - For extended slices, if weights have different length than slice

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0, 4.0])

sampler[0] = 10.0           # Update single weight
sampler[1] = 0              # Exclude from sampling
sampler[0:2] = [5.0, 6.0]   # Update multiple weights
sampler[1:3] = [7.0]        # Replace 2 elements with 1 (shrinks list)
sampler[1:2] = [8.0, 9.0]   # Replace 1 element with 2 (expands list)
```

---

#### \_\_delitem\_\_

```python
__delitem__(index: int) -> None
```

Delete the element at the given index.

**Parameters:**

- `index` - Integer index (can be negative)

**Raises:**

- `IndexError` - If index is out of bounds

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
del sampler[1]
list(sampler)  # [1.0, 3.0]
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

#### insert

```python
insert(index: int, weight: float) -> None
```

Insert a weight at the given index.

**Parameters:**

- `index` - Index at which to insert (supports negative indices)
- `weight` - Positive weight value

**Raises:**

- `ValueError` - If weight is non-positive, infinite, or NaN

**Example:**

```python
sampler = SamplerList([1.0, 3.0])
sampler.insert(1, 2.0)
list(sampler)  # [1.0, 2.0, 3.0]
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

#### remove

```python
remove(weight: float) -> None
```

Remove the first element with the given weight.

Uses approximate comparison (tolerance 1e-10).

**Parameters:**

- `weight` - Weight value to remove

**Raises:**

- `ValueError` - If no element with this weight exists

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0, 2.0])
sampler.remove(2.0)
list(sampler)  # [1.0, 3.0, 2.0]
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

#### reverse

```python
reverse() -> None
```

Reverse the order of elements in place.

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
sampler.reverse()
list(sampler)  # [3.0, 2.0, 1.0]
```

---

#### copy

```python
copy() -> list[float]
```

Return a list copy of all weights.

**Returns:** A new list containing all weights

**Example:**

```python
sampler = SamplerList([1.0, 2.0, 3.0])
weights = sampler.copy()  # [1.0, 2.0, 3.0]
```

---

#### sort

```python
sort(*, reverse: bool = False) -> None
```

Sort elements in place.

**Parameters:**

- `reverse` - If True, sort in descending order (keyword-only)

**Example:**

```python
sampler = SamplerList([3.0, 1.0, 2.0])
sampler.sort()
list(sampler)  # [1.0, 2.0, 3.0]

sampler.sort(reverse=True)
list(sampler)  # [3.0, 2.0, 1.0]
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

## SamplerDict

A dictionary-like type with weighted random sampling.

Keys are strings. Values are non-negative floats representing weights. The `sample()` method returns a random key with probability proportional to its weight.

### Constructor

```python
SamplerDict(weights: dict[str, float] | None = None, *, seed: int | None = None)
```

Create a new `SamplerDict` with optional initial weights.

**Parameters:**

- `weights` - Optional dictionary of key-weight pairs
- `seed` - Optional seed for the random number generator (keyword-only)

**Example:**

```python
from dynamic_random_sampler import SamplerDict

# Create empty sampler
sampler = SamplerDict()

# Initialize with weights
sampler = SamplerDict({"apple": 5.0, "banana": 3.0, "cherry": 2.0})

# With seed for reproducibility
sampler = SamplerDict({"a": 1.0, "b": 2.0}, seed=42)
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
sampler = SamplerDict({"a": 1.0, "b": 2.0})
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
| `ValueError` | Invalid weight (negative, infinite, NaN), cannot sample from empty/all-zero |
| `IndexError` | Index out of bounds, pop from empty list |
| `KeyError` | Key not found in SamplerDict |
| `TypeError` | Invalid index type |
