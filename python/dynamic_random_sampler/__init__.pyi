"""Dynamic Random Sampler - High-performance weighted random sampling.

This module provides efficient weighted random sampling with O(log* N) operations,
implementing the algorithm from "Dynamic Generation of Discrete Random Variates"
by Matias, Vitter, and Ni (1993/2003).
"""

from collections.abc import Iterator
from typing import overload

__version__: str

class SamplerList:
    """A dynamic weighted random sampler that behaves like a Python list.

    Implements efficient weighted random sampling where each index j is returned
    with probability w_j / sum(w_i). Supports dynamic weight updates in O(log* N)
    amortized expected time.

    Supports most list operations including indexing, slicing, append, extend,
    pop, insert, remove, reverse, sort, and clear. Setting weight to 0 excludes
    an element from sampling but keeps it in the list.

    Examples:
        Basic usage::

            >>> sampler = SamplerList([1.0, 2.0, 3.0, 4.0])
            >>> idx = sampler.sample()  # Returns 0-3, weighted by probability
            >>> sampler[0] = 10.0  # Update weight dynamically
            >>> sampler[1] = 0  # Exclude index 1 from sampling

        Empty sampler::

            >>> sampler = SamplerList()  # Create empty sampler
            >>> sampler.append(1.0)
            >>> sampler.extend([2.0, 3.0])

        Reproducible sampling::

            >>> sampler = SamplerList([1.0, 2.0, 3.0], seed=42)
            >>> results = [sampler.sample() for _ in range(5)]
    """

    def __init__(
        self, weights: list[float] | None = None, *, seed: int | None = None
    ) -> None:
        """Create a new sampler from an optional list of weights.

        Args:
            weights: Optional list of positive weights. If None or empty,
                creates an empty sampler.
            seed: Optional seed for the random number generator (keyword-only).
                If None, uses system entropy.

        Raises:
            ValueError: If weights contains non-positive values.
            ValueError: If any weight is infinite or NaN.
        """
        ...

    def sample(self) -> int:
        """Sample a random index according to the weight distribution.

        Returns an index j with probability w_j / sum(w_i).
        Uses O(log* N) expected time.
        Elements with weight 0 are excluded from sampling.

        Returns:
            The sampled index.

        Raises:
            ValueError: If the sampler is empty.
            ValueError: If all elements have weight 0.
        """
        ...

    def seed(self, seed: int) -> None:
        """Reseed the internal random number generator.

        Args:
            seed: New seed value for the RNG.
        """
        ...

    def append(self, weight: float) -> None:
        """Append a weight to the end.

        Args:
            weight: Positive weight value.

        Raises:
            ValueError: If weight is non-positive, infinite, or NaN.
        """
        ...

    def extend(self, weights: list[float]) -> None:
        """Extend the sampler with multiple weights.

        Args:
            weights: List of positive weight values.

        Raises:
            ValueError: If any weight is non-positive, infinite, or NaN.
        """
        ...

    def pop(self) -> float:
        """Remove and return the last weight.

        Returns:
            The removed weight value.

        Raises:
            IndexError: If the sampler is empty.
        """
        ...

    def insert(self, index: int, weight: float) -> None:
        """Insert a weight at the given index.

        Args:
            index: Index at which to insert. Supports negative indices.
            weight: Positive weight value.

        Raises:
            ValueError: If weight is non-positive, infinite, or NaN.
        """
        ...

    def remove(self, weight: float) -> None:
        """Remove the first element with the given weight.

        Uses approximate comparison (tolerance 1e-10).

        Args:
            weight: Weight value to remove.

        Raises:
            ValueError: If no element with this weight exists.
        """
        ...

    def reverse(self) -> None:
        """Reverse the order of elements in place."""
        ...

    def copy(self) -> list[float]:
        """Return a list copy of all weights."""
        ...

    def sort(self, *, reverse: bool = False) -> None:
        """Sort elements in place.

        Args:
            reverse: If True, sort in descending order (keyword-only).
        """
        ...

    def clear(self) -> None:
        """Remove all elements."""
        ...

    def index(self, weight: float) -> int:
        """Find the first index of an element with the given weight.

        Uses approximate comparison (tolerance 1e-10).

        Args:
            weight: Weight value to search for.

        Returns:
            Index of the first matching element.

        Raises:
            ValueError: If no element with this weight exists.
        """
        ...

    def count(self, weight: float) -> int:
        """Count the number of elements with the given weight.

        Uses approximate comparison (tolerance 1e-10).

        Args:
            weight: Weight value to count.

        Returns:
            Number of elements with this weight.
        """
        ...

    def __len__(self) -> int:
        """Return the number of elements."""
        ...

    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> list[float]: ...
    def __getitem__(self, index: int | slice) -> float | list[float]:
        """Get the weight at the given index or slice.

        Supports negative indices like Python lists.

        Args:
            index: Integer index or slice (can be negative).

        Returns:
            Weight value at the index, or list of weights for slices.

        Raises:
            IndexError: If index is out of bounds.
        """
        ...

    @overload
    def __setitem__(self, index: int, weight: float) -> None: ...
    @overload
    def __setitem__(self, index: slice, weights: list[float]) -> None: ...
    def __setitem__(
        self, index: int | slice, weight: float | list[float]
    ) -> None:
        """Set the weight at the given index or slice.

        Setting weight to 0 excludes the element from sampling
        but keeps it in the list.

        For slices, the list may be resized if the replacement
        has a different length (except for extended slices with step != 1).

        Args:
            index: Integer index or slice (can be negative).
            weight: New weight value(s) (non-negative).

        Raises:
            ValueError: If weight is negative, infinite, or NaN.
            IndexError: If index is out of bounds.
            ValueError: For extended slices with step != 1, if the
                replacement list has a different length than the slice.
        """
        ...

    def __delitem__(self, index: int) -> None:
        """Delete the element at the given index.

        Args:
            index: Integer index (can be negative).

        Raises:
            IndexError: If index is out of bounds.
        """
        ...

    def __contains__(self, weight: float) -> bool:
        """Check if a weight value exists among elements.

        Uses approximate comparison (tolerance 1e-10).
        """
        ...

    def __iter__(self) -> Iterator[float]:
        """Return an iterator over all weights."""
        ...


class SamplerDict:
    """A dictionary-like type with weighted random sampling.

    Keys are strings. Values are non-negative floats representing weights.
    The sample() method returns a random key with probability proportional
    to its weight.

    Examples:
        Basic usage::

            >>> sampler = SamplerDict()
            >>> sampler["apple"] = 5.0
            >>> sampler["banana"] = 3.0
            >>> key = sampler.sample()  # Returns "apple" or "banana"

        Initialize with weights::

            >>> sampler = SamplerDict({"a": 1.0, "b": 2.0, "c": 3.0})
            >>> key = sampler.sample()

        With seed for reproducibility::

            >>> sampler = SamplerDict({"a": 1.0, "b": 2.0}, seed=42)
            >>> key = sampler.sample()
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        """Create a new SamplerDict with optional initial weights.

        Args:
            weights: Optional dictionary of key-weight pairs.
            seed: Optional seed for the random number generator (keyword-only).
        """
        ...

    def sample(self) -> str:
        """Sample a random key according to the weight distribution.

        Returns a key with probability proportional to its weight.
        Keys with weight 0 are excluded from sampling.

        Returns:
            The sampled key.

        Raises:
            ValueError: If the dictionary is empty.
            ValueError: If all weights are 0.
        """
        ...

    def seed(self, seed: int) -> None:
        """Reseed the internal random number generator.

        Args:
            seed: New seed value for the RNG.
        """
        ...

    def keys(self) -> list[str]:
        """Return a list of all keys."""
        ...

    def values(self) -> list[float]:
        """Return a list of all weights (values)."""
        ...

    def items(self) -> list[tuple[str, float]]:
        """Return a list of (key, weight) tuples."""
        ...

    def get(self, key: str, default: float | None = None) -> float | None:
        """Get the weight for a key, or a default value if not present.

        Args:
            key: The key to look up.
            default: Value to return if key is not present (default: None).

        Returns:
            The weight for the key, or default if not present.
        """
        ...

    def pop(self, key: str) -> float:
        """Remove and return the weight for a key.

        Args:
            key: The key to remove.

        Returns:
            The removed weight value.

        Raises:
            KeyError: If the key is not present.
        """
        ...

    def update(self, other: dict[str, float]) -> None:
        """Update the dictionary with key-weight pairs from another dict.

        Args:
            other: Dictionary of key-weight pairs to add/update.

        Raises:
            ValueError: If any weight is invalid.
        """
        ...

    def clear(self) -> None:
        """Remove all keys from the dictionary."""
        ...

    def setdefault(self, key: str, default: float) -> float:
        """Set a key's weight if not already present.

        Args:
            key: The key to set.
            default: Weight value to set if key is not present.

        Returns:
            The weight for the key (new or existing).

        Raises:
            ValueError: If the weight is invalid.
        """
        ...

    def __len__(self) -> int:
        """Return the number of keys."""
        ...

    def __getitem__(self, key: str) -> float:
        """Get the weight for a key.

        Args:
            key: The key to look up.

        Returns:
            The weight for the key.

        Raises:
            KeyError: If the key is not present.
        """
        ...

    def __setitem__(self, key: str, weight: float) -> None:
        """Set the weight for a key.

        If the key already exists, updates its weight.
        If the key is new, inserts it.
        Setting weight to 0 keeps the key present but excludes it from sampling.

        Args:
            key: The key to set.
            weight: The weight value (non-negative).

        Raises:
            ValueError: If weight is negative, infinite, or NaN.
        """
        ...

    def __delitem__(self, key: str) -> None:
        """Delete a key from the dictionary.

        Uses swap-remove internally for efficiency.

        Args:
            key: The key to delete.

        Raises:
            KeyError: If the key is not present.
        """
        ...

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the dictionary."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over keys."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...
