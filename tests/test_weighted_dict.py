"""Tests for WeightedDict basic operations and sampling distribution."""

import math
from typing import Any

import pytest

# =============================================================================
# Basic Dict Operations
# =============================================================================


def test_empty_dict() -> None:
    """Test creating an empty WeightedDict."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    assert len(wd) == 0
    assert list(wd.keys()) == []
    assert list(wd.values()) == []
    assert list(wd.items()) == []


def test_setitem_and_getitem() -> None:
    """Test setting and getting items."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0
    wd["c"] = 3.0

    assert wd["a"] == 1.0
    assert wd["b"] == 2.0
    assert wd["c"] == 3.0


def test_getitem_missing_key() -> None:
    """Test getting a missing key raises KeyError."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0

    with pytest.raises(KeyError):
        _ = wd["missing"]


def test_setitem_update() -> None:
    """Test updating an existing key's weight."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    assert abs(wd["a"] - 1.0) < 1e-9

    wd["a"] = 5.0
    assert abs(wd["a"] - 5.0) < 1e-9
    assert len(wd) == 1  # No duplicate key


def test_setitem_zero_weight() -> None:
    """Test setting weight to zero keeps the key but excludes from sampling."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0

    # Set one to zero
    wd["a"] = 0.0

    # Key still exists
    assert "a" in wd
    assert wd["a"] == 0.0
    assert len(wd) == 2

    # But only b can be sampled
    wd.seed(42)
    for _ in range(100):
        assert wd.sample() == "b"


def test_setitem_invalid_weight() -> None:
    """Test setting invalid weights raises ValueError."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()

    with pytest.raises(ValueError):
        wd["a"] = -1.0

    with pytest.raises(ValueError):
        wd["a"] = math.inf

    with pytest.raises(ValueError):
        wd["a"] = math.nan


def test_delitem() -> None:
    """Test deleting a key."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0
    wd["c"] = 3.0

    del wd["b"]

    assert "b" not in wd
    assert len(wd) == 2
    # Remaining keys
    assert set(wd.keys()) == {"a", "c"}


def test_delitem_missing_key() -> None:
    """Test deleting a missing key raises KeyError."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0

    with pytest.raises(KeyError):
        del wd["missing"]


def test_delitem_swap_remove_preserves_weights() -> None:
    """Test that swap-remove correctly preserves other keys' weights."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["first"] = 1.0
    wd["middle"] = 2.0
    wd["last"] = 3.0

    # Delete the middle one - "last" should be swapped in
    del wd["middle"]

    assert wd["first"] == 1.0
    assert wd["last"] == 3.0
    assert len(wd) == 2


def test_contains() -> None:
    """Test checking if a key exists."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0

    assert "a" in wd
    assert "b" not in wd


def test_contains_zero_weight() -> None:
    """Test that keys with zero weight are still 'in' the dict."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["a"] = 0.0

    assert "a" in wd


def test_len() -> None:
    """Test getting the number of keys."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    assert len(wd) == 0

    wd["a"] = 1.0
    assert len(wd) == 1

    wd["b"] = 2.0
    assert len(wd) == 2

    del wd["a"]
    assert len(wd) == 1


def test_iter() -> None:
    """Test iterating over keys."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0
    wd["c"] = 3.0

    keys = list(wd)
    assert set(keys) == {"a", "b", "c"}


def test_keys() -> None:
    """Test keys() method."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0

    assert set(wd.keys()) == {"a", "b"}


def test_values() -> None:
    """Test values() method."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0

    values = wd.values()
    assert sorted(values) == [1.0, 2.0]


def test_items() -> None:
    """Test items() method."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0

    items = dict(wd.items())
    assert items == {"a": 1.0, "b": 2.0}


def test_get_existing() -> None:
    """Test get() with existing key."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0

    assert wd.get("a") == 1.0
    assert wd.get("a", 99.0) == 1.0


def test_get_missing() -> None:
    """Test get() with missing key."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()

    assert wd.get("missing") is None
    assert wd.get("missing", 99.0) == 99.0


def test_pop_existing() -> None:
    """Test pop() removes and returns the weight."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0

    weight = wd.pop("a")
    assert weight == 1.0
    assert "a" not in wd
    assert len(wd) == 1


def test_pop_missing() -> None:
    """Test pop() with missing key raises KeyError."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()

    with pytest.raises(KeyError):
        wd.pop("missing")


def test_update() -> None:
    """Test update() method."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0

    wd.update({"b": 2.0, "c": 3.0})

    assert wd["a"] == 1.0
    assert wd["b"] == 2.0
    assert wd["c"] == 3.0


def test_update_overwrites() -> None:
    """Test update() overwrites existing keys."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0

    wd.update({"a": 10.0})
    assert abs(wd["a"] - 10.0) < 1e-9


def test_clear() -> None:
    """Test clear() removes all keys."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 2.0

    wd.clear()

    assert len(wd) == 0
    assert list(wd.keys()) == []


def test_setdefault_missing() -> None:
    """Test setdefault() with missing key."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()

    result = wd.setdefault("a", 5.0)
    assert abs(result - 5.0) < 1e-9
    assert abs(wd["a"] - 5.0) < 1e-9


def test_setdefault_existing() -> None:
    """Test setdefault() with existing key returns existing value."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0

    result = wd.setdefault("a", 99.0)
    assert result == 1.0
    assert wd["a"] == 1.0  # Not changed


# =============================================================================
# Sampling Tests
# =============================================================================


def test_sample_basic() -> None:
    """Test basic sampling returns valid keys."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict(seed=42)
    wd["a"] = 1.0
    wd["b"] = 2.0
    wd["c"] = 3.0

    for _ in range(100):
        key = wd.sample()
        assert key in {"a", "b", "c"}


def test_sample_empty_raises() -> None:
    """Test sampling from empty dict raises ValueError."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()

    with pytest.raises(ValueError, match="empty"):
        wd.sample()


def test_sample_all_zero_raises() -> None:
    """Test sampling when all weights are zero raises ValueError."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 0.0
    wd["b"] = 0.0

    with pytest.raises(ValueError, match="all weights are 0"):
        wd.sample()


def test_sample_distribution() -> None:
    """Test that sampling distribution is correct."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict(seed=42)
    # Weight 7 should be sampled ~70% of the time
    wd["low"] = 1.0
    wd["low2"] = 1.0
    wd["low3"] = 1.0
    wd["high"] = 7.0

    counts: dict[str, int] = {}
    for _ in range(1000):
        key = wd.sample()
        counts[key] = counts.get(key, 0) + 1

    # "high" should be sampled about 700 times (70%)
    high_count = counts.get("high", 0)
    assert 600 < high_count < 800, f"Expected ~700, got {high_count}"


def test_sample_single_key() -> None:
    """Test sampling with single key always returns that key."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict(seed=42)
    wd["only"] = 1.0

    for _ in range(50):
        assert wd.sample() == "only"


def test_sample_after_delete() -> None:
    """Test sampling works correctly after deleting keys."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict(seed=42)
    wd["a"] = 1.0
    wd["b"] = 100.0  # Dominant
    wd["c"] = 1.0

    # Delete the dominant one
    del wd["b"]

    # Now sampling should only return a or c
    for _ in range(100):
        key = wd.sample()
        assert key in {"a", "c"}


def test_sample_after_update() -> None:
    """Test sampling works correctly after updating weights."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict(seed=42)
    wd["a"] = 1.0
    wd["b"] = 1.0

    # Make a dominant
    wd["a"] = 100.0

    samples = [wd.sample() for _ in range(100)]
    a_count = sum(1 for s in samples if s == "a")

    # a should dominate (>90%)
    assert a_count > 90, f"Expected >90, got {a_count}"


# =============================================================================
# Seeding Tests
# =============================================================================


def test_seed_affects_sampling() -> None:
    """Test that different seeds produce different results."""
    from dynamic_random_sampler import WeightedDict

    wd1: Any = WeightedDict(seed=42)
    wd2: Any = WeightedDict(seed=12345)

    for wd in [wd1, wd2]:
        wd["a"] = 1.0
        wd["b"] = 1.0
        wd["c"] = 1.0
        wd["d"] = 1.0
        wd["e"] = 1.0

    samples1 = [wd1.sample() for _ in range(100)]
    samples2 = [wd2.sample() for _ in range(100)]

    # Different seeds should produce different sequences
    assert samples1 != samples2


def test_seed_method() -> None:
    """Test the seed() method for reseeding."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    wd["b"] = 1.0
    wd["c"] = 1.0

    # Seed and sample
    wd.seed(42)
    samples1 = [wd.sample() for _ in range(50)]

    # Reseed with same seed and sample again
    wd.seed(42)
    samples2 = [wd.sample() for _ in range(50)]

    # Should be identical
    assert samples1 == samples2


# =============================================================================
# Edge Cases
# =============================================================================


def test_many_inserts_and_deletes() -> None:
    """Test many inserts and deletes maintain consistency."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict(seed=42)

    # Insert 100 keys
    for i in range(100):
        wd[f"key_{i}"] = float(i + 1)

    assert len(wd) == 100

    # Delete 50 keys
    for i in range(0, 100, 2):
        del wd[f"key_{i}"]

    assert len(wd) == 50

    # Check remaining keys
    for i in range(1, 100, 2):
        assert f"key_{i}" in wd
        assert abs(wd[f"key_{i}"] - float(i + 1)) < 1e-9


def test_repr() -> None:
    """Test __repr__ method."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0

    repr_str = repr(wd)
    assert "WeightedDict" in repr_str
    assert "a" in repr_str


def test_reinsert_after_delete() -> None:
    """Test reinserting a key after deletion."""
    from dynamic_random_sampler import WeightedDict

    wd: Any = WeightedDict()
    wd["a"] = 1.0
    del wd["a"]
    wd["a"] = 2.0

    assert wd["a"] == 2.0
    assert len(wd) == 1
