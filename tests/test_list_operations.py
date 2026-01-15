"""Tests for Python list-like operations on SamplerList.

The SamplerList uses stable indices - elements can only be added at the end
(append) or removed from the end (pop). There is no __delitem__ or remove().
Setting weight to 0 excludes an element from sampling but keeps its index valid.
"""

import math
from typing import Any

import pytest

# =============================================================================
# Construction Tests
# =============================================================================


def test_empty_weights_allowed() -> None:
    """Verify empty weight list creates an empty sampler."""
    from dynamic_random_sampler import SamplerList

    sampler = SamplerList([])
    assert len(sampler) == 0
    # Also test with no argument
    sampler2 = SamplerList()
    assert len(sampler2) == 0


def test_negative_weights_rejected() -> None:
    """Verify negative weights are rejected during construction."""
    from dynamic_random_sampler import SamplerList

    with pytest.raises(ValueError):
        SamplerList([1.0, -1.0])


def test_zero_weight_rejected_in_construction() -> None:
    """Verify zero weights are rejected during construction."""
    from dynamic_random_sampler import SamplerList

    with pytest.raises(ValueError):
        SamplerList([1.0, 0.0])


# =============================================================================
# Item Access Tests (__getitem__, __setitem__) - Single Index
# =============================================================================


def test_getitem_positive_index() -> None:
    """Test getting weight with positive index."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 2.0) < 1e-10
    assert abs(sampler[2] - 3.0) < 1e-10


def test_getitem_negative_index() -> None:
    """Test getting weight with negative index."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    assert abs(sampler[-1] - 3.0) < 1e-10
    assert abs(sampler[-2] - 2.0) < 1e-10
    assert abs(sampler[-3] - 1.0) < 1e-10


def test_getitem_out_of_bounds() -> None:
    """Test getting weight with out of bounds index."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    with pytest.raises(IndexError):
        _ = sampler[3]
    with pytest.raises(IndexError):
        _ = sampler[-4]


def test_setitem_positive_index() -> None:
    """Test setting weight with positive index."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    sampler[1] = 5.0
    assert abs(sampler[1] - 5.0) < 1e-10


def test_setitem_negative_index() -> None:
    """Test setting weight with negative index."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    sampler[-1] = 5.0
    assert abs(sampler[-1] - 5.0) < 1e-10


def test_setitem_to_zero_excludes_from_sampling() -> None:
    """Test setting weight to zero excludes from sampling but keeps element."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    sampler[1] = 0.0
    # Element still exists at index 1
    assert sampler[1] == 0.0
    # Length unchanged
    assert len(sampler) == 3
    # Other elements unchanged
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[2] - 3.0) < 1e-10


def test_setitem_invalid_weight() -> None:
    """Test setting invalid weights raises error."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        sampler[0] = -1.0
    with pytest.raises(ValueError):
        sampler[0] = math.inf
    with pytest.raises(ValueError):
        sampler[0] = math.nan


# =============================================================================
# Slice Tests (__getitem__, __setitem__ with slices)
# =============================================================================


def test_getitem_slice_basic() -> None:
    """Test getting weights with a slice."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sampler[1:4]
    assert len(result) == 3
    assert abs(result[0] - 2.0) < 1e-10
    assert abs(result[1] - 3.0) < 1e-10
    assert abs(result[2] - 4.0) < 1e-10


def test_getitem_slice_negative() -> None:
    """Test getting weights with negative slice indices."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sampler[-3:-1]
    assert len(result) == 2
    assert abs(result[0] - 3.0) < 1e-10
    assert abs(result[1] - 4.0) < 1e-10


def test_getitem_slice_step() -> None:
    """Test getting weights with slice step."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sampler[::2]  # Every other element
    assert len(result) == 3
    assert abs(result[0] - 1.0) < 1e-10
    assert abs(result[1] - 3.0) < 1e-10
    assert abs(result[2] - 5.0) < 1e-10


def test_getitem_slice_empty() -> None:
    """Test getting empty slice."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    result = sampler[1:1]
    assert len(result) == 0


def test_setitem_slice_basic() -> None:
    """Test setting weights with a slice."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])
    sampler[1:4] = [10.0, 20.0, 30.0]
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 10.0) < 1e-10
    assert abs(sampler[2] - 20.0) < 1e-10
    assert abs(sampler[3] - 30.0) < 1e-10
    assert abs(sampler[4] - 5.0) < 1e-10


def test_setitem_slice_different_length() -> None:
    """Test setting slice with different length resizes the list."""
    from dynamic_random_sampler import SamplerList

    # Test shrinking
    sampler: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])
    sampler[1:4] = [10.0, 20.0]  # Replace 3 elements with 2
    assert len(sampler) == 4
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 10.0) < 1e-10
    assert abs(sampler[2] - 20.0) < 1e-10
    assert abs(sampler[3] - 5.0) < 1e-10

    # Test expanding
    sampler2: Any = SamplerList([1.0, 2.0, 3.0])
    sampler2[1:2] = [10.0, 20.0, 30.0]  # Replace 1 element with 3
    assert len(sampler2) == 5
    assert abs(sampler2[0] - 1.0) < 1e-10
    assert abs(sampler2[1] - 10.0) < 1e-10
    assert abs(sampler2[2] - 20.0) < 1e-10
    assert abs(sampler2[3] - 30.0) < 1e-10
    assert abs(sampler2[4] - 3.0) < 1e-10


def test_setitem_extended_slice_wrong_length() -> None:
    """Test extended slice (step != 1) with wrong length raises error."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="attempt to assign sequence"):
        sampler[::2] = [10.0, 20.0]  # Extended slice needs exact length


def test_getitem_slice_negative_step() -> None:
    """Test getting weights with negative slice step (reverse)."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sampler[::-1]  # Reverse
    assert len(result) == 5
    assert abs(result[0] - 5.0) < 1e-10
    assert abs(result[4] - 1.0) < 1e-10


def test_setitem_slice_negative_step() -> None:
    """Test setting weights with negative slice step."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0])
    sampler[4:1:-1] = [50.0, 40.0, 30.0]  # Set indices 4, 3, 2
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 2.0) < 1e-10
    assert abs(sampler[2] - 30.0) < 1e-10
    assert abs(sampler[3] - 40.0) < 1e-10
    assert abs(sampler[4] - 50.0) < 1e-10


def test_getitem_full_slice() -> None:
    """Test getting all elements with full slice."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    result = sampler[:]
    assert len(result) == 3
    assert result == list(sampler)


# =============================================================================
# Contains Tests (__contains__)
# =============================================================================


def test_contains_existing_weight() -> None:
    """Test checking for existing weight."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    assert 1.0 in sampler
    assert 2.0 in sampler
    assert 3.0 in sampler


def test_contains_nonexistent_weight() -> None:
    """Test checking for nonexistent weight."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    assert 5.0 not in sampler


def test_contains_zero_weight() -> None:
    """Test checking for zero weight (element still exists)."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    sampler[0] = 0.0
    # Zero weight element still exists
    assert 0.0 in sampler
    # Original weight is gone
    assert 1.0 not in sampler


# =============================================================================
# Iteration Tests (__iter__)
# =============================================================================


def test_iter_returns_all_weights() -> None:
    """Test iteration returns all weights."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    weights = list(sampler)
    assert len(weights) == 3
    assert abs(weights[0] - 1.0) < 1e-10
    assert abs(weights[1] - 2.0) < 1e-10
    assert abs(weights[2] - 3.0) < 1e-10


def test_iter_includes_zero_weight_elements() -> None:
    """Test iteration includes elements with weight 0."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    sampler[1] = 0.0  # Set to zero but don't delete
    weights = list(sampler)
    assert len(weights) == 3
    assert abs(weights[0] - 1.0) < 1e-10
    assert weights[1] == 0.0
    assert abs(weights[2] - 3.0) < 1e-10


def test_list_conversion() -> None:
    """Test list() works on sampler."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    weights = list(sampler)
    assert len(weights) == 3
    assert abs(weights[0] - 1.0) < 1e-10
    assert abs(weights[1] - 2.0) < 1e-10
    assert abs(weights[2] - 3.0) < 1e-10


# =============================================================================
# Append/Extend Tests
# =============================================================================


def test_append_adds_element() -> None:
    """Test append adds element to end."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0])
    sampler.append(3.0)
    assert len(sampler) == 3
    assert abs(sampler[2] - 3.0) < 1e-10


def test_append_invalid_weight() -> None:
    """Test append with invalid weight raises error."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0])
    with pytest.raises(ValueError):
        sampler.append(0.0)
    with pytest.raises(ValueError):
        sampler.append(-1.0)


def test_extend_adds_multiple() -> None:
    """Test extend adds multiple elements."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0])
    sampler.extend([2.0, 3.0, 4.0])
    assert len(sampler) == 4
    assert abs(sampler[1] - 2.0) < 1e-10
    assert abs(sampler[3] - 4.0) < 1e-10


def test_extend_empty_list() -> None:
    """Test extend with empty list does nothing."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0])
    sampler.extend([])
    assert len(sampler) == 2


# =============================================================================
# Pop/Clear Tests
# =============================================================================


def test_pop_returns_last_weight() -> None:
    """Test pop returns and removes last weight."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    weight = sampler.pop()
    assert abs(weight - 3.0) < 1e-10
    assert len(sampler) == 2


def test_pop_empty_raises() -> None:
    """Test pop on empty sampler raises error."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0])
    sampler.pop()  # Pop the only element
    with pytest.raises(IndexError):
        sampler.pop()


def test_clear_removes_all() -> None:
    """Test clear removes all elements."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    sampler.clear()
    assert len(sampler) == 0


def test_clear_then_append() -> None:
    """Test that append works after clear."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    sampler.clear()
    sampler.append(5.0)
    assert len(sampler) == 1
    assert abs(sampler[0] - 5.0) < 1e-10


# =============================================================================
# Index/Count Tests
# =============================================================================


def test_index_finds_first() -> None:
    """Test index finds first occurrence."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 2.0, 3.0])
    assert sampler.index(2.0) == 1


def test_index_not_found() -> None:
    """Test index raises when not found."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        sampler.index(5.0)


def test_count_existing() -> None:
    """Test count counts occurrences."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 2.0, 2.0, 3.0])
    assert sampler.count(2.0) == 3


def test_count_nonexistent() -> None:
    """Test count returns 0 for nonexistent."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    assert sampler.count(5.0) == 0


# =============================================================================
# Pop Efficiency Tests
# =============================================================================


def test_multiple_pops_work_correctly() -> None:
    """Test that multiple pops work correctly."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([float(i) for i in range(1, 101)])
    assert len(sampler) == 100

    # Pop 50 elements from the end
    for i in range(50):
        weight = sampler.pop()
        expected = float(100 - i)
        assert abs(weight - expected) < 1e-10

    assert len(sampler) == 50
    # Verify remaining elements are correct
    for i in range(50):
        assert abs(sampler[i] - float(i + 1)) < 1e-10


def test_pop_and_append_cycle() -> None:
    """Test that pop and append can be interleaved."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])

    # Pop last
    weight = sampler.pop()
    assert abs(weight - 3.0) < 1e-10
    assert len(sampler) == 2

    # Append new
    sampler.append(4.0)
    assert len(sampler) == 3
    assert abs(sampler[2] - 4.0) < 1e-10

    # Pop again
    weight = sampler.pop()
    assert abs(weight - 4.0) < 1e-10
    assert len(sampler) == 2

    # Verify remaining
    assert list(sampler) == [1.0, 2.0]
