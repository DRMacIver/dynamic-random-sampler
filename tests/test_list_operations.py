"""Tests for Python list-like operations on DynamicSampler."""

import math
from typing import Any

import pytest

# =============================================================================
# Item Access Tests (__getitem__, __setitem__, __delitem__) - Single Index
# =============================================================================


def test_getitem_positive_index() -> None:
    """Test getting weight with positive index."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 2.0) < 1e-10
    assert abs(sampler[2] - 3.0) < 1e-10


def test_getitem_negative_index() -> None:
    """Test getting weight with negative index."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    assert abs(sampler[-1] - 3.0) < 1e-10
    assert abs(sampler[-2] - 2.0) < 1e-10
    assert abs(sampler[-3] - 1.0) < 1e-10


def test_getitem_out_of_bounds() -> None:
    """Test getting weight with out of bounds index."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    with pytest.raises(IndexError):
        _ = sampler[3]
    with pytest.raises(IndexError):
        _ = sampler[-4]


def test_setitem_positive_index() -> None:
    """Test setting weight with positive index."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    sampler[1] = 5.0
    assert abs(sampler[1] - 5.0) < 1e-10


def test_setitem_negative_index() -> None:
    """Test setting weight with negative index."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    sampler[-1] = 5.0
    assert abs(sampler[-1] - 5.0) < 1e-10


def test_setitem_to_zero_excludes_from_sampling() -> None:
    """Test setting weight to zero excludes from sampling but keeps element."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
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
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        sampler[0] = -1.0
    with pytest.raises(ValueError):
        sampler[0] = math.inf
    with pytest.raises(ValueError):
        sampler[0] = math.nan


def test_delitem_positive_index() -> None:
    """Test deleting with positive index removes element."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[1]
    # Length decreases
    assert len(sampler) == 2
    # Elements shift
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 3.0) < 1e-10  # Was at index 2


def test_delitem_negative_index() -> None:
    """Test deleting with negative index removes element."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[-1]
    # Length decreases
    assert len(sampler) == 2
    # Last element gone
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 2.0) < 1e-10


# =============================================================================
# Slice Tests (__getitem__, __setitem__, __delitem__ with slices)
# =============================================================================


def test_getitem_slice_basic() -> None:
    """Test getting weights with a slice."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sampler[1:4]
    assert len(result) == 3
    assert abs(result[0] - 2.0) < 1e-10
    assert abs(result[1] - 3.0) < 1e-10
    assert abs(result[2] - 4.0) < 1e-10


def test_getitem_slice_negative() -> None:
    """Test getting weights with negative slice indices."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sampler[-3:-1]
    assert len(result) == 2
    assert abs(result[0] - 3.0) < 1e-10
    assert abs(result[1] - 4.0) < 1e-10


def test_getitem_slice_step() -> None:
    """Test getting weights with slice step."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sampler[::2]  # Every other element
    assert len(result) == 3
    assert abs(result[0] - 1.0) < 1e-10
    assert abs(result[1] - 3.0) < 1e-10
    assert abs(result[2] - 5.0) < 1e-10


def test_getitem_slice_empty() -> None:
    """Test getting empty slice."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    result = sampler[1:1]
    assert len(result) == 0


def test_setitem_slice_basic() -> None:
    """Test setting weights with a slice."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0, 4.0, 5.0])
    sampler[1:4] = [10.0, 20.0, 30.0]
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 10.0) < 1e-10
    assert abs(sampler[2] - 20.0) < 1e-10
    assert abs(sampler[3] - 30.0) < 1e-10
    assert abs(sampler[4] - 5.0) < 1e-10


def test_setitem_slice_wrong_length() -> None:
    """Test setting slice with wrong length raises error."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="attempt to assign sequence"):
        sampler[1:4] = [10.0, 20.0]  # Wrong length


def test_delitem_slice_basic() -> None:
    """Test deleting elements with a slice."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0, 4.0, 5.0])
    del sampler[1:4]
    assert len(sampler) == 2
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 5.0) < 1e-10


def test_delitem_slice_step() -> None:
    """Test deleting elements with slice step."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0, 4.0, 5.0])
    del sampler[::2]  # Delete elements at indices 0, 2, 4
    assert len(sampler) == 2
    assert abs(sampler[0] - 2.0) < 1e-10
    assert abs(sampler[1] - 4.0) < 1e-10


# =============================================================================
# Contains Tests (__contains__)
# =============================================================================


def test_contains_existing_weight() -> None:
    """Test checking for existing weight."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    assert 1.0 in sampler
    assert 2.0 in sampler
    assert 3.0 in sampler


def test_contains_nonexistent_weight() -> None:
    """Test checking for nonexistent weight."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    assert 5.0 not in sampler


def test_contains_deleted_weight() -> None:
    """Test checking for deleted weight (element removed)."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[0]
    assert 1.0 not in sampler


# =============================================================================
# Iteration Tests (__iter__)
# =============================================================================


def test_iter_returns_all_weights() -> None:
    """Test iteration returns all weights."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    weights = list(sampler)
    assert len(weights) == 3
    assert abs(weights[0] - 1.0) < 1e-10
    assert abs(weights[1] - 2.0) < 1e-10
    assert abs(weights[2] - 3.0) < 1e-10


def test_iter_excludes_deleted_elements() -> None:
    """Test iteration excludes deleted elements."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[1]
    weights = list(sampler)
    # Only 2 elements remain after deletion
    assert len(weights) == 2
    assert abs(weights[0] - 1.0) < 1e-10
    assert abs(weights[1] - 3.0) < 1e-10


def test_iter_includes_zero_weight_elements() -> None:
    """Test iteration includes elements with weight 0."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    sampler[1] = 0.0  # Set to zero but don't delete
    weights = list(sampler)
    assert len(weights) == 3
    assert abs(weights[0] - 1.0) < 1e-10
    assert weights[1] == 0.0
    assert abs(weights[2] - 3.0) < 1e-10


def test_list_conversion() -> None:
    """Test list() works on sampler."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
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
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0])
    sampler.append(3.0)
    assert len(sampler) == 3
    assert abs(sampler[2] - 3.0) < 1e-10


def test_append_invalid_weight() -> None:
    """Test append with invalid weight raises error."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0])
    with pytest.raises(ValueError):
        sampler.append(0.0)
    with pytest.raises(ValueError):
        sampler.append(-1.0)


def test_extend_adds_multiple() -> None:
    """Test extend adds multiple elements."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0])
    sampler.extend([2.0, 3.0, 4.0])
    assert len(sampler) == 4
    assert abs(sampler[1] - 2.0) < 1e-10
    assert abs(sampler[3] - 4.0) < 1e-10


def test_extend_empty_list() -> None:
    """Test extend with empty list does nothing."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0])
    sampler.extend([])
    assert len(sampler) == 2


# =============================================================================
# Pop/Clear Tests
# =============================================================================


def test_pop_returns_last_weight() -> None:
    """Test pop returns and removes last weight."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    weight = sampler.pop()
    assert abs(weight - 3.0) < 1e-10
    assert len(sampler) == 2


def test_pop_after_delete() -> None:
    """Test pop works after delete."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[2]  # Delete last
    # Now only [1.0, 2.0] remain
    weight = sampler.pop()  # Should pop 2.0
    assert abs(weight - 2.0) < 1e-10
    assert len(sampler) == 1


def test_pop_empty_raises() -> None:
    """Test pop on empty sampler raises error."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0])
    sampler.pop()  # Pop the only element
    with pytest.raises(IndexError):
        sampler.pop()


def test_clear_removes_all() -> None:
    """Test clear removes all elements."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    sampler.clear()
    assert len(sampler) == 0


# =============================================================================
# Index/Count Tests
# =============================================================================


def test_index_finds_first() -> None:
    """Test index finds first occurrence."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 2.0, 3.0])
    assert sampler.index(2.0) == 1


def test_index_not_found() -> None:
    """Test index raises when not found."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        sampler.index(5.0)


def test_index_after_delete() -> None:
    """Test index works after delete."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[0]  # Remove 1.0
    # Now [2.0, 3.0]
    with pytest.raises(ValueError):
        sampler.index(1.0)  # 1.0 no longer exists
    assert sampler.index(2.0) == 0  # 2.0 is now at index 0


def test_count_existing() -> None:
    """Test count counts occurrences."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 2.0, 2.0, 3.0])
    assert sampler.count(2.0) == 3


def test_count_nonexistent() -> None:
    """Test count returns 0 for nonexistent."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    assert sampler.count(5.0) == 0


def test_count_after_delete() -> None:
    """Test count works after delete."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 2.0, 3.0])
    del sampler[1]  # Remove first 2.0
    # Now [1.0, 2.0, 3.0]
    assert sampler.count(2.0) == 1


# =============================================================================
# Remove Tests
# =============================================================================


def test_remove_existing() -> None:
    """Test remove removes first occurrence."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 2.0, 3.0])
    sampler.remove(2.0)
    assert len(sampler) == 3
    assert sampler.count(2.0) == 1


def test_remove_nonexistent() -> None:
    """Test remove raises for nonexistent weight."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        sampler.remove(5.0)
