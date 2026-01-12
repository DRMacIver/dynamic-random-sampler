"""Tests for Python list-like operations on DynamicSampler."""

import math
from typing import Any

import pytest

# =============================================================================
# Item Access Tests (__getitem__, __setitem__, __delitem__)
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


def test_setitem_to_zero_deletes() -> None:
    """Test setting weight to zero deletes the element."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    sampler[1] = 0.0
    assert sampler[1] == 0.0
    assert sampler.is_deleted(1)


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
    """Test deleting with positive index."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[1]
    assert sampler.is_deleted(1)
    assert sampler[1] == 0.0


def test_delitem_negative_index() -> None:
    """Test deleting with negative index."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[-1]
    assert sampler.is_deleted(2)


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
    """Test checking for deleted weight."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[0]
    assert 1.0 not in sampler


# =============================================================================
# Iteration Tests (__iter__, to_list)
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


def test_iter_includes_deleted_as_zero() -> None:
    """Test iteration includes deleted elements as 0.0."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[1]
    weights = list(sampler)
    assert len(weights) == 3
    assert abs(weights[0] - 1.0) < 1e-10
    assert weights[1] == 0.0
    assert abs(weights[2] - 3.0) < 1e-10


def test_to_list_same_as_iter() -> None:
    """Test to_list returns same as list()."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    assert sampler.to_list() == list(sampler)


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
    """Test pop returns and deletes last weight."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    weight = sampler.pop()
    assert abs(weight - 3.0) < 1e-10
    assert sampler.is_deleted(2)


def test_pop_skips_deleted() -> None:
    """Test pop skips deleted elements."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[2]  # Delete last
    weight = sampler.pop()  # Should pop index 1
    assert abs(weight - 2.0) < 1e-10
    assert sampler.is_deleted(1)


def test_pop_empty_raises() -> None:
    """Test pop on empty sampler raises error."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0])
    sampler.pop()  # Pop the only element
    with pytest.raises(IndexError):
        sampler.pop()


def test_clear_deletes_all() -> None:
    """Test clear deletes all elements."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    sampler.clear()
    assert sampler.active_count() == 0
    for i in range(3):
        assert sampler.is_deleted(i)


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


def test_index_ignores_deleted() -> None:
    """Test index ignores deleted elements."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    del sampler[0]
    with pytest.raises(ValueError):
        sampler.index(1.0)


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


def test_count_ignores_deleted() -> None:
    """Test count ignores deleted elements."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 2.0, 3.0])
    del sampler[1]
    assert sampler.count(2.0) == 1
