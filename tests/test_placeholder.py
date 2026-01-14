"""Placeholder tests - will be expanded with Hypothesis property-based tests."""

from typing import Any

import pytest


def test_import_works() -> None:
    """Verify the Rust module can be imported."""
    from dynamic_random_sampler import SamplerList

    assert SamplerList is not None


def test_basic_construction() -> None:
    """Verify basic sampler construction works."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    assert len(sampler) == 3


def test_weight_retrieval() -> None:
    """Verify weights can be retrieved via indexing."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 4.0])
    assert abs(sampler[0] - 1.0) < 1e-10
    assert abs(sampler[1] - 2.0) < 1e-10
    assert abs(sampler[2] - 4.0) < 1e-10


def test_sample_returns_valid_index() -> None:
    """Verify sample returns a valid index."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    for _ in range(100):
        idx: int = sampler.sample()
        assert 0 <= idx < 3


def test_empty_weights_rejected() -> None:
    """Verify empty weight list is rejected."""
    from dynamic_random_sampler import SamplerList

    with pytest.raises(ValueError):
        SamplerList([])


def test_negative_weights_rejected() -> None:
    """Verify negative weights are rejected."""
    from dynamic_random_sampler import SamplerList

    with pytest.raises(ValueError):
        SamplerList([1.0, -1.0])


def test_zero_weight_rejected() -> None:
    """Verify zero weights are rejected."""
    from dynamic_random_sampler import SamplerList

    with pytest.raises(ValueError):
        SamplerList([1.0, 0.0])
