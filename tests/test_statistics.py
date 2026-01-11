"""Statistical tests for verifying sampling distribution correctness."""

from typing import Any

import pytest


def test_chi_squared_result_attributes() -> None:
    """Verify ChiSquaredResult has expected attributes."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    result = sampler.test_distribution(1000)

    assert hasattr(result, "chi_squared")
    assert hasattr(result, "degrees_of_freedom")
    assert hasattr(result, "p_value")
    assert hasattr(result, "num_samples")
    assert hasattr(result, "passes")


def test_chi_squared_result_values() -> None:
    """Verify ChiSquaredResult contains valid values."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0, 3.0])
    result = sampler.test_distribution(1000)

    assert result.chi_squared >= 0.0
    assert result.degrees_of_freedom == 2  # 3 elements - 1
    assert 0.0 <= result.p_value <= 1.0
    assert result.num_samples == 1000


def test_chi_squared_passes_for_correct_distribution() -> None:
    """Verify chi-squared test passes for correct distribution.

    With 10000 samples, the sampler should produce a distribution that
    passes the chi-squared test at the 0.01 significance level.
    """
    from dynamic_random_sampler import DynamicSampler

    # Test with various weight distributions
    test_cases = [
        [1.0, 1.0, 1.0],  # Uniform
        [1.0, 2.0, 3.0],  # Linear
        [1.0, 4.0, 16.0],  # Geometric
        [0.1, 0.9],  # Skewed
    ]

    for weights in test_cases:
        sampler: Any = DynamicSampler(weights)
        result = sampler.test_distribution(10000)

        # Test should pass at 0.01 level (99% confidence)
        assert result.passes(0.01), (
            f"Chi-squared test failed for weights {weights}: "
            f"chi2={result.chi_squared:.2f}, p={result.p_value:.4f}"
        )


def test_chi_squared_repr() -> None:
    """Verify ChiSquaredResult has a useful repr."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0])
    result = sampler.test_distribution(100)

    repr_str = repr(result)
    assert "ChiSquaredResult" in repr_str
    assert "chi_squared" in repr_str
    assert "p_value" in repr_str


def test_chi_squared_default_num_samples() -> None:
    """Verify default num_samples is 10000."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0])
    result = sampler.test_distribution()

    assert result.num_samples == 10000


def test_chi_squared_custom_num_samples() -> None:
    """Verify custom num_samples works."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler([1.0, 2.0])

    for n in [100, 500, 5000]:
        result = sampler.test_distribution(n)
        assert result.num_samples == n


def test_chi_squared_import() -> None:
    """Verify ChiSquaredResult can be imported."""
    from dynamic_random_sampler import (
        ChiSquaredResult,  # pyright: ignore[reportAttributeAccessIssue]
    )

    assert ChiSquaredResult is not None


@pytest.mark.slow
def test_chi_squared_high_confidence() -> None:
    """High-cost test with large sample size for strong confidence.

    This test uses 100000 samples and should be very unlikely to fail
    if the distribution is correct (< 0.1% chance of false negative).
    """
    from dynamic_random_sampler import DynamicSampler

    weights = [1.0, 2.0, 4.0, 8.0, 16.0]
    sampler: Any = DynamicSampler(weights)
    result = sampler.test_distribution(100000)

    # With 100k samples, even small deviations would be detected
    # Test at 0.001 level (99.9% confidence)
    assert result.passes(0.001), (
        f"High-confidence chi-squared test failed: "
        f"chi2={result.chi_squared:.2f}, df={result.degrees_of_freedom}, "
        f"p={result.p_value:.6f}"
    )
