"""Likelihood-based statistical tests for the dynamic random sampler.

This module contains a Hypothesis-based property test that verifies the sampler
produces correct probability distributions when weights are dynamically updated.

The test uses a likelihood ratio approach: if the sampler is correct, the total
log-likelihood of observed samples should be normally distributed with known
mean and variance. A two-tailed test checks if the observed likelihood is
too extreme to be explained by correct sampling.
"""

import hypothesis.strategies as st
import pytest
from hypothesis import given, note, settings

# Strategy for generating valid positive weights
positive_weights = st.floats(
    min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False
)


@pytest.mark.slow
@given(
    initial_weights=st.lists(positive_weights, min_size=1, max_size=50).filter(
        lambda ws: any(w > 0 for w in ws)
    ),
    num_samples=st.integers(min_value=100, max_value=10_000),
    num_assignments=st.integers(min_value=0, max_value=100),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
@settings(max_examples=100, deadline=None)
def test_likelihood_statistical_test(
    initial_weights: list[float],
    num_samples: int,
    num_assignments: int,
    seed: int,
    data: st.DataObject,
) -> None:
    """Test that the sampler produces correct distributions with dynamic updates.

    This test:
    1. Draws initial non-empty weights (not all zero)
    2. Draws N samples (100 to 10k)
    3. Draws 0 to 100 assignments
    4. Each assignment specifies (sample_index, weight_index, new_weight)
       where sample_index < N, weight_index <= 10 * len(weights), and
       new_weight is a valid non-negative weight
    5. Runs the likelihood-based statistical test
    6. Fails if p-value < 1e-6
    """
    from dynamic_random_sampler import _likelihood_test

    # Generate assignments
    max_weight_index = 10 * len(initial_weights)
    assignments: list[tuple[int, int, float]] = []

    for _ in range(num_assignments):
        sample_index = data.draw(
            st.integers(min_value=0, max_value=num_samples - 1),
            label="sample_index"
        )
        weight_index = data.draw(
            st.integers(min_value=0, max_value=max_weight_index),
            label="weight_index"
        )
        # Weight can be zero (to exclude) or positive
        new_weight = data.draw(
            st.floats(
                min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            label="new_weight",
        )
        assignments.append((sample_index, weight_index, new_weight))

    note(f"Initial weights: {initial_weights[:5]}... ({len(initial_weights)} total)")
    note(f"Num samples: {num_samples}")
    note(f"Num assignments: {len(assignments)}")
    note(f"Seed: {seed}")

    # Run the likelihood test
    # Skip if assignments cause all weights to become zero (not a valid test scenario)
    try:
        result = _likelihood_test(initial_weights, num_samples, assignments, seed=seed)
    except ValueError as e:
        if "all weights became zero" in str(e):
            from hypothesis import assume

            assume(False)  # Skip this test case
        raise

    note(f"Result: z={result.z_score:.4f}, p={result.p_value:.6f}")
    note(f"Observed LL: {result.observed_log_likelihood:.4f}")
    note(f"Expected LL: {result.expected_log_likelihood:.4f}")
    note(f"Variance: {result.variance:.4f}")

    # The test should pass - p-value should be > 1e-6
    # With max_examples=100 and alpha=1e-6, we expect < 0.0001 false positives per run
    assert result.passes(1e-6), (
        f"Likelihood test failed: z={result.z_score:.4f}, p={result.p_value:.10f}\n"
        f"Observed LL: {result.observed_log_likelihood:.4f}\n"
        f"Expected LL: {result.expected_log_likelihood:.4f}\n"
        f"Variance: {result.variance:.4f}\n"
        f"This indicates the sampler may not be producing correct distributions.\n"
        f"Seed: {seed}, initial_weights: {initial_weights[:5]}..., "
        f"num_assignments: {len(assignments)}"
    )


@pytest.mark.slow
@given(
    weights=st.lists(
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_likelihood_no_assignments(weights: list[float], seed: int) -> None:
    """Test likelihood with no dynamic updates (baseline correctness)."""
    from dynamic_random_sampler import _likelihood_test

    # With no assignments, this is a simpler test
    result = _likelihood_test(weights, num_samples=1000, assignments=[], seed=seed)

    note(f"Weights: {weights}")
    note(f"z={result.z_score:.4f}, p={result.p_value:.6f}")

    assert result.passes(1e-6), (
        f"Baseline likelihood test failed: "
        f"z={result.z_score:.4f}, p={result.p_value:.10f}"
    )


@pytest.mark.slow
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_likelihood_with_weight_growth(seed: int) -> None:
    """Test that weights can grow beyond initial array size."""
    from dynamic_random_sampler import _likelihood_test

    # Start with just 2 weights
    initial_weights = [1.0, 1.0]

    # Add assignments that extend the weight array
    assignments: list[tuple[int, int, float]] = [
        (0, 5, 2.0),   # Before sample 0, set weight[5] = 2.0 (extends array)
        (50, 10, 3.0), # Before sample 50, set weight[10] = 3.0
        (75, 2, 0.5),  # Before sample 75, modify weight[2]
    ]

    result = _likelihood_test(
        initial_weights, num_samples=100, assignments=assignments, seed=seed
    )

    note(f"z={result.z_score:.4f}, p={result.p_value:.6f}")

    assert result.passes(1e-6), (
        f"Weight growth test failed: z={result.z_score:.4f}, p={result.p_value:.10f}"
    )


def test_likelihood_test_validation() -> None:
    """Test that the likelihood test validates its inputs correctly."""
    from dynamic_random_sampler import _likelihood_test

    # Test num_samples < 100
    with pytest.raises(ValueError, match="num_samples must be at least 100"):
        _likelihood_test([1.0], num_samples=50, assignments=[])

    # Test empty initial_weights
    with pytest.raises(ValueError, match="initial_weights cannot be empty"):
        _likelihood_test([], num_samples=100, assignments=[])

    # Test assignment sample_index >= num_samples
    with pytest.raises(ValueError, match=r"sample_index .* is >= num_samples"):
        _likelihood_test([1.0], num_samples=100, assignments=[(100, 0, 1.0)])

    # Test assignment with negative weight
    with pytest.raises(ValueError, match="weight must be non-negative"):
        _likelihood_test([1.0], num_samples=100, assignments=[(0, 0, -1.0)])

    # Test all weights becoming zero
    with pytest.raises(ValueError, match="all weights became zero"):
        _likelihood_test([1.0], num_samples=100, assignments=[(0, 0, 0.0)])


def test_likelihood_result_passes_method() -> None:
    """Test the passes() method of the result."""
    from dynamic_random_sampler import _likelihood_test

    result = _likelihood_test([1.0, 2.0, 3.0], num_samples=100, assignments=[], seed=42)

    # With reasonable weights and seed, p-value should be reasonable
    # The passes method should work correctly
    if result.p_value > 0.5:
        assert result.passes(0.5)
    if result.p_value > 0.01:
        assert result.passes(0.01)

    # Always check this - a properly working sampler should have p > 1e-10
    # (with very high probability)
    assert result.passes(1e-10) or result.p_value > 1e-10, (
        f"Suspiciously low p-value: {result.p_value}"
    )
