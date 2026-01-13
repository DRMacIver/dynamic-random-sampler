"""Hypothesis-based property tests for the dynamic random sampler.

This module contains extensive property-based tests using Hypothesis,
including rule-based stateful testing for the dynamic update functionality.
"""

from collections import Counter
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, note, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

# -----------------------------------------------------------------------------
# Basic Property Tests
# -----------------------------------------------------------------------------


@given(st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=1, max_size=100))
def test_construction_with_positive_weights(weights: list[float]) -> None:
    """Any list of positive weights should construct successfully."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler(weights)
    assert len(sampler) == len(weights)


@given(st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=1, max_size=100))
def test_weights_are_preserved(weights: list[float]) -> None:
    """Weights should be retrievable after construction."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler(weights)
    for i, expected in enumerate(weights):
        actual = sampler[i]
        # Allow for floating point imprecision from log2/exp2 conversion
        assert abs(actual - expected) / max(expected, 1e-10) < 1e-10


@given(st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=1, max_size=50))
@settings(max_examples=50)
def test_sample_returns_valid_indices(weights: list[float]) -> None:
    """Sample should always return a valid index."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler(weights)
    for _ in range(100):
        idx = sampler.sample()
        assert 0 <= idx < len(weights)


@given(
    st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=1, max_size=20),
    st.integers(min_value=0, max_value=19),
    st.floats(min_value=0.01, max_value=1e6),
)
def test_update_changes_weight(
    weights: list[float], index: int, new_weight: float
) -> None:
    """Updating a weight should change the stored weight."""
    from dynamic_random_sampler import DynamicSampler

    assume(index < len(weights))

    sampler: Any = DynamicSampler(weights)
    sampler[index] = new_weight
    actual = sampler[index]
    assert abs(actual - new_weight) / max(new_weight, 1e-10) < 1e-10


@given(
    st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=2, max_size=20),
    st.integers(min_value=0, max_value=19),
    st.floats(min_value=0.01, max_value=1e6),
)
def test_update_preserves_other_weights(
    weights: list[float], index: int, new_weight: float
) -> None:
    """Updating one weight should not affect other weights."""
    from dynamic_random_sampler import DynamicSampler

    assume(index < len(weights))

    sampler: Any = DynamicSampler(weights)
    sampler[index] = new_weight

    for i, expected in enumerate(weights):
        if i != index:
            actual = sampler[i]
            assert abs(actual - expected) / max(expected, 1e-10) < 1e-10


# -----------------------------------------------------------------------------
# Rule-Based Stateful Testing
# -----------------------------------------------------------------------------

indices = st.runner().flatmap(
    lambda self: st.integers(min_value=0, max_value=len(self.weights) - 1)
    if self.weights
    else st.nothing()
)


class DynamicSamplerStateMachine(RuleBasedStateMachine):
    """Stateful test machine for the DynamicSampler.

    This machine performs many random operations on a sampler and verifies
    invariants hold throughout. At the end, it checks statistical conformance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sampler: Any = None
        self.weights: list[float] = []
        self.sample_counts: Counter[int] = Counter()
        self.total_samples: int = 0
        self.chi_squared_seed: int = 0

    @initialize(
        weights=st.lists(
            st.floats(min_value=0.1, max_value=100.0), min_size=2, max_size=20
        ),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def init_sampler(self, weights: list[float], seed: int) -> None:
        """Initialize the sampler with random weights."""
        from dynamic_random_sampler import DynamicSampler

        self.sampler = DynamicSampler(weights)
        self.weights = list(weights)
        self.sample_counts = Counter()
        self.total_samples = 0
        self.chi_squared_seed = seed
        note(f"Initialized with {len(weights)} weights: {weights[:5]}...")
        note(f"Chi-squared seed: {seed}")

    @rule(
        n_to_add=st.integers(1, 100),
        weight=st.floats(min_value=0.1, max_value=1e6),
    )
    @precondition(lambda self: self.sampler is not None)
    def add_many_weights(self, n_to_add: int, weight: float) -> None:
        """Add multiple weights with the same value."""
        for _ in range(n_to_add):
            self.sampler.append(weight)
        self.weights.extend([weight] * n_to_add)
        note(f"Added {n_to_add} weights with value {weight:.2e}")

    @rule(index=indices, new_weight=st.floats(min_value=0.1, max_value=100.0))
    @precondition(lambda self: self.sampler is not None and len(self.weights) > 0)
    def update_weight(self, index: int, new_weight: float) -> None:
        """Update a weight to a new positive value."""
        old_weight = self.weights[index]
        self.sampler[index] = new_weight
        self.weights[index] = new_weight
        note(f"Updated index {index}: {old_weight:.2f} -> {new_weight:.2f}")

    @rule(count=st.integers(min_value=1, max_value=100))
    @precondition(lambda self: self.sampler is not None)
    def take_samples(self, count: int) -> None:
        """Take multiple samples and record them."""
        for _ in range(count):
            idx = self.sampler.sample()
            self.sample_counts[idx] += 1
            self.total_samples += 1

    @rule()
    @precondition(lambda self: self.sampler is not None)
    def take_single_sample(self) -> None:
        """Take a single sample."""
        idx = self.sampler.sample()
        self.sample_counts[idx] += 1
        self.total_samples += 1

    @rule(
        index=st.integers(min_value=0, max_value=100),
        factor=st.floats(min_value=0.1, max_value=10.0),
    )
    @precondition(lambda self: self.sampler is not None and len(self.weights) > 0)
    def scale_weight(self, index: int, factor: float) -> None:
        """Scale a weight by a factor."""
        index = index % len(self.weights)
        new_weight = max(0.1, self.weights[index] * factor)
        self.sampler[index] = new_weight
        self.weights[index] = new_weight
        note(f"Scaled index {index} by {factor:.2f}")

    @rule()
    @precondition(lambda self: self.sampler is not None and len(self.weights) > 0)
    def make_one_dominant(self) -> None:
        """Make one element have much higher weight than others."""
        import random

        dominant_idx = random.randrange(len(self.weights))
        total_others = sum(w for i, w in enumerate(self.weights) if i != dominant_idx)
        new_weight = total_others * 100  # 100x all others combined
        self.sampler[dominant_idx] = new_weight
        self.weights[dominant_idx] = new_weight
        note(f"Made index {dominant_idx} dominant with weight {new_weight:.2f}")

    @rule()
    @precondition(lambda self: self.sampler is not None and len(self.weights) > 0)
    def equalize_weights(self) -> None:
        """Set all weights to be equal."""
        equal_weight = 1.0
        for i in range(len(self.weights)):
            self.sampler[i] = equal_weight
            self.weights[i] = equal_weight
        note("Equalized all weights to 1.0")

    @rule(index=st.integers(min_value=0, max_value=100))
    @precondition(lambda self: self.sampler is not None and len(self.weights) > 1)
    def effectively_remove_element(self, index: int) -> None:
        """Set a weight to effectively zero (simulating removal).

        We use 1e-100 as a stand-in for zero since the sampler currently
        requires positive weights. This tests that near-zero weights are
        effectively never sampled.
        """
        index = index % len(self.weights)
        # Only remove if we have at least one reasonable weight left
        reasonable_weights = sum(1 for w in self.weights if w >= 0.01)
        if reasonable_weights <= 1 and self.weights[index] >= 0.01:
            # Don't remove the last reasonable weight
            return

        near_zero = 1e-100
        self.sampler[index] = near_zero
        self.weights[index] = near_zero
        note(f"Effectively removed index {index} (set to {near_zero})")

    @rule(index=st.integers(min_value=0, max_value=100))
    @precondition(lambda self: self.sampler is not None and len(self.weights) > 0)
    def restore_removed_element(self, index: int) -> None:
        """Restore an element that was effectively removed."""
        index = index % len(self.weights)
        if self.weights[index] < 0.01:
            new_weight = 1.0
            self.sampler[index] = new_weight
            self.weights[index] = new_weight
            note(f"Restored index {index} to weight {new_weight}")

    # -------------------------------------------------------------------------
    # Invariants - checked after every operation
    # -------------------------------------------------------------------------

    @invariant()
    def length_matches(self) -> None:
        """Sampler length should always match our tracked weights."""
        if self.sampler is not None:
            assert len(self.sampler) == len(self.weights)

    @invariant()
    def weights_are_positive(self) -> None:
        """All tracked weights should be positive."""
        for w in self.weights:
            assert w > 0, f"Found non-positive weight: {w}"

    @invariant()
    def sample_returns_valid_index(self) -> None:
        """Any sample should return a valid index."""
        if self.sampler is not None and len(self.weights) > 0:
            idx = self.sampler.sample()
            assert 0 <= idx < len(self.weights), f"Invalid sample index: {idx}"

    @invariant()
    def never_samples_effectively_removed(self) -> None:
        """We should never sample elements with absurdly low weights.

        When there are elements with reasonable weights (>= 0.01), we should
        effectively never sample elements with weight < 1e-50. The probability
        is so astronomically low it should never happen in practice.
        """
        if self.sampler is None or len(self.weights) == 0:
            return

        # Check if we have any reasonable weights
        max_weight = max(self.weights)
        if max_weight < 0.01:
            return  # All weights are tiny, sampling any is fine

        # Take a sample and verify it's not from an effectively-removed element
        idx = self.sampler.sample()
        sampled_weight = self.weights[idx]

        # If max_weight is reasonable and sampled weight is absurdly low, fail
        # The threshold of 1e-50 is still incredibly unlikely but gives margin
        assert sampled_weight >= 1e-50, (
            f"Sampled index {idx} with weight {sampled_weight} when max weight "
            f"is {max_weight}. This should be astronomically unlikely!"
        )

    @invariant()
    def weights_match_sampler(self) -> None:
        """Our tracked weights should match the sampler's weights."""
        if self.sampler is not None:
            for i, expected in enumerate(self.weights):
                actual = self.sampler[i]
                rel_error = abs(actual - expected) / max(expected, 1e-10)
                assert rel_error < 1e-9, (
                    f"Weight mismatch at {i}: expected {expected}, got {actual}"
                )

    def teardown(self) -> None:
        """Run statistical conformance check at end of test.

        This tests that the sampler in its CURRENT state produces correct
        distributions. We take fresh samples after all mutations are done.
        """
        if self.sampler is None:
            return

        note(f"Total samples taken during test: {self.total_samples}")
        note(f"Final weights: {self.weights}")

        # Run chi-squared test on the final state with fresh samples
        # Use enough samples for statistical power without being too slow
        # Pass seed for reproducibility (allows Hypothesis to shrink failing cases)
        result = self.sampler.test_distribution(10000, seed=self.chi_squared_seed)
        note(
            f"Chi-squared test: chi2={result.chi_squared:.2f}, p={result.p_value:.4f}, "
            f"seed={self.chi_squared_seed}, excluded={result.excluded_count}, "
            f"unexpected={result.unexpected_samples}"
        )

        # The test should pass at a reasonable significance level
        # With max_examples=100 and alpha=0.001, we expect ~0.1 false failures per run
        # The chi-squared test properly handles small weights:
        # - Indices with expected >= 5 are included in chi-squared
        # - Indices with expected 0.001-5 are excluded (low but possible)
        # - Indices with expected < 0.001 are excluded AND must have 0 samples
        assert result.passes(0.001), (
            f"Statistical conformance failed: chi2={result.chi_squared:.2f}, "
            f"p_value={result.p_value:.6f}, seed={self.chi_squared_seed}, "
            f"excluded={result.excluded_count}, "
            f"unexpected={result.unexpected_samples}. "
            f"Final weights: {self.weights}"
        )


# Create the test class that pytest will discover
@pytest.mark.slow
class TestDynamicSamplerStateful(DynamicSamplerStateMachine.TestCase):  # pyright: ignore[reportUntypedBaseClass]
    """Stateful test class - slow due to comprehensive coverage."""

    pass


TestDynamicSamplerStateful.settings = settings(
    max_examples=100, stateful_step_count=50, deadline=None
)


# -----------------------------------------------------------------------------
# Additional Sampling Property Tests
# -----------------------------------------------------------------------------


@given(st.data())
@settings(max_examples=20)
def test_dominant_weight_gets_most_samples(data: st.DataObject) -> None:
    """An element with vastly higher weight should get almost all samples."""
    from dynamic_random_sampler import DynamicSampler

    n = data.draw(st.integers(min_value=2, max_value=10))
    dominant_idx = data.draw(st.integers(min_value=0, max_value=n - 1))

    # Create weights where one is 10000x the others (very dominant)
    weights = [1.0] * n
    weights[dominant_idx] = 10000.0

    sampler: Any = DynamicSampler(weights)

    # Take many samples
    counts: Counter[int] = Counter()
    num_samples = 1000
    for _ in range(num_samples):
        counts[sampler.sample()] += 1

    # The dominant element should have gotten > 98% of samples
    # (allowing some margin for statistical variation)
    dominant_fraction = counts[dominant_idx] / num_samples
    assert dominant_fraction > 0.98, (
        f"Dominant element only got {dominant_fraction:.1%} of samples (expected >98%)"
    )


@given(st.lists(st.floats(min_value=1.0, max_value=10.0), min_size=2, max_size=10))
@settings(max_examples=30)
def test_all_elements_can_be_sampled(weights: list[float]) -> None:
    """With similar weights, all elements should eventually be sampled."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler(weights)

    sampled: set[int] = set()
    # With similar weights, 1000 samples should hit all elements
    for _ in range(1000):
        sampled.add(sampler.sample())
        if len(sampled) == len(weights):
            break

    assert len(sampled) == len(weights), (
        f"After 1000 samples, only {len(sampled)}/{len(weights)} elements "
        f"were sampled. Weights: {weights}"
    )


@given(
    st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=2, max_size=5),
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=4),
            st.floats(min_value=0.1, max_value=100.0),
        ),
        min_size=1,
        max_size=20,
    ),
)
@settings(max_examples=20)
def test_updates_followed_by_samples_are_valid(
    initial_weights: list[float], updates: list[tuple[int, float]]
) -> None:
    """After any sequence of updates, samples should still be valid."""
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler(initial_weights)

    for idx, new_weight in updates:
        if idx < len(initial_weights):
            sampler[idx] = new_weight

    # Samples should always be valid
    for _ in range(50):
        sample_idx = sampler.sample()
        assert 0 <= sample_idx < len(initial_weights)


@pytest.mark.slow
@given(st.lists(st.floats(min_value=1.0, max_value=10.0), min_size=3, max_size=10))
@settings(max_examples=10, deadline=None)
def test_chi_squared_passes_after_construction(weights: list[float]) -> None:
    """Chi-squared test should pass for a freshly constructed sampler.

    Note: We use weights in [1.0, 10.0] range to avoid extreme skew that
    can cause chi-squared tests to be unstable with finite samples.
    """
    from dynamic_random_sampler import DynamicSampler

    sampler: Any = DynamicSampler(weights)
    # Use 10k samples - enough for good statistical power without being too slow
    result = sampler.test_distribution(10000)

    # Use alpha=1e-6 for negligible false positive rate (~1e-5 per run).
    # Power: with 10k samples, >99.9999% power to detect 50% sampling error.
    assert result.passes(1e-6), (
        f"Chi-squared test failed: chi2={result.chi_squared:.2f}, "
        f"p_value={result.p_value:.6f}, weights={weights}"
    )


# -----------------------------------------------------------------------------
# Extreme Weight Range Tests
# -----------------------------------------------------------------------------


def test_extreme_weight_range() -> None:
    """Test that sampling works correctly with extremely different weight magnitudes.

    Uses the Gumbel-max trick internally which works entirely in log space,
    allowing handling of weight ratios up to 10^300.
    """
    from dynamic_random_sampler import DynamicSampler

    # Weights spanning 200 orders of magnitude
    # Element 0: 1e-100 (extremely tiny)
    # Element 1: 1.0 (normal)
    # Element 2: 1e100 (extremely large)
    weights = [1e-100, 1.0, 1e100]
    sampler: Any = DynamicSampler(weights)

    # Take samples - element 2 should get almost all of them
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    num_samples = 1000
    for _ in range(num_samples):
        idx = sampler.sample()
        counts[idx] += 1

    # Element 2 has weight 1e100 / (1e100 + 1 + 1e-100) ≈ 1.0
    # It should get virtually all samples
    assert counts[2] > 990, f"Element 2 should get >99% of samples, got {counts[2]/10}%"

    # Elements 0 and 1 should get effectively 0 samples
    # (probability is about 1e-100 for element 0 and 1e-100 for element 1)
    assert counts[0] == 0, f"Element 0 (1e-100) shouldn't be sampled, got {counts[0]}"
    assert counts[1] < 10, f"Element 1 (1.0) rarely sampled, got {counts[1]}"


@given(st.data())
@settings(max_examples=10, deadline=None)
def test_extreme_dominant_weight(data: st.DataObject) -> None:
    """Test that a dominant element with extreme weight gets all samples.

    Even with weight ratios of 10^100+, the dominant element should get
    effectively all samples.
    """
    from dynamic_random_sampler import DynamicSampler

    # Create weights where one dominates by 100+ orders of magnitude
    n = data.draw(st.integers(min_value=2, max_value=5))
    dominant_idx = data.draw(st.integers(min_value=0, max_value=n - 1))

    # All elements get weight 1.0 except the dominant one
    weights = [1.0] * n
    weights[dominant_idx] = 1e100  # Dominant by 100 orders of magnitude

    sampler: Any = DynamicSampler(weights)

    # All samples should go to the dominant element
    for _ in range(100):
        idx = sampler.sample()
        assert idx == dominant_idx, (
            f"Expected all samples to go to index {dominant_idx}, got {idx}"
        )
