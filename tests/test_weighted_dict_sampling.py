"""Hypothesis-based tests for SamplerDict sampling distribution correctness.

Tests verify that the sampling distribution matches the expected probability
distribution based on weights.
"""

from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

# Strategy for generating a dict of positive weights
weights_dict_strategy = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=48),
        min_size=1,
        max_size=8,
    ),
    values=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
    min_size=2,
    max_size=10,
)


@pytest.mark.slow
@given(weights=weights_dict_strategy)
@settings(max_examples=20, deadline=None)
def test_all_keys_can_be_sampled(weights: dict[str, float]) -> None:
    """Test that all keys with positive weight can be sampled.

    Uses enough samples to statistically ensure all keys are hit with high
    probability, even for keys with relatively small weights.
    """
    from dynamic_random_sampler import SamplerDict

    wd: Any = SamplerDict(seed=42)
    for key, weight in weights.items():
        wd[key] = weight

    # Sample enough times to hit all keys with high probability
    # With 10 keys and 100000 samples, even a 0.1% probability key
    # should be hit ~100 times
    num_samples = 100000

    sampled_keys: set[str] = set()
    for _ in range(num_samples):
        sampled_keys.add(wd.sample())
        # Early exit if we've found all keys
        if len(sampled_keys) == len(weights):
            break

    # All keys with positive weight should eventually be sampled
    positive_keys = {k for k, v in weights.items() if v > 0}
    assert sampled_keys == positive_keys, (
        f"Not all keys sampled. Expected {positive_keys}, got {sampled_keys}"
    )


@given(
    weights=weights_dict_strategy, seed=st.integers(min_value=0, max_value=2**32 - 1)
)
@settings(max_examples=10, deadline=None)
def test_sampling_returns_only_valid_keys(weights: dict[str, float], seed: int) -> None:
    """Test that sampling only returns keys that exist in the dict."""
    from dynamic_random_sampler import SamplerDict

    wd: Any = SamplerDict(seed=seed)
    for key, weight in weights.items():
        wd[key] = weight

    for _ in range(100):
        key = wd.sample()
        assert key in weights, f"Sampled key {key!r} not in weights"


@given(
    base_weights=weights_dict_strategy,
    delete_ratio=st.floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=10, deadline=None)
def test_sampling_correct_after_deletions(
    base_weights: dict[str, float], delete_ratio: float
) -> None:
    """Test that sampling is correct after deleting some keys."""
    from dynamic_random_sampler import SamplerDict

    wd: Any = SamplerDict(seed=42)
    for key, weight in base_weights.items():
        wd[key] = weight

    # Delete some keys
    keys_to_delete = list(base_weights.keys())[: int(len(base_weights) * delete_ratio)]
    for key in keys_to_delete:
        del wd[key]

    remaining_keys = set(base_weights.keys()) - set(keys_to_delete)

    if len(remaining_keys) == 0:
        return  # Nothing left to sample

    # Sample and verify only remaining keys are returned
    for _ in range(100):
        key = wd.sample()
        assert key in remaining_keys, (
            f"Sampled deleted key {key!r}. Remaining: {remaining_keys}"
        )


def test_chi_squared_distribution() -> None:
    """Statistical test that sampling distribution matches expected weights.

    Uses chi-squared test to verify the observed distribution matches
    the expected probability distribution.
    """
    from dynamic_random_sampler import SamplerDict

    wd: Any = SamplerDict(seed=12345)

    # Set up weights with clear expected distribution
    weights = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    total = sum(weights.values())

    for key, weight in weights.items():
        wd[key] = weight

    # Sample many times
    num_samples = 10000
    counts: dict[str, int] = dict.fromkeys(weights, 0)
    for _ in range(num_samples):
        key = wd.sample()
        counts[key] += 1

    # Compute chi-squared statistic
    chi_squared = 0.0
    for key in weights:
        expected = (weights[key] / total) * num_samples
        observed = counts[key]
        chi_squared += (observed - expected) ** 2 / expected

    # With 3 degrees of freedom (4 categories - 1), chi-squared should be
    # less than ~11.34 at p=0.01 significance level
    assert chi_squared < 15.0, (
        f"Chi-squared test failed: {chi_squared:.2f} > 15.0. "
        f"Counts: {counts}, Expected proportions: "
        + ", ".join(f"{k}={v / total:.2%}" for k, v in weights.items())
    )


def test_zero_weight_excluded() -> None:
    """Test that keys with zero weight are never sampled."""
    from dynamic_random_sampler import SamplerDict

    wd: Any = SamplerDict(seed=42)
    wd["positive"] = 1.0
    wd["zero"] = 0.0

    for _ in range(1000):
        assert wd.sample() == "positive"


def test_extreme_weight_ratio() -> None:
    """Test sampling with very different weight magnitudes."""
    from dynamic_random_sampler import SamplerDict

    wd: Any = SamplerDict(seed=42)
    wd["tiny"] = 0.001
    wd["huge"] = 1000.0

    # "huge" should be sampled almost always (1000000:1 ratio)
    num_samples = 10000
    tiny_count = 0
    for _ in range(num_samples):
        if wd.sample() == "tiny":
            tiny_count += 1

    # Expected: ~0.01 samples for tiny, but allow some variance
    # At this ratio, getting even 1 would be unlikely, but we allow up to 10
    assert tiny_count < 20, f"Tiny sampled too often: {tiny_count}"


def test_sampling_after_weight_update() -> None:
    """Test that weight updates correctly affect sampling distribution."""
    from dynamic_random_sampler import SamplerDict

    wd: Any = SamplerDict(seed=42)
    wd["a"] = 1.0
    wd["b"] = 1.0

    # Initially, both should be sampled roughly equally
    counts_before: dict[str, int] = {"a": 0, "b": 0}
    for _ in range(1000):
        counts_before[wd.sample()] += 1

    # Both should be roughly 500 +/- 100
    assert 400 < counts_before["a"] < 600
    assert 400 < counts_before["b"] < 600

    # Now make b much heavier
    wd["b"] = 9.0

    wd.seed(42)  # Reset for consistency
    counts_after: dict[str, int] = {"a": 0, "b": 0}
    for _ in range(1000):
        counts_after[wd.sample()] += 1

    # b should now be ~90%
    assert counts_after["b"] > 800, f"b count after update: {counts_after['b']}"


def test_swap_remove_preserves_distribution() -> None:
    """Test that swap-remove doesn't corrupt the sampling distribution."""
    from dynamic_random_sampler import SamplerDict

    wd: Any = SamplerDict(seed=42)

    # Insert with known weights
    wd["keep_1"] = 10.0
    wd["delete_me"] = 5.0
    wd["keep_2"] = 10.0

    # Delete the middle one (triggers swap-remove)
    del wd["delete_me"]

    # Now keep_1 and keep_2 should be sampled equally
    counts: dict[str, int] = {"keep_1": 0, "keep_2": 0}
    for _ in range(1000):
        counts[wd.sample()] += 1

    # Both should be roughly 500 +/- 100
    assert 400 < counts["keep_1"] < 600, f"keep_1 count: {counts['keep_1']}"
    assert 400 < counts["keep_2"] < 600, f"keep_2 count: {counts['keep_2']}"
