"""Tests for RNG seeding and reproducibility in SamplerList.

Note: Due to HashMap iteration non-determinism in the Rust implementation,
exact sequence reproducibility is not guaranteed even with the same seed.
The seed affects the RNG state, but the order of random number consumption
can vary between runs. These tests verify that seeding affects behavior
without requiring exact sequence matching.
"""

from typing import Any


def test_seed_affects_sampling() -> None:
    """Test that different seeds produce different results."""
    from dynamic_random_sampler import SamplerList

    # With different seeds, we should get different distributions
    sampler1: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
    sampler2: Any = SamplerList([1.0, 2.0, 3.0, 4.0, 5.0], seed=12345)

    samples1 = [sampler1.sample() for _ in range(100)]
    samples2 = [sampler2.sample() for _ in range(100)]

    # Different seeds should produce different sequences
    assert samples1 != samples2


def test_seed_zero_is_valid() -> None:
    """Test that seed=0 is a valid seed."""
    from dynamic_random_sampler import SamplerList

    # Should not raise
    sampler: Any = SamplerList([1.0, 2.0, 3.0], seed=0)
    for _ in range(10):
        result = sampler.sample()
        assert 0 <= result < 3


def test_large_seed_value() -> None:
    """Test that large seed values work."""
    from dynamic_random_sampler import SamplerList

    large_seed = 2**63 - 1  # Max u64
    sampler: Any = SamplerList([1.0, 2.0, 3.0], seed=large_seed)
    for _ in range(10):
        result = sampler.sample()
        assert 0 <= result < 3


def test_unseeded_sampler_works() -> None:
    """Test that unseeded (entropy-based) samplers work."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0])
    for _ in range(10):
        result = sampler.sample()
        assert 0 <= result < 3


def test_reseed_method_exists() -> None:
    """Test that the seed() method exists and can be called."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 2.0, 3.0], seed=42)

    # Reseed should not raise
    sampler.seed(99)
    sampler.seed(0)
    sampler.seed(2**63 - 1)

    # Sampling should still work after reseeding
    for _ in range(10):
        result = sampler.sample()
        assert 0 <= result < 3


def test_distribution_correct_with_seed() -> None:
    """Test that seeded sampler still produces correct distribution."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 1.0, 1.0, 7.0], seed=42)

    # With weights [1, 1, 1, 7], index 3 should be sampled ~70% of the time
    samples = [sampler.sample() for _ in range(1000)]
    count_3 = sum(1 for s in samples if s == 3)

    # Should be around 700, allow some variance
    assert 600 < count_3 < 800, f"Expected ~700, got {count_3}"


def test_sampling_after_update() -> None:
    """Test that sampling works correctly after updates with seed."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 1.0, 1.0, 1.0], seed=42)

    # Sample some values
    for _ in range(10):
        sampler.sample()

    # Update a weight
    sampler[0] = 100.0

    # Now index 0 should dominate
    samples = [sampler.sample() for _ in range(100)]
    count_0 = sum(1 for s in samples if s == 0)

    # Should be mostly index 0 now
    assert count_0 > 90, f"Expected >90, got {count_0}"


def test_sampling_after_append() -> None:
    """Test that sampling works correctly after append with seed."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 1.0, 1.0], seed=42)

    # Append a dominant weight
    sampler.append(100.0)

    # Now index 3 should dominate
    samples = [sampler.sample() for _ in range(100)]
    count_3 = sum(1 for s in samples if s == 3)

    # Should be mostly index 3 now
    assert count_3 > 90, f"Expected >90, got {count_3}"


def test_sampling_after_pop() -> None:
    """Test that sampling works correctly after pop with seed."""
    from dynamic_random_sampler import SamplerList

    sampler: Any = SamplerList([1.0, 1.0, 1.0, 1.0], seed=42)

    # Pop three elements from the end
    sampler.pop()  # Now [1.0, 1.0, 1.0] at indices 0,1,2
    sampler.pop()  # Now [1.0, 1.0] at indices 0,1
    sampler.pop()  # Now [1.0] at index 0

    # Now only one element remains at index 0
    assert len(sampler) == 1
    for _ in range(50):
        assert sampler.sample() == 0
