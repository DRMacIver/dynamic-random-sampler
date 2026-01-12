//! Statistical correctness tests for various weight distributions.
//!
//! These tests verify that sampling produces correct distributions using
//! chi-squared tests with sufficient power to detect 50% errors.
//!
//! Power analysis (for each test):
//! - Alpha = 1e-6 (false positive rate per test)
//! - Samples = 10,000 (gives >99.9999% power to detect 50% error)
//! - Expected flakiness: ~1 in 100,000 runs

// Clippy config for tests - don't need production-level strictness
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::must_use_candidate)]

use dynamic_random_sampler::core::{chi_squared_from_counts, sample, Tree};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Number of samples for chi-squared tests.
/// Power analysis: 10k samples gives >99.9999% power to detect 50% error at alpha=1e-6.
const NUM_SAMPLES: usize = 10_000;

/// Significance level for tests.
/// With ~10 tests, expect ~1e-5 false positive rate per run.
const ALPHA: f64 = 1e-6;

/// Weight distribution types.
#[derive(Debug, Clone, Copy)]
enum Distribution {
    Uniform,
    PowerLaw { alpha: f64 },
    OneHot { hot_index: usize },
    Exponential { lambda: f64 },
}

impl Distribution {
    fn generate_weights(&self, n: usize) -> Vec<f64> {
        match self {
            Self::Uniform => vec![1.0; n],
            Self::PowerLaw { alpha } => (0..n)
                .map(|i| 1.0 / (i as f64 + 1.0).powf(*alpha))
                .collect(),
            Self::OneHot { hot_index } => {
                let mut weights = vec![1e-10; n];
                if *hot_index < n {
                    weights[*hot_index] = 1.0;
                }
                weights
            }
            Self::Exponential { lambda } => (0..n).map(|i| (-lambda * i as f64).exp()).collect(),
        }
    }
}

fn to_log_weights(weights: &[f64]) -> Vec<f64> {
    weights.iter().map(|w| w.log2()).collect()
}

/// Run chi-squared test for a given distribution and size.
fn test_distribution(dist: Distribution, n: usize, seed: u64) {
    let weights = dist.generate_weights(n);
    let log_weights = to_log_weights(&weights);
    let tree = Tree::new(log_weights);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Sample and count
    let mut counts = vec![0usize; n];
    for _ in 0..NUM_SAMPLES {
        if let Some(idx) = sample(&tree, &mut rng) {
            counts[idx] += 1;
        }
    }

    // Run chi-squared test
    let result = chi_squared_from_counts(&counts, &weights, NUM_SAMPLES);

    assert!(
        result.passes(ALPHA),
        "Chi-squared test failed for {:?} with n={}: chi2={:.2}, p={:.8}, df={}",
        dist,
        n,
        result.chi_squared,
        result.p_value,
        result.degrees_of_freedom
    );
}

// =============================================================================
// Uniform distribution tests
// =============================================================================

#[test]
fn test_uniform_10() {
    test_distribution(Distribution::Uniform, 10, 1001);
}

#[test]
fn test_uniform_100() {
    test_distribution(Distribution::Uniform, 100, 1002);
}

#[test]
fn test_uniform_1000() {
    test_distribution(Distribution::Uniform, 1000, 1003);
}

#[test]
fn test_uniform_10000() {
    test_distribution(Distribution::Uniform, 10000, 1004);
}

// =============================================================================
// Power law distribution tests
// =============================================================================

#[test]
fn test_power_law_10() {
    test_distribution(Distribution::PowerLaw { alpha: 1.0 }, 10, 2001);
}

#[test]
fn test_power_law_100() {
    test_distribution(Distribution::PowerLaw { alpha: 1.0 }, 100, 2002);
}

#[test]
fn test_power_law_1000() {
    test_distribution(Distribution::PowerLaw { alpha: 1.0 }, 1000, 2003);
}

#[test]
fn test_power_law_steep_100() {
    test_distribution(Distribution::PowerLaw { alpha: 2.0 }, 100, 2101);
}

// =============================================================================
// Exponential distribution tests
// =============================================================================

#[test]
fn test_exponential_10() {
    test_distribution(Distribution::Exponential { lambda: 0.1 }, 10, 3001);
}

#[test]
fn test_exponential_100() {
    test_distribution(Distribution::Exponential { lambda: 0.01 }, 100, 3002);
}

#[test]
fn test_exponential_1000() {
    test_distribution(Distribution::Exponential { lambda: 0.001 }, 1000, 3003);
}

#[test]
fn test_exponential_steep_100() {
    test_distribution(Distribution::Exponential { lambda: 0.1 }, 100, 3101);
}

// =============================================================================
// One-hot distribution tests (dominant element)
// =============================================================================

#[test]
fn test_one_hot_first() {
    // One-hot is heavily skewed, chi-squared won't work well.
    // Instead, verify the dominant element gets almost all samples.
    let n = 100;
    let dist = Distribution::OneHot { hot_index: 0 };
    let weights = dist.generate_weights(n);
    let log_weights = to_log_weights(&weights);
    let tree = Tree::new(log_weights);
    let mut rng = ChaCha8Rng::seed_from_u64(4001);

    let mut counts = vec![0usize; n];
    for _ in 0..NUM_SAMPLES {
        if let Some(idx) = sample(&tree, &mut rng) {
            counts[idx] += 1;
        }
    }

    // The hot element should get > 99.9% of samples
    let hot_fraction = counts[0] as f64 / NUM_SAMPLES as f64;
    assert!(
        hot_fraction > 0.999,
        "Hot element only got {:.2}% of samples",
        hot_fraction * 100.0
    );
}

#[test]
fn test_one_hot_middle() {
    let n = 100;
    let hot_idx = 50;
    let dist = Distribution::OneHot { hot_index: hot_idx };
    let weights = dist.generate_weights(n);
    let log_weights = to_log_weights(&weights);
    let tree = Tree::new(log_weights);
    let mut rng = ChaCha8Rng::seed_from_u64(4002);

    let mut counts = vec![0usize; n];
    for _ in 0..NUM_SAMPLES {
        if let Some(idx) = sample(&tree, &mut rng) {
            counts[idx] += 1;
        }
    }

    let hot_fraction = counts[hot_idx] as f64 / NUM_SAMPLES as f64;
    assert!(
        hot_fraction > 0.999,
        "Hot element only got {:.2}% of samples",
        hot_fraction * 100.0
    );
}

// =============================================================================
// Mixed weight tests
// =============================================================================

#[test]
fn test_two_groups() {
    // 50 elements at weight 1, 50 at weight 10
    // Group 2 should get 10x more samples
    let mut weights = vec![1.0; 50];
    weights.extend(vec![10.0; 50]);
    let log_weights = to_log_weights(&weights);
    let tree = Tree::new(log_weights);
    let mut rng = ChaCha8Rng::seed_from_u64(5001);

    let mut counts = vec![0usize; 100];
    for _ in 0..NUM_SAMPLES {
        if let Some(idx) = sample(&tree, &mut rng) {
            counts[idx] += 1;
        }
    }

    let result = chi_squared_from_counts(&counts, &weights, NUM_SAMPLES);
    assert!(
        result.passes(ALPHA),
        "Chi-squared test failed: chi2={:.2}, p={:.8}",
        result.chi_squared,
        result.p_value
    );
}

#[test]
fn test_wide_range() {
    // Weights spanning 6 orders of magnitude
    let weights: Vec<f64> = (0..100).map(|i| 10.0_f64.powf(i as f64 / 16.0)).collect();
    let log_weights = to_log_weights(&weights);
    let tree = Tree::new(log_weights);
    let mut rng = ChaCha8Rng::seed_from_u64(5002);

    let mut counts = vec![0usize; 100];
    for _ in 0..NUM_SAMPLES {
        if let Some(idx) = sample(&tree, &mut rng) {
            counts[idx] += 1;
        }
    }

    let result = chi_squared_from_counts(&counts, &weights, NUM_SAMPLES);
    assert!(
        result.passes(ALPHA),
        "Chi-squared test failed: chi2={:.2}, p={:.8}",
        result.chi_squared,
        result.p_value
    );
}
