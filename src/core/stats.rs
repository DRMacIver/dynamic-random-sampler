//! Statistical tests for verifying sampling distribution correctness.
//!
//! This module provides statistical functions including a chi-squared
//! goodness-of-fit test to verify sampling distributions.

#![allow(clippy::many_single_char_names)] // Standard notation for gamma functions
#![allow(clippy::cast_precision_loss)] // Acceptable for statistical calculations
#![allow(clippy::excessive_precision)] // Lanczos coefficients need high precision

/// Result of a chi-squared goodness-of-fit test.
#[derive(Debug, Clone, Copy)]
pub struct ChiSquaredResult {
    /// The chi-squared statistic.
    pub chi_squared: f64,
    /// Degrees of freedom (number of categories - 1).
    pub degrees_of_freedom: usize,
    /// The p-value (probability of observing this result or more extreme).
    pub p_value: f64,
    /// Number of samples taken.
    pub num_samples: usize,
}

impl ChiSquaredResult {
    /// Returns true if the test passes at the given significance level.
    ///
    /// A test "passes" if the p-value is greater than alpha, meaning we cannot
    /// reject the null hypothesis that the observed distribution matches expected.
    #[must_use]
    pub const fn passes(&self, alpha: f64) -> bool {
        self.p_value > alpha
    }
}

/// Performs a chi-squared goodness-of-fit test.
///
/// Given observed counts and expected probabilities (weights), calculates
/// the chi-squared statistic and p-value.
///
/// # Arguments
///
/// * `observed` - Observed counts for each category
/// * `weights` - Expected weights (will be normalized to probabilities)
/// * `num_samples` - Total number of samples taken
///
/// # Returns
///
/// A `ChiSquaredResult` containing the test results.
#[must_use]
pub fn chi_squared_from_counts(
    observed: &[usize],
    weights: &[f64],
    num_samples: usize,
) -> ChiSquaredResult {
    assert_eq!(
        observed.len(),
        weights.len(),
        "observed and weights must have same length"
    );

    let n = weights.len();
    assert!(n > 0, "cannot test empty distribution");

    // Normalize weights to get expected probabilities
    let total_weight: f64 = weights.iter().sum();
    assert!(total_weight > 0.0, "total weight must be positive");

    // Calculate chi-squared statistic
    let num_samples_f64 = num_samples as f64;
    let mut chi_squared = 0.0;

    for (i, &obs) in observed.iter().enumerate() {
        let expected = (weights[i] / total_weight) * num_samples_f64;
        if expected > 0.0 {
            let diff = obs as f64 - expected;
            chi_squared += (diff * diff) / expected;
        }
    }

    let degrees_of_freedom = n - 1;
    let p_value = chi_squared_sf(chi_squared, degrees_of_freedom);

    ChiSquaredResult {
        chi_squared,
        degrees_of_freedom,
        p_value,
        num_samples,
    }
}

/// Result of a likelihood-based statistical test.
#[derive(Debug, Clone)]
pub struct LikelihoodTestResult {
    /// The observed sum of log-likelihoods.
    pub observed_log_likelihood: f64,
    /// The expected sum of log-likelihoods under null hypothesis.
    pub expected_log_likelihood: f64,
    /// The variance of the log-likelihood sum under null hypothesis.
    pub variance: f64,
    /// The z-score (standardized test statistic).
    pub z_score: f64,
    /// The two-tailed p-value.
    pub p_value: f64,
    /// Number of samples taken.
    pub num_samples: usize,
}

impl LikelihoodTestResult {
    /// Returns true if the test passes at the given significance level.
    ///
    /// A test "passes" if the p-value is greater than alpha.
    #[must_use]
    pub const fn passes(&self, alpha: f64) -> bool {
        self.p_value > alpha
    }
}

/// Assignment for the likelihood test: right before sample i, set weight[j] = x.
#[derive(Debug, Clone, Copy)]
pub struct Assignment {
    /// The sample index (0-indexed) before which to apply this assignment.
    pub sample_index: usize,
    /// The weight index to update.
    pub weight_index: usize,
    /// The new weight value (in linear space, not log).
    pub new_weight: f64,
}

/// Performs a likelihood-based statistical test on a sampler.
///
/// This test verifies that the sampler produces samples according to the
/// correct probability distribution, accounting for dynamic weight updates.
///
/// # Algorithm
///
/// 1. Create a sampler with initial weights
/// 2. For each sample i from 0 to N-1:
///    - Apply any assignments where sample_index == i
///    - Take a sample and record its log-probability
/// 3. Under the null hypothesis:
///    - Each sample's log-probability has known mean and variance
///    - The sum of log-likelihoods is approximately normal (CLT)
/// 4. Compute z-score and two-tailed p-value
///
/// # Arguments
///
/// * `initial_weights` - Initial weights (linear space, must be positive and not all zero)
/// * `num_samples` - Number of samples to take (must be >= 100)
/// * `assignments` - Assignments to apply (sorted by sample_index internally)
/// * `rng` - Random number generator
///
/// # Returns
///
/// `Ok(LikelihoodTestResult)` with the test statistics and p-value, or
/// `Err` if assignments caused all weights to become zero during the test.
///
/// # Panics
///
/// Panics if num_samples < 100 or if initial_weights is empty or all zero.
pub fn likelihood_test<R: rand::Rng>(
    initial_weights: &[f64],
    num_samples: usize,
    assignments: &[Assignment],
    rng: &mut R,
) -> Result<LikelihoodTestResult, &'static str> {
    use crate::core::{sample, MutableTree};

    assert!(num_samples >= 100, "num_samples must be at least 100");
    assert!(!initial_weights.is_empty(), "initial_weights cannot be empty");
    assert!(
        initial_weights.iter().any(|&w| w > 0.0),
        "at least one weight must be positive"
    );

    // Convert to log space
    let log_weights: Vec<f64> = initial_weights
        .iter()
        .map(|&w| if w <= 0.0 { f64::NEG_INFINITY } else { w.log2() })
        .collect();

    let mut tree = MutableTree::new(log_weights);

    // Sort assignments by sample_index
    let mut sorted_assignments: Vec<Assignment> = assignments.to_vec();
    sorted_assignments.sort_by_key(|a| a.sample_index);

    let mut assignment_idx = 0;

    let mut total_log_likelihood = 0.0;
    let mut expected_mean = 0.0;
    let mut expected_variance = 0.0;

    for sample_i in 0..num_samples {
        // Apply any assignments for this sample index
        while assignment_idx < sorted_assignments.len()
            && sorted_assignments[assignment_idx].sample_index == sample_i
        {
            let assignment = &sorted_assignments[assignment_idx];
            let weight_idx = assignment.weight_index;
            let new_weight = assignment.new_weight;

            // Extend tree if necessary (zero-extend)
            while tree.len() <= weight_idx {
                tree.insert(f64::NEG_INFINITY); // Insert with zero weight
            }

            // Update the weight
            let log_weight = if new_weight <= 0.0 {
                f64::NEG_INFINITY
            } else {
                new_weight.log2()
            };
            tree.update(weight_idx, log_weight);

            assignment_idx += 1;
        }

        // Get current log probabilities
        let log_probs = tree.log_probabilities();

        // Compute expected mean and variance for this sample
        // E[log P(J)] = Σ p_i * log p_i (negative entropy)
        // Var[log P(J)] = E[(log P(J))^2] - E[log P(J)]^2
        let mut sample_mean = 0.0;
        let mut sample_second_moment = 0.0;

        for &log_p in &log_probs {
            if log_p.is_finite() {
                let p = log_p.exp2(); // Convert from log2 to probability
                let log_p_nat = log_p * std::f64::consts::LN_2; // Convert to natural log
                sample_mean += p * log_p_nat;
                sample_second_moment += p * log_p_nat * log_p_nat;
            }
        }

        let sample_variance = sample_second_moment - sample_mean * sample_mean;

        expected_mean += sample_mean;
        expected_variance += sample_variance.max(0.0); // Ensure non-negative due to numerical issues

        // Take a sample
        let immutable_tree = tree.as_tree();
        let sampled_idx = sample(&immutable_tree, rng)
            .ok_or("all weights became zero during test")?;

        // Record log-likelihood (in natural log)
        let log_p = log_probs[sampled_idx];
        let log_p_nat = if log_p.is_finite() {
            log_p * std::f64::consts::LN_2
        } else {
            f64::NEG_INFINITY
        };
        total_log_likelihood += log_p_nat;
    }

    // Compute z-score
    let std_dev = expected_variance.sqrt();
    let z_score = if std_dev > 0.0 {
        (total_log_likelihood - expected_mean) / std_dev
    } else {
        // If variance is zero, all samples have the same probability
        // In this case, observed should equal expected
        if (total_log_likelihood - expected_mean).abs() < 1e-10 {
            0.0
        } else {
            f64::INFINITY
        }
    };

    // Two-tailed p-value
    let p_value = 2.0 * standard_normal_cdf(-z_score.abs());

    Ok(LikelihoodTestResult {
        observed_log_likelihood: total_log_likelihood,
        expected_log_likelihood: expected_mean,
        variance: expected_variance,
        z_score,
        p_value,
        num_samples,
    })
}

/// Chi-squared survival function (1 - CDF).
///
/// Returns P(X > x) where X follows a chi-squared distribution with k degrees of freedom.
#[must_use]
pub fn chi_squared_sf(x: f64, k: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if k == 0 {
        return 0.0;
    }

    let a = k as f64 / 2.0;
    let z = x / 2.0;

    1.0 - regularized_gamma_p(a, z)
}

/// Regularized lower incomplete gamma function P(a, x).
///
/// # Preconditions
/// This function is only called from `chi_squared_sf` which validates:
/// - `x > 0` (from `x/2` where `x > 0`)
/// - `a > 0` (from `k/2` where `k >= 1`)
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    // These are invariants guaranteed by chi_squared_sf
    debug_assert!(x > 0.0, "regularized_gamma_p called with x <= 0");
    debug_assert!(a > 0.0, "regularized_gamma_p called with a <= 0");

    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Series expansion for regularized incomplete gamma function.
fn gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;

    let gln = ln_gamma(a);

    let mut sum = 1.0 / a;
    let mut term = sum;

    for n in 1..max_iter {
        term *= x / (a + f64::from(n));
        sum += term;
        if term.abs() < sum.abs() * eps {
            break;
        }
    }

    sum * (a.mul_add(x.ln(), -x) - gln).exp()
}

/// Continued fraction expansion for complementary incomplete gamma function.
///
/// This function contains standard numerical underflow guards (d.abs() < fpmin, c.abs() < fpmin)
/// that are extremely difficult to trigger but are necessary for numerical stability.
#[cfg_attr(coverage_nightly, coverage(off))]
fn gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let fpmin = 1e-300;

    let gln = ln_gamma(a);

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..max_iter {
        let i_f64 = f64::from(i);
        let an = -i_f64 * (i_f64 - a);
        b += 2.0;
        d = an.mul_add(d, b);
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }

    (a.mul_add(x.ln(), -x) - gln).exp() * h
}

/// Standard normal CDF (cumulative distribution function).
///
/// Returns P(X ≤ x) where X ~ N(0,1).
/// Uses the error function approximation.
#[must_use]
pub fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation.
///
/// Uses Horner's method with coefficients from Abramowitz and Stegun.
fn erf(x: f64) -> f64 {
    // Constants for the approximation
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    // Save the sign of x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let y = 1.0 - (A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5) * (-x * x).exp();

    sign * y
}

/// Natural logarithm of the gamma function using Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    const LANCZOS_G: f64 = 7.0;
    const LANCZOS_COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (x + i as f64);
    }

    let t = x + LANCZOS_G + 0.5;
    let log_2pi_half = 0.5 * (2.0 * std::f64::consts::PI).ln();
    (x + 0.5).mul_add(t.ln(), log_2pi_half) - t + sum.ln()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_chi_squared_from_counts_uniform() {
        // Perfect uniform distribution
        let observed = vec![250, 250, 250, 250];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let result = chi_squared_from_counts(&observed, &weights, 1000);

        assert_eq!(result.degrees_of_freedom, 3);
        assert_eq!(result.num_samples, 1000);
        assert!(result.chi_squared < 1e-10); // Perfect match
        assert!(result.p_value > 0.99); // Very high p-value
    }

    #[test]
    fn test_chi_squared_from_counts_weighted() {
        // Distribution matching weights 1:2:3:4
        let observed = vec![100, 200, 300, 400];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let result = chi_squared_from_counts(&observed, &weights, 1000);

        assert!(result.chi_squared < 1e-10); // Perfect match
        assert!(result.p_value > 0.99);
    }

    #[test]
    fn test_chi_squared_from_counts_mismatch() {
        // Observed uniform but expected weighted
        let observed = vec![250, 250, 250, 250];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let result = chi_squared_from_counts(&observed, &weights, 1000);

        // Should detect mismatch
        assert!(result.chi_squared > 10.0);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_chi_squared_sf_known_values() {
        // For df=1, chi2=3.841 corresponds to p=0.05
        let p = chi_squared_sf(3.841, 1);
        assert!((p - 0.05).abs() < 0.01, "got {p}");

        // For df=1, chi2=6.635 corresponds to p=0.01
        let p = chi_squared_sf(6.635, 1);
        assert!((p - 0.01).abs() < 0.005, "got {p}");

        // For df=2, chi2=5.991 corresponds to p=0.05
        let p = chi_squared_sf(5.991, 2);
        assert!((p - 0.05).abs() < 0.01, "got {p}");
    }

    #[test]
    fn test_ln_gamma_known_values() {
        // Gamma(1) = 1, ln(1) = 0
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);

        // Gamma(2) = 1, ln(1) = 0
        assert!((ln_gamma(2.0) - 0.0).abs() < 1e-10);

        // Gamma(3) = 2, ln(2) ~= 0.693
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);

        // Gamma(4) = 6, ln(6) ~= 1.791
        assert!((ln_gamma(4.0) - 6.0_f64.ln()).abs() < 1e-10);

        // Gamma(0.5) = sqrt(pi)
        let expected = std::f64::consts::PI.sqrt().ln();
        assert!((ln_gamma(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_result_passes() {
        let result = ChiSquaredResult {
            chi_squared: 5.0,
            degrees_of_freedom: 3,
            p_value: 0.17,
            num_samples: 1000,
        };

        assert!(result.passes(0.05)); // 0.17 > 0.05
        assert!(result.passes(0.10)); // 0.17 > 0.10
        assert!(!result.passes(0.20)); // 0.17 < 0.20
    }

    // -------------------------------------------------------------------------
    // Additional Coverage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_chi_squared_sf_zero_df() {
        // k = 0 degrees of freedom should return 0
        let p = chi_squared_sf(5.0, 0);
        assert!((p - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_sf_negative_x() {
        // x <= 0 should return 1.0
        let p = chi_squared_sf(-1.0, 3);
        assert!((p - 1.0).abs() < 1e-10);

        let p_zero = chi_squared_sf(0.0, 3);
        assert!((p_zero - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_gamma_small_values() {
        // Test reflection formula: x < 0.5
        let result = ln_gamma(0.3);
        // Gamma(0.3) = Gamma(1.3) / 0.3
        // We just check it returns a finite value
        assert!(result.is_finite());
    }

    #[test]
    fn test_regularized_gamma_p_edge_cases() {
        // Test internal functions via chi_squared_sf
        // For very small x, P(a, x) should be near 0
        let p = chi_squared_sf(0.001, 10);
        assert!(p > 0.99);
    }

    #[test]
    fn test_gamma_cf_branch() {
        // Test the continued fraction branch (x >= a + 1)
        // For large x, chi_squared_sf should return near 0
        let p = chi_squared_sf(100.0, 2);
        assert!(p < 0.001);
    }

    // -------------------------------------------------------------------------
    // Standard Normal CDF Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_standard_normal_cdf_known_values() {
        // CDF(0) = 0.5
        let p = standard_normal_cdf(0.0);
        assert!((p - 0.5).abs() < 1e-6, "CDF(0) = {p}, expected 0.5");

        // CDF(-infinity) -> 0, CDF(+infinity) -> 1
        let p_neg = standard_normal_cdf(-10.0);
        assert!(p_neg < 1e-10, "CDF(-10) should be near 0, got {p_neg}");

        let p_pos = standard_normal_cdf(10.0);
        assert!((p_pos - 1.0).abs() < 1e-10, "CDF(10) should be near 1, got {p_pos}");

        // CDF(1.96) ~= 0.975 (97.5th percentile)
        let p196 = standard_normal_cdf(1.96);
        assert!((p196 - 0.975).abs() < 0.001, "CDF(1.96) = {p196}, expected ~0.975");

        // CDF(-1.96) ~= 0.025 (2.5th percentile)
        let p_neg196 = standard_normal_cdf(-1.96);
        assert!((p_neg196 - 0.025).abs() < 0.001, "CDF(-1.96) = {p_neg196}, expected ~0.025");
    }

    #[test]
    fn test_standard_normal_cdf_symmetry() {
        // CDF(x) + CDF(-x) = 1
        for x in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let sum = standard_normal_cdf(x) + standard_normal_cdf(-x);
            assert!((sum - 1.0).abs() < 1e-10, "CDF({x}) + CDF(-{x}) = {sum}");
        }
    }

    // -------------------------------------------------------------------------
    // Likelihood Test Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_likelihood_test_uniform_weights() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Uniform weights - should pass
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = likelihood_test(&weights, 1000, &[], &mut rng).unwrap();

        // With correct sampling, p-value should be reasonable
        assert!(result.passes(1e-6), "p-value too low: {}", result.p_value);
        assert_eq!(result.num_samples, 1000);
    }

    #[test]
    fn test_likelihood_test_weighted() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Non-uniform weights
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let result = likelihood_test(&weights, 1000, &[], &mut rng).unwrap();

        assert!(result.passes(1e-6), "p-value too low: {}", result.p_value);
    }

    #[test]
    fn test_likelihood_test_with_assignments() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Start with uniform, then change weights
        let weights = vec![1.0, 1.0];
        let assignments = vec![
            Assignment { sample_index: 50, weight_index: 0, new_weight: 10.0 },
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(456);

        let result = likelihood_test(&weights, 100, &assignments, &mut rng).unwrap();

        assert!(result.passes(1e-6), "p-value too low: {}", result.p_value);
    }

    #[test]
    fn test_likelihood_test_array_extension() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Start with 2 weights, extend to 5
        let weights = vec![1.0, 1.0];
        let assignments = vec![
            Assignment { sample_index: 0, weight_index: 4, new_weight: 2.0 },
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(789);

        let result = likelihood_test(&weights, 100, &assignments, &mut rng).unwrap();

        assert!(result.passes(1e-6), "p-value too low: {}", result.p_value);
    }

    #[test]
    fn test_likelihood_test_all_weights_zero_error() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Start with one weight, set it to zero
        let weights = vec![1.0];
        let assignments = vec![
            Assignment { sample_index: 0, weight_index: 0, new_weight: 0.0 },
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(999);

        let result = likelihood_test(&weights, 100, &assignments, &mut rng);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "all weights became zero during test");
    }

    #[test]
    fn test_likelihood_test_result_passes() {
        let result = LikelihoodTestResult {
            observed_log_likelihood: -100.0,
            expected_log_likelihood: -100.5,
            variance: 10.0,
            z_score: 0.158,
            p_value: 0.87,
            num_samples: 100,
        };

        assert!(result.passes(0.05));
        assert!(result.passes(0.1));
        assert!(!result.passes(0.9));
    }

    #[test]
    #[should_panic(expected = "num_samples must be at least 100")]
    fn test_likelihood_test_too_few_samples() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let weights = vec![1.0, 1.0];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _ = likelihood_test(&weights, 50, &[], &mut rng);
    }

    #[test]
    #[should_panic(expected = "initial_weights cannot be empty")]
    fn test_likelihood_test_empty_weights() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let weights: Vec<f64> = vec![];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _ = likelihood_test(&weights, 100, &[], &mut rng);
    }
}
