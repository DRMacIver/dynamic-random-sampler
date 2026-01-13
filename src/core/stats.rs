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
}
