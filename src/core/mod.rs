//! Core algorithm implementation for dynamic random sampling.
//!
//! This module contains the pure Rust implementation of the data structure
//! from "Dynamic Generation of Discrete Random Variates" (Matias, Vitter, Ni, 1993/2003).
//!
//! The implementation is separated from `PyO3` bindings to allow standalone testing.
//!
//! # Section 4 Optimizations
//!
//! This implementation includes the Section 4 optimizations for achieving
//! O(log* N) amortized update time:
//!
//! - **Tolerance factor b**: Allows weights to vary within an expanded interval
//!   without triggering parent changes, reducing update propagation.
//!
//! - **Degree bound d**: Requires at least d children for a range to have a parent,
//!   which bounds the tree height and update complexity.
//!
//! See [`OptimizationConfig`] for configuration options.

// Allow some pedantic lints that are not applicable for this mathematical implementation
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

pub mod config;
pub mod level;
pub mod range;
pub mod sampler;
pub mod stats;
pub mod tree;
pub mod update;

pub use config::OptimizationConfig;
pub use level::Level;
pub use range::Range;
pub use sampler::{sample, sample_n};
pub use stats::{chi_squared_from_counts, chi_squared_sf, ChiSquaredResult};
pub use tree::Tree;
pub use update::MutableTree;

/// Sentinel value for deleted elements.
///
/// Deleted elements have their log-weight set to `NEG_INFINITY`,
/// which corresponds to a weight of 0 (since log₂(0) = -∞).
pub const DELETED_LOG_WEIGHT: f64 = f64::NEG_INFINITY;

/// Check if a log-weight represents a deleted element.
#[inline]
#[must_use]
pub fn is_deleted_weight(log_weight: f64) -> bool {
    log_weight == DELETED_LOG_WEIGHT
}

/// Compute the range number j for a given log-weight.
///
/// Given log₂(w), returns j such that w ∈ [2^(j-1), 2^j).
/// This is equivalent to j = floor(log₂(w)) + 1.
///
/// # Arguments
/// * `log_weight` - The log₂ of the weight
///
/// # Returns
/// The range number j
#[inline]
#[must_use]
#[allow(clippy::missing_const_for_fn)] // floor() is not const
pub fn compute_range_number(log_weight: f64) -> i32 {
    log_weight.floor() as i32 + 1
}

/// Check if a weight (given as log₂(w)) belongs in range j.
///
/// Range j covers the interval [2^(j-1), 2^j), which in log space
/// is [j-1, j).
///
/// # Arguments
/// * `range_number` - The range number j
/// * `log_weight` - The log₂ of the weight
///
/// # Returns
/// True if the weight belongs in this range
#[inline]
#[must_use]
pub fn weight_in_range(range_number: i32, log_weight: f64) -> bool {
    let j = range_number;
    let lower = f64::from(j - 1);
    let upper = f64::from(j);
    log_weight >= lower && log_weight < upper
}

/// Compute `log₂(sum(2^log_weights))` using the log-sum-exp trick for numerical stability.
///
/// This computes log₂(w₁ + w₂ + ... + wₙ) given log₂(wᵢ) values.
/// Uses: log₂(Σwᵢ) = max(log₂wᵢ) + log₂(Σ2^(log₂wᵢ - max))
///
/// # Arguments
/// * `log_weights` - Iterator over log₂ weights
///
/// # Returns
/// The log₂ of the sum of weights, or `NEG_INFINITY` if empty
pub fn log_sum_exp<I: Iterator<Item = f64>>(log_weights: I) -> f64 {
    let log_weights: Vec<f64> = log_weights.collect();
    if log_weights.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_log = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_log.is_infinite() {
        return f64::NEG_INFINITY;
    }

    // sum = Σ 2^(log_w - max_log)
    let sum: f64 = log_weights.iter().map(|&lw| (lw - max_log).exp2()).sum();

    max_log + sum.log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_range_number() {
        // Weight 1.0: log₂(1) = 0, so j = 0 + 1 = 1
        assert_eq!(compute_range_number(0.0), 1);
        // Weight 2.0: log₂(2) = 1, so j = 1 + 1 = 2
        assert_eq!(compute_range_number(1.0), 2);
        // Weight 4.0: log₂(4) = 2, so j = 2 + 1 = 3
        assert_eq!(compute_range_number(2.0), 3);
        // Weight 0.5: log₂(0.5) = -1, so j = -1 + 1 = 0
        assert_eq!(compute_range_number(-1.0), 0);
        // Weight 0.25: log₂(0.25) = -2, so j = -2 + 1 = -1
        assert_eq!(compute_range_number(-2.0), -1);
        // Weight 1.5: log₂(1.5) ≈ 0.585, so j = floor(0.585) + 1 = 1
        assert_eq!(compute_range_number(1.5_f64.log2()), 1);
    }

    #[test]
    fn test_weight_in_range() {
        // Range 1: [2^0, 2^1) = [1, 2) → log space: [0, 1)
        assert!(weight_in_range(1, 0.0)); // weight 1.0
        assert!(weight_in_range(1, 0.5)); // weight ~1.41
        assert!(!weight_in_range(1, 1.0)); // weight 2.0, upper bound exclusive
        assert!(!weight_in_range(1, -0.1)); // weight < 1.0

        // Range 2: [2^1, 2^2) = [2, 4) → log space: [1, 2)
        assert!(weight_in_range(2, 1.0)); // weight 2.0
        assert!(weight_in_range(2, 1.9)); // weight ~3.73
        assert!(!weight_in_range(2, 2.0)); // weight 4.0

        // Range 0: [2^-1, 2^0) = [0.5, 1) → log space: [-1, 0)
        assert!(weight_in_range(0, -1.0)); // weight 0.5
        assert!(weight_in_range(0, -0.1)); // weight ~0.93
        assert!(!weight_in_range(0, 0.0)); // weight 1.0
    }

    #[test]
    fn test_log_sum_exp_single() {
        // Single weight: sum = weight
        let result = log_sum_exp([2.0].into_iter());
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_two_equal() {
        // Two equal weights: log₂(2^x + 2^x) = log₂(2 * 2^x) = 1 + x
        let result = log_sum_exp([3.0, 3.0].into_iter());
        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_different() {
        // log₂(2 + 4) = log₂(6) ≈ 2.585
        let result = log_sum_exp([1.0, 2.0].into_iter());
        assert!((result - 6.0_f64.log2()).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_empty() {
        let result = log_sum_exp([].into_iter());
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_log_sum_exp_large_values() {
        // Test numerical stability with large values
        // log₂(2^100 + 2^100) = log₂(2 * 2^100) = 101
        let result = log_sum_exp([100.0, 100.0].into_iter());
        assert!((result - 101.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_mixed_large_small() {
        // log₂(2^100 + 2^0) ≈ 100 (the small value is negligible)
        let result = log_sum_exp([100.0, 0.0].into_iter());
        // 2^100 + 1 ≈ 2^100, so result should be very close to 100
        assert!((result - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_deleted_weight() {
        assert!(is_deleted_weight(f64::NEG_INFINITY));
        assert!(is_deleted_weight(DELETED_LOG_WEIGHT));
        assert!(!is_deleted_weight(0.0));
        assert!(!is_deleted_weight(-100.0));
        assert!(!is_deleted_weight(100.0));
    }

    #[test]
    fn test_log_sum_exp_with_deleted() {
        // Deleted weights (NEG_INFINITY) should not contribute to the sum
        let result = log_sum_exp([2.0, DELETED_LOG_WEIGHT, 2.0].into_iter());
        // log₂(4 + 0 + 4) = log₂(8) = 3
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_all_deleted() {
        // All deleted should return NEG_INFINITY
        let result = log_sum_exp([DELETED_LOG_WEIGHT, DELETED_LOG_WEIGHT].into_iter());
        assert!(result == f64::NEG_INFINITY);
    }
}
