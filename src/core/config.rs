//! Configuration for Section 4 optimizations.
//!
//! This module implements the tolerance factor and degree bound from Section 4
//! of the paper "Dynamic Generation of Discrete Random Variates" that enable
//! O(log* N) amortized update time.
//!
//! Key parameters:
//! - **Tolerance factor b (`$0 \leq b < 1$`)**: Allows "lazy updating" by expanding
//!   the weight interval for range j from [2^(j-1), 2^j) to [(1-b)2^(j-1), (2+b)2^(j-1)).
//!   This prevents cascading updates when weights change slightly.
//!
//! - **Degree bound d (`$d \geq 2$`)**: The minimum number of children required for a
//!   range to be considered non-root. The paper recommends `$d \geq 16$` for optimal
//!   amortized bounds, with `$d = 1/2 * ((2+b)/(1-b))^{2^{2c}}$` for integer `$c \geq 0$`.
//!
//! With b = 0.4 and d = 32, we get O(log* N) amortized update time.

/// Configuration for the Section 4 optimizations.
///
/// Default configuration uses the paper's recommended values of b = 0.4 and d = 32.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct OptimizationConfig {
    /// Tolerance factor b, where `$0 \leq b < 1$`.
    ///
    /// Controls the expanded range interval:
    /// - Standard interval for range j: [2^(j-1), 2^j)
    /// - With tolerance: `$[(1-b) \cdot 2^{j-1}, (2+b) \cdot 2^{j-1})$`
    ///
    /// Higher values reduce update propagation but increase rejection rate during sampling.
    /// Recommended: 0.4 (from paper).
    tolerance: f64,

    /// Minimum degree bound d for non-root ranges.
    ///
    /// A range must have at least d children to have a parent in the next level.
    /// Ranges with fewer than d children become roots stored in the level table.
    ///
    /// Recommended: 32 (from paper, with b = 0.4 and c = 1).
    min_degree: usize,
}

impl Default for OptimizationConfig {
    /// Create configuration with the paper's recommended values.
    ///
    /// Uses b = 0.4 and d = 32 which achieve O(log* N) amortized update time.
    fn default() -> Self {
        Self::optimized()
    }
}

impl OptimizationConfig {
    /// Create a new configuration with custom parameters.
    ///
    /// # Arguments
    /// * `tolerance` - Tolerance factor b (must be in [0, 1))
    /// * `min_degree` - Minimum degree bound d (must be >= 2)
    ///
    /// # Panics
    /// Panics if tolerance is not in [0, 1) or `min_degree` is less than 2.
    #[must_use]
    pub fn new(tolerance: f64, min_degree: usize) -> Self {
        assert!(
            (0.0..1.0).contains(&tolerance),
            "Tolerance must be in [0, 1), got {tolerance}"
        );
        assert!(
            min_degree >= 2,
            "Minimum degree must be at least 2, got {min_degree}"
        );
        Self {
            tolerance,
            min_degree,
        }
    }

    /// Create configuration with the paper's recommended optimized values.
    ///
    /// Uses b = 0.4 and d = 32 for O(log* N) amortized update time.
    #[must_use]
    pub const fn optimized() -> Self {
        Self {
            tolerance: 0.4,
            min_degree: 32,
        }
    }

    /// Create basic configuration without Section 4 optimizations.
    ///
    /// Uses b = 0 (no tolerance) and d = 2 (standard non-root definition).
    /// This gives O(2^(log* N)) update time.
    #[must_use]
    pub const fn basic() -> Self {
        Self {
            tolerance: 0.0,
            min_degree: 2,
        }
    }

    /// Get the tolerance factor b.
    #[inline]
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Get the minimum degree bound d.
    #[inline]
    #[must_use]
    pub const fn min_degree(&self) -> usize {
        self.min_degree
    }

    /// Check if a degree qualifies as a root (has parent in next level).
    ///
    /// A range is a root if it has fewer than `min_degree` children.
    #[inline]
    #[must_use]
    pub const fn is_root_degree(&self, degree: usize) -> bool {
        degree < self.min_degree
    }

    /// Check if a degree qualifies as non-root (becomes child of next level).
    ///
    /// A range is non-root if it has at least `min_degree` children.
    #[inline]
    #[must_use]
    pub const fn is_non_root_degree(&self, degree: usize) -> bool {
        degree >= self.min_degree
    }

    /// Compute the lower bound of the tolerated interval for range j.
    ///
    /// With tolerance b, range j accepts weights from `$[(1-b) \cdot 2^{j-1}, (2+b) \cdot 2^{j-1})$`.
    /// In log space: `$\log_2((1-b) \cdot 2^{j-1}) = \log_2(1-b) + j - 1$`
    #[inline]
    #[must_use]
    pub fn tolerated_lower_log(&self, range_number: i32) -> f64 {
        let j = f64::from(range_number);
        if self.tolerance == 0.0 {
            j - 1.0
        } else {
            (1.0 - self.tolerance).log2() + j - 1.0
        }
    }

    /// Compute the upper bound of the tolerated interval for range j.
    ///
    /// With tolerance b, range j accepts weights up to `$(2+b) \cdot 2^{j-1}$`.
    /// In log space: `$\log_2((2+b) \cdot 2^{j-1}) = \log_2(2+b) + j - 1$`
    #[inline]
    #[must_use]
    pub fn tolerated_upper_log(&self, range_number: i32) -> f64 {
        let j = f64::from(range_number);
        if self.tolerance == 0.0 {
            j
        } else {
            (2.0 + self.tolerance).log2() + j - 1.0
        }
    }

    /// Check if a weight (given as `$\log_2(w)$`) is within the tolerated interval for range j.
    ///
    /// With tolerance b, the tolerated interval is `$[(1-b) \cdot 2^{j-1}, (2+b) \cdot 2^{j-1})$`.
    /// The element only needs to change range if it falls outside this interval.
    #[inline]
    #[must_use]
    pub fn weight_in_tolerated_range(&self, range_number: i32, log_weight: f64) -> bool {
        let lower = self.tolerated_lower_log(range_number);
        let upper = self.tolerated_upper_log(range_number);
        log_weight >= lower && log_weight < upper
    }

    /// Compute the tolerance amount (in log space) for a range.
    ///
    /// This is the minimum weight change needed to potentially move an element
    /// out of its current range: `$b \cdot 2^{j-1}$`.
    /// In log space: `$\log_2(b \cdot 2^{j-1}) = \log_2(b) + j - 1$`
    ///
    /// Returns `NEG_INFINITY` if tolerance is 0.
    #[inline]
    #[must_use]
    pub fn tolerance_amount_log(&self, range_number: i32) -> f64 {
        if self.tolerance == 0.0 {
            f64::NEG_INFINITY
        } else {
            self.tolerance.log2() + f64::from(range_number) - 1.0
        }
    }

    /// Compute the expected acceptance probability for rejection sampling.
    ///
    /// With tolerance b, the acceptance probability is at least `$1/(2+b) \approx 0.42$` for b=0.4.
    #[inline]
    #[must_use]
    pub fn min_acceptance_probability(&self) -> f64 {
        1.0 / (2.0 + self.tolerance)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Configuration Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let config = OptimizationConfig::default();
        assert!((config.tolerance() - 0.4).abs() < 1e-10);
        assert_eq!(config.min_degree(), 32);
    }

    #[test]
    fn test_optimized_config() {
        let config = OptimizationConfig::optimized();
        assert!((config.tolerance() - 0.4).abs() < 1e-10);
        assert_eq!(config.min_degree(), 32);
    }

    #[test]
    fn test_basic_config() {
        let config = OptimizationConfig::basic();
        assert!((config.tolerance()).abs() < 1e-10);
        assert_eq!(config.min_degree(), 2);
    }

    #[test]
    fn test_custom_config() {
        let config = OptimizationConfig::new(0.3, 16);
        assert!((config.tolerance() - 0.3).abs() < 1e-10);
        assert_eq!(config.min_degree(), 16);
    }

    #[test]
    #[should_panic(expected = "Tolerance must be in [0, 1)")]
    fn test_invalid_tolerance_too_high() {
        let _ = OptimizationConfig::new(1.0, 16);
    }

    #[test]
    #[should_panic(expected = "Tolerance must be in [0, 1)")]
    fn test_invalid_tolerance_negative() {
        let _ = OptimizationConfig::new(-0.1, 16);
    }

    #[test]
    #[should_panic(expected = "Minimum degree must be at least 2")]
    fn test_invalid_min_degree() {
        let _ = OptimizationConfig::new(0.4, 1);
    }

    // -------------------------------------------------------------------------
    // Root/Non-Root Degree Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_root_degree_basic() {
        let config = OptimizationConfig::basic();
        assert!(config.is_root_degree(0));
        assert!(config.is_root_degree(1));
        assert!(!config.is_root_degree(2));
        assert!(!config.is_root_degree(10));
    }

    #[test]
    fn test_is_root_degree_optimized() {
        let config = OptimizationConfig::optimized();
        assert!(config.is_root_degree(0));
        assert!(config.is_root_degree(1));
        assert!(config.is_root_degree(31));
        assert!(!config.is_root_degree(32));
        assert!(!config.is_root_degree(100));
    }

    #[test]
    fn test_is_non_root_degree() {
        let config = OptimizationConfig::new(0.4, 16);
        assert!(!config.is_non_root_degree(0));
        assert!(!config.is_non_root_degree(15));
        assert!(config.is_non_root_degree(16));
        assert!(config.is_non_root_degree(100));
    }

    // -------------------------------------------------------------------------
    // Tolerated Interval Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tolerated_lower_log_basic() {
        let config = OptimizationConfig::basic();
        // With b=0, lower bound is 2^(j-1), so log_2 = j-1
        assert!((config.tolerated_lower_log(1) - 0.0).abs() < 1e-10);
        assert!((config.tolerated_lower_log(2) - 1.0).abs() < 1e-10);
        assert!((config.tolerated_lower_log(3) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_tolerated_upper_log_basic() {
        let config = OptimizationConfig::basic();
        // With b=0, upper bound is 2^j, so log_2 = j
        assert!((config.tolerated_upper_log(1) - 1.0).abs() < 1e-10);
        assert!((config.tolerated_upper_log(2) - 2.0).abs() < 1e-10);
        assert!((config.tolerated_upper_log(3) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tolerated_interval_optimized() {
        let config = OptimizationConfig::optimized();
        // With b=0.4, range j=2:
        // Lower: (1-0.4)*2^1 = 0.6*2 = 1.2, log_2 ~= 0.263
        // Upper: (2+0.4)*2^1 = 2.4*2 = 4.8, log_2 ~= 2.263
        let lower = config.tolerated_lower_log(2);
        let upper = config.tolerated_upper_log(2);

        let expected_lower = (0.6_f64 * 2.0).log2();
        let expected_upper = (2.4_f64 * 2.0).log2();

        assert!((lower - expected_lower).abs() < 1e-10);
        assert!((upper - expected_upper).abs() < 1e-10);
    }

    #[test]
    fn test_weight_in_tolerated_range_basic() {
        let config = OptimizationConfig::basic();
        // Range 2: [2, 4) in linear space, [1, 2) in log space
        assert!(config.weight_in_tolerated_range(2, 1.0)); // weight 2
        assert!(config.weight_in_tolerated_range(2, 1.5)); // weight ~2.83
        assert!(!config.weight_in_tolerated_range(2, 2.0)); // weight 4, at boundary
        assert!(!config.weight_in_tolerated_range(2, 0.9)); // weight ~1.87
    }

    #[test]
    fn test_weight_in_tolerated_range_optimized() {
        let config = OptimizationConfig::optimized();
        // Range 2 with b=0.4: [1.2, 4.8) in linear space
        // log_2(1.2) ~= 0.263, log_2(4.8) ~= 2.263

        // Weight 2 (log=1) should be in tolerated range
        assert!(config.weight_in_tolerated_range(2, 1.0));

        // Weight 1.5 (log ~= 0.585) should be in tolerated range
        assert!(config.weight_in_tolerated_range(2, 1.5_f64.log2()));

        // Weight 1.0 (log=0) should NOT be in tolerated range (< 1.2)
        assert!(!config.weight_in_tolerated_range(2, 0.0));

        // Weight 5 (log ~= 2.32) should NOT be in tolerated range (> 4.8)
        assert!(!config.weight_in_tolerated_range(2, 5.0_f64.log2()));
    }

    // -------------------------------------------------------------------------
    // Tolerance Amount Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tolerance_amount_log_basic() {
        let config = OptimizationConfig::basic();
        // With b=0, tolerance amount is 0, log_2 = -inf
        assert!(config.tolerance_amount_log(2).is_infinite());
        assert!(config.tolerance_amount_log(2) < 0.0);
    }

    #[test]
    fn test_tolerance_amount_log_optimized() {
        let config = OptimizationConfig::optimized();
        // With b=0.4, range j=2: tolerance = 0.4*2^1 = 0.8
        // log_2(0.8) ~= -0.322
        let tolerance = config.tolerance_amount_log(2);
        let expected = 0.8_f64.log2();
        assert!((tolerance - expected).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Acceptance Probability Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_min_acceptance_probability_basic() {
        let config = OptimizationConfig::basic();
        // With b=0, acceptance probability is 1/2 = 0.5
        assert!((config.min_acceptance_probability() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_min_acceptance_probability_optimized() {
        let config = OptimizationConfig::optimized();
        // With b=0.4, acceptance probability is 1/2.4 ~= 0.417
        let expected = 1.0 / 2.4;
        assert!((config.min_acceptance_probability() - expected).abs() < 1e-10);
    }
}
