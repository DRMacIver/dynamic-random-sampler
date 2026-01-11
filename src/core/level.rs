//! Level data structure for the dynamic random sampler.
//!
//! A Level represents a collection of ranges at a specific height in the tree.
//! At level 1, ranges contain actual elements (from level 0, the implicit leaf level).
//! At higher levels, ranges contain ranges from the previous level.
//!
//! Key components from the paper:
//! - Level table `T_ℓ`: Contains root ranges (degree 1) at level ℓ
//! - Internal ranges: Ranges with degree ≥ 2 that have parents in the next level
//! - `weight(T_ℓ)`: Total weight of all root ranges at level ℓ

use std::collections::HashMap;

use crate::core::{compute_range_number, log_sum_exp, Range};

/// A unique identifier for a range within a level.
pub type RangeId = usize;

/// A level in the tree data structure.
///
/// Manages all ranges at a specific level, tracking which are roots
/// (stored in the level table) and which have parents at the next level.
#[derive(Debug)]
pub struct Level {
    /// The level number (1-indexed; level 0 is the implicit element level)
    level_number: usize,
    /// All ranges at this level, keyed by their range number j
    /// A range `R_j^(ℓ)` is stored at key j
    ranges: HashMap<i32, Range>,
    /// Cache of total weight of all root ranges
    cached_root_total_log_weight: Option<f64>,
}

impl Level {
    /// Create a new empty level.
    ///
    /// # Arguments
    /// * `level_number` - The level number (must be ≥ 1)
    #[must_use]
    pub fn new(level_number: usize) -> Self {
        assert!(level_number >= 1, "Level number must be at least 1");
        Self {
            level_number,
            ranges: HashMap::new(),
            cached_root_total_log_weight: None,
        }
    }

    /// Get the level number.
    #[must_use]
    pub const fn level_number(&self) -> usize {
        self.level_number
    }

    /// Get the number of ranges at this level.
    #[must_use]
    pub fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Check if the level has no ranges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Get a range by its range number.
    #[must_use]
    pub fn get_range(&self, range_number: i32) -> Option<&Range> {
        self.ranges.get(&range_number)
    }

    /// Get a mutable reference to a range by its range number.
    pub fn get_range_mut(&mut self, range_number: i32) -> Option<&mut Range> {
        self.cached_root_total_log_weight = None;
        self.ranges.get_mut(&range_number)
    }

    /// Get or create a range with the given range number.
    ///
    /// If the range doesn't exist, creates an empty one.
    pub fn get_or_create_range(&mut self, range_number: i32) -> &mut Range {
        self.cached_root_total_log_weight = None;
        self.ranges
            .entry(range_number)
            .or_insert_with(|| Range::new(range_number))
    }

    /// Insert a child into the appropriate range based on its log-weight.
    ///
    /// Automatically determines the range number from the log-weight.
    ///
    /// # Arguments
    /// * `child_index` - The child's index
    /// * `child_log_weight` - The log₂ of the child's weight
    pub fn insert_child(&mut self, child_index: usize, child_log_weight: f64) {
        let range_number = compute_range_number(child_log_weight);
        let range = self.get_or_create_range(range_number);
        range.add_child(child_index, child_log_weight);
    }

    /// Remove a child from its range.
    ///
    /// # Arguments
    /// * `range_number` - The range number where the child is located
    /// * `child_index` - The child's index
    ///
    /// # Returns
    /// The removed child's log-weight, or None if not found
    pub fn remove_child(&mut self, range_number: i32, child_index: usize) -> Option<f64> {
        self.cached_root_total_log_weight = None;
        if let Some(range) = self.ranges.get_mut(&range_number) {
            let result = range.remove_child(child_index);
            // Remove empty ranges
            if range.is_empty() {
                self.ranges.remove(&range_number);
            }
            result
        } else {
            None
        }
    }

    /// Check if a range exists and is a root range (degree == 1).
    #[must_use]
    pub fn is_root_range(&self, range_number: i32) -> bool {
        self.ranges.get(&range_number).is_some_and(Range::is_root)
    }

    /// Get all root ranges at this level.
    ///
    /// Root ranges have exactly one child and are stored in the level table.
    pub fn root_ranges(&self) -> impl Iterator<Item = (i32, &Range)> {
        self.ranges
            .iter()
            .filter(|(_, r)| Range::is_root(r))
            .map(|(&n, r)| (n, r))
    }

    /// Get all non-root ranges at this level.
    ///
    /// Non-root ranges have degree ≥ 2 and will have parents at the next level.
    pub fn non_root_ranges(&self) -> impl Iterator<Item = (i32, &Range)> {
        self.ranges
            .iter()
            .filter(|(_, r)| !Range::is_root(r) && !Range::is_empty(r))
            .map(|(&n, r)| (n, r))
    }

    /// Get the number of root ranges.
    #[must_use]
    pub fn root_count(&self) -> usize {
        self.ranges.values().filter(|r| Range::is_root(r)).count()
    }

    /// Get the number of non-root ranges (ranges with degree ≥ 2).
    #[must_use]
    pub fn non_root_count(&self) -> usize {
        self.ranges
            .values()
            .filter(|r| !Range::is_root(r) && !Range::is_empty(r))
            .count()
    }

    /// Compute the total log-weight of all root ranges (weight of `T_ℓ`).
    ///
    /// This is used in Step 1 of the sampling algorithm.
    #[must_use]
    pub fn root_total_log_weight(&mut self) -> f64 {
        if let Some(cached) = self.cached_root_total_log_weight {
            return cached;
        }

        let total = log_sum_exp(
            self.ranges
                .values_mut()
                .filter(|r| Range::is_root(r))
                .map(Range::total_log_weight),
        );
        self.cached_root_total_log_weight = Some(total);
        total
    }

    /// Compute the total log-weight of all root ranges (immutable version).
    #[must_use]
    pub fn compute_root_total_log_weight(&self) -> f64 {
        log_sum_exp(
            self.ranges
                .values()
                .filter(|r| Range::is_root(r))
                .map(Range::compute_total_log_weight),
        )
    }

    /// Iterate over all ranges at this level.
    pub fn ranges(&self) -> impl Iterator<Item = (i32, &Range)> {
        self.ranges.iter().map(|(&n, r)| (n, r))
    }

    /// Get the sum of `2^j` for all non-empty root ranges.
    ///
    /// This is the `roots(T_ℓ)` value from the paper, used to find
    /// the first (largest) root range number.
    #[must_use]
    pub fn roots_sum(&self) -> f64 {
        self.ranges
            .iter()
            .filter(|(_, r)| Range::is_root(r))
            .map(|(&j, _)| 2.0_f64.powi(j))
            .sum()
    }

    /// Get the largest range number among root ranges.
    ///
    /// Returns None if there are no root ranges.
    #[must_use]
    pub fn largest_root_range_number(&self) -> Option<i32> {
        self.ranges
            .iter()
            .filter(|(_, r)| Range::is_root(r))
            .map(|(&j, _)| j)
            .max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Level Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_level() {
        let level = Level::new(1);
        assert_eq!(level.level_number(), 1);
        assert_eq!(level.range_count(), 0);
        assert!(level.is_empty());
    }

    #[test]
    fn test_create_level_at_higher_number() {
        let level = Level::new(5);
        assert_eq!(level.level_number(), 5);
    }

    #[test]
    #[should_panic(expected = "Level number must be at least 1")]
    fn test_create_level_zero_panics() {
        let _ = Level::new(0);
    }

    // -------------------------------------------------------------------------
    // Range Management Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_or_create_range() {
        let mut level = Level::new(1);

        // Create a new range
        let range = level.get_or_create_range(2);
        assert_eq!(range.range_number(), 2);
        assert!(range.is_empty());

        // Get the same range
        level.get_or_create_range(2).add_child(0, 1.0);
        let range = level.get_range(2).unwrap();
        assert_eq!(range.degree(), 1);
    }

    #[test]
    fn test_insert_child_creates_range() {
        let mut level = Level::new(1);

        // Insert child with weight 2 (log₂(2) = 1) → range 2
        level.insert_child(0, 1.0);

        assert_eq!(level.range_count(), 1);
        let range = level.get_range(2).unwrap();
        assert_eq!(range.degree(), 1);
    }

    #[test]
    fn test_insert_multiple_children_same_range() {
        let mut level = Level::new(1);

        // Insert children with weights in [2, 4) → range 2
        level.insert_child(0, 1.0); // weight 2
        level.insert_child(1, 1.5); // weight ~2.83
        level.insert_child(2, 1.9); // weight ~3.73

        assert_eq!(level.range_count(), 1);
        let range = level.get_range(2).unwrap();
        assert_eq!(range.degree(), 3);
    }

    #[test]
    fn test_insert_children_different_ranges() {
        let mut level = Level::new(1);

        // Insert children with different weight ranges
        level.insert_child(0, 0.0); // weight 1 → range 1
        level.insert_child(1, 1.0); // weight 2 → range 2
        level.insert_child(2, 2.0); // weight 4 → range 3

        assert_eq!(level.range_count(), 3);
        assert!(level.get_range(1).is_some());
        assert!(level.get_range(2).is_some());
        assert!(level.get_range(3).is_some());
    }

    #[test]
    fn test_remove_child() {
        let mut level = Level::new(1);

        level.insert_child(0, 1.0); // range 2
        level.insert_child(1, 1.5); // range 2

        let removed = level.remove_child(2, 0);
        assert_eq!(removed, Some(1.0));
        assert_eq!(level.get_range(2).unwrap().degree(), 1);
    }

    #[test]
    fn test_remove_last_child_removes_range() {
        let mut level = Level::new(1);

        level.insert_child(0, 1.0); // range 2

        let removed = level.remove_child(2, 0);
        assert!(removed.is_some());
        assert!(level.get_range(2).is_none());
        assert!(level.is_empty());
    }

    // -------------------------------------------------------------------------
    // Root Range Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_root_vs_non_root_ranges() {
        let mut level = Level::new(1);

        // Single child → root range
        level.insert_child(0, 1.0); // range 2, degree 1
                                    // Multiple children → non-root range
        level.insert_child(1, 2.0); // range 3
        level.insert_child(2, 2.5); // range 3

        assert!(level.is_root_range(2)); // degree 1
        assert!(!level.is_root_range(3)); // degree 2
    }

    #[test]
    fn test_root_count() {
        let mut level = Level::new(1);

        level.insert_child(0, 1.0); // range 2, degree 1 (root)
        level.insert_child(1, 2.0); // range 3
        level.insert_child(2, 2.5); // range 3, degree 2 (non-root)
        level.insert_child(3, 0.0); // range 1, degree 1 (root)

        assert_eq!(level.root_count(), 2);
        assert_eq!(level.non_root_count(), 1);
    }

    #[test]
    fn test_iterate_root_ranges() {
        let mut level = Level::new(1);

        level.insert_child(0, 1.0); // range 2, root
        level.insert_child(1, 2.0); // range 3
        level.insert_child(2, 2.5); // range 3, non-root
        level.insert_child(3, 0.0); // range 1, root

        let roots: Vec<_> = level.root_ranges().collect();
        assert_eq!(roots.len(), 2);

        let root_numbers: Vec<_> = roots.iter().map(|(n, _)| *n).collect();
        assert!(root_numbers.contains(&1));
        assert!(root_numbers.contains(&2));
    }

    #[test]
    fn test_iterate_non_root_ranges() {
        let mut level = Level::new(1);

        level.insert_child(0, 1.0); // range 2, root
        level.insert_child(1, 2.0); // range 3
        level.insert_child(2, 2.5); // range 3, non-root

        let non_roots: Vec<_> = level.non_root_ranges().collect();
        assert_eq!(non_roots.len(), 1);
        assert_eq!(non_roots[0].0, 3);
    }

    // -------------------------------------------------------------------------
    // Total Weight Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_root_total_log_weight_empty() {
        let mut level = Level::new(1);
        let total = level.root_total_log_weight();
        assert!(total.is_infinite() && total < 0.0);
    }

    #[test]
    fn test_root_total_log_weight_single_root() {
        let mut level = Level::new(1);
        level.insert_child(0, 1.0); // weight 2, root

        let total = level.root_total_log_weight().exp2();
        assert!((total - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_root_total_log_weight_multiple_roots() {
        let mut level = Level::new(1);
        level.insert_child(0, 1.0); // weight 2, range 2, root
        level.insert_child(1, 2.0); // weight 4, range 3, root

        // Total of roots = 2 + 4 = 6
        let total = level.root_total_log_weight().exp2();
        assert!((total - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_root_total_excludes_non_roots() {
        let mut level = Level::new(1);
        level.insert_child(0, 1.0); // weight 2, range 2, root
        level.insert_child(1, 2.0); // weight 4, range 3
        level.insert_child(2, 2.5); // weight ~5.66, range 3 → non-root

        // Total of roots = 2 (only range 2)
        let total = level.root_total_log_weight().exp2();
        assert!((total - 2.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Roots Sum and Largest Range Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_roots_sum() {
        let mut level = Level::new(1);
        level.insert_child(0, 1.0); // range 2, root → contributes 2^2 = 4
        level.insert_child(1, 0.0); // range 1, root → contributes 2^1 = 2

        let sum = level.roots_sum();
        assert!((sum - 6.0).abs() < 1e-10); // 4 + 2 = 6
    }

    #[test]
    fn test_largest_root_range_number() {
        let mut level = Level::new(1);
        level.insert_child(0, 1.0); // range 2
        level.insert_child(1, 0.0); // range 1
        level.insert_child(2, 2.0); // range 3
        level.insert_child(3, 2.5); // range 3 (now degree 2, non-root)

        // Root ranges are 1 and 2; range 3 is non-root
        assert_eq!(level.largest_root_range_number(), Some(2));
    }

    #[test]
    fn test_largest_root_range_number_empty() {
        let level = Level::new(1);
        assert_eq!(level.largest_root_range_number(), None);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_negative_range_numbers() {
        let mut level = Level::new(1);

        // Weight 0.5 → log₂(0.5) = -1 → range 0
        level.insert_child(0, -1.0);
        // Weight 0.25 → log₂(0.25) = -2 → range -1
        level.insert_child(1, -2.0);

        assert!(level.get_range(0).is_some());
        assert!(level.get_range(-1).is_some());
    }

    #[test]
    fn test_caching_invalidation() {
        let mut level = Level::new(1);
        level.insert_child(0, 1.0);

        let total1 = level.root_total_log_weight();

        // Modification invalidates cache
        level.insert_child(1, 0.0);
        let total2 = level.root_total_log_weight();

        assert!(total2 > total1); // Added another root
    }
}
