//! Range data structure for the dynamic random sampler.
//!
//! A Range `R_j^(ℓ)` represents a collection of elements at level ℓ whose weights
//! fall in the interval `[2^(j-1), 2^j)`. At level 1, these are actual elements;
//! at higher levels, these are ranges from the previous level.
//!
//! Key properties from the paper:
//! - Range number j determines the weight interval `[2^(j-1), 2^j)`
//! - Total weight of a range is the sum of all children's weights
//! - Degree is the number of children
//! - Root ranges have degree 1 (only one child)
//! - Non-root ranges have degree >= 2

use std::collections::HashMap;

use crate::core::log_sum_exp;

/// A child entry in a range, identified by its index and log-weight.
#[derive(Debug, Clone, Copy)]
pub struct Child {
    /// Index of the child (element index at level 1, or range ID at higher levels)
    pub index: usize,
    /// Log₂ of the child's weight
    pub log_weight: f64,
}

/// A range in the tree structure.
///
/// Stores children whose weights fall in `[2^(j-1), 2^j)` where j is the range number.
#[derive(Debug)]
pub struct Range {
    /// The range number j, determining the weight interval `[2^(j-1), 2^j)`
    range_number: i32,
    /// Children stored in this range, keyed by child index for O(1) lookup
    children: HashMap<usize, f64>,
    /// Cached total log-weight (invalidated on modifications)
    cached_total_log_weight: Option<f64>,
}

impl Range {
    /// Create a new empty range with the given range number.
    ///
    /// # Arguments
    /// * `range_number` - The range number j, determining interval `[2^(j-1), 2^j)`
    #[must_use]
    pub fn new(range_number: i32) -> Self {
        Self {
            range_number,
            children: HashMap::new(),
            cached_total_log_weight: None,
        }
    }

    /// Get the range number j.
    #[must_use]
    pub const fn range_number(&self) -> i32 {
        self.range_number
    }

    /// Get the number of children (degree) in this range.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.children.len()
    }

    /// Check if the range is empty (has no children).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Check if this is a root range (has exactly one child).
    ///
    /// Root ranges are stored in level tables rather than parent ranges.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.children.len() == 1
    }

    /// Add a child to this range.
    ///
    /// # Arguments
    /// * `index` - The child's index (element index or range ID)
    /// * `log_weight` - The log₂ of the child's weight
    ///
    /// # Panics
    /// Panics if a child with this index already exists. Use `update_child_weight` instead.
    pub fn add_child(&mut self, index: usize, log_weight: f64) {
        assert!(
            !self.children.contains_key(&index),
            "Child with index {index} already exists"
        );
        self.children.insert(index, log_weight);
        self.cached_total_log_weight = None;
    }

    /// Remove a child from this range.
    ///
    /// # Arguments
    /// * `index` - The child's index to remove
    ///
    /// # Returns
    /// The log-weight of the removed child, or None if not found
    pub fn remove_child(&mut self, index: usize) -> Option<f64> {
        let result = self.children.remove(&index);
        if result.is_some() {
            self.cached_total_log_weight = None;
        }
        result
    }

    /// Update the weight of an existing child.
    ///
    /// # Arguments
    /// * `index` - The child's index
    /// * `new_log_weight` - The new log₂ weight
    ///
    /// # Returns
    /// The old log-weight, or None if child not found
    pub fn update_child_weight(&mut self, index: usize, new_log_weight: f64) -> Option<f64> {
        if let Some(old_weight) = self.children.get_mut(&index) {
            let old = *old_weight;
            *old_weight = new_log_weight;
            self.cached_total_log_weight = None;
            Some(old)
        } else {
            None
        }
    }

    /// Get a child by index.
    ///
    /// # Arguments
    /// * `index` - The child's index
    ///
    /// # Returns
    /// The child's log-weight if found
    #[must_use]
    pub fn get_child(&self, index: usize) -> Option<f64> {
        self.children.get(&index).copied()
    }

    /// Check if a child with the given index exists.
    #[must_use]
    pub fn contains_child(&self, index: usize) -> bool {
        self.children.contains_key(&index)
    }

    /// Get the total log-weight of all children.
    ///
    /// Uses log-sum-exp for numerical stability.
    /// Returns `NEG_INFINITY` if the range is empty.
    #[must_use]
    pub fn total_log_weight(&mut self) -> f64 {
        if let Some(cached) = self.cached_total_log_weight {
            return cached;
        }

        let total = log_sum_exp(self.children.values().copied());
        self.cached_total_log_weight = Some(total);
        total
    }

    /// Get the total log-weight without caching (immutable version).
    #[must_use]
    pub fn compute_total_log_weight(&self) -> f64 {
        log_sum_exp(self.children.values().copied())
    }

    /// Iterate over all children.
    ///
    /// # Returns
    /// Iterator yielding `(index, log_weight)` pairs
    pub fn children(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.children.iter().map(|(&idx, &lw)| (idx, lw))
    }

    /// Get a random child by bucket index.
    ///
    /// For the rejection method, we need to select children uniformly at random.
    /// This returns the child at a given "bucket" position (0 to degree-1).
    ///
    /// Note: `HashMap` iteration order is arbitrary but consistent, which is
    /// fine for uniform random selection.
    ///
    /// # Arguments
    /// * `bucket` - The bucket index (0 to degree-1)
    ///
    /// # Returns
    /// `(child_index, child_log_weight)` or None if bucket is out of range
    #[must_use]
    pub fn get_child_by_bucket(&self, bucket: usize) -> Option<(usize, f64)> {
        self.children
            .iter()
            .nth(bucket)
            .map(|(&idx, &lw)| (idx, lw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Range Creation and Basic Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_empty_range() {
        let range = Range::new(2);
        assert_eq!(range.range_number(), 2);
        assert_eq!(range.degree(), 0);
        assert!(range.is_empty());
    }

    #[test]
    fn test_create_range_with_negative_number() {
        let range = Range::new(-3);
        assert_eq!(range.range_number(), -3);
        assert!(range.is_empty());
    }

    #[test]
    fn test_add_child_to_range() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0); // log₂(2) = 1, so weight 2
        assert_eq!(range.degree(), 1);
        assert!(!range.is_empty());
    }

    #[test]
    fn test_add_multiple_children() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0);
        range.add_child(1, 1.5);
        range.add_child(2, 1.9);
        assert_eq!(range.degree(), 3);
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_add_duplicate_child_panics() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0);
        range.add_child(0, 1.5); // Should panic
    }

    #[test]
    fn test_remove_child_from_range() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0);
        range.add_child(1, 1.5);
        assert_eq!(range.degree(), 2);

        let removed = range.remove_child(0);
        assert_eq!(removed, Some(1.0));
        assert_eq!(range.degree(), 1);
        assert!(!range.contains_child(0));
        assert!(range.contains_child(1));
    }

    #[test]
    fn test_remove_nonexistent_child() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0);
        let removed = range.remove_child(999);
        assert_eq!(removed, None);
        assert_eq!(range.degree(), 1);
    }

    // -------------------------------------------------------------------------
    // Total Weight Computation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_total_weight_empty_range() {
        let mut range = Range::new(2);
        let total = range.total_log_weight();
        assert!(total.is_infinite() && total < 0.0);
    }

    #[test]
    fn test_total_weight_single_child() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0); // weight 2
        let total_log_weight = range.total_log_weight();
        let total_weight = total_log_weight.exp2();
        assert!((total_weight - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_weight_multiple_children() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0); // weight 2
        range.add_child(1, 2.0_f64.log2().log2()); // Actually, let me fix: log₂(3) ≈ 1.585
                                                   // Let's use clearer values:
                                                   // weight 2.0 has log₂ = 1.0
                                                   // weight 3.0 has log₂ ≈ 1.585
        let mut range2 = Range::new(2);
        range2.add_child(0, 2.0_f64.log2()); // weight 2, log = 1
        range2.add_child(1, 3.0_f64.log2()); // weight 3, log ≈ 1.585
        let total_weight = range2.total_log_weight().exp2();
        assert!((total_weight - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_weight_numerical_stability() {
        // Test with very large weights
        let mut range = Range::new(100);
        range.add_child(0, 100.0); // 2^100
        range.add_child(1, 100.0); // 2^100
        let total_log = range.total_log_weight();
        // Expected: log₂(2 * 2^100) = 101
        assert!((total_log - 101.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_weight_caching() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0);
        range.add_child(1, 2.0);

        // First call computes
        let total1 = range.total_log_weight();
        // Second call should use cache (same result)
        let total2 = range.total_log_weight();
        assert!((total1 - total2).abs() < 1e-15);

        // Modification invalidates cache
        range.add_child(2, 1.5);
        let total3 = range.total_log_weight();
        assert!(total3 > total1); // Should be different now
    }

    // -------------------------------------------------------------------------
    // Degree and Root Status Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_degree_counts_children() {
        let mut range = Range::new(2);
        assert_eq!(range.degree(), 0);
        range.add_child(0, 1.0);
        assert_eq!(range.degree(), 1);
        range.add_child(1, 1.5);
        assert_eq!(range.degree(), 2);
        range.add_child(2, 1.9);
        assert_eq!(range.degree(), 3);
    }

    #[test]
    fn test_root_range_has_degree_one() {
        let mut range = Range::new(2);
        assert!(!range.is_root()); // Empty is not root

        range.add_child(0, 1.0);
        assert!(range.is_root()); // Degree 1 is root

        range.add_child(1, 1.5);
        assert!(!range.is_root()); // Degree 2 is not root
    }

    // -------------------------------------------------------------------------
    // Child Access Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_child() {
        let mut range = Range::new(2);
        range.add_child(5, 1.5);
        range.add_child(10, 2.5);

        assert_eq!(range.get_child(5), Some(1.5));
        assert_eq!(range.get_child(10), Some(2.5));
        assert_eq!(range.get_child(999), None);
    }

    #[test]
    fn test_contains_child() {
        let mut range = Range::new(2);
        range.add_child(5, 1.5);

        assert!(range.contains_child(5));
        assert!(!range.contains_child(6));
    }

    #[test]
    fn test_iterate_over_children() {
        let mut range = Range::new(2);
        range.add_child(5, 1.5);
        range.add_child(10, 2.5);

        let children: Vec<_> = range.children().collect();
        assert_eq!(children.len(), 2);

        // Check both children are present (order may vary)
        let has_5 = children
            .iter()
            .any(|&(idx, lw)| idx == 5 && (lw - 1.5).abs() < 1e-10);
        let has_10 = children
            .iter()
            .any(|&(idx, lw)| idx == 10 && (lw - 2.5).abs() < 1e-10);
        assert!(has_5);
        assert!(has_10);
    }

    #[test]
    fn test_get_child_by_bucket() {
        let mut range = Range::new(2);
        range.add_child(5, 1.5);
        range.add_child(10, 2.5);

        // Bucket 0 and 1 should return the two children
        let child0 = range.get_child_by_bucket(0);
        let child1 = range.get_child_by_bucket(1);
        let child2 = range.get_child_by_bucket(2);

        assert!(child0.is_some());
        assert!(child1.is_some());
        assert!(child2.is_none());

        // Both children should be different
        assert_ne!(child0.unwrap().0, child1.unwrap().0);
    }

    // -------------------------------------------------------------------------
    // Weight Update Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_child_weight() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0); // weight 2
        range.add_child(1, 3.0_f64.log2()); // weight 3
                                            // Initial total: 5.0

        let old = range.update_child_weight(0, 2.5_f64.log2());
        assert_eq!(old, Some(1.0));

        // New total: 2.5 + 3 = 5.5
        let total = range.total_log_weight().exp2();
        assert!((total - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_update_nonexistent_child() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0);

        let old = range.update_child_weight(999, 2.0);
        assert_eq!(old, None);
    }

    #[test]
    fn test_compute_total_log_weight_immutable() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0);
        range.add_child(1, 2.0);

        // Immutable version works without mutable reference
        let total = range.compute_total_log_weight();
        assert!(total > 0.0);

        // Can call multiple times on immutable reference
        let total2 = range.compute_total_log_weight();
        assert!((total - total2).abs() < 1e-15);
    }
}
