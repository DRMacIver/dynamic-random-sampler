//! Range data structure for the dynamic random sampler.
//!
//! A Range `$R_j^{(\ell)}$` represents a collection of elements at level `$\ell$` whose weights
//! fall in the interval `[2^(j-1), 2^j)`. At level 1, these are actual elements;
//! at higher levels, these are ranges from the previous level.
//!
//! Key properties from the paper:
//! - Range number j determines the weight interval `[2^(j-1), 2^j)`
//! - Total weight of a range is the sum of all children's weights
//! - Degree is the number of children
//! - Root ranges have degree 1 (only one child)
//! - Non-root ranges have degree >= 2
//!
//! # Implementation Note
//!
//! Children are stored in a `Vec` for O(1) random access (critical for rejection
//! sampling performance) with a `HashMap` index for O(1) lookup by child ID.

use std::collections::HashMap;

use crate::core::log_sum_exp;

/// A child entry in a range, identified by its index and log-weight.
#[derive(Debug, Clone, Copy)]
pub struct Child {
    /// Index of the child (element index at level 1, or range ID at higher levels)
    pub index: usize,
    /// `$\log_2$` of the child's weight
    pub log_weight: f64,
}

/// A range in the tree structure.
///
/// Stores children whose weights fall in `[2^(j-1), 2^j)` where j is the range number.
///
/// Uses dual storage for efficient operations:
/// - `children_vec`: O(1) random access for rejection sampling
/// - `children_idx`: O(1) lookup by child index for updates/removals
#[derive(Debug)]
pub struct Range {
    /// The range number j, determining the weight interval `[2^(j-1), 2^j)`
    range_number: i32,
    /// Children stored as a Vec for O(1) random access
    children_vec: Vec<Child>,
    /// Index mapping: `child_index` -> position in `children_vec`
    children_idx: HashMap<usize, usize>,
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
            children_vec: Vec::new(),
            children_idx: HashMap::new(),
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
    pub const fn degree(&self) -> usize {
        self.children_vec.len()
    }

    /// Check if the range is empty (has no children).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.children_vec.is_empty()
    }

    /// Check if this is a root range (has exactly one child).
    ///
    /// Root ranges are stored in level tables rather than parent ranges.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.children_vec.len() == 1
    }

    /// Add a child to this range.
    ///
    /// # Arguments
    /// * `index` - The child's index (element index or range ID)
    /// * `log_weight` - The `$\log_2$` of the child's weight
    ///
    /// # Panics
    /// Panics if a child with this index already exists. Use `update_child_weight` instead.
    pub fn add_child(&mut self, index: usize, log_weight: f64) {
        assert!(
            !self.children_idx.contains_key(&index),
            "Child with index {index} already exists"
        );
        let pos = self.children_vec.len();
        self.children_vec.push(Child { index, log_weight });
        self.children_idx.insert(index, pos);
        self.cached_total_log_weight = None;
    }

    /// Remove a child from this range.
    ///
    /// Uses swap-remove for O(1) removal while maintaining O(1) random access.
    ///
    /// # Arguments
    /// * `index` - The child's index to remove
    ///
    /// # Returns
    /// The log-weight of the removed child, or None if not found
    pub fn remove_child(&mut self, index: usize) -> Option<f64> {
        let pos = self.children_idx.remove(&index)?;
        let removed = self.children_vec.swap_remove(pos);

        // If we swapped an element (not removing last), update the moved element's index
        if pos < self.children_vec.len() {
            let moved_child_index = self.children_vec[pos].index;
            self.children_idx.insert(moved_child_index, pos);
        }

        self.cached_total_log_weight = None;
        Some(removed.log_weight)
    }

    /// Update the weight of an existing child.
    ///
    /// # Arguments
    /// * `index` - The child's index
    /// * `new_log_weight` - The new `$\log_2$` weight
    ///
    /// # Returns
    /// The old log-weight, or None if child not found
    pub fn update_child_weight(&mut self, index: usize, new_log_weight: f64) -> Option<f64> {
        let &pos = self.children_idx.get(&index)?;
        let old = self.children_vec[pos].log_weight;
        self.children_vec[pos].log_weight = new_log_weight;
        self.cached_total_log_weight = None;
        Some(old)
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
        let &pos = self.children_idx.get(&index)?;
        Some(self.children_vec[pos].log_weight)
    }

    /// Check if a child with the given index exists.
    #[must_use]
    pub fn contains_child(&self, index: usize) -> bool {
        self.children_idx.contains_key(&index)
    }

    /// Get the total log-weight of all children (mutable version with caching).
    ///
    /// Uses log-sum-exp for numerical stability.
    /// Returns `NEG_INFINITY` if the range is empty.
    /// Caches the result for future calls.
    #[must_use]
    pub fn total_log_weight(&mut self) -> f64 {
        if let Some(cached) = self.cached_total_log_weight {
            return cached;
        }

        let total = log_sum_exp(self.children_vec.iter().map(|c| c.log_weight));
        self.cached_total_log_weight = Some(total);
        total
    }

    /// Get the total log-weight of all children (immutable version without caching).
    ///
    /// Uses log-sum-exp for numerical stability.
    /// Returns `NEG_INFINITY` if the range is empty.
    #[must_use]
    pub fn compute_total_log_weight(&self) -> f64 {
        log_sum_exp(self.children_vec.iter().map(|c| c.log_weight))
    }

    /// Iterate over all children.
    ///
    /// # Returns
    /// Iterator yielding `(index, log_weight)` pairs
    pub fn children(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.children_vec.iter().map(|c| (c.index, c.log_weight))
    }

    /// Get a random child by bucket index in O(1) time.
    ///
    /// For the rejection method, we need to select children uniformly at random.
    /// This returns the child at a given "bucket" position (0 to degree-1).
    ///
    /// # Arguments
    /// * `bucket` - The bucket index (0 to degree-1)
    ///
    /// # Returns
    /// `(child_index, child_log_weight)` or None if bucket is out of range
    #[must_use]
    #[inline]
    pub fn get_child_by_bucket(&self, bucket: usize) -> Option<(usize, f64)> {
        self.children_vec
            .get(bucket)
            .map(|c| (c.index, c.log_weight))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
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
        range.add_child(0, 1.0); // log_2(2) = 1, so weight 2
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
        range.add_child(1, 2.0_f64.log2().log2()); // Actually, let me fix: log_2(3) ~= 1.585
                                                   // Let's use clearer values:
                                                   // weight 2.0 has log_2 = 1.0
                                                   // weight 3.0 has log_2 ~= 1.585
        let mut range2 = Range::new(2);
        range2.add_child(0, 2.0_f64.log2()); // weight 2, log = 1
        range2.add_child(1, 3.0_f64.log2()); // weight 3, log ~= 1.585
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
        // Expected: log_2(2 * 2^100) = 101
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

    // -------------------------------------------------------------------------
    // Random Access Tests (for rejection sampling optimization)
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_child_by_bucket_covers_all_children() {
        let mut range = Range::new(2);
        range.add_child(100, 1.0);
        range.add_child(200, 1.5);
        range.add_child(300, 1.9);

        // Collect all children via bucket access
        let mut collected_indices: Vec<usize> = Vec::new();
        for bucket in 0..3 {
            if let Some((idx, _)) = range.get_child_by_bucket(bucket) {
                collected_indices.push(idx);
            }
        }

        // Should have exactly 3 children
        assert_eq!(collected_indices.len(), 3);
        collected_indices.sort_unstable();
        assert_eq!(collected_indices, vec![100, 200, 300]);
    }

    #[test]
    fn test_get_child_by_bucket_consistency() {
        // Bucket access should be consistent across multiple calls
        let mut range = Range::new(2);
        for i in 0..10 {
            #[allow(clippy::cast_possible_wrap)]
            let weight = f64::from(i as i32).mul_add(0.1, 1.0);
            range.add_child(i * 7, weight);
        }

        // Access the same bucket multiple times
        let first_access = range.get_child_by_bucket(3);
        let second_access = range.get_child_by_bucket(3);
        assert_eq!(first_access, second_access);
    }

    #[test]
    fn test_get_child_by_bucket_after_removal() {
        let mut range = Range::new(2);
        range.add_child(0, 1.0);
        range.add_child(1, 1.5);
        range.add_child(2, 1.9);

        // Remove middle element
        range.remove_child(1);

        // Should only have 2 accessible buckets now
        assert!(range.get_child_by_bucket(0).is_some());
        assert!(range.get_child_by_bucket(1).is_some());
        assert!(range.get_child_by_bucket(2).is_none());

        // The remaining children should be 0 and 2
        let mut remaining: Vec<usize> = (0..2)
            .filter_map(|b| range.get_child_by_bucket(b).map(|(idx, _)| idx))
            .collect();
        remaining.sort_unstable();
        assert_eq!(remaining, vec![0, 2]);
    }

    #[test]
    fn test_bucket_access_with_many_children() {
        let mut range = Range::new(5);
        let n = 100;
        for i in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let weight = (i as f64).mul_add(0.001, 4.0);
            range.add_child(i, weight);
        }

        // All buckets should be accessible
        for bucket in 0..n {
            assert!(
                range.get_child_by_bucket(bucket).is_some(),
                "Bucket {bucket} should be accessible"
            );
        }

        // Bucket n should not be accessible
        assert!(range.get_child_by_bucket(n).is_none());
    }

    #[test]
    fn test_child_weights_in_bucket_access() {
        let mut range = Range::new(2);
        range.add_child(10, 1.2);
        range.add_child(20, 1.4);
        range.add_child(30, 1.6);

        // Verify that bucket access returns correct weights
        let children_via_bucket: Vec<(usize, f64)> = (0..3)
            .filter_map(|b| range.get_child_by_bucket(b))
            .collect();

        // Each child should have its correct weight
        for (idx, weight) in &children_via_bucket {
            let expected_weight = match *idx {
                10 => 1.2,
                20 => 1.4,
                30 => 1.6,
                _ => panic!("Unexpected index {idx}"),
            };
            assert!(
                (weight - expected_weight).abs() < 1e-10,
                "Weight mismatch for index {idx}: expected {expected_weight}, got {weight}"
            );
        }
    }

    #[test]
    fn test_bucket_access_matches_children_iterator() {
        let mut range = Range::new(3);
        range.add_child(5, 2.0);
        range.add_child(15, 2.5);
        range.add_child(25, 2.8);
        range.add_child(35, 2.9);

        // Collect via iterator
        let mut iter_children: Vec<(usize, f64)> = range.children().collect();
        iter_children.sort_by_key(|(idx, _)| *idx);

        // Collect via bucket access
        let mut bucket_children: Vec<(usize, f64)> = (0..range.degree())
            .filter_map(|b| range.get_child_by_bucket(b))
            .collect();
        bucket_children.sort_by_key(|(idx, _)| *idx);

        // Should be identical
        assert_eq!(iter_children.len(), bucket_children.len());
        for ((iter_idx, iter_w), (bucket_idx, bucket_w)) in
            iter_children.iter().zip(bucket_children.iter())
        {
            assert_eq!(iter_idx, bucket_idx);
            assert!((iter_w - bucket_w).abs() < 1e-15);
        }
    }
}
