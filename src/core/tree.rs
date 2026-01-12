//! Tree data structure for the dynamic random sampler.
//!
//! This module implements the forest of trees as described in Section 2 of the paper.
//! Elements are organized into ranges based on their weights, and ranges with
//! multiple children (degree ≥ d) propagate up to higher levels, forming trees.
//!
//! Key concepts:
//! - Level 0 (implicit): Contains the actual elements with their weights
//! - Level 1+: Contains ranges that group children from the previous level
//! - A range with degree < d becomes a "root" and is stored in the level table `T_ℓ`
//! - A range with degree ≥ d has a parent in the next level
//!
//! The tree has height at most L ≤ `log* N + 1`, where log\* is the iterated logarithm.
//!
//! # Section 4 Optimizations
//!
//! With Section 4 optimizations enabled (d ≥ 16, b > 0):
//! - The degree bound d determines root vs non-root classification
//! - Tolerance factor b allows lazy updates without parent changes
//! - This achieves O(log* N) amortized update time

use crate::core::{log_sum_exp, Level, OptimizationConfig};

/// The forest of trees data structure.
///
/// Manages a collection of levels, each containing ranges that group
/// children from the previous level. The tree is built bottom-up
/// from the elements.
///
/// With Section 4 optimizations, the degree bound `d` and tolerance `b`
/// can be configured via [`OptimizationConfig`].
#[derive(Debug)]
pub struct Tree {
    /// Element log-weights (level 0)
    element_log_weights: Vec<f64>,
    /// Levels 1 through L (level 0 is implicit)
    levels: Vec<Level>,
    /// Optimization configuration
    config: OptimizationConfig,
}

impl Tree {
    /// Build a tree from element weights using basic configuration (d=2).
    ///
    /// The tree is constructed bottom-up:
    /// 1. Insert elements into ranges at level 1 based on their weights
    /// 2. For each level, ranges with degree ≥ d propagate to the next level
    /// 3. Continue until no ranges have degree ≥ d
    ///
    /// # Arguments
    /// * `log_weights` - The log₂ of each element's weight
    #[must_use]
    pub fn new(log_weights: Vec<f64>) -> Self {
        Self::with_config(log_weights, OptimizationConfig::basic())
    }

    /// Build a tree from element weights with Section 4 optimizations.
    ///
    /// Uses the paper's recommended values of b=0.4 and d=32.
    ///
    /// # Arguments
    /// * `log_weights` - The log₂ of each element's weight
    #[must_use]
    pub fn new_optimized(log_weights: Vec<f64>) -> Self {
        Self::with_config(log_weights, OptimizationConfig::optimized())
    }

    /// Build a tree from element weights with custom optimization configuration.
    ///
    /// The tree is constructed bottom-up:
    /// 1. Insert elements into ranges at level 1 based on their weights
    /// 2. For each level, ranges with degree ≥ `min_degree` propagate to the next level
    /// 3. Continue until no ranges have degree ≥ `min_degree`
    ///
    /// # Arguments
    /// * `log_weights` - The log₂ of each element's weight
    /// * `config` - The optimization configuration
    #[must_use]
    pub fn with_config(log_weights: Vec<f64>, config: OptimizationConfig) -> Self {
        let mut tree = Self {
            element_log_weights: log_weights,
            levels: Vec::new(),
            config,
        };
        let log_weights = &tree.element_log_weights;

        if log_weights.is_empty() {
            return tree;
        }

        // Build level 1: insert elements into ranges
        let mut level1 = Level::with_config(1, tree.config);
        for (idx, &log_weight) in log_weights.iter().enumerate() {
            level1.insert_child(idx, log_weight);
        }
        tree.levels.push(level1);

        // Build higher levels: ranges with degree >= min_degree propagate up
        loop {
            let current_level_num = tree.levels.len();
            let current_level = &tree.levels[current_level_num - 1];

            // Collect non-root ranges (degree >= min_degree) that need parents
            let non_roots: Vec<_> = current_level
                .non_root_ranges()
                .map(|(j, r)| (j, r.compute_total_log_weight()))
                .collect();

            if non_roots.is_empty() {
                break;
            }

            // Create next level with same config
            let mut next_level = Level::with_config(current_level_num + 1, tree.config);
            for (range_number, range_log_weight) in non_roots {
                // The range becomes a child in the next level
                // Use the range_number as a unique identifier
                #[allow(clippy::cast_sign_loss)]
                let child_idx = range_number as usize;
                next_level.insert_child(child_idx, range_log_weight);
            }
            tree.levels.push(next_level);
        }

        tree
    }

    /// Get the optimization configuration.
    #[must_use]
    pub const fn config(&self) -> &OptimizationConfig {
        &self.config
    }

    /// Get the number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.element_log_weights.len()
    }

    /// Check if the tree is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.element_log_weights.is_empty()
    }

    /// Get the number of levels (excluding the implicit level 0).
    #[must_use]
    pub const fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Get the height of the tree (number of levels including level 0).
    ///
    /// Returns 0 for an empty tree, 1 if there are elements but no ranges, etc.
    #[must_use]
    pub const fn height(&self) -> usize {
        if self.element_log_weights.is_empty() {
            0
        } else {
            self.levels.len() + 1
        }
    }

    /// Get a level by its number (1-indexed).
    #[must_use]
    pub fn get_level(&self, level_number: usize) -> Option<&Level> {
        if level_number == 0 || level_number > self.levels.len() {
            None
        } else {
            Some(&self.levels[level_number - 1])
        }
    }

    /// Get a mutable level by its number (1-indexed).
    pub fn get_level_mut(&mut self, level_number: usize) -> Option<&mut Level> {
        if level_number == 0 || level_number > self.levels.len() {
            None
        } else {
            Some(&mut self.levels[level_number - 1])
        }
    }

    /// Get an element's log-weight.
    #[must_use]
    pub fn element_log_weight(&self, index: usize) -> Option<f64> {
        self.element_log_weights.get(index).copied()
    }

    /// Get the total log-weight of all elements.
    #[must_use]
    pub fn total_log_weight(&self) -> f64 {
        log_sum_exp(self.element_log_weights.iter().copied())
    }

    /// Get the total log-weight of all root ranges at a given level.
    ///
    /// This corresponds to `weight(T_ℓ)` in the paper.
    #[must_use]
    pub fn level_root_total(&self, level_number: usize) -> f64 {
        self.get_level(level_number)
            .map_or(f64::NEG_INFINITY, Level::compute_root_total_log_weight)
    }

    /// Get the number of root ranges at a given level.
    #[must_use]
    pub fn root_count_at_level(&self, level_number: usize) -> usize {
        self.get_level(level_number).map_or(0, Level::root_count)
    }

    /// Get the total number of root ranges across all levels.
    #[must_use]
    pub fn total_root_count(&self) -> usize {
        self.levels.iter().map(Level::root_count).sum()
    }

    /// Get the maximum level number (L in the paper).
    #[must_use]
    pub const fn max_level(&self) -> usize {
        self.levels.len()
    }

    /// Iterate over all levels.
    pub fn levels(&self) -> impl Iterator<Item = &Level> {
        self.levels.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Empty Tree Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_tree() {
        let tree = Tree::new(vec![]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.level_count(), 0);
        assert_eq!(tree.height(), 0);
    }

    // -------------------------------------------------------------------------
    // Single Element Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_single_element_tree() {
        let tree = Tree::new(vec![1.0]); // weight 2
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.level_count(), 1);
        assert_eq!(tree.height(), 2); // level 0 (element) + level 1
    }

    #[test]
    fn test_single_element_creates_root() {
        let tree = Tree::new(vec![1.0]); // weight 2 → range 2
        let level1 = tree.get_level(1).unwrap();
        assert_eq!(level1.root_count(), 1);
        assert_eq!(level1.non_root_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Two Elements Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_two_elements_same_range() {
        // Both elements in range 2: weights in [2, 4)
        let tree = Tree::new(vec![1.0, 1.5]); // weights 2, 2.83
        assert_eq!(tree.len(), 2);

        let level1 = tree.get_level(1).unwrap();
        // One range with 2 elements → not a root
        assert_eq!(level1.non_root_count(), 1);

        // Level 2 should exist with the range as a root
        assert!(tree.get_level(2).is_some());
        let level2 = tree.get_level(2).unwrap();
        assert_eq!(level2.root_count(), 1);
    }

    #[test]
    fn test_two_elements_different_ranges() {
        // Elements in different ranges
        let tree = Tree::new(vec![0.0, 2.0]); // weights 1 (range 1), 4 (range 3)
        assert_eq!(tree.len(), 2);

        let level1 = tree.get_level(1).unwrap();
        // Two ranges, each with 1 element → both are roots
        assert_eq!(level1.root_count(), 2);
        assert_eq!(level1.non_root_count(), 0);

        // No level 2 needed
        assert_eq!(tree.level_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Level Building Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_level_building_stops_at_roots() {
        // 4 elements all in the same range → should create multiple levels
        let tree = Tree::new(vec![1.0, 1.1, 1.2, 1.3]); // all in range 2

        let level1 = tree.get_level(1).unwrap();
        assert_eq!(level1.non_root_count(), 1); // degree 4, not a root

        // Level 2 should have the range as a child
        let level2 = tree.get_level(2).unwrap();
        assert_eq!(level2.root_count(), 1); // now a root
    }

    #[test]
    fn test_tree_height_bound() {
        // Tree height should be O(log* N)
        // For reasonable N, this is at most 5-6 levels
        let weights: Vec<f64> = (0..100).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect();
        let tree = Tree::new(weights);

        // Height should be reasonable (definitely less than log N)
        assert!(tree.height() <= 10);
    }

    // -------------------------------------------------------------------------
    // Weight Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_total_log_weight() {
        let tree = Tree::new(vec![0.0, 1.0]); // weights 1, 2 → total 3
        let total = tree.total_log_weight().exp2();
        assert!((total - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_element_log_weight() {
        let tree = Tree::new(vec![0.0, 1.0, 2.0]);
        assert_eq!(tree.element_log_weight(0), Some(0.0));
        assert_eq!(tree.element_log_weight(1), Some(1.0));
        assert_eq!(tree.element_log_weight(2), Some(2.0));
        assert_eq!(tree.element_log_weight(3), None);
    }

    #[test]
    fn test_level_root_total() {
        // Two roots at level 1: weights 1 and 4
        let tree = Tree::new(vec![0.0, 2.0]); // weights 1 (range 1), 4 (range 3)
        let root_total = tree.level_root_total(1).exp2();
        assert!((root_total - 5.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Root Count Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_root_count_at_level() {
        let tree = Tree::new(vec![0.0, 2.0, 3.0]); // weights 1, 4, 8

        let level1 = tree.get_level(1).unwrap();
        assert_eq!(level1.root_count(), 3);
    }

    #[test]
    fn test_total_root_count() {
        // Some elements in same range, some different
        let tree = Tree::new(vec![0.0, 0.5, 2.0]); // weights 1, 1.41, 4
                                                   // Range 1: 2 elements (not root), Range 3: 1 element (root)
                                                   // Level 2: 1 root from range 1

        let total_roots = tree.total_root_count();
        assert_eq!(total_roots, 2); // one at each level
    }

    // -------------------------------------------------------------------------
    // Access Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_level_out_of_bounds() {
        let tree = Tree::new(vec![1.0]);
        assert!(tree.get_level(0).is_none()); // level 0 is implicit
        assert!(tree.get_level(1).is_some());
        assert!(tree.get_level(2).is_none());
    }

    #[test]
    fn test_iterate_levels() {
        let tree = Tree::new(vec![1.0, 1.1]); // same range → 2 levels
        let level_nums: Vec<_> = tree.levels().map(Level::level_number).collect();
        assert_eq!(level_nums, vec![1, 2]);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_negative_log_weights() {
        // Weights less than 1
        let tree = Tree::new(vec![-1.0, -2.0]); // weights 0.5, 0.25
        assert_eq!(tree.len(), 2);

        let total = tree.total_log_weight().exp2();
        assert!((total - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_wide_weight_range() {
        // Weights spanning many orders of magnitude
        let tree = Tree::new(vec![-10.0, 0.0, 10.0]); // weights 2^-10, 1, 2^10
        assert_eq!(tree.len(), 3);

        // Each in a different range → all roots at level 1
        let level1 = tree.get_level(1).unwrap();
        assert_eq!(level1.root_count(), 3);
    }

    #[test]
    fn test_many_elements_same_range() {
        // Many elements in the same range
        let weights: Vec<f64> = (0..10).map(|i| f64::from(i).mul_add(0.05, 1.0)).collect();
        let tree = Tree::new(weights);

        // All in range 2 (weights between 2 and 4)
        let level1 = tree.get_level(1).unwrap();
        assert_eq!(level1.range_count(), 1);
        assert_eq!(level1.non_root_count(), 1); // degree 10, not a root
    }

    #[test]
    fn test_tree_structure_consistency() {
        let tree = Tree::new(vec![0.0, 0.5, 1.0, 1.5, 2.0]);

        // Every non-root range at level ℓ should have a corresponding
        // child in level ℓ+1
        for level_num in 1..tree.max_level() {
            let level = tree.get_level(level_num).unwrap();
            let non_root_count = level.non_root_count();

            if level_num < tree.max_level() {
                let next_level = tree.get_level(level_num + 1).unwrap();
                // The next level should have entries for non-root ranges
                assert!(next_level.range_count() > 0 || non_root_count == 0);
            }
        }
    }
}
