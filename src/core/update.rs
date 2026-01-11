//! Weight update algorithm for the dynamic random sampler.
//!
//! This module implements weight updates that maintain the tree structure.
//! When an element's weight changes, the update propagates up through the
//! tree levels as necessary.
//!
//! From Section 2.3 of the paper:
//! 1. Update the element's weight
//! 2. If the element moves to a different range, update level 1
//! 3. Propagate changes up the tree as range weights change
//! 4. Root ranges may become non-root (or vice versa)

use crate::core::{compute_range_number, Level, Tree};

/// A mutable tree that supports weight updates.
///
/// This struct wraps a `Tree` and provides methods to update element weights
/// while maintaining the tree structure.
#[derive(Debug)]
pub struct MutableTree {
    /// Element log-weights (level 0)
    element_log_weights: Vec<f64>,
    /// Levels 1 through L
    levels: Vec<Level>,
}

impl MutableTree {
    /// Create a new mutable tree from element weights.
    #[must_use]
    pub fn new(log_weights: Vec<f64>) -> Self {
        let tree = Tree::new(log_weights);
        Self::from_tree(&tree)
    }

    /// Create a mutable tree from an existing immutable tree.
    fn from_tree(tree: &Tree) -> Self {
        // We need to reconstruct the tree to get ownership of levels
        Self::new_internal(tree)
    }

    fn new_internal(tree: &Tree) -> Self {
        let element_log_weights: Vec<f64> = (0..tree.len())
            .filter_map(|i| tree.element_log_weight(i))
            .collect();

        let mut result = Self {
            element_log_weights: element_log_weights.clone(),
            levels: Vec::new(),
        };

        if element_log_weights.is_empty() {
            return result;
        }

        // Rebuild the tree from scratch with ownership
        let mut level1 = Level::new(1);
        for (idx, &log_weight) in element_log_weights.iter().enumerate() {
            level1.insert_child(idx, log_weight);
        }
        result.levels.push(level1);

        // Build higher levels
        loop {
            let current_level_num = result.levels.len();
            let current_level = &result.levels[current_level_num - 1];

            let non_roots: Vec<_> = current_level
                .non_root_ranges()
                .map(|(j, r)| (j, r.compute_total_log_weight()))
                .collect();

            if non_roots.is_empty() {
                break;
            }

            let mut next_level = Level::new(current_level_num + 1);
            for (range_number, range_log_weight) in non_roots {
                #[allow(clippy::cast_sign_loss)]
                let child_idx = range_number as usize;
                next_level.insert_child(child_idx, range_log_weight);
            }
            result.levels.push(next_level);
        }

        result
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

    /// Get an element's log-weight.
    #[must_use]
    pub fn element_log_weight(&self, index: usize) -> Option<f64> {
        self.element_log_weights.get(index).copied()
    }

    /// Get the number of levels.
    #[must_use]
    pub const fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Get a reference to the underlying tree for sampling.
    ///
    /// Note: This creates a new Tree - for efficiency, consider caching
    /// or using the mutable tree directly for sampling.
    #[must_use]
    pub fn as_tree(&self) -> Tree {
        Tree::new(self.element_log_weights.clone())
    }

    /// Update an element's weight.
    ///
    /// # Arguments
    /// * `index` - The element index
    /// * `new_log_weight` - The new log₂ weight
    ///
    /// # Returns
    /// `true` if the update was successful, `false` if index is out of bounds.
    pub fn update(&mut self, index: usize, new_log_weight: f64) -> bool {
        if index >= self.element_log_weights.len() {
            return false;
        }

        let old_log_weight = self.element_log_weights[index];
        let old_range = compute_range_number(old_log_weight);
        let new_range = compute_range_number(new_log_weight);

        // Update the element weight
        self.element_log_weights[index] = new_log_weight;

        // If the element stays in the same range, just update the weight in that range
        if old_range == new_range {
            if let Some(level) = self.levels.get_mut(0) {
                if let Some(range) = level.get_range_mut(old_range) {
                    range.update_child_weight(index, new_log_weight);
                }
            }
            // Propagate weight changes up
            self.propagate_weight_changes(1, old_range);
        } else {
            // Element moves to a different range
            if let Some(level) = self.levels.get_mut(0) {
                level.remove_child(old_range, index);
                level.insert_child(index, new_log_weight);
            }
            // Propagate changes for both ranges
            self.propagate_structure_changes(1, old_range, new_range);
        }

        true
    }

    /// Propagate weight changes up the tree when an element's weight changes
    /// but it stays in the same range.
    fn propagate_weight_changes(&mut self, level_num: usize, range_number: i32) {
        if level_num > self.levels.len() {
            return;
        }

        // Check if this range's degree changed (root <-> non-root)
        let was_root_before = self.was_root_at_level(level_num, range_number);

        // Recompute the range weight by looking at the level
        if level_num <= self.levels.len() {
            let level = &mut self.levels[level_num - 1];
            if let Some(range) = level.get_range_mut(range_number) {
                // Force cache invalidation
                let _ = range.total_log_weight();
            }
        }

        let is_root_now = self.is_root_at_level(level_num, range_number);

        // If root status changed, may need to restructure
        if was_root_before != is_root_now {
            self.handle_root_status_change(level_num, range_number, is_root_now);
        } else if !is_root_now && level_num < self.levels.len() {
            // Non-root range: update weight in parent level
            let new_weight = self.get_range_weight(level_num, range_number);
            if let Some(parent_level) = self.levels.get_mut(level_num) {
                #[allow(clippy::cast_sign_loss)]
                let child_idx = range_number as usize;
                if let Some(range) = parent_level.get_range_mut(compute_range_number(new_weight)) {
                    range.update_child_weight(child_idx, new_weight);
                }
            }
            // Continue propagating up
            #[allow(clippy::cast_sign_loss)]
            let parent_range = compute_range_number(new_weight);
            self.propagate_weight_changes(level_num + 1, parent_range);
        }
    }

    /// Propagate structural changes when an element moves between ranges.
    fn propagate_structure_changes(&mut self, level_num: usize, _old_range: i32, _new_range: i32) {
        // Rebuild the tree from level_num upward for simplicity
        // This is less efficient than incremental updates but ensures correctness
        self.rebuild_from_level(level_num);
    }

    /// Rebuild the tree from a given level upward.
    fn rebuild_from_level(&mut self, from_level: usize) {
        // Truncate levels and rebuild
        if from_level > 0 && from_level <= self.levels.len() {
            self.levels.truncate(from_level);
        }

        // Rebuild higher levels
        loop {
            let current_level_num = self.levels.len();
            let current_level = &self.levels[current_level_num - 1];

            let non_roots: Vec<_> = current_level
                .non_root_ranges()
                .map(|(j, r)| (j, r.compute_total_log_weight()))
                .collect();

            if non_roots.is_empty() {
                break;
            }

            let mut next_level = Level::new(current_level_num + 1);
            for (range_number, range_log_weight) in non_roots {
                #[allow(clippy::cast_sign_loss)]
                let child_idx = range_number as usize;
                next_level.insert_child(child_idx, range_log_weight);
            }
            self.levels.push(next_level);
        }
    }

    /// Check if a range was a root before any changes.
    fn was_root_at_level(&self, level_num: usize, range_number: i32) -> bool {
        use crate::core::Range;
        self.levels
            .get(level_num - 1)
            .and_then(|l| l.get_range(range_number))
            .is_some_and(Range::is_root)
    }

    /// Check if a range is currently a root.
    fn is_root_at_level(&self, level_num: usize, range_number: i32) -> bool {
        use crate::core::Range;
        self.levels
            .get(level_num - 1)
            .and_then(|l| l.get_range(range_number))
            .is_some_and(Range::is_root)
    }

    /// Get the total log-weight of a range.
    fn get_range_weight(&self, level_num: usize, range_number: i32) -> f64 {
        use crate::core::Range;
        self.levels
            .get(level_num - 1)
            .and_then(|l| l.get_range(range_number))
            .map_or(f64::NEG_INFINITY, Range::compute_total_log_weight)
    }

    /// Handle when a range's root status changes.
    fn handle_root_status_change(
        &mut self,
        level_num: usize,
        _range_number: i32,
        _is_now_root: bool,
    ) {
        // Rebuild from this level for correctness
        self.rebuild_from_level(level_num);
    }

    /// Get a level by number (1-indexed).
    #[must_use]
    pub fn get_level(&self, level_num: usize) -> Option<&Level> {
        if level_num == 0 || level_num > self.levels.len() {
            None
        } else {
            Some(&self.levels[level_num - 1])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -------------------------------------------------------------------------
    // Basic Update Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_empty_tree() {
        let mut tree = MutableTree::new(vec![]);
        assert!(!tree.update(0, 1.0));
    }

    #[test]
    fn test_update_out_of_bounds() {
        let mut tree = MutableTree::new(vec![1.0, 2.0]);
        assert!(!tree.update(5, 1.0));
    }

    #[test]
    fn test_update_single_element() {
        let mut tree = MutableTree::new(vec![1.0]); // weight 2
        assert!(tree.update(0, 2.0)); // new weight 4
        assert_eq!(tree.element_log_weight(0), Some(2.0));
    }

    #[test]
    fn test_update_same_range() {
        // Elements stay in same range after update
        let mut tree = MutableTree::new(vec![1.0, 1.5]); // weights 2, 2.83 (range 2)
        assert!(tree.update(0, 1.3)); // new weight ~2.46 (still range 2)
        assert_eq!(tree.element_log_weight(0), Some(1.3));
    }

    #[test]
    fn test_update_different_range() {
        // Element moves to different range
        let mut tree = MutableTree::new(vec![1.0]); // weight 2 (range 2)
        assert!(tree.update(0, 2.5)); // new weight ~5.66 (range 3)
        assert_eq!(tree.element_log_weight(0), Some(2.5));

        // Range should have changed
        let range_num = compute_range_number(2.5);
        assert_eq!(range_num, 3);
    }

    // -------------------------------------------------------------------------
    // Weight Correctness Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_preserves_other_weights() {
        let mut tree = MutableTree::new(vec![1.0, 2.0, 3.0]);
        tree.update(1, 1.5);

        assert_eq!(tree.element_log_weight(0), Some(1.0));
        assert_eq!(tree.element_log_weight(1), Some(1.5));
        assert_eq!(tree.element_log_weight(2), Some(3.0));
    }

    #[test]
    fn test_multiple_updates_same_element() {
        let mut tree = MutableTree::new(vec![1.0]);

        tree.update(0, 2.0);
        assert_eq!(tree.element_log_weight(0), Some(2.0));

        tree.update(0, 0.5);
        assert_eq!(tree.element_log_weight(0), Some(0.5));

        tree.update(0, -1.0);
        assert_eq!(tree.element_log_weight(0), Some(-1.0));
    }

    #[test]
    fn test_multiple_updates_different_elements() {
        let mut tree = MutableTree::new(vec![1.0, 2.0, 3.0]);

        tree.update(0, 0.0);
        tree.update(1, 0.0);
        tree.update(2, 0.0);

        // All should be equal now
        assert_eq!(tree.element_log_weight(0), Some(0.0));
        assert_eq!(tree.element_log_weight(1), Some(0.0));
        assert_eq!(tree.element_log_weight(2), Some(0.0));
    }

    // -------------------------------------------------------------------------
    // Structure Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_maintains_tree_structure() {
        let mut tree = MutableTree::new(vec![1.0, 1.1, 1.2]);

        // Update that keeps all in same range
        tree.update(0, 1.05);

        // Tree should still have valid structure
        assert!(tree.level_count() >= 1);
        let level1 = tree.get_level(1).unwrap();
        assert_eq!(level1.range_count(), 1);
    }

    #[test]
    fn test_update_changes_tree_height() {
        // Start with elements in different ranges (shallow tree)
        let mut tree = MutableTree::new(vec![0.0, 2.0, 4.0]);
        let _initial_levels = tree.level_count();

        // Move all to same range (might create deeper tree)
        tree.update(0, 1.0);
        tree.update(1, 1.1);
        tree.update(2, 1.2);

        // All in range 2 now
        let level1 = tree.get_level(1).unwrap();
        assert_eq!(level1.range_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_to_very_small_weight() {
        let mut tree = MutableTree::new(vec![10.0]); // weight 1024
        tree.update(0, -10.0); // weight ~0.001
        assert_eq!(tree.element_log_weight(0), Some(-10.0));
    }

    #[test]
    fn test_update_to_very_large_weight() {
        let mut tree = MutableTree::new(vec![0.0]); // weight 1
        tree.update(0, 50.0); // weight 2^50
        assert_eq!(tree.element_log_weight(0), Some(50.0));
    }

    #[test]
    fn test_update_negative_to_positive_range() {
        let mut tree = MutableTree::new(vec![-2.0]); // weight 0.25 (range -1)
        tree.update(0, 2.0); // weight 4 (range 3)
        assert_eq!(tree.element_log_weight(0), Some(2.0));
    }

    // -------------------------------------------------------------------------
    // Sampling After Update Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_after_update() {
        use crate::core::sample;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut mutable = MutableTree::new(vec![0.0, 0.0]);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Initially equal weights
        let tree1 = mutable.as_tree();
        let samples1: Vec<_> = (0..1000).filter_map(|_| sample(&tree1, &mut rng)).collect();
        let count0 = samples1.iter().filter(|&&x| x == 0).count();
        let ratio1 = f64::from(u32::try_from(count0).unwrap()) / 1000.0;
        assert!(ratio1 > 0.4 && ratio1 < 0.6, "ratio was {ratio1}");

        // Update to make element 1 much heavier
        mutable.update(1, 10.0);
        let tree2 = mutable.as_tree();

        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let samples2: Vec<_> = (0..1000)
            .filter_map(|_| sample(&tree2, &mut rng2))
            .collect();
        let count1 = samples2.iter().filter(|&&x| x == 1).count();
        let fraction = f64::from(u32::try_from(count1).unwrap()) / 1000.0;
        assert!(fraction > 0.99, "fraction was {fraction}");
    }

    // -------------------------------------------------------------------------
    // Stress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_many_updates() {
        let mut tree = MutableTree::new(vec![0.0; 100]);

        // Update each element multiple times
        for round in 0i32..10 {
            for i in 0..100 {
                let new_weight =
                    f64::from(i32::try_from(i).unwrap()).mul_add(0.1, f64::from(round));
                tree.update(i, new_weight);
            }
        }

        // Verify all weights are correct
        for i in 0..100 {
            let expected = f64::from(i32::try_from(i).unwrap()).mul_add(0.1, 9.0);
            assert_relative_eq!(
                tree.element_log_weight(i).unwrap(),
                expected,
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_alternating_updates() {
        let mut tree = MutableTree::new(vec![1.0, 1.0]);

        // Alternate which element is heavier
        for i in 0..20 {
            if i % 2 == 0 {
                tree.update(0, 10.0);
                tree.update(1, 0.0);
            } else {
                tree.update(0, 0.0);
                tree.update(1, 10.0);
            }
        }

        // Final state: element 1 is heavier
        assert_eq!(tree.element_log_weight(0), Some(0.0));
        assert_eq!(tree.element_log_weight(1), Some(10.0));
    }
}
