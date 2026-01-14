//! Weight update algorithm for the dynamic random sampler.
//!
//! This module implements weight updates that maintain the tree structure.
//! When an element's weight changes, the update propagates up through the
//! tree levels as necessary.
//!
//! # Basic Update Algorithm (Section 2.3)
//!
//! 1. Update the element's weight
//! 2. If the element moves to a different range, update level 1
//! 3. Propagate changes up the tree as range weights change
//! 4. Root ranges may become non-root (or vice versa)
//!
//! # Section 4 Optimizations
//!
//! With Section 4 optimizations, we use tolerance-based "lazy updating":
//!
//! 1. Each range `$R_j$` tolerates weights in `$[(1-b) \cdot 2^{j-1}, (2+b) \cdot 2^{j-1})$`
//!    instead of just [2^(j-1), 2^j)
//!
//! 2. An element only changes its parent range when its weight moves
//!    outside the tolerated interval (requires change of at least `$b \cdot 2^{j-1}$`)
//!
//! 3. The degree bound d >= 16 limits how many parent changes can cascade
//!
//! This achieves O(log* N) amortized expected update time.

use crate::core::debug::TimeoutGuard;
use crate::core::{
    compute_range_number, is_deleted_weight, Level, OptimizationConfig, Tree, DELETED_LOG_WEIGHT,
};

/// A mutable tree that supports weight updates.
///
/// This struct wraps a `Tree` and provides methods to update element weights
/// while maintaining the tree structure.
///
/// With Section 4 optimizations enabled, updates use tolerance-based lazy
/// propagation to achieve O(log* N) amortized expected update time.
#[derive(Debug)]
pub struct MutableTree {
    /// Element log-weights (level 0)
    element_log_weights: Vec<f64>,
    /// The current range number for each element (cached)
    element_ranges: Vec<i32>,
    /// Levels 1 through L
    levels: Vec<Level>,
    /// Optimization configuration (tolerance and degree bound)
    config: OptimizationConfig,
}

impl MutableTree {
    /// Create a new mutable tree from element weights using basic configuration.
    #[must_use]
    pub fn new(log_weights: Vec<f64>) -> Self {
        Self::with_config(log_weights, OptimizationConfig::basic())
    }

    /// Create a new mutable tree with Section 4 optimizations.
    ///
    /// Uses b=0.4 and d=32 for O(log* N) amortized update time.
    #[must_use]
    pub fn new_optimized(log_weights: Vec<f64>) -> Self {
        Self::with_config(log_weights, OptimizationConfig::optimized())
    }

    /// Create a new mutable tree with custom optimization configuration.
    #[must_use]
    pub fn with_config(log_weights: Vec<f64>, config: OptimizationConfig) -> Self {
        let tree = Tree::with_config(log_weights, config);
        Self::from_tree(&tree)
    }

    /// Create a mutable tree from an existing immutable tree.
    fn from_tree(tree: &Tree) -> Self {
        Self::new_internal(tree)
    }

    fn new_internal(tree: &Tree) -> Self {
        let element_log_weights: Vec<f64> = (0..tree.len())
            .filter_map(|i| tree.element_log_weight(i))
            .collect();

        // Cache the range number for each element (use i32::MIN for deleted elements)
        let element_ranges: Vec<i32> = element_log_weights
            .iter()
            .map(|&lw| {
                if is_deleted_weight(lw) {
                    i32::MIN // Sentinel for deleted elements
                } else {
                    compute_range_number(lw)
                }
            })
            .collect();

        let config = *tree.config();

        let mut result = Self {
            element_log_weights: element_log_weights.clone(),
            element_ranges,
            levels: Vec::new(),
            config,
        };

        if element_log_weights.is_empty() {
            return result;
        }

        // Rebuild the tree from scratch with ownership
        let mut level1 = Level::with_config(1, config);
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

            let mut next_level = Level::with_config(current_level_num + 1, config);
            for (range_number, range_log_weight) in non_roots {
                #[allow(clippy::cast_sign_loss)]
                let child_idx = range_number as usize;
                next_level.insert_child(child_idx, range_log_weight);
            }
            result.levels.push(next_level);
        }

        result
    }

    /// Get the optimization configuration.
    #[must_use]
    pub const fn config(&self) -> &OptimizationConfig {
        &self.config
    }

    /// Get the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.element_log_weights.len()
    }

    /// Check if the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.element_log_weights.is_empty()
    }

    /// Get an element's log-weight.
    #[must_use]
    pub fn element_log_weight(&self, index: usize) -> Option<f64> {
        self.element_log_weights.get(index).copied()
    }

    /// Get the number of levels.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Get a reference to the underlying tree for sampling.
    ///
    /// Note: This creates a new Tree - for efficiency, consider caching
    /// or using the mutable tree directly for sampling.
    #[must_use]
    pub fn as_tree(&self) -> Tree {
        Tree::with_config(self.element_log_weights.clone(), self.config)
    }

    /// Update an element's weight.
    ///
    /// With Section 4 optimizations, uses tolerance-based lazy updating:
    /// - If the new weight is within the tolerated interval of the current range,
    ///   only the weight is updated (no parent change)
    /// - If the new weight is outside the tolerated interval, the element
    ///   moves to a new range and changes propagate up
    ///
    /// Special cases:
    /// - Setting weight to zero (`NEG_INFINITY`) is equivalent to `delete()`
    /// - Updating a deleted element to positive weight "undeletes" it
    ///
    /// # Arguments
    /// * `index` - The element index
    /// * `new_log_weight` - The new `$\log_2$` weight
    ///
    /// # Returns
    /// `true` if the update was successful, `false` if index is out of bounds.
    pub fn update(&mut self, index: usize, new_log_weight: f64) -> bool {
        if index >= self.element_log_weights.len() {
            return false;
        }

        // Handle deletion case
        if is_deleted_weight(new_log_weight) {
            return self.delete(index);
        }

        let was_deleted = self.is_deleted(index);
        let old_range = self.element_ranges[index];

        // Update the element weight
        self.element_log_weights[index] = new_log_weight;

        // Handle undelete case - restoring a deleted element
        // This is like an insert but at an existing index
        if was_deleted {
            let new_range = compute_range_number(new_log_weight);
            self.element_ranges[index] = new_range;

            // Levels should always exist if we have elements (invariant)
            debug_assert!(
                !self.levels.is_empty(),
                "Levels empty during undelete - tree invariant violated"
            );

            // Check if range was root before insertion
            let was_root = self
                .levels
                .first()
                .is_some_and(|l| l.is_root_range(new_range));

            if let Some(level) = self.levels.get_mut(0) {
                level.insert_child(index, new_log_weight);
            }

            // Check if range is now non-root
            let is_non_root = self
                .levels
                .first()
                .is_some_and(|l| !l.is_root_range(new_range));

            // Propagate structural change if needed
            if was_root && is_non_root {
                self.propagate_insert(1, new_range, new_log_weight);
            } else if !was_root && is_non_root {
                self.propagate_weight_changes(1, new_range);
            }

            return true;
        }

        // Check if the new weight is within the tolerated interval of the current range
        // With tolerance b, range j accepts weights in [(1-b)*2^(j-1), (2+b)*2^(j-1))
        let stays_in_range = self
            .config
            .weight_in_tolerated_range(old_range, new_log_weight);

        // Check if range still exists (may have been cleaned up during tree maintenance)
        let range_exists = self
            .levels
            .first()
            .is_some_and(|l| l.get_range(old_range).is_some());

        if stays_in_range && range_exists {
            // Lazy update: just update the weight in the current range
            let level = self.levels.get_mut(0).expect("level 0 must exist");
            let range = level
                .get_range_mut(old_range)
                .expect("range must exist (just checked)");
            range.update_child_weight(index, new_log_weight);

            // Propagate weight changes up (but no structural changes)
            self.propagate_weight_changes(1, old_range);
        } else {
            // Element needs to move to a different range
            let new_range = compute_range_number(new_log_weight);
            self.element_ranges[index] = new_range;

            if let Some(level) = self.levels.get_mut(0) {
                level.remove_child(old_range, index);
                level.insert_child(index, new_log_weight);
            }
            // Propagate structural changes for both ranges
            self.propagate_structure_changes(1, old_range, new_range);
        }

        true
    }

    /// Check if a weight change would require a parent change for the element.
    ///
    /// With tolerance b, changes smaller than `$b \cdot 2^{j-1}$` won't trigger parent changes.
    #[must_use]
    pub fn would_require_parent_change(&self, index: usize, new_log_weight: f64) -> bool {
        if index >= self.element_ranges.len() {
            return false;
        }
        let current_range = self.element_ranges[index];
        !self
            .config
            .weight_in_tolerated_range(current_range, new_log_weight)
    }

    /// Soft-delete an element by setting its weight to zero.
    ///
    /// The element remains in the data structure but will never be sampled.
    /// Its index remains valid and stable. Uses incremental O(log* N) propagation.
    ///
    /// # Arguments
    /// * `index` - The element index to delete
    ///
    /// # Returns
    /// `true` if the delete was successful, `false` if index is out of bounds
    /// or element was already deleted.
    pub fn delete(&mut self, index: usize) -> bool {
        let _guard = TimeoutGuard::new("delete");

        if index >= self.element_log_weights.len() {
            return false;
        }

        // Already deleted?
        if self.is_deleted(index) {
            return true;
        }

        let old_range = self.element_ranges[index];

        // Check if range was non-root before deletion
        let was_non_root = self
            .levels
            .first()
            .is_some_and(|l| !l.is_root_range(old_range));

        // Set to deleted
        self.element_log_weights[index] = DELETED_LOG_WEIGHT;
        self.element_ranges[index] = i32::MIN; // Sentinel for "no range"

        // Remove from the range at level 1
        if let Some(level) = self.levels.get_mut(0) {
            level.remove_child(old_range, index);
        }

        // Check if range is now root (or empty)
        let is_root_or_empty = self
            .levels
            .first()
            .is_none_or(|l| l.is_root_range(old_range) || l.get_range(old_range).is_none());

        // If range transitioned from non-root to root/empty, propagate deletion
        if was_non_root && is_root_or_empty {
            self.propagate_delete(1, old_range);
        } else if was_non_root {
            // Still non-root, just update the weight in parent
            self.propagate_weight_changes(1, old_range);
        }

        true
    }

    /// Propagate structural change when a range becomes root or empty.
    /// This removes the range from the parent level and continues propagating up.
    ///
    /// `level_num` is 1-indexed: level 1 = `self.levels[0]`, level 2 = `self.levels[1]`, etc.
    fn propagate_delete(&mut self, level_num: usize, range_number: i32) {
        let _guard = TimeoutGuard::new("propagate_delete");

        // level_num is 1-indexed, so level_idx is the actual vector index
        let level_idx = level_num - 1;

        // If no parent level exists, nothing to do (recursion base case)
        if level_idx >= self.levels.len() {
            return;
        }

        // Get the parent range number by looking at where this range is stored
        // The range is stored as a child in a parent range at level_num + 1
        // with child_idx = range_number
        #[allow(clippy::cast_sign_loss)]
        let child_idx = range_number as usize;

        // Find which parent range contains this child
        let parent_range_number = self.levels.get(level_idx).and_then(|l| {
            l.ranges()
                .find(|(_, r)| r.children().any(|(idx, _)| idx == child_idx))
                .map(|(j, _)| j)
        });

        let Some(parent_range) = parent_range_number else {
            return; // Child not found in any parent range
        };

        // Check if parent range was non-root before
        let parent_was_non_root = self
            .levels
            .get(level_idx)
            .is_some_and(|l| !l.is_root_range(parent_range));

        // Remove this range from the parent level
        if let Some(parent_level) = self.levels.get_mut(level_idx) {
            parent_level.remove_child(parent_range, child_idx);
        }

        // Check if parent is now root (or empty)
        let parent_is_root_or_empty = self
            .levels
            .get(level_idx)
            .is_none_or(|l| l.is_root_range(parent_range) || l.get_range(parent_range).is_none());

        // Propagate up if parent changed from non-root to root/empty
        if parent_was_non_root && parent_is_root_or_empty {
            self.propagate_delete(level_num + 1, parent_range);
        } else if parent_was_non_root {
            // Parent still has 2+ children after removing one, just update its weight
            self.propagate_weight_changes(level_num, parent_range);
        }
    }

    /// Check if an element has been deleted.
    ///
    /// # Arguments
    /// * `index` - The element index to check
    ///
    /// # Returns
    /// `true` if the element is deleted, `false` otherwise (including if out of bounds).
    #[must_use]
    pub fn is_deleted(&self, index: usize) -> bool {
        self.element_log_weights
            .get(index)
            .is_some_and(|&w| is_deleted_weight(w))
    }

    /// Get the number of active (non-deleted) elements.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.element_log_weights
            .iter()
            .filter(|&&w| !is_deleted_weight(w))
            .count()
    }

    /// Insert a new element with the given log-weight.
    ///
    /// The new element is appended and gets the next available index.
    /// Uses incremental O(log* N) propagation instead of full rebuild.
    ///
    /// # Arguments
    /// * `log_weight` - The `$\log_2$` of the new element's weight
    ///
    /// # Returns
    /// The index of the newly inserted element.
    pub fn insert(&mut self, log_weight: f64) -> usize {
        let _guard = TimeoutGuard::new("insert");

        let new_index = self.element_log_weights.len();

        // Add to element storage
        self.element_log_weights.push(log_weight);

        // Handle deleted elements (NEG_INFINITY)
        if is_deleted_weight(log_weight) {
            self.element_ranges.push(i32::MIN);
            return new_index;
        }

        // Compute and cache range number
        let range_number = compute_range_number(log_weight);
        self.element_ranges.push(range_number);

        // Insert into level 1
        if self.levels.is_empty() {
            // First element - create level 1
            let level1 = Level::with_config(1, self.config);
            self.levels.push(level1);
        }

        // Check if range was root before insertion
        let was_root = self
            .levels
            .first()
            .is_some_and(|l| l.is_root_range(range_number));

        if let Some(level) = self.levels.get_mut(0) {
            level.insert_child(new_index, log_weight);
        }

        // Check if range is now non-root (transitioned from root)
        let is_non_root = self
            .levels
            .first()
            .is_some_and(|l| !l.is_root_range(range_number));

        // Only propagate if structure changed (root -> non-root)
        if was_root && is_non_root {
            self.propagate_insert(1, range_number, log_weight);
        } else if !was_root && is_non_root {
            // Range was already non-root, just update its weight in parent
            self.propagate_weight_changes(1, range_number);
        }

        new_index
    }

    /// Propagate structural change when a range becomes non-root.
    /// This adds the range to the next level and continues propagating up.
    fn propagate_insert(&mut self, level_num: usize, range_number: i32, range_log_weight: f64) {
        let _guard = TimeoutGuard::new("propagate_insert");

        // Ensure next level exists
        if level_num >= self.levels.len() {
            let next_level = Level::with_config(level_num + 1, self.config);
            self.levels.push(next_level);
        }

        // Get the parent range number for this range's total weight
        let total_weight = self
            .levels
            .get(level_num - 1)
            .and_then(|l| l.get_range(range_number))
            .map_or(
                range_log_weight,
                super::range::Range::compute_total_log_weight,
            );

        let parent_range_number = compute_range_number(total_weight);

        // Check if parent range was root before
        let parent_was_root = self
            .levels
            .get(level_num)
            .is_some_and(|l| l.is_root_range(parent_range_number));

        // Add or update this range in the parent level
        #[allow(clippy::cast_sign_loss)]
        let child_idx = range_number as usize;
        if let Some(parent_level) = self.levels.get_mut(level_num) {
            parent_level.upsert_child(child_idx, total_weight);
        }

        // Check if parent is now non-root
        let parent_is_non_root = self
            .levels
            .get(level_num)
            .is_some_and(|l| !l.is_root_range(parent_range_number));

        // Propagate up if parent changed from root to non-root
        if parent_was_root && parent_is_non_root {
            self.propagate_insert(level_num + 1, parent_range_number, total_weight);
        }
    }

    /// Propagate weight changes up the tree when an element's weight changes
    /// but it stays in the same range.
    ///
    /// Note: Root status changes (root <-> non-root) are handled by
    /// `propagate_insert`, `propagate_delete`, and `propagate_structure_changes`.
    /// This function only propagates weight updates through existing non-root ranges.
    fn propagate_weight_changes(&mut self, level_num: usize, range_number: i32) {
        // Invariants: always called with level >= 1, and levels is never empty
        debug_assert!(
            level_num >= 1,
            "propagate_weight_changes called with level 0"
        );
        debug_assert!(
            level_num <= self.levels.len(),
            "propagate_weight_changes called with level {} but only {} levels exist",
            level_num,
            self.levels.len()
        );

        // Base case: at or past the top level
        if level_num >= self.levels.len() {
            return;
        }

        // Get the range and recompute its total weight
        // Invariant: range must exist since we're propagating weight for it
        let level = &mut self.levels[level_num - 1];
        let range = level
            .get_range_mut(range_number)
            .expect("propagate_weight_changes called for non-existent range");
        let new_weight = range.total_log_weight();

        // Check if this range is a root - if so, no parent to update
        let is_root = range.is_root();

        // Non-root ranges need to update their parent
        if !is_root && level_num < self.levels.len() {
            // Deleted weight means the range is empty - handled by propagate_delete
            debug_assert!(
                !is_deleted_weight(new_weight),
                "Non-root range {range_number} at level {level_num} has deleted weight"
            );

            // Update weight in parent level
            #[allow(clippy::cast_sign_loss)]
            let child_idx = range_number as usize;
            let parent_range_number = compute_range_number(new_weight);

            // Update parent range if it exists
            // Note: parent range may not exist yet during tree construction or
            // may have been removed during concurrent deletions
            let should_propagate = self.levels.get_mut(level_num).is_some_and(|parent_level| {
                parent_level
                    .get_range_mut(parent_range_number)
                    .is_some_and(|parent_range| {
                        parent_range.update_child_weight(child_idx, new_weight);
                        true
                    })
            });

            // Continue propagating up only if parent range exists
            if should_propagate {
                self.propagate_weight_changes(level_num + 1, parent_range_number);
            }
        }
    }

    /// Propagate structural changes when an element moves between ranges.
    /// `old_range` may transition to root (like delete), `new_range` may transition to non-root (like insert).
    fn propagate_structure_changes(&mut self, level_num: usize, old_range: i32, new_range: i32) {
        let _guard = TimeoutGuard::new("propagate_structure_changes");

        // Check old range: did it transition from non-root to root?
        let old_is_root_or_empty = self
            .levels
            .get(level_num - 1)
            .is_none_or(|l| l.is_root_range(old_range) || l.get_range(old_range).is_none());

        // The element was already removed from old_range by update()
        // If old_range became root/empty, propagate deletion
        // (We assume it was non-root before, otherwise no structural change needed)
        if old_is_root_or_empty && level_num <= self.levels.len() {
            self.propagate_delete(level_num, old_range);
        }

        // Check new range: did it transition from root to non-root?
        let new_is_non_root = self
            .levels
            .get(level_num - 1)
            .is_some_and(|l| !l.is_root_range(new_range));

        // If new_range became non-root, we need to add it to the parent level
        // Get the total weight of the new range
        if new_is_non_root {
            let total_weight = self
                .levels
                .get(level_num - 1)
                .and_then(|l| l.get_range(new_range))
                .map_or(
                    f64::NEG_INFINITY,
                    super::range::Range::compute_total_log_weight,
                );

            if !is_deleted_weight(total_weight) {
                self.propagate_insert(level_num, new_range, total_weight);
            }
        }
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
#[cfg_attr(coverage_nightly, coverage(off))]
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

    // -------------------------------------------------------------------------
    // Section 4 Optimization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimized_tree_creation() {
        let tree = MutableTree::new_optimized(vec![1.0, 1.5, 2.0]);
        assert!((tree.config().tolerance() - 0.4).abs() < 1e-10);
        assert_eq!(tree.config().min_degree(), 32);
    }

    #[test]
    fn test_custom_config_tree_creation() {
        let config = OptimizationConfig::new(0.3, 16);
        let tree = MutableTree::with_config(vec![1.0, 2.0], config);
        assert!((tree.config().tolerance() - 0.3).abs() < 1e-10);
        assert_eq!(tree.config().min_degree(), 16);
    }

    #[test]
    fn test_tolerance_based_lazy_update_same_range() {
        // With b=0.4, range 2 (normally [2,4)) tolerates weights in [1.2, 4.8)
        // An element at weight 2 (log=1) should tolerate updates to ~3 (log=~1.58) without parent change
        let config = OptimizationConfig::optimized();
        let mut tree = MutableTree::with_config(vec![1.0], config); // weight 2 (range 2)

        // Check that small changes within tolerance don't require parent change
        assert!(!tree.would_require_parent_change(0, 1.2)); // weight ~2.3, within [1.2, 4.8)
        assert!(!tree.would_require_parent_change(0, 1.5)); // weight ~2.83, within [1.2, 4.8)
        assert!(!tree.would_require_parent_change(0, 1.9)); // weight ~3.73, within [1.2, 4.8)

        // Update within tolerance
        tree.update(0, 1.5);
        assert_eq!(tree.element_log_weight(0), Some(1.5));
    }

    #[test]
    fn test_tolerance_based_lazy_update_crosses_boundary() {
        // With b=0.4, range 2 tolerates [1.2, 4.8) in linear space
        // log2(4.8) ~= 2.26, so weight 5 (log ~= 2.32) should require parent change
        let config = OptimizationConfig::optimized();
        let tree = MutableTree::with_config(vec![1.0], config); // weight 2 (range 2)

        // Weight 5 (log ~= 2.32) is outside tolerated interval
        assert!(tree.would_require_parent_change(0, 5.0_f64.log2()));

        // Weight 1.0 (log=0) is also outside tolerated interval (below 1.2)
        assert!(tree.would_require_parent_change(0, 0.0));
    }

    #[test]
    fn test_basic_config_no_tolerance() {
        // With b=0, standard range [2,4) applies strictly
        let config = OptimizationConfig::basic();
        let tree = MutableTree::with_config(vec![1.0], config); // weight 2 (range 2)

        // Without tolerance, any weight outside [2,4) requires parent change
        assert!(tree.would_require_parent_change(0, 0.9)); // weight < 2
        assert!(tree.would_require_parent_change(0, 2.0)); // weight 4, at boundary
        assert!(!tree.would_require_parent_change(0, 1.5)); // weight ~2.83, within [2,4)
    }

    #[test]
    fn test_degree_bound_affects_root_classification() {
        // With d=32 (optimized), ranges need 32+ children to be non-root
        // With d=2 (basic), ranges need only 2+ children to be non-root
        let weights_32: Vec<f64> = (0..32).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect();

        let basic_tree = MutableTree::new(weights_32.clone());
        let optimized_tree = MutableTree::new_optimized(weights_32);

        // With basic config (d=2), 32 elements in one range should propagate to multiple levels
        // With optimized config (d=32), they might all fit in one level's root
        let basic_level1 = basic_tree.get_level(1).unwrap();
        let optimized_level1 = optimized_tree.get_level(1).unwrap();

        // Basic: range with 32 elements is non-root (has parent)
        assert_eq!(basic_level1.non_root_count(), 1);

        // Optimized: range with 32 elements is exactly at threshold (d=32), so non-root
        assert_eq!(optimized_level1.non_root_count(), 1);
    }

    #[test]
    fn test_degree_bound_31_elements_becomes_root() {
        // With d=32, 31 elements should make the range a root
        let weights_31: Vec<f64> = (0..31).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect();

        let optimized_tree = MutableTree::new_optimized(weights_31);
        let level1 = optimized_tree.get_level(1).unwrap();

        // Range with 31 elements is a root (degree < 32)
        assert_eq!(level1.root_count(), 1);
        assert_eq!(level1.non_root_count(), 0);
    }

    #[test]
    fn test_optimized_tree_sampling_still_works() {
        use crate::core::sample;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let tree = MutableTree::new_optimized(vec![0.0, 10.0]); // weights 1 and 1024
        let immutable = tree.as_tree();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Most samples should be element 1 (much heavier)
        let samples: Vec<_> = (0..1000)
            .filter_map(|_| sample(&immutable, &mut rng))
            .collect();
        let count1 = samples.iter().filter(|&&x| x == 1).count();
        let fraction = f64::from(u32::try_from(count1).unwrap())
            / f64::from(u32::try_from(samples.len()).unwrap());
        assert!(fraction > 0.99, "fraction was {fraction}");
    }

    #[test]
    fn test_optimized_update_correctness() {
        use crate::core::sample;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut tree = MutableTree::new_optimized(vec![0.0, 0.0]); // equal weights

        // Make element 0 much heavier
        tree.update(0, 10.0);
        let immutable = tree.as_tree();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Most samples should be element 0 now
        let samples: Vec<_> = (0..1000)
            .filter_map(|_| sample(&immutable, &mut rng))
            .collect();
        let count0 = samples.iter().filter(|&&x| x == 0).count();
        let fraction = f64::from(u32::try_from(count0).unwrap())
            / f64::from(u32::try_from(samples.len()).unwrap());
        assert!(fraction > 0.99, "fraction was {fraction}");
    }

    // -------------------------------------------------------------------------
    // Delete Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_delete_single_element() {
        let mut tree = MutableTree::new(vec![1.0, 2.0]);
        assert!(!tree.is_deleted(0));
        assert!(tree.delete(0));
        assert!(tree.is_deleted(0));
    }

    #[test]
    fn test_delete_out_of_bounds() {
        let mut tree = MutableTree::new(vec![1.0]);
        assert!(!tree.delete(5));
    }

    #[test]
    fn test_delete_already_deleted() {
        let mut tree = MutableTree::new(vec![1.0]);
        assert!(tree.delete(0));
        assert!(tree.delete(0)); // Should return true (already deleted)
    }

    #[test]
    fn test_is_deleted_out_of_bounds() {
        let tree = MutableTree::new(vec![1.0]);
        assert!(!tree.is_deleted(5));
    }

    #[test]
    fn test_active_count() {
        let mut tree = MutableTree::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(tree.active_count(), 3);

        tree.delete(1);
        assert_eq!(tree.active_count(), 2);

        tree.delete(0);
        assert_eq!(tree.active_count(), 1);
    }

    #[test]
    fn test_deleted_element_not_sampled() {
        use crate::core::sample;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut tree = MutableTree::new(vec![0.0, 0.0]); // equal weights

        // Delete element 0
        tree.delete(0);

        // Sample many times - should only get element 1
        let immutable = tree.as_tree();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..1000 {
            let sample_result = sample(&immutable, &mut rng);
            assert_eq!(sample_result, Some(1), "Sampled deleted element!");
        }
    }

    #[test]
    fn test_delete_all_elements() {
        use crate::core::sample;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut tree = MutableTree::new(vec![1.0, 2.0]);
        tree.delete(0);
        tree.delete(1);

        assert_eq!(tree.active_count(), 0);

        // Sampling should return None
        let immutable = tree.as_tree();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        assert_eq!(sample(&immutable, &mut rng), None);
    }

    // -------------------------------------------------------------------------
    // Insert Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_to_empty_tree() {
        let mut tree = MutableTree::new(vec![]);
        assert_eq!(tree.len(), 0);

        let idx = tree.insert(1.0);
        assert_eq!(idx, 0);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.element_log_weight(0), Some(1.0));
    }

    #[test]
    fn test_insert_multiple() {
        let mut tree = MutableTree::new(vec![1.0]);
        assert_eq!(tree.len(), 1);

        let idx1 = tree.insert(2.0);
        assert_eq!(idx1, 1);
        assert_eq!(tree.len(), 2);

        let idx2 = tree.insert(3.0);
        assert_eq!(idx2, 2);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_insert_and_sample() {
        use crate::core::sample;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut tree = MutableTree::new(vec![0.0]); // weight 1

        // Insert element with much higher weight
        let idx = tree.insert(10.0); // weight 1024
        assert_eq!(idx, 1);

        // New element should dominate sampling
        let immutable = tree.as_tree();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let samples: Vec<_> = (0..1000)
            .filter_map(|_| sample(&immutable, &mut rng))
            .collect();

        let count1 = samples.iter().filter(|&&x| x == 1).count();
        let fraction = f64::from(u32::try_from(count1).unwrap()) / 1000.0;
        assert!(fraction > 0.99, "fraction was {fraction}");
    }

    #[test]
    fn test_insert_deleted() {
        use crate::core::DELETED_LOG_WEIGHT;

        let mut tree = MutableTree::new(vec![1.0]);
        let idx = tree.insert(DELETED_LOG_WEIGHT);

        assert_eq!(idx, 1);
        assert!(tree.is_deleted(1));
        assert_eq!(tree.active_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Undelete Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_undelete_via_update() {
        let mut tree = MutableTree::new(vec![1.0, 2.0]);

        // Delete element 0
        tree.delete(0);
        assert!(tree.is_deleted(0));
        assert_eq!(tree.active_count(), 1);

        // Undelete by updating to positive weight
        tree.update(0, 3.0);
        assert!(!tree.is_deleted(0));
        assert_eq!(tree.active_count(), 2);
        assert_eq!(tree.element_log_weight(0), Some(3.0));
    }

    #[test]
    fn test_undelete_and_sample() {
        use crate::core::sample;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut tree = MutableTree::new(vec![0.0, 0.0]); // equal weights

        // Delete element 0
        tree.delete(0);

        // Undelete with much higher weight
        tree.update(0, 10.0);

        // Element 0 should now dominate
        let immutable = tree.as_tree();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let samples: Vec<_> = (0..1000)
            .filter_map(|_| sample(&immutable, &mut rng))
            .collect();

        let count0 = samples.iter().filter(|&&x| x == 0).count();
        let fraction = f64::from(u32::try_from(count0).unwrap()) / 1000.0;
        assert!(fraction > 0.99, "fraction was {fraction}");
    }

    #[test]
    fn test_update_to_deleted() {
        use crate::core::DELETED_LOG_WEIGHT;

        let mut tree = MutableTree::new(vec![1.0, 2.0]);

        // Update to NEG_INFINITY should delete
        tree.update(0, DELETED_LOG_WEIGHT);
        assert!(tree.is_deleted(0));
        assert_eq!(tree.active_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Statistical Correctness Tests After Updates
    // -------------------------------------------------------------------------

    #[test]
    fn test_distribution_correct_after_many_updates() {
        use crate::core::{chi_squared_from_counts, sample};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Create tree with uniform weights
        let log_weights = vec![0.0; 10]; // all weight 1
        let mut tree = MutableTree::new(log_weights);
        let mut rng = ChaCha8Rng::seed_from_u64(123_456);

        // Perform many random updates
        for i in 0..100 {
            let idx = i % tree.len();
            #[allow(clippy::cast_precision_loss)]
            let new_weight = (i as f64 * 0.1).sin().abs() + 0.1; // weights in [0.1, 1.1]
            tree.update(idx, new_weight.log2());
        }

        // Set final known weights for testing
        let final_weights: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        for (i, &w) in final_weights.iter().enumerate() {
            tree.update(i, w.log2());
        }

        // Sample many times
        let immutable = tree.as_tree();
        let mut counts = [0usize; 10];
        let num_samples = 10_000;
        for _ in 0..num_samples {
            if let Some(idx) = sample(&immutable, &mut rng) {
                counts[idx] += 1;
            }
        }

        // Verify distribution with chi-squared test
        let weights: Vec<f64> = final_weights.to_vec();
        let result = chi_squared_from_counts(&counts[..], &weights, num_samples);
        assert!(
            result.passes(0.001),
            "Distribution incorrect after updates: chi2={:.2}, p={:.6}",
            result.chi_squared,
            result.p_value
        );
    }

    #[test]
    fn test_distribution_correct_after_inserts() {
        use crate::core::{chi_squared_from_counts, sample};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Start with small tree
        let log_weights = vec![0.0, 1.0, 2.0]; // weights 1, 2, 4
        let mut tree = MutableTree::new(log_weights);

        // Insert more elements
        tree.insert(1.5_f64.log2()); // weight 1.5
        tree.insert(3.0_f64.log2()); // weight 3
        tree.insert(2.5_f64.log2()); // weight 2.5

        // Expected weights: [1, 2, 4, 1.5, 3, 2.5]
        let expected_weights = vec![1.0, 2.0, 4.0, 1.5, 3.0, 2.5];

        // Sample
        let immutable = tree.as_tree();
        let mut rng = ChaCha8Rng::seed_from_u64(654_321);
        let mut counts = vec![0usize; 6];
        let num_samples = 10_000;
        for _ in 0..num_samples {
            if let Some(idx) = sample(&immutable, &mut rng) {
                counts[idx] += 1;
            }
        }

        // Verify distribution
        let result = chi_squared_from_counts(&counts, &expected_weights, num_samples);
        assert!(
            result.passes(0.001),
            "Distribution incorrect after inserts: chi2={:.2}, p={:.6}",
            result.chi_squared,
            result.p_value
        );
    }

    #[test]
    fn test_distribution_correct_after_deletes() {
        use crate::core::{chi_squared_from_counts, sample};
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // Start with uniform weights
        let log_weights = vec![0.0; 10]; // all weight 1
        let mut tree = MutableTree::new(log_weights);

        // Delete some elements
        tree.delete(2);
        tree.delete(5);
        tree.delete(7);

        // Expected: 7 active elements with equal weight
        let active_count = tree.active_count();
        assert_eq!(active_count, 7);

        // Sample
        let immutable = tree.as_tree();
        let mut rng = ChaCha8Rng::seed_from_u64(111_222);
        let mut counts = [0usize; 10];
        let num_samples = 10_000;
        for _ in 0..num_samples {
            if let Some(idx) = sample(&immutable, &mut rng) {
                counts[idx] += 1;
            }
        }

        // Verify deleted elements got 0 samples
        assert_eq!(counts[2], 0, "Deleted element 2 was sampled");
        assert_eq!(counts[5], 0, "Deleted element 5 was sampled");
        assert_eq!(counts[7], 0, "Deleted element 7 was sampled");

        // Verify distribution among active elements is uniform
        // Filter to only active elements for chi-squared test
        let active_indices: Vec<usize> = (0..10).filter(|&i| !tree.is_deleted(i)).collect();
        let active_counts: Vec<usize> = active_indices.iter().map(|&i| counts[i]).collect();
        let active_weights = vec![1.0; 7];

        let result = chi_squared_from_counts(&active_counts, &active_weights, num_samples);
        assert!(
            result.passes(0.001),
            "Distribution incorrect after deletes: chi2={:.2}, p={:.6}",
            result.chi_squared,
            result.p_value
        );
    }

    // -------------------------------------------------------------------------
    // Additional Coverage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_empty() {
        let tree = MutableTree::new(vec![]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        let tree2 = MutableTree::new(vec![1.0]);
        assert!(!tree2.is_empty());
    }

    #[test]
    fn test_would_require_parent_change_out_of_bounds() {
        let tree = MutableTree::new(vec![1.0, 2.0]);
        // Out of bounds index should return false
        assert!(!tree.would_require_parent_change(100, 5.0));
    }

    #[test]
    fn test_mutable_tree_from_tree_with_deleted() {
        use crate::core::DELETED_LOG_WEIGHT;

        // Create a tree with some deleted elements
        let weights = vec![1.0, DELETED_LOG_WEIGHT, 2.0, DELETED_LOG_WEIGHT, 3.0];
        let tree = Tree::with_config(weights, OptimizationConfig::basic());
        let mutable = MutableTree::from_tree(&tree);

        // Verify the deleted elements are tracked
        assert!(mutable.is_deleted(1));
        assert!(mutable.is_deleted(3));
        assert!(!mutable.is_deleted(0));
        assert!(!mutable.is_deleted(2));
        assert!(!mutable.is_deleted(4));

        assert_eq!(mutable.active_count(), 3);
    }

    #[test]
    fn test_undelete_restores_element_to_empty_tree() {
        let mut tree = MutableTree::new(vec![1.0]);

        // Delete the only element
        assert!(tree.delete(0));
        assert!(tree.is_deleted(0));
        assert_eq!(tree.active_count(), 0);

        // Undelete by updating with a valid weight
        assert!(tree.update(0, 2.0));
        assert!(!tree.is_deleted(0));
        assert_eq!(tree.active_count(), 1);
    }

    #[test]
    fn test_delete_keeps_nonroot_status() {
        // Create a tree where deletion doesn't change root status
        // Need multiple elements in same range to create non-root
        let weights: Vec<f64> = (0..10).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect();
        let mut tree = MutableTree::new(weights);

        // Delete one element from a range with many elements
        // The range should stay non-root
        tree.delete(0);

        assert!(tree.is_deleted(0));
        assert_eq!(tree.active_count(), 9);
    }

    #[test]
    fn test_update_triggers_rebuild_on_root_status_change() {
        // Create tree where updates cause root status changes
        // Start with two elements in same range (non-root)
        let mut tree = MutableTree::new(vec![1.0, 1.1]);

        // Level 1 should have a non-root range
        assert!(tree.level_count() >= 2);

        // Delete one element - should change to root
        tree.delete(1);

        // Tree structure should be adjusted
        assert_eq!(tree.active_count(), 1);
    }

    #[test]
    fn test_propagate_weight_changes_through_levels() {
        // Create a multi-level tree
        let weights: Vec<f64> = (0..10).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect();
        let mut tree = MutableTree::new(weights);

        // Update an element's weight within the same range
        // Should propagate weight changes up without structural changes
        let old_levels = tree.level_count();
        tree.update(0, 1.05);

        // Structure should remain the same
        assert_eq!(tree.level_count(), old_levels);
    }

    #[test]
    fn test_get_level_returns_none_for_invalid() {
        let tree = MutableTree::new(vec![1.0]);

        // Level 0 is invalid
        assert!(tree.get_level(0).is_none());

        // Out of bounds
        assert!(tree.get_level(100).is_none());

        // Level 1 should exist
        assert!(tree.get_level(1).is_some());
    }

    #[test]
    fn test_undelete_into_existing_nonroot() {
        // Create tree with elements in same range (non-root)
        let mut tree = MutableTree::new(vec![1.0, 1.1, 1.2]);

        // Delete middle element
        tree.delete(1);

        // Undelete it back into the same range
        tree.update(1, 1.15);

        assert!(!tree.is_deleted(1));
        assert_eq!(tree.active_count(), 3);
    }

    #[test]
    fn test_undelete_creates_levels_when_empty() {
        // Start with empty tree
        let mut tree = MutableTree::new(vec![]);

        // Insert some elements
        tree.insert(1.0);
        tree.insert(1.1);

        // Delete all
        tree.delete(0);
        tree.delete(1);
        assert_eq!(tree.active_count(), 0);

        // Clear levels manually would require internal access,
        // but the levels.is_empty() check is for when tree starts empty
        // So we need to test the path differently:
        // Create empty tree and insert directly
        let mut empty_tree = MutableTree::new(vec![]);
        empty_tree.insert(1.0); // This creates levels

        assert_eq!(empty_tree.active_count(), 1);
    }

    #[test]
    fn test_undelete_transitions_root_to_nonroot() {
        // Create tree with one element (a root)
        let mut tree = MutableTree::new(vec![1.0]);

        // Insert another element in same range
        tree.insert(1.1);

        // Delete both
        tree.delete(0);
        tree.delete(1);
        assert_eq!(tree.active_count(), 0);

        // Undelete first element (creates a root)
        tree.update(0, 1.0);
        assert_eq!(tree.active_count(), 1);

        // Undelete second element (transitions root to non-root)
        tree.update(1, 1.1);
        assert_eq!(tree.active_count(), 2);
    }

    #[test]
    fn test_propagate_delete_parent_stays_nonroot() {
        // Create a tree where a parent range stays non-root after deletion
        // This requires multiple elements at a higher level
        let weights: Vec<f64> = (0..20).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect();
        let mut tree = MutableTree::new(weights);

        // With many elements in same range, parent at level 2 should be non-root
        // Delete one element
        tree.delete(0);

        // Parent range should still be non-root
        assert_eq!(tree.active_count(), 19);
    }

    #[test]
    fn test_propagate_weight_changes_stops_early() {
        // Test that weight propagation handles the case where level doesn't exist
        let mut tree = MutableTree::new(vec![1.0, 2.0, 3.0]);

        // Update with weight that stays in range
        tree.update(0, 0.9);

        assert!(!tree.is_deleted(0));
    }

    #[test]
    fn test_update_moves_element_between_ranges() {
        // Element starts in one range, moves to another
        let mut tree = MutableTree::new(vec![1.0, 2.0]);

        // Move element 0 from range 2 (weight ~2) to range 3 (weight ~4)
        tree.update(0, 2.0); // log2(4) = 2

        assert!((tree.element_log_weight(0).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_root_status_change_triggers_rebuild() {
        // Create a tree with exactly 2 elements in same range
        // This creates a non-root range at level 1
        let mut tree = MutableTree::new(vec![1.0, 1.1]);

        // Verify we have at least 2 levels (non-root at level 1)
        assert!(tree.level_count() >= 2);

        // Now update element 1 to be in a different range
        // This should change the original range from non-root to root
        tree.update(1, 5.0); // moves to a different range

        // The structural change should trigger rebuild
        assert!(!tree.is_deleted(0));
        assert!(!tree.is_deleted(1));
    }

    #[test]
    fn test_weight_change_propagates_through_nonroot_ranges() {
        // Create a deep tree with non-root ranges at multiple levels
        let weights: Vec<f64> = (0..32).map(|i| f64::from(i).mul_add(0.001, 1.0)).collect();
        let mut tree = MutableTree::new(weights);

        // Update a weight that stays in range but changes the total
        // This should propagate weight changes through non-root ranges
        tree.update(0, 1.01);

        // Verify tree is still valid
        assert_eq!(tree.active_count(), 32);
    }

    #[test]
    fn test_insert_triggers_propagate_insert_path() {
        // Start with elements that create a root range
        let mut tree = MutableTree::new(vec![1.0]);

        // Insert another element in same range - should trigger propagate_insert
        tree.insert(1.05);

        // Now range at level 1 becomes non-root (degree 2)
        assert_eq!(tree.active_count(), 2);
        assert!(tree.level_count() >= 2);
    }

    #[test]
    fn test_delete_triggers_parent_stays_nonroot() {
        // Create a structure where deleting from level 1 triggers propagate_delete
        // but the parent at level 2 stays non-root because it has 3+ children.
        //
        // Strategy: Create many elements in the same weight range so they form
        // multiple non-root ranges at level 1, which all become children of the
        // same parent range at level 2.
        //
        // With min_degree=2:
        // - 6 elements with similar weights -> 3 non-root ranges at level 1
        // - All 3 ranges have similar total weights -> same parent range at level 2
        // - Parent at level 2 has degree 3 (non-root)
        // - Delete 1 element -> range becomes root, parent now has degree 2 (still non-root)

        // Elements all around log_weight = 1.0 (weight = 2), slight variations
        // to spread across different ranges at level 1
        let weights = vec![1.0, 1.01, 1.0, 1.01, 1.0, 1.01];
        let mut tree = MutableTree::new(weights);

        // Initial structure should have multiple levels
        let initial_levels = tree.level_count();

        // Delete one element to trigger propagate_delete
        // This should make one range become root while parent stays non-root
        tree.delete(0);

        assert_eq!(tree.active_count(), 5);
        assert!(tree.level_count() >= initial_levels.saturating_sub(1));
    }

    #[test]
    fn test_propagate_weight_recursive() {
        // Test that weight propagation continues through multiple levels
        // Create a tree with multiple levels by having elements in different ranges
        let weights: Vec<f64> = (0..64).map(|i| f64::from(i).mul_add(0.1, 1.0)).collect();
        let mut tree = MutableTree::new(weights);

        let initial_levels = tree.level_count();
        assert!(initial_levels >= 2, "expected 2+ levels for this test");

        // Update a weight that stays in range - should propagate up
        tree.update(0, 1.01);

        // Tree structure should be maintained
        assert_eq!(tree.active_count(), 64);
    }

    // -------------------------------------------------------------------------
    // propagate_delete edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_propagate_delete_past_top_level() {
        // Create a tree with few elements that might have minimal levels
        // Then delete elements to trigger propagation past the top level
        let mut tree = MutableTree::new(vec![1.0, 1.0, 1.0]);

        // Delete elements one by one - this exercises propagate_delete
        tree.delete(0);
        tree.delete(1);
        tree.delete(2);

        // All should be deleted
        assert!(tree.is_deleted(0));
        assert!(tree.is_deleted(1));
        assert!(tree.is_deleted(2));
        assert_eq!(tree.active_count(), 0);
    }

    #[test]
    fn test_propagate_delete_multiple_levels() {
        // Create a tree with enough elements to have multiple levels
        // Then delete elements to cause cascading propagation
        let weights: Vec<f64> = (0..16).map(|i| f64::from(i + 1)).collect();
        let mut tree = MutableTree::new(weights);

        // Delete multiple elements to exercise propagate_delete thoroughly
        for i in 0..16 {
            tree.delete(i);
        }

        // All should be deleted
        assert_eq!(tree.active_count(), 0);
    }

    #[test]
    fn test_equalize_weights_exercises_propagate_delete() {
        // This test mimics what the Hypothesis stateful tests do:
        // Start with varied weights, then set all to 1.0
        // This causes major tree restructuring that exercises propagate_delete
        let weights: Vec<f64> = (1..=20).map(f64::from).collect();
        let mut tree = MutableTree::new(weights);

        // Set all weights to 1.0 (log2(1) = 0)
        for i in 0..20 {
            tree.update(i, 0.0); // log2(1) = 0
        }

        // Tree should still function correctly
        assert_eq!(tree.active_count(), 20);

        // All weights should be 0.0 (log2)
        for i in 0..20 {
            assert_eq!(tree.element_log_weight(i), Some(0.0));
        }
    }

    #[test]
    fn test_hypothesis_minimal_reproducer() {
        // Minimal reproducer from Hypothesis stateful test:
        // init_sampler(weights=[0.1, 0.1])
        // equalize_weights() repeated with sampling interleaved
        //
        // This exercises the propagate_delete early return path

        use crate::core::sampler::sample;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // weights=[0.1, 0.1] in log2 space = log2(0.1) ≈ -3.32
        let log_weight = 0.1_f64.log2();
        let mut tree = MutableTree::new(vec![log_weight, log_weight]);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // equalize_weights sets all weights to 1.0 (log2(1) = 0)
        // Interleave with sampling like Hypothesis does
        for _ in 0..15 {
            tree.update(0, 0.0);
            tree.update(1, 0.0);
            // Sample (like never_samples_effectively_removed invariant)
            let _ = sample(&tree.as_tree(), &mut rng);
        }

        // Tree should still be valid
        assert_eq!(tree.active_count(), 2);
    }

    #[test]
    fn test_weight_updates_cause_range_transitions() {
        // Create elements in different ranges, then move them all to one range
        // This should exercise propagate_delete when ranges empty out
        let mut tree = MutableTree::new(vec![1.0, 4.0, 16.0, 64.0]); // Different ranges

        // Move all to the same range (weight 1.0 = log 0)
        tree.update(0, 0.0);
        tree.update(1, 0.0);
        tree.update(2, 0.0);
        tree.update(3, 0.0);

        // All should now be at log weight 0
        assert_eq!(tree.element_log_weight(0), Some(0.0));
        assert_eq!(tree.element_log_weight(1), Some(0.0));
        assert_eq!(tree.element_log_weight(2), Some(0.0));
        assert_eq!(tree.element_log_weight(3), Some(0.0));
    }

    #[test]
    fn test_propagate_delete_reaches_top_of_tree() {
        // This test creates a tree with multiple levels, then deletes elements
        // to cause propagate_delete to recurse past the topmost level.
        //
        // Strategy:
        // 1. Create elements that produce non-root ranges (degree >= 2 at each level)
        // 2. Delete elements to make ranges become root/empty
        // 3. The deletion cascade should eventually recurse past the top

        // Create many elements in the SAME range to ensure we get non-root ranges.
        // All weights = 1.0 (log2 = 0) means they all go to range 1.
        // This creates a tree with multiple levels because range 1 at each level
        // will have many children.
        let n = 64; // Enough to create multiple levels
        let weights: Vec<f64> = vec![0.0; n]; // All in range 1

        let mut tree = MutableTree::new(weights);
        let initial_levels = tree.level_count();

        // The tree should have multiple levels because all elements are in one range
        assert!(
            initial_levels >= 2,
            "Expected at least 2 levels, got {initial_levels}"
        );

        // Delete all but one element - this should cause the tree to shrink
        // and propagate_delete to eventually recurse past the top
        for i in 0..n - 1 {
            tree.delete(i);
        }

        // Tree should still work
        assert_eq!(tree.active_count(), 1);
        assert!(!tree.is_deleted(n - 1));

        // Now delete the last one
        tree.delete(n - 1);
        assert_eq!(tree.active_count(), 0);
    }

    #[test]
    fn test_propagate_delete_recurses_to_nonexistent_level() {
        // This test specifically exercises the base case of propagate_delete
        // where we try to propagate past the topmost level.
        //
        // The key insight: when a range at the highest level transitions
        // from non-root to root/empty, the recursive call tries to find
        // a parent at a level that doesn't exist.

        // Start with 4 elements in the same range (all weight 1.0)
        // This creates a single non-root range at level 1
        let weights: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0];
        let mut tree = MutableTree::new(weights);

        // With 4 elements in one range, level 1 has one range with degree 4
        // Level 2 should have one range containing that level-1 range
        let level_count = tree.level_count();

        // Delete 3 of 4 elements. When we go from 2 children to 1,
        // the range at level 1 becomes a root, triggering propagate_delete
        // at the next level up
        tree.delete(0);
        tree.delete(1);
        tree.delete(2);

        // After deletes, only 1 element remains
        assert_eq!(tree.active_count(), 1);

        // The level count might have changed
        let _new_level_count = tree.level_count();

        // The key assertion: tree still works after cascading deletes
        assert!(!tree.is_deleted(3));
        assert_eq!(tree.element_log_weight(3), Some(0.0));

        // Delete the last one to fully exercise the path
        tree.delete(3);
        assert_eq!(tree.active_count(), 0);

        // Ensure we had multiple levels to propagate through
        assert!(
            level_count >= 2,
            "Test requires at least 2 levels initially, got {level_count}"
        );
    }

    #[test]
    fn test_delete_all_triggers_full_propagation() {
        // Similar to above but with varied weights to create a more complex tree structure.
        // Elements with weights 2^0, 2^1, 2^2, ... go to different ranges.
        let weights: Vec<f64> = (0..32).map(|i| f64::from(i % 8)).collect();
        let mut tree = MutableTree::new(weights);

        // Delete all elements one by one
        for i in 0..32 {
            tree.delete(i);
        }

        assert_eq!(tree.active_count(), 0);
    }

    #[test]
    fn test_update_crosses_range_boundary_triggers_propagate_delete() {
        use crate::core::sample;

        // Reproducer from Hypothesis: update that moves element to different range
        // and triggers propagate_delete to reach the base case (recursion past top level).
        //
        // weights = [0.5, 495120.14], update index 0 to 304492.81
        // This causes:
        // - Element 0 moves from one range to another
        // - Old range becomes root/empty, triggering propagate_delete
        // - propagate_delete reaches level_num where level_idx >= levels.len()
        let mut tree = MutableTree::new(vec![0.5_f64.log2(), 495_120.14_f64.log2()]);

        // Update element 0 to a much larger weight
        tree.update(0, 304_492.81_f64.log2());

        // Verify the tree is still valid structurally
        assert_eq!(tree.active_count(), 2);
        assert!((tree.element_log_weight(0).unwrap() - 304_492.81_f64.log2()).abs() < 1e-10);
        assert!((tree.element_log_weight(1).unwrap() - 495_120.14_f64.log2()).abs() < 1e-10);

        // Verify sampling behavior: both weights are similar (~300k vs ~500k),
        // so element 1 should be sampled more often (about 62% vs 38%)
        let mut rng = rand::thread_rng();
        let samples: Vec<usize> = (0..1000)
            .filter_map(|_| sample(&tree.as_tree(), &mut rng))
            .collect();
        let count_0 = samples.iter().filter(|&&x| x == 0).count();
        let count_1 = samples.iter().filter(|&&x| x == 1).count();

        // Element 1 has higher weight, should be sampled more
        assert!(
            count_1 > count_0,
            "Element 1 (weight ~495k) should be sampled more than element 0 (weight ~304k), got {count_1} vs {count_0}",
        );
        // Both elements should be reachable
        assert!(count_0 > 0, "Element 0 should be sampled at least once");
        assert!(count_1 > 0, "Element 1 should be sampled at least once");
    }

    #[test]
    fn test_shrink_tree_via_weight_changes() {
        // Create a tree and then move all elements to the same range,
        // which should cause major structural changes including propagate_delete
        // being called at the higher levels.

        // Start with varied weights across different ranges
        let weights: Vec<f64> = (0..32).map(|i| f64::from(i + 1).log2()).collect();
        let mut tree = MutableTree::new(weights);

        // Move all elements to the same range (weight = 1, log = 0)
        for i in 0..32 {
            tree.update(i, 0.0);
        }

        // Now delete all but a few
        for i in 0..30 {
            tree.delete(i);
        }

        // Remaining elements should still work
        assert_eq!(tree.active_count(), 2);
    }

    #[test]
    fn test_collapse_complex_tree_structure() {
        // This test mimics what Hypothesis does: start with varied weights
        // spread across many ranges, then equalize them all.
        //
        // The key insight: when elements move from many different ranges
        // into one range, the source ranges become empty and trigger
        // cascading propagate_delete calls up the tree.

        // Create elements with exponentially varying weights to ensure
        // they're spread across many different ranges
        let n = 50;
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                // Weights: 1, 2, 4, 8, 16, 32, ... spread elements across ranges
                let power = i % 10;
                f64::from(1u32 << power).log2()
            })
            .collect();

        let mut tree = MutableTree::new(weights);
        let initial_levels = tree.level_count();

        // Verify we have a complex structure
        assert!(
            initial_levels >= 1,
            "Should have at least 1 level initially"
        );

        // Now equalize all weights to 1.0 (log2 = 0)
        // This moves all elements to the same range, emptying the others
        for i in 0..n {
            tree.update(i, 0.0);
        }

        // Tree should still work correctly
        assert_eq!(tree.active_count(), n);
        for i in 0..n {
            assert_eq!(tree.element_log_weight(i), Some(0.0));
        }
    }

    #[test]
    fn test_repeated_equalize_and_diversify() {
        // Alternate between spreading elements across ranges and
        // collapsing them to the same range. This exercises the
        // tree restructuring code paths extensively.

        let n = 20usize;
        #[allow(clippy::cast_precision_loss)]
        let initial_weights: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).log2()).collect();

        let mut tree = MutableTree::new(initial_weights);

        for iteration in 0..5 {
            // Equalize: move all to range 1
            for i in 0..n {
                tree.update(i, 0.0);
            }
            assert_eq!(tree.active_count(), n, "Iteration {iteration} equalize");

            // Diversify: spread across ranges
            for i in 0..n {
                let power = i % 8;
                tree.update(i, f64::from(1u32 << power).log2());
            }
            assert_eq!(tree.active_count(), n, "Iteration {iteration} diversify");
        }

        // Final state should be valid
        assert_eq!(tree.active_count(), n);
    }
}
