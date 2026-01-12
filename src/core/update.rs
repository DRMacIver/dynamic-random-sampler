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

            // Insert into level 1
            if self.levels.is_empty() {
                let level1 = Level::with_config(1, self.config);
                self.levels.push(level1);
            }

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

        if stays_in_range {
            // Lazy update: just update the weight in the current range
            if let Some(level) = self.levels.get_mut(0) {
                if let Some(range) = level.get_range_mut(old_range) {
                    range.update_child_weight(index, new_log_weight);
                }
            }
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
    fn propagate_delete(&mut self, level_num: usize, range_number: i32) {
        let _guard = TimeoutGuard::new("propagate_delete");

        // If no parent level exists, nothing to do
        if level_num >= self.levels.len() {
            return;
        }

        // Get the parent range number by looking at where this range is stored
        // The range is stored as a child in a parent range at level_num + 1
        // with child_idx = range_number
        #[allow(clippy::cast_sign_loss)]
        let child_idx = range_number as usize;

        // Find which parent range contains this child
        let parent_range_number = self.levels.get(level_num).and_then(|l| {
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
            .get(level_num)
            .is_some_and(|l| !l.is_root_range(parent_range));

        // Remove this range from the parent level
        if let Some(parent_level) = self.levels.get_mut(level_num) {
            parent_level.remove_child(parent_range, child_idx);
        }

        // Check if parent is now root (or empty)
        let parent_is_root_or_empty = self
            .levels
            .get(level_num)
            .is_none_or(|l| l.is_root_range(parent_range) || l.get_range(parent_range).is_none());

        // Propagate up if parent changed from non-root to root/empty
        if parent_was_non_root && parent_is_root_or_empty {
            self.propagate_delete(level_num + 1, parent_range);
        } else if parent_was_non_root {
            // Still non-root, just update weight
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
    fn propagate_weight_changes(&mut self, level_num: usize, range_number: i32) {
        if level_num > self.levels.len() {
            return;
        }

        // Check if this range's degree changed (root <-> non-root)
        let was_root_before = self.was_root_at_level(level_num, range_number);

        // Recompute and cache the range weight
        let new_weight = if level_num <= self.levels.len() {
            let level = &mut self.levels[level_num - 1];
            if let Some(range) = level.get_range_mut(range_number) {
                range.total_log_weight()
            } else {
                return;
            }
        } else {
            return;
        };

        let is_root_now = self.is_root_at_level(level_num, range_number);

        // If root status changed, may need to restructure
        if was_root_before != is_root_now {
            self.handle_root_status_change(level_num, range_number, is_root_now);
        } else if !is_root_now && level_num < self.levels.len() {
            // Non-root range: update weight in parent level

            // If the range is empty/gone (NEG_INFINITY), nothing to propagate
            if is_deleted_weight(new_weight) {
                return;
            }

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

            let mut next_level = Level::with_config(current_level_num + 1, self.config);
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
}
