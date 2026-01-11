//! Sampling algorithm for the dynamic random sampler.
//!
//! This module implements the `O(log* N)` expected time sampling algorithm
//! from Section 2.2 of the paper.
//!
//! The sampling algorithm works in three steps:
//! 1. Select a level table `T_ℓ` with probability proportional to `weight(T_ℓ)`
//! 2. From `T_ℓ`, select a root range `R_j` using the first-fit method
//! 3. Walk down the tree from `R_j` using rejection sampling until reaching an element

use rand::Rng;

use crate::core::{log_sum_exp, Range, Tree};

/// Sample a random element from the tree according to the weight distribution.
///
/// Returns the index of the sampled element.
///
/// # Algorithm (Section 2.2)
/// 1. Select level ℓ with probability `weight(T_ℓ) / Σweight(T_i)`
/// 2. Select root range `R_j` from `T_ℓ` using first-fit method
/// 3. Walk down from `R_j` using rejection sampling
///
/// Expected time: `O(log* N)`
pub fn sample<R: Rng>(tree: &Tree, rng: &mut R) -> Option<usize> {
    if tree.is_empty() {
        return None;
    }

    // Step 1: Select a level table with probability proportional to its weight
    let level_num = select_level(tree, rng)?;

    // Step 2: Select a root range from the level using first-fit
    let level = tree.get_level(level_num)?;
    let range = select_root_range(level, rng)?;

    // Step 3: Walk down the tree to select an element
    walk_down(tree, level_num, range, rng)
}

/// Select a level with probability proportional to its root total weight.
fn select_level<R: Rng>(tree: &Tree, rng: &mut R) -> Option<usize> {
    let max_level = tree.max_level();
    if max_level == 0 {
        return None;
    }

    // Compute total weight across all level tables
    let level_weights: Vec<f64> = (1..=max_level).map(|l| tree.level_root_total(l)).collect();

    let total_log_weight = log_sum_exp(level_weights.iter().copied());
    if total_log_weight.is_infinite() && total_log_weight < 0.0 {
        return None;
    }

    // Sample a level using weighted sampling
    let u: f64 = rng.gen();
    let target = u * total_log_weight.exp2();

    let mut cumulative = 0.0;
    for (i, &log_weight) in level_weights.iter().enumerate() {
        cumulative += log_weight.exp2();
        if cumulative >= target {
            return Some(i + 1);
        }
    }

    Some(max_level)
}

/// Select a root range from a level using the first-fit method.
///
/// From Section 2.2: Generate random number `u ∈ [0, roots(T_ℓ))`,
/// then find first j such that `u < 2^j` and `R_j` is non-empty root.
fn select_root_range<'a, R: Rng>(level: &'a crate::core::Level, rng: &mut R) -> Option<&'a Range> {
    let roots_sum = level.roots_sum();
    if roots_sum == 0.0 {
        return None;
    }

    // Generate u ∈ [0, roots_sum)
    let u: f64 = rng.gen::<f64>() * roots_sum;

    // First-fit: find first j where u < cumulative sum of 2^j
    // Root ranges are sorted by range number, so iterate in order
    let mut root_ranges: Vec<_> = level.root_ranges().collect();
    root_ranges.sort_unstable_by_key(|(j, _)| *j);

    let mut cumulative = 0.0;
    for (j, range) in root_ranges {
        cumulative += 2.0_f64.powi(j);
        if u < cumulative {
            return Some(range);
        }
    }

    // Should not reach here, but return last root as fallback
    level.root_ranges().next().map(|(_, r)| r)
}

/// Walk down from a range at a given level to select an element.
///
/// Uses rejection sampling at each level.
fn walk_down<R: Rng>(
    tree: &Tree,
    start_level: usize,
    start_range: &Range,
    rng: &mut R,
) -> Option<usize> {
    let mut current_level = start_level;
    let mut current_range = start_range;

    loop {
        // At level 1, children are elements - sample directly
        if current_level == 1 {
            return sample_from_range(current_range, rng);
        }

        // At higher levels, children are ranges from the previous level
        // Use rejection sampling to select a child
        let child_index = sample_from_range(current_range, rng)?;

        // The child_index at level > 1 refers to a range number at level - 1
        #[allow(clippy::cast_possible_wrap)]
        let child_range_number = child_index as i32;

        let next_level = tree.get_level(current_level - 1)?;
        let next_range = next_level.get_range(child_range_number)?;

        current_level -= 1;
        current_range = next_range;
    }
}

/// Sample a child from a range using rejection sampling.
///
/// Samples proportional to child weights.
fn sample_from_range<R: Rng>(range: &Range, rng: &mut R) -> Option<usize> {
    if range.is_empty() {
        return None;
    }

    let j = range.range_number();
    let upper_bound = 2.0_f64.powi(j); // Maximum weight in this range

    // Collect children for rejection sampling
    let children: Vec<_> = range.children().collect();
    if children.is_empty() {
        return None;
    }

    loop {
        // Pick a random child uniformly
        let idx = rng.gen_range(0..children.len());
        let (child_idx, log_weight) = children[idx];

        // Accept with probability weight / upper_bound
        let weight = log_weight.exp2();
        let accept_prob = weight / upper_bound;

        if rng.gen::<f64>() < accept_prob {
            return Some(child_idx);
        }
    }
}

/// Sample multiple elements from the tree.
///
/// Returns a vector of sampled indices.
pub fn sample_n<R: Rng>(tree: &Tree, n: usize, rng: &mut R) -> Vec<usize> {
    (0..n).filter_map(|_| sample(tree, rng)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn make_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(12345)
    }

    // -------------------------------------------------------------------------
    // Basic Sampling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_empty_tree() {
        let tree = Tree::new(vec![]);
        let mut rng = make_rng();
        assert_eq!(sample(&tree, &mut rng), None);
    }

    #[test]
    fn test_sample_single_element() {
        let tree = Tree::new(vec![1.0]); // weight 2
        let mut rng = make_rng();

        // Should always return 0
        for _ in 0..10 {
            assert_eq!(sample(&tree, &mut rng), Some(0));
        }
    }

    #[test]
    fn test_sample_returns_valid_index() {
        let tree = Tree::new(vec![1.0, 2.0, 3.0]);
        let mut rng = make_rng();

        for _ in 0..100 {
            let idx = sample(&tree, &mut rng);
            assert!(idx.is_some());
            assert!(idx.unwrap() < 3);
        }
    }

    // -------------------------------------------------------------------------
    // Distribution Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_distribution_two_elements() {
        // Weight 1 vs weight 2 → should sample element 1 twice as often
        let tree = Tree::new(vec![0.0, 1.0]); // weights 1, 2
        let mut rng = make_rng();

        let samples = sample_n(&tree, 10000, &mut rng);
        let count_0: usize = samples.iter().filter(|&&x| x == 0).count();
        let count_1: usize = samples.iter().filter(|&&x| x == 1).count();

        // Expected ratio: 1:2, so count_1 should be about 2x count_0
        let ratio = count_1 as f64 / count_0 as f64;
        assert!(ratio > 1.5 && ratio < 2.5, "ratio was {ratio}");
    }

    #[test]
    fn test_sample_distribution_equal_weights() {
        let tree = Tree::new(vec![0.0, 0.0, 0.0]); // all weight 1
        let mut rng = make_rng();

        let samples = sample_n(&tree, 10000, &mut rng);
        let counts: Vec<usize> = (0..3)
            .map(|i| samples.iter().filter(|&&x| x == i).count())
            .collect();

        // Each should be about 1/3 of total
        for &count in &counts {
            let fraction = count as f64 / 10000.0;
            assert!(
                fraction > 0.25 && fraction < 0.42,
                "fraction was {fraction}"
            );
        }
    }

    #[test]
    fn test_sample_distribution_highly_skewed() {
        // weight 1 vs weight 1024 → element 1 should dominate
        let tree = Tree::new(vec![0.0, 10.0]); // weights 1, 1024
        let mut rng = make_rng();

        let samples = sample_n(&tree, 10000, &mut rng);
        let count_1: usize = samples.iter().filter(|&&x| x == 1).count();

        // Element 1 should be sampled ~1024/1025 of the time
        let fraction = count_1 as f64 / 10000.0;
        assert!(fraction > 0.99, "fraction was {fraction}");
    }

    // -------------------------------------------------------------------------
    // Multi-Level Tree Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_multi_level_tree() {
        // Elements in same range create multi-level tree
        let tree = Tree::new(vec![1.0, 1.1, 1.2, 1.3]);
        let mut rng = make_rng();

        // Verify tree has multiple levels
        assert!(tree.level_count() >= 2);

        // Sampling should still work
        for _ in 0..100 {
            let idx = sample(&tree, &mut rng);
            assert!(idx.is_some());
            assert!(idx.unwrap() < 4);
        }
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_negative_log_weights() {
        // Weights less than 1
        let tree = Tree::new(vec![-1.0, -2.0]); // weights 0.5, 0.25
        let mut rng = make_rng();

        let samples = sample_n(&tree, 10000, &mut rng);
        let count_0: usize = samples.iter().filter(|&&x| x == 0).count();
        let count_1: usize = samples.iter().filter(|&&x| x == 1).count();

        // Weight ratio 0.5:0.25 = 2:1
        let ratio = count_0 as f64 / count_1 as f64;
        assert!(ratio > 1.5 && ratio < 2.5, "ratio was {ratio}");
    }

    #[test]
    fn test_sample_wide_weight_range() {
        // Weights spanning many orders of magnitude
        let tree = Tree::new(vec![0.0, 10.0, 20.0]); // weights 1, 1024, 1048576
        let mut rng = make_rng();

        let samples = sample_n(&tree, 10000, &mut rng);
        let count_2: usize = samples.iter().filter(|&&x| x == 2).count();

        // Element 2 should be sampled almost all the time
        let fraction = count_2 as f64 / 10000.0;
        assert!(fraction > 0.99, "fraction was {fraction}");
    }

    #[test]
    fn test_sample_many_elements() {
        let weights: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();
        let tree = Tree::new(weights);
        let mut rng = make_rng();

        // Sampling should work efficiently
        for _ in 0..1000 {
            let idx = sample(&tree, &mut rng);
            assert!(idx.is_some());
            assert!(idx.unwrap() < 100);
        }
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_select_level_single_level() {
        let tree = Tree::new(vec![0.0, 2.0]); // different ranges → single level
        let mut rng = make_rng();

        let level = select_level(&tree, &mut rng);
        assert_eq!(level, Some(1));
    }

    #[test]
    fn test_sample_from_range_single_child() {
        let mut range = Range::new(2);
        range.add_child(5, 1.0);
        let mut rng = make_rng();

        // Should always return 5
        for _ in 0..10 {
            assert_eq!(sample_from_range(&range, &mut rng), Some(5));
        }
    }

    #[test]
    fn test_sample_n() {
        let tree = Tree::new(vec![1.0, 2.0]);
        let mut rng = make_rng();

        let samples = sample_n(&tree, 100, &mut rng);
        assert_eq!(samples.len(), 100);
    }

    // -------------------------------------------------------------------------
    // Determinism Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sampling_is_deterministic_with_seed() {
        let tree = Tree::new(vec![0.0, 1.0, 2.0]);

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        let samples1 = sample_n(&tree, 100, &mut rng1);
        let samples2 = sample_n(&tree, 100, &mut rng2);

        assert_eq!(samples1, samples2);
    }
}
