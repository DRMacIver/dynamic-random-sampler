//! Benchmarks for sampling performance.
//!
//! These benchmarks test various weight distributions.
//! Correctness is verified by separate tests in the test suite.

// Clippy config for benchmarks - don't need production-level strictness
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use dynamic_random_sampler::core::{sample, sample_n, MutableTree, Tree};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Duration;

/// Weight distribution types for benchmarking.
#[derive(Debug, Clone, Copy)]
pub enum Distribution {
    /// All weights equal (uniform sampling).
    Uniform,
    /// Weights follow power law: w_i = 1 / (i + 1)^alpha.
    PowerLaw { alpha: f64 },
    /// Single element has all the weight.
    OneHot { hot_index: usize },
    /// Exponential decay: w_i = exp(-lambda * i).
    Exponential { lambda: f64 },
}

impl Distribution {
    fn name(&self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::PowerLaw { .. } => "power_law",
            Self::OneHot { .. } => "one_hot",
            Self::Exponential { .. } => "exponential",
        }
    }

    /// Generate weights for this distribution.
    pub fn generate_weights(&self, n: usize) -> Vec<f64> {
        match self {
            Self::Uniform => vec![1.0; n],
            Self::PowerLaw { alpha } => (0..n)
                .map(|i| 1.0 / (i as f64 + 1.0).powf(*alpha))
                .collect(),
            Self::OneHot { hot_index } => {
                let mut weights = vec![1e-10; n];
                if *hot_index < n {
                    weights[*hot_index] = 1.0;
                }
                weights
            }
            Self::Exponential { lambda } => (0..n).map(|i| (-lambda * i as f64).exp()).collect(),
        }
    }
}

/// Convert linear weights to log weights for tree construction.
fn to_log_weights(weights: &[f64]) -> Vec<f64> {
    weights.iter().map(|w| w.log2()).collect()
}

/// Benchmark tree construction.
fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let distributions = [Distribution::Uniform, Distribution::PowerLaw { alpha: 1.0 }];

    let sizes = [100, 1000];

    for dist in &distributions {
        for &n in &sizes {
            let weights = dist.generate_weights(n);
            let log_weights = to_log_weights(&weights);

            group.bench_with_input(
                BenchmarkId::new(dist.name(), n),
                &log_weights,
                |b, log_weights| {
                    b.iter(|| MutableTree::new(black_box(log_weights.clone())));
                },
            );
        }
    }

    group.finish();
}

/// Benchmark single sample performance.
fn bench_single_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_sample");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let distributions = [
        Distribution::Uniform,
        Distribution::PowerLaw { alpha: 1.0 },
        Distribution::OneHot { hot_index: 0 },
    ];

    let sizes = [100, 1000];

    for dist in &distributions {
        for &n in &sizes {
            let weights = dist.generate_weights(n);
            let log_weights = to_log_weights(&weights);
            let tree = Tree::new(log_weights);
            let mut rng = ChaCha8Rng::seed_from_u64(12345);

            group.bench_with_input(BenchmarkId::new(dist.name(), n), &tree, |b, tree| {
                b.iter(|| sample(black_box(tree), &mut rng));
            });
        }
    }

    group.finish();
}

/// Benchmark batch sampling (1000 samples at a time).
fn bench_batch_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_1000");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let distributions = [Distribution::Uniform, Distribution::PowerLaw { alpha: 1.0 }];

    let sizes = [100, 1000];

    for dist in &distributions {
        for &n in &sizes {
            let weights = dist.generate_weights(n);
            let log_weights = to_log_weights(&weights);
            let tree = Tree::new(log_weights);
            let mut rng = ChaCha8Rng::seed_from_u64(12345);

            group.bench_with_input(BenchmarkId::new(dist.name(), n), &tree, |b, tree| {
                b.iter(|| sample_n(black_box(tree), 1000, &mut rng));
            });
        }
    }

    group.finish();
}

/// Benchmark weight update performance.
fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let sizes = [100, 1000];

    for &n in &sizes {
        let weights = Distribution::Uniform.generate_weights(n);
        let log_weights = to_log_weights(&weights);
        let mut tree = MutableTree::new(log_weights);
        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        // Benchmark same-range updates (no structural changes)
        group.bench_function(BenchmarkId::new("same_range", n), |b| {
            b.iter(|| {
                // Update to a weight still in the same range (small change)
                let idx = rng.gen_range(0..n);
                let current = tree.element_log_weight(idx).unwrap();
                // Stay within same range by small adjustment
                let new_weight = current + 0.01;
                tree.update(black_box(idx), black_box(new_weight));
            });
        });

        // Reset tree for next benchmark
        let log_weights = to_log_weights(&weights);
        let mut tree = MutableTree::new(log_weights);

        // Benchmark cross-range updates (structural changes)
        group.bench_function(BenchmarkId::new("cross_range", n), |b| {
            b.iter(|| {
                let idx = rng.gen_range(0..n);
                // Large change that likely crosses range boundaries
                let new_weight = rng.gen_range(-5.0..5.0);
                tree.update(black_box(idx), black_box(new_weight));
            });
        });
    }

    group.finish();
}

/// Benchmark insert and delete operations.
fn bench_insert_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_delete");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let sizes = [100, 1000];

    for &n in &sizes {
        let weights = Distribution::Uniform.generate_weights(n);
        let log_weights = to_log_weights(&weights);
        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        // Benchmark insert - use a steady-state tree with insert+delete pairs
        // to keep size bounded and measure true insert cost
        let mut tree = MutableTree::new(log_weights.clone());
        let mut inserted_indices: Vec<usize> = Vec::new();

        group.bench_function(BenchmarkId::new("insert", n), |b| {
            b.iter(|| {
                let weight = rng.gen_range(-2.0..2.0);
                let idx = tree.insert(black_box(weight));
                inserted_indices.push(idx);

                // Keep tree size bounded by deleting old inserted elements
                if inserted_indices.len() > n {
                    let old_idx = inserted_indices.remove(0);
                    tree.delete(old_idx);
                }
            });
        });

        // Benchmark delete (soft delete) - delete random elements, reset when half deleted
        let mut tree = MutableTree::new(log_weights.clone());
        let mut available: Vec<usize> = (0..n).collect();
        let mut deleted_count = 0;

        group.bench_function(BenchmarkId::new("delete", n), |b| {
            b.iter(|| {
                if deleted_count >= n / 2 {
                    // Reset: rebuild tree
                    tree = MutableTree::new(log_weights.clone());
                    available = (0..n).collect();
                    deleted_count = 0;
                }

                // Pick and delete a random available element
                if !available.is_empty() {
                    let pos = rng.gen_range(0..available.len());
                    let idx = available.swap_remove(pos);
                    tree.delete(black_box(idx));
                    deleted_count += 1;
                }
            });
        });
    }

    group.finish();
}

/// Benchmark mixed sample + update workflow.
fn bench_mixed_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let sizes = [100, 1000];

    for &n in &sizes {
        let weights = Distribution::PowerLaw { alpha: 1.0 }.generate_weights(n);
        let log_weights = to_log_weights(&weights);
        let tree = MutableTree::new(log_weights);
        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        // 90% samples, 10% updates
        group.bench_function(BenchmarkId::new("90_sample_10_update", n), |b| {
            let mut tree = tree.as_tree();
            let mut mutable = MutableTree::new(to_log_weights(&weights));
            b.iter(|| {
                if rng.gen_ratio(9, 10) {
                    black_box(sample(&tree, &mut rng));
                } else {
                    let idx = rng.gen_range(0..n);
                    let new_weight = rng.gen_range(-2.0..2.0);
                    mutable.update(idx, new_weight);
                    tree = mutable.as_tree();
                }
            });
        });

        // 50% samples, 50% updates
        group.bench_function(BenchmarkId::new("50_sample_50_update", n), |b| {
            let mut tree = tree.as_tree();
            let mut mutable = MutableTree::new(to_log_weights(&weights));
            b.iter(|| {
                if rng.gen_bool(0.5) {
                    black_box(sample(&tree, &mut rng));
                } else {
                    let idx = rng.gen_range(0..n);
                    let new_weight = rng.gen_range(-2.0..2.0);
                    mutable.update(idx, new_weight);
                    tree = mutable.as_tree();
                }
            });
        });
    }

    group.finish();
}

/// Benchmark bulk delete operations.
fn bench_bulk_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_delete");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let sizes = [100, 1000];

    for &n in &sizes {
        let weights = Distribution::Uniform.generate_weights(n);
        let log_weights = to_log_weights(&weights);

        // Benchmark deleting k elements from the END (should be optimizable)
        for k in [1, 10, 50] {
            if k > n / 2 {
                continue;
            }
            let mut tree = MutableTree::new(log_weights.clone());
            let mut deleted_count = 0;

            group.bench_function(BenchmarkId::new(format!("from_end_{k}"), n), |b| {
                b.iter(|| {
                    // Reset tree when we've deleted too many
                    if deleted_count >= n / 2 {
                        tree = MutableTree::new(log_weights.clone());
                        deleted_count = 0;
                    }

                    // Delete k elements from the end
                    let current_len = tree.len();
                    for i in 0..k {
                        let idx = current_len - 1 - i;
                        if !tree.is_deleted(idx) {
                            tree.delete(black_box(idx));
                        }
                    }
                    deleted_count += k;
                });
            });
        }

        // Benchmark deleting k RANDOM elements (baseline comparison)
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        for k in [1, 10, 50] {
            if k > n / 2 {
                continue;
            }
            let mut tree = MutableTree::new(log_weights.clone());
            let mut available: Vec<usize> = (0..n).collect();

            group.bench_function(BenchmarkId::new(format!("random_{k}"), n), |b| {
                b.iter(|| {
                    // Reset when we've used too many
                    if available.len() < k * 2 {
                        tree = MutableTree::new(log_weights.clone());
                        available = (0..n).collect();
                    }

                    // Delete k random elements
                    for _ in 0..k {
                        let pos = rng.gen_range(0..available.len());
                        let idx = available.swap_remove(pos);
                        tree.delete(black_box(idx));
                    }
                });
            });
        }
    }

    group.finish();
}

/// Benchmark bulk insert (extend) operations.
fn bench_bulk_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_insert");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let sizes = [100, 1000];
    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    for &n in &sizes {
        let weights = Distribution::Uniform.generate_weights(n);
        let log_weights = to_log_weights(&weights);

        // Benchmark inserting k elements at the end
        for k in [1, 10, 50] {
            let mut tree = MutableTree::new(log_weights.clone());

            group.bench_function(BenchmarkId::new(format!("extend_{k}"), n), |b| {
                b.iter(|| {
                    // Keep tree size bounded
                    if tree.len() > n * 2 {
                        tree = MutableTree::new(log_weights.clone());
                    }

                    // Insert k elements
                    for _ in 0..k {
                        let weight = rng.gen_range(-2.0..2.0);
                        tree.insert(black_box(weight));
                    }
                });
            });
        }
    }

    group.finish();
}

/// Benchmark pop operations (delete from end, one at a time).
fn bench_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    let sizes = [100, 1000];

    for &n in &sizes {
        let weights = Distribution::Uniform.generate_weights(n);
        let log_weights = to_log_weights(&weights);

        let mut tree = MutableTree::new(log_weights.clone());
        let mut active_count = n;

        group.bench_function(BenchmarkId::new("single", n), |b| {
            b.iter(|| {
                // Reset when we've popped too many
                if active_count < n / 2 {
                    tree = MutableTree::new(log_weights.clone());
                    active_count = n;
                }

                // Pop by deleting the last non-deleted element
                // Find last active element
                for i in (0..tree.len()).rev() {
                    if !tree.is_deleted(i) {
                        tree.delete(black_box(i));
                        active_count -= 1;
                        break;
                    }
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_single_sample,
    bench_batch_sample,
    bench_update,
    bench_insert_delete,
    bench_mixed_workflow,
    bench_bulk_delete,
    bench_bulk_insert,
    bench_pop,
);
criterion_main!(benches);
