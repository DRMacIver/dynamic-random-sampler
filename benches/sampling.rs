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
use rand::SeedableRng;
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

criterion_group!(
    benches,
    bench_construction,
    bench_single_sample,
    bench_batch_sample,
);
criterion_main!(benches);
