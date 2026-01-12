//! Simple binary for profiling sampling performance.
//!
//! Run with: `cargo build --release --example profile_sampling`
//! Profile with: `samply record ./target/release/examples/profile_sampling`

use dynamic_random_sampler::core::{sample, Tree};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    // Create a tree with 1000 uniform weights
    let log_weights: Vec<f64> = vec![0.0; 1000];
    let tree = Tree::new(log_weights);
    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    // Sample 10 million times
    let iterations = 10_000_000;
    let mut sum: usize = 0;

    for _ in 0..iterations {
        if let Some(idx) = sample(&tree, &mut rng) {
            sum = sum.wrapping_add(idx);
        }
    }

    // Print to prevent dead code elimination
    println!("Sum: {sum}");
}
