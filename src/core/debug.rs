//! Debug utilities for the dynamic random sampler.
//!
//! This module provides debugging tools that are enabled with the `debug-timeout` feature:
//! - Operation timeout checking (panics if operations exceed 1 second)
//! - Iteration counters for detecting infinite loops
//! - Detailed assertions about internal state

use std::time::Duration;
#[cfg(feature = "debug-timeout")]
use std::time::Instant;

/// Default timeout duration for operations (1 second).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_millis(100);

/// Maximum iterations allowed in rejection sampling before we consider it stuck.
pub const MAX_REJECTION_ITERATIONS: usize = 1_000_000;

/// A guard that tracks operation duration and panics if it exceeds the timeout.
///
/// Only active when the `debug-timeout` feature is enabled.
#[cfg(feature = "debug-timeout")]
pub struct TimeoutGuard {
    start: Instant,
    timeout: Duration,
    operation: &'static str,
}

#[cfg(feature = "debug-timeout")]
impl TimeoutGuard {
    /// Create a new timeout guard for an operation.
    #[must_use]
    pub fn new(operation: &'static str) -> Self {
        Self {
            start: Instant::now(),
            timeout: DEFAULT_TIMEOUT,
            operation,
        }
    }

    /// Check if we've exceeded the timeout and panic if so.
    ///
    /// Call this periodically in long-running operations.
    pub fn check(&self) {
        let elapsed = self.start.elapsed();
        let operation = self.operation;
        let timeout = self.timeout;
        assert!(
            elapsed <= self.timeout,
            "Operation '{operation}' exceeded timeout of {timeout:?} (elapsed: {elapsed:?})",
        );
    }

    /// Check and also report progress for debugging.
    pub fn check_with_context(&self, context: &str) {
        let elapsed = self.start.elapsed();
        let operation = self.operation;
        let timeout = self.timeout;
        assert!(
            elapsed <= self.timeout,
            "Operation '{operation}' exceeded timeout of {timeout:?} (elapsed: {elapsed:?}). Context: {context}",
        );
    }
}

#[cfg(feature = "debug-timeout")]
impl Drop for TimeoutGuard {
    fn drop(&mut self) {
        // Log completion time if it took significant time
        let elapsed = self.start.elapsed();
        if elapsed > Duration::from_millis(100) {
            eprintln!(
                "[DEBUG] Operation '{}' completed in {:?}",
                self.operation, elapsed
            );
        }
    }
}

/// No-op timeout guard when debug-timeout feature is disabled.
#[cfg(not(feature = "debug-timeout"))]
pub struct TimeoutGuard;

#[cfg(not(feature = "debug-timeout"))]
impl TimeoutGuard {
    #[inline]
    #[must_use]
    pub const fn new(_operation: &'static str) -> Self {
        Self
    }

    #[inline]
    pub const fn check(&self) {}

    #[inline]
    pub const fn check_with_context(&self, _context: &str) {}
}

/// Assert with detailed context, only in debug-timeout mode.
#[cfg(feature = "debug-timeout")]
#[macro_export]
macro_rules! debug_assert_timeout {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            panic!("Debug assertion failed: {}", format!($($arg)*));
        }
    };
}

#[cfg(not(feature = "debug-timeout"))]
#[macro_export]
macro_rules! debug_assert_timeout {
    ($cond:expr, $($arg:tt)*) => {};
}

/// Track iteration count and panic if it exceeds the maximum.
///
/// In release mode (without debug-timeout), this only checks every 1024 iterations
/// to minimize overhead while still catching infinite loops.
#[cfg(feature = "debug-timeout")]
pub struct IterationCounter {
    count: usize,
    max: usize,
    operation: &'static str,
    guard: TimeoutGuard,
}

#[cfg(feature = "debug-timeout")]
impl IterationCounter {
    /// Create a new iteration counter.
    #[must_use]
    pub fn new(operation: &'static str, max: usize) -> Self {
        Self {
            count: 0,
            max,
            operation,
            guard: TimeoutGuard::new(operation),
        }
    }

    /// Increment the counter and check limits.
    ///
    /// Panics if:
    /// - Iteration count exceeds max (always)
    /// - Time exceeds timeout (only with debug-timeout feature)
    pub fn tick(&mut self) {
        self.count += 1;

        // Always check iteration count
        let count = self.count;
        let max = self.max;
        let operation = self.operation;
        assert!(
            count <= max,
            "Operation '{operation}' exceeded maximum iterations ({max}) - likely infinite loop",
        );

        // Check timeout periodically (every 1000 iterations)
        if count.is_multiple_of(1000) {
            self.guard.check_with_context(&format!("iteration {count}"));
        }
    }

    /// Get the current iteration count.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
}

/// Lightweight iteration counter for release mode.
///
/// Only checks every 1024 iterations to minimize overhead.
#[cfg(not(feature = "debug-timeout"))]
pub struct IterationCounter {
    count: usize,
    max: usize,
    operation: &'static str,
}

#[cfg(not(feature = "debug-timeout"))]
impl IterationCounter {
    /// Create a new iteration counter.
    #[inline]
    #[must_use]
    pub const fn new(operation: &'static str, max: usize) -> Self {
        Self {
            count: 0,
            max,
            operation,
        }
    }

    /// Increment the counter and check limits periodically.
    ///
    /// Only checks every 1024 iterations to minimize overhead.
    #[inline]
    pub fn tick(&mut self) {
        self.count += 1;

        // Only check every 1024 iterations for minimal overhead (1024 = 2^10, cheap bitwise)
        if self.count & 0x3FF == 0 {
            let count = self.count;
            let max = self.max;
            let operation = self.operation;
            assert!(
                count <= max,
                "Operation '{operation}' exceeded maximum iterations ({max}) - likely infinite loop",
            );
        }
    }

    /// Get the current iteration count.
    #[inline]
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
}

/// Debug helper to dump range state.
#[cfg(feature = "debug-timeout")]
pub fn dump_range_state(range: &crate::core::Range) {
    let range_num = range.range_number();
    let degree = range.degree();
    let is_empty = range.is_empty();
    let total_lw = range.compute_total_log_weight();
    let upper_bound = 2.0_f64.powi(range_num);
    eprintln!("[DEBUG] Range {range_num} state:");
    eprintln!("  - degree: {degree}");
    eprintln!("  - is_empty: {is_empty}");
    eprintln!("  - total_log_weight: {total_lw}");
    eprintln!("  - upper_bound (2^j): {upper_bound}");
    for (idx, log_weight) in range.children() {
        let weight = log_weight.exp2();
        let accept_prob = weight / upper_bound;
        eprintln!(
            "    child {idx}: log_weight={log_weight:.4}, weight={weight:.6}, accept_prob={accept_prob:.6}",
        );
    }
}

#[cfg(not(feature = "debug-timeout"))]
pub fn dump_range_state(_range: &crate::core::Range) {}

/// Debug helper to dump level state.
#[cfg(feature = "debug-timeout")]
pub fn dump_level_state(level: &crate::core::Level) {
    eprintln!("[DEBUG] Level {} state:", level.level_number());
    eprintln!("  - root_count: {}", level.root_count());
    eprintln!(
        "  - root_total_log_weight: {}",
        level.compute_root_total_log_weight()
    );
    for (j, range) in level.root_ranges() {
        eprintln!("  - root range {}: degree={}", j, range.degree());
    }
}

#[cfg(not(feature = "debug-timeout"))]
pub fn dump_level_state(_level: &crate::core::Level) {}

/// Debug helper to dump tree state.
#[cfg(feature = "debug-timeout")]
pub fn dump_tree_state(tree: &crate::core::Tree) {
    eprintln!("[DEBUG] Tree state:");
    eprintln!("  - len: {}", tree.len());
    eprintln!("  - max_level: {}", tree.max_level());
    eprintln!("  - level_count: {}", tree.level_count());
    for level_num in 1..=tree.max_level() {
        if let Some(level) = tree.get_level(level_num) {
            eprintln!(
                "  - level {}: {} roots, total_weight={}",
                level_num,
                level.root_count(),
                level.compute_root_total_log_weight().exp2()
            );
        }
    }
}

#[cfg(not(feature = "debug-timeout"))]
pub fn dump_tree_state(_tree: &crate::core::Tree) {}
