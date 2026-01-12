//! Dynamic Random Sampler - Rust implementation
//!
//! This module implements the data structure from "Dynamic Generation of Discrete
//! Random Variates" by Matias, Vitter, and Ni (1993/2003).
//!
//! The implementation will be completed inside the devcontainer.

#![allow(clippy::redundant_pub_crate)]

pub mod core;

#[cfg(feature = "python")]
mod python_bindings {
    use pyo3::prelude::*;

    use crate::core::{sample, MutableTree, DELETED_LOG_WEIGHT};

    /// A dynamic weighted random sampler with O(log* N) operations.
    ///
    /// Implements the data structure from "Dynamic Generation of Discrete Random Variates"
    /// by Matias, Vitter, and Ni (1993/2003).
    ///
    /// Supports efficient sampling from a discrete probability distribution
    /// with dynamically changing weights.
    #[pyclass]
    pub struct DynamicSampler {
        /// The mutable tree data structure from the paper
        tree: MutableTree,
    }

    #[pymethods]
    impl DynamicSampler {
        /// Create a new sampler from a list of weights.
        ///
        /// Weights must be positive. They are converted to log space internally.
        ///
        /// # Errors
        ///
        /// Returns error if weights is empty or contains non-positive values.
        #[new]
        #[allow(clippy::needless_pass_by_value)]
        pub fn new(weights: Vec<f64>) -> PyResult<Self> {
            if weights.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weights cannot be empty",
                ));
            }
            if weights.iter().any(|&w| w <= 0.0) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "all weights must be positive",
                ));
            }
            let log_weights: Vec<f64> = weights.iter().map(|w| w.log2()).collect();
            // Use basic config for correct sampling distribution with any tree size.
            // The optimized config (min_degree=32) requires large trees for correct
            // multi-level structure; basic config (min_degree=2) works for all sizes.
            let tree = MutableTree::new(log_weights);
            Ok(Self { tree })
        }

        /// Return the number of elements (including deleted).
        #[allow(clippy::missing_const_for_fn)] // pymethod cannot be const
        fn __len__(&self) -> usize {
            self.tree.len()
        }

        /// Get the weight of element at index (in original space, not log space).
        ///
        /// Returns 0.0 for deleted elements.
        ///
        /// # Errors
        ///
        /// Returns error if index is out of bounds.
        pub fn weight(&self, index: usize) -> PyResult<f64> {
            self.tree
                .element_log_weight(index)
                .map(|lw| {
                    if lw == DELETED_LOG_WEIGHT {
                        0.0
                    } else {
                        lw.exp2()
                    }
                })
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyIndexError, _>("index out of bounds")
                })
        }

        /// Update the weight of element at index.
        ///
        /// Setting weight to 0 is equivalent to calling `delete(index)`.
        /// Updating a deleted element to a positive weight "undeletes" it.
        ///
        /// # Errors
        ///
        /// Returns error if weight is negative or index is out of bounds.
        pub fn update(&mut self, index: usize, weight: f64) -> PyResult<()> {
            if weight < 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be non-negative",
                ));
            }
            if index >= self.tree.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "index out of bounds",
                ));
            }
            let log_weight = if weight == 0.0 {
                DELETED_LOG_WEIGHT
            } else {
                weight.log2()
            };
            self.tree.update(index, log_weight);
            Ok(())
        }

        /// Insert a new element with the given weight.
        ///
        /// The new element is appended and gets the next available index.
        ///
        /// # Arguments
        ///
        /// * `weight` - The weight of the new element (must be positive)
        ///
        /// # Returns
        ///
        /// The index of the newly inserted element.
        ///
        /// # Errors
        ///
        /// Returns error if weight is non-positive.
        pub fn insert(&mut self, weight: f64) -> PyResult<usize> {
            if weight <= 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be positive",
                ));
            }
            let log_weight = weight.log2();
            Ok(self.tree.insert(log_weight))
        }

        /// Soft-delete an element by setting its weight to zero.
        ///
        /// The element remains in the data structure but will never be sampled.
        /// Its index remains valid and stable. Use `update(index, weight)` with
        /// a positive weight to "undelete" the element.
        ///
        /// # Errors
        ///
        /// Returns error if index is out of bounds.
        pub fn delete(&mut self, index: usize) -> PyResult<()> {
            if index >= self.tree.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "index out of bounds",
                ));
            }
            self.tree.delete(index);
            Ok(())
        }

        /// Check if an element has been deleted.
        ///
        /// # Errors
        ///
        /// Returns error if index is out of bounds.
        pub fn is_deleted(&self, index: usize) -> PyResult<bool> {
            if index >= self.tree.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "index out of bounds",
                ));
            }
            Ok(self.tree.is_deleted(index))
        }

        /// Get the number of active (non-deleted) elements.
        #[must_use]
        pub fn active_count(&self) -> usize {
            self.tree.active_count()
        }

        /// Sample a random index according to the weight distribution.
        ///
        /// Returns index j with probability `w_j / sum(w_i)`.
        /// Deleted elements (weight 0) are never returned.
        /// Uses O(log* N) expected time.
        ///
        /// # Errors
        ///
        /// Returns error if all elements are deleted (nothing to sample).
        pub fn sample(&self) -> PyResult<usize> {
            let tree = self.tree.as_tree();
            let mut rng = rand::thread_rng();
            sample(&tree, &mut rng).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "no active elements to sample (all deleted)",
                )
            })
        }

        /// Run a chi-squared goodness-of-fit test on this sampler.
        ///
        /// Takes `num_samples` samples and tests whether the observed distribution
        /// matches the expected distribution based on weights.
        ///
        /// # Arguments
        ///
        /// * `num_samples` - Number of samples to take (default: 10000)
        /// * `seed` - Optional random seed for reproducibility (default: None)
        ///
        /// # Returns
        ///
        /// A `PyChiSquaredResult` containing the test statistics.
        #[pyo3(signature = (num_samples=10000, seed=None))]
        #[allow(clippy::items_after_statements)] // Helper function and constants placed near usage
        #[allow(clippy::too_many_lines)] // Logic is cohesive, splitting would reduce clarity
        pub fn test_distribution(
            &self,
            num_samples: usize,
            seed: Option<u64>,
        ) -> PyChiSquaredResult {
            use rand::prelude::*;
            use rand_chacha::ChaCha8Rng;

            // Get log weights from tree, filtering out deleted elements
            let n = self.tree.len();
            let log_weights: Vec<f64> = (0..n)
                .map(|i| {
                    self.tree
                        .element_log_weight(i)
                        .unwrap_or(DELETED_LOG_WEIGHT)
                })
                .collect();

            // Convert log weights to regular weights (for expected probability calculation)
            let max_log = log_weights
                .iter()
                .copied()
                .filter(|&lw| lw != DELETED_LOG_WEIGHT)
                .fold(f64::NEG_INFINITY, f64::max);

            // If all deleted, return early
            if max_log == f64::NEG_INFINITY {
                return PyChiSquaredResult {
                    chi_squared: 0.0,
                    degrees_of_freedom: 0,
                    p_value: 1.0,
                    num_samples: 0,
                    excluded_count: n,
                    unexpected_samples: 0,
                };
            }

            let weights: Vec<f64> = log_weights
                .iter()
                .map(|&lw| {
                    if lw == DELETED_LOG_WEIGHT {
                        0.0
                    } else {
                        (lw - max_log).exp2()
                    }
                })
                .collect();
            let total_weight: f64 = weights.iter().sum();

            // Helper to do sampling with a given RNG using the actual tree-based algorithm
            fn do_sampling<R: Rng>(
                tree: &crate::core::Tree,
                rng: &mut R,
                num_samples: usize,
                n: usize,
            ) -> Vec<usize> {
                let mut observed = vec![0usize; n];
                for _ in 0..num_samples {
                    if let Some(idx) = sample(tree, rng) {
                        if idx < n {
                            observed[idx] += 1;
                        }
                    }
                }
                observed
            }

            // Count observed occurrences by sampling using the tree-based algorithm
            let tree = self.tree.as_tree();
            let observed = seed.map_or_else(
                || {
                    let mut rng = rand::thread_rng();
                    do_sampling(&tree, &mut rng, num_samples, n)
                },
                |s| {
                    let mut rng = ChaCha8Rng::seed_from_u64(s);
                    do_sampling(&tree, &mut rng, num_samples, n)
                },
            );

            // Calculate expected counts and identify excluded indices
            // Chi-squared test requires expected counts >= ~5 for validity
            // We use two thresholds:
            // - MIN_EXPECTED_CHI2: exclude from chi-squared test (low but possible)
            // - MIN_EXPECTED_FAIL: if we see samples here, it's a definite failure
            const MIN_EXPECTED_CHI2: f64 = 5.0; // Standard chi-squared assumption
            const MIN_EXPECTED_FAIL: f64 = 0.001; // Truly impossible samples
            #[allow(clippy::cast_precision_loss)] // Acceptable for statistical calculations
            let num_samples_f64 = num_samples as f64;

            let mut included_observed = Vec::new();
            let mut included_weights = Vec::new();
            let mut excluded_count = 0usize;
            let mut unexpected_samples = 0usize;

            for (i, &w) in weights.iter().enumerate() {
                let expected = (w / total_weight) * num_samples_f64;
                if expected >= MIN_EXPECTED_CHI2 {
                    // Include in chi-squared test
                    included_observed.push(observed[i]);
                    included_weights.push(w);
                } else if expected >= MIN_EXPECTED_FAIL {
                    // Low but possible - exclude from chi-squared but don't fail if sampled
                    excluded_count += 1;
                } else {
                    // Effectively zero - should never be sampled
                    excluded_count += 1;
                    if observed[i] > 0 {
                        unexpected_samples += observed[i];
                    }
                }
            }

            // If we have unexpected samples in "impossible" indices, fail immediately
            if unexpected_samples > 0 {
                return PyChiSquaredResult {
                    chi_squared: f64::INFINITY,
                    degrees_of_freedom: 0,
                    p_value: 0.0,
                    num_samples,
                    excluded_count,
                    unexpected_samples,
                };
            }

            // If no indices are included (all weights too small), skip chi-squared
            if included_observed.is_empty() {
                return PyChiSquaredResult {
                    chi_squared: 0.0,
                    degrees_of_freedom: 0,
                    p_value: 1.0,
                    num_samples,
                    excluded_count,
                    unexpected_samples: 0,
                };
            }

            // Recalculate total samples for included indices
            let included_total: usize = included_observed.iter().sum();

            // Run chi-squared only on included indices
            let result = crate::core::chi_squared_from_counts(
                &included_observed,
                &included_weights,
                included_total,
            );

            PyChiSquaredResult {
                chi_squared: result.chi_squared,
                degrees_of_freedom: result.degrees_of_freedom,
                p_value: result.p_value,
                num_samples: result.num_samples,
                excluded_count,
                unexpected_samples: 0,
            }
        }
    }

    /// Result of a chi-squared goodness-of-fit test (Python wrapper).
    #[pyclass(name = "ChiSquaredResult")]
    #[derive(Clone)]
    pub struct PyChiSquaredResult {
        /// The chi-squared statistic.
        #[pyo3(get)]
        pub chi_squared: f64,
        /// Degrees of freedom (number of categories - 1).
        #[pyo3(get)]
        pub degrees_of_freedom: usize,
        /// The p-value (probability of observing this or more extreme result).
        #[pyo3(get)]
        pub p_value: f64,
        /// Number of samples taken.
        #[pyo3(get)]
        pub num_samples: usize,
        /// Number of indices excluded from chi-squared (expected < threshold).
        #[pyo3(get)]
        pub excluded_count: usize,
        /// Number of unexpected samples in excluded indices.
        #[pyo3(get)]
        pub unexpected_samples: usize,
    }

    #[pymethods]
    impl PyChiSquaredResult {
        /// Returns true if the test passes at the given significance level.
        ///
        /// A test "passes" if the p-value is greater than alpha, meaning we cannot
        /// reject the null hypothesis that the observed distribution matches expected.
        #[must_use]
        pub fn passes(&self, alpha: f64) -> bool {
            self.p_value > alpha
        }

        fn __repr__(&self) -> String {
            format!(
                "ChiSquaredResult(chi_squared={:.4}, df={}, p_value={:.6}, n={}, excluded={}, unexpected={})",
                self.chi_squared, self.degrees_of_freedom, self.p_value, self.num_samples,
                self.excluded_count, self.unexpected_samples
            )
        }
    }

    /// Python module definition
    #[pymodule]
    #[allow(clippy::missing_errors_doc)]
    pub fn dynamic_random_sampler(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
        m.add_class::<DynamicSampler>()?;
        m.add_class::<PyChiSquaredResult>()?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::*;
