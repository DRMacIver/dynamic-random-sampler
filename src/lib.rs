//! Dynamic Random Sampler - Rust implementation
//!
//! This module implements the data structure from "Dynamic Generation of Discrete
//! Random Variates" by Matias, Vitter, and Ni (1993/2003).
//!
//! The implementation will be completed inside the devcontainer.

// Enable coverage attribute when running with coverage_nightly cfg
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![allow(clippy::redundant_pub_crate)]

pub mod core;

#[cfg(feature = "python")]
mod python_bindings {
    use pyo3::prelude::*;
    use pyo3::types::PySlice;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    use crate::core::{sample, MutableTree, DELETED_LOG_WEIGHT};

    /// A dynamic weighted random sampler that behaves like a Python list.
    ///
    /// Implements the data structure from "Dynamic Generation of Discrete Random Variates"
    /// by Matias, Vitter, and Ni (1993/2003).
    ///
    /// Uses stable indices - indices never shift. Elements can only be added
    /// at the end (append) or removed from the end (pop). Setting weight to 0
    /// excludes an element from sampling but keeps its index valid.
    #[pyclass]
    pub struct SamplerList {
        /// The mutable tree data structure
        tree: MutableTree,
        /// Current logical length (number of elements added minus popped)
        len: usize,
        /// Internal random number generator (`ChaCha8` for reproducibility)
        rng: ChaCha8Rng,
    }

    impl SamplerList {
        /// Get the weight at an internal tree index.
        fn get_weight_internal(&self, internal_idx: usize) -> f64 {
            self.tree
                .element_log_weight(internal_idx)
                .map_or(0.0, |lw| {
                    if lw == f64::NEG_INFINITY {
                        0.0
                    } else {
                        lw.exp2()
                    }
                })
        }

        /// Validate a weight value for construction/append (must be positive).
        fn validate_positive_weight(weight: f64) -> PyResult<()> {
            if weight <= 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be positive",
                ));
            }
            if !weight.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be finite (not infinity or NaN)",
                ));
            }
            Ok(())
        }

        /// Validate a weight value for update (can be zero for soft exclusion).
        fn validate_nonnegative_weight(weight: f64) -> PyResult<()> {
            if weight < 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be non-negative",
                ));
            }
            if !weight.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be finite (not infinity or NaN)",
                ));
            }
            Ok(())
        }

        /// Map a Python index (possibly negative) to a valid internal index.
        #[allow(clippy::cast_sign_loss)]
        fn map_index(&self, index: isize) -> PyResult<usize> {
            let len = self.len;
            let idx = if index < 0 {
                let positive = (-index) as usize;
                if positive > len {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        "index out of bounds",
                    ));
                }
                len - positive
            } else {
                index as usize
            };
            if idx >= len {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "index out of bounds",
                ));
            }
            Ok(idx)
        }
    }

    /// Get Python indices from a slice, given the current length.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn slice_indices(slice: &Bound<'_, PySlice>, len: usize) -> PyResult<Vec<usize>> {
        let indices = slice.indices(len as isize)?;
        let mut result = Vec::new();
        let mut i = indices.start;
        if indices.step > 0 {
            while i < indices.stop {
                result.push(i as usize);
                i += indices.step;
            }
        } else {
            while i > indices.stop {
                result.push(i as usize);
                i += indices.step;
            }
        }
        Ok(result)
    }

    #[pymethods]
    impl SamplerList {
        /// Create a new sampler from a list of weights.
        ///
        /// Weights must be positive (or empty to start with an empty sampler).
        ///
        /// # Arguments
        ///
        /// * `weights` - List of positive weights (can be empty)
        /// * `seed` - Optional seed for the random number generator. If None, uses entropy
        ///
        /// # Errors
        ///
        /// Returns error if weights contains non-positive values.
        #[new]
        #[pyo3(signature = (weights=None, *, seed=None))]
        #[allow(clippy::needless_pass_by_value)]
        pub fn new(weights: Option<Vec<f64>>, seed: Option<u64>) -> PyResult<Self> {
            let weights = weights.unwrap_or_default();
            for &w in &weights {
                Self::validate_positive_weight(w)?;
            }
            let len = weights.len();
            let log_weights: Vec<f64> = weights.iter().map(|w| w.log2()).collect();
            let tree = MutableTree::new(log_weights);
            let rng = seed.map_or_else(ChaCha8Rng::from_entropy, ChaCha8Rng::seed_from_u64);
            Ok(Self { tree, len, rng })
        }

        /// Return the number of elements.
        #[allow(clippy::missing_const_for_fn)] // pymethod cannot be const
        fn __len__(&self) -> usize {
            self.len
        }

        /// Get the weight at the given index or slice.
        ///
        /// Supports negative indices and slices like Python lists.
        ///
        /// # Errors
        ///
        /// Returns error if index is out of bounds.
        #[allow(deprecated)]
        fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            if let Ok(slice) = key.downcast::<PySlice>() {
                // Handle slice
                let py_indices = slice_indices(slice, self.len)?;
                let weights: Vec<f64> = py_indices
                    .iter()
                    .map(|&i| self.get_weight_internal(i))
                    .collect();
                Ok(weights.into_pyobject(py)?.into_any().unbind())
            } else if let Ok(index) = key.extract::<isize>() {
                // Handle integer index
                let idx = self.map_index(index)?;
                let weight = self.get_weight_internal(idx);
                Ok(weight.into_pyobject(py)?.into_any().unbind())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "indices must be integers or slices",
                ))
            }
        }

        /// Set the weight at the given index or slice.
        ///
        /// Setting weight to 0 excludes the element from sampling but keeps it
        /// in the list (indices stay stable).
        ///
        /// For slices, if the value has a different length than the slice,
        /// the list is resized accordingly (like Python lists).
        ///
        /// # Errors
        ///
        /// Returns error if weight is negative, infinite, NaN, or index is out of bounds.
        #[allow(deprecated)]
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
        fn __setitem__(
            &mut self,
            key: &Bound<'_, PyAny>,
            value: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            if let Ok(slice) = key.downcast::<PySlice>() {
                // Handle slice
                let new_weights: Vec<f64> = value.extract()?;
                for &w in &new_weights {
                    Self::validate_nonnegative_weight(w)?;
                }

                let indices = slice.indices(self.len as isize)?;
                let py_indices = slice_indices(slice, self.len)?;

                // For extended slices (step != 1), require exact length match (like Python)
                if indices.step != 1 {
                    if new_weights.len() != py_indices.len() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "attempt to assign sequence of size {} to extended slice of size {}",
                            new_weights.len(),
                            py_indices.len()
                        )));
                    }
                    // Same size, just update in place
                    for (&idx, &weight) in py_indices.iter().zip(new_weights.iter()) {
                        let log_weight = if weight == 0.0 {
                            f64::NEG_INFINITY
                        } else {
                            weight.log2()
                        };
                        self.tree.update(idx, log_weight);
                    }
                    return Ok(());
                }

                // For step=1 slices, support resizing (like Python)
                if new_weights.len() == py_indices.len() {
                    // Same size, just update in place (faster)
                    for (&idx, &weight) in py_indices.iter().zip(new_weights.iter()) {
                        let log_weight = if weight == 0.0 {
                            f64::NEG_INFINITY
                        } else {
                            weight.log2()
                        };
                        self.tree.update(idx, log_weight);
                    }
                } else {
                    // Different size, need to rebuild
                    let start = indices.start.max(0) as usize;
                    let stop = indices.stop.max(0) as usize;
                    let start = start.min(self.len);
                    let stop = stop.min(self.len);

                    // Get current weights
                    let mut current: Vec<f64> =
                        (0..self.len).map(|i| self.get_weight_internal(i)).collect();

                    // Replace the slice with new weights
                    let _removed: Vec<f64> = current.splice(start..stop, new_weights).collect();

                    // Rebuild from new weights
                    self.rebuild_from_weights(current);
                }
                Ok(())
            } else if let Ok(index) = key.extract::<isize>() {
                // Handle integer index
                let weight: f64 = value.extract()?;
                Self::validate_nonnegative_weight(weight)?;
                let idx = self.map_index(index)?;
                let log_weight = if weight == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    weight.log2()
                };
                self.tree.update(idx, log_weight);
                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "indices must be integers or slices",
                ))
            }
        }

        /// Check if a weight value exists among elements.
        fn __contains__(&self, weight: f64) -> bool {
            (0..self.len).any(|i| (self.get_weight_internal(i) - weight).abs() < 1e-10)
        }

        /// Return an iterator over all weights.
        fn __iter__(&self) -> PyWeightIterator {
            let weights: Vec<f64> = (0..self.len).map(|i| self.get_weight_internal(i)).collect();
            PyWeightIterator { weights, index: 0 }
        }

        // =====================================================================
        // Python list-like operations (stable indices only)
        // =====================================================================

        /// Append a weight to the end.
        ///
        /// If we previously popped elements, reuses the soft-deleted slot.
        /// Otherwise, inserts a new element.
        ///
        /// # Errors
        ///
        /// Returns error if weight is non-positive, infinite, or NaN.
        pub fn append(&mut self, weight: f64) -> PyResult<()> {
            Self::validate_positive_weight(weight)?;
            let log_weight = weight.log2();

            if self.len < self.tree.len() {
                // Reuse a popped slot - just update the weight
                self.tree.update(self.len, log_weight);
            } else {
                // Insert a new element
                self.tree.insert(log_weight);
            }
            self.len += 1;
            Ok(())
        }

        /// Extend the sampler with multiple weights.
        ///
        /// # Errors
        ///
        /// Returns error if any weight is non-positive, infinite, or NaN.
        #[allow(clippy::needless_pass_by_value)]
        pub fn extend(&mut self, weights: Vec<f64>) -> PyResult<()> {
            for &w in &weights {
                Self::validate_positive_weight(w)?;
            }
            for w in weights {
                self.append(w)?;
            }
            Ok(())
        }

        /// Remove and return the last weight.
        ///
        /// # Errors
        ///
        /// Returns error if the sampler is empty.
        pub fn pop(&mut self) -> PyResult<f64> {
            if self.len == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "pop from empty list",
                ));
            }
            let last_idx = self.len - 1;
            let weight = self.get_weight_internal(last_idx);
            self.tree.delete(last_idx);
            self.len -= 1;
            Ok(weight)
        }

        /// Find the first index of an element with the given weight.
        ///
        /// # Errors
        ///
        /// Returns error if no element with this weight exists.
        #[allow(clippy::missing_errors_doc)]
        pub fn index(&self, weight: f64) -> PyResult<usize> {
            for i in 0..self.len {
                let w = self.get_weight_internal(i);
                if (w - weight).abs() < 1e-10 {
                    return Ok(i);
                }
            }
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{weight} is not in list"
            )))
        }

        /// Count the number of elements with the given weight.
        #[must_use]
        pub fn count(&self, weight: f64) -> usize {
            (0..self.len)
                .filter(|&i| (self.get_weight_internal(i) - weight).abs() < 1e-10)
                .count()
        }

        /// Remove all elements.
        ///
        /// After calling `clear()`, the sampler will be empty (len = 0).
        pub fn clear(&mut self) {
            self.tree = MutableTree::new(vec![]);
            self.len = 0;
        }

        /// Sample a random index according to the weight distribution.
        ///
        /// Returns an index j with probability `w_j / sum(w_i)`.
        /// Uses O(log* N) expected time.
        ///
        /// Elements with weight 0 are excluded from sampling.
        ///
        /// Uses the internal RNG. For reproducible results, create the sampler
        /// with a seed: `SamplerList(weights, seed=12345)`.
        ///
        /// # Errors
        ///
        /// Returns error if the sampler is empty or all elements have weight 0.
        pub fn sample(&mut self) -> PyResult<usize> {
            if self.len == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot sample from empty list",
                ));
            }
            let tree = self.tree.as_tree();
            sample(&tree, &mut self.rng).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot sample: all elements have weight 0",
                )
            })
        }

        /// Reseed the internal random number generator.
        ///
        /// # Arguments
        ///
        /// * `seed` - New seed value for the RNG
        pub fn seed(&mut self, seed: u64) {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }

        // =====================================================================
        // List mutation methods (rebuild-based implementation)
        // =====================================================================

        /// Helper to rebuild the tree from current weights.
        fn rebuild_from_weights(&mut self, weights: Vec<f64>) {
            let log_weights: Vec<f64> = weights
                .iter()
                .map(|&w| if w == 0.0 { f64::NEG_INFINITY } else { w.log2() })
                .collect();
            self.tree = MutableTree::new(log_weights);
            self.len = weights.len();
        }

        /// Delete the element at the given index.
        ///
        /// This shifts all subsequent indices down by one.
        ///
        /// # Errors
        ///
        /// Returns error if index is out of bounds.
        fn __delitem__(&mut self, index: isize) -> PyResult<()> {
            let idx = self.map_index(index)?;
            let mut weights: Vec<f64> = (0..self.len).map(|i| self.get_weight_internal(i)).collect();
            weights.remove(idx);
            self.rebuild_from_weights(weights);
            Ok(())
        }

        /// Insert a weight at the given index.
        ///
        /// All elements at and after this index are shifted right.
        ///
        /// # Errors
        ///
        /// Returns error if weight is non-positive, infinite, or NaN.
        #[allow(clippy::cast_sign_loss)]
        pub fn insert(&mut self, index: isize, weight: f64) -> PyResult<()> {
            Self::validate_positive_weight(weight)?;
            // Allow inserting at len (append equivalent)
            let idx = if index < 0 {
                let positive = (-index) as usize;
                if positive > self.len {
                    0 // Clamp to beginning like Python
                } else {
                    self.len - positive
                }
            } else {
                let idx = index as usize;
                if idx > self.len {
                    self.len // Clamp to end like Python
                } else {
                    idx
                }
            };
            let mut weights: Vec<f64> = (0..self.len).map(|i| self.get_weight_internal(i)).collect();
            weights.insert(idx, weight);
            self.rebuild_from_weights(weights);
            Ok(())
        }

        /// Remove the first element with the given weight.
        ///
        /// # Errors
        ///
        /// Returns error if no element with this weight exists.
        pub fn remove(&mut self, weight: f64) -> PyResult<()> {
            let idx = self.index(weight)?;
            let mut weights: Vec<f64> = (0..self.len).map(|i| self.get_weight_internal(i)).collect();
            weights.remove(idx);
            self.rebuild_from_weights(weights);
            Ok(())
        }

        /// Reverse the order of elements in place.
        pub fn reverse(&mut self) {
            let mut weights: Vec<f64> = (0..self.len).map(|i| self.get_weight_internal(i)).collect();
            weights.reverse();
            self.rebuild_from_weights(weights);
        }

        /// Return a copy of the weights as a Python list.
        fn copy(&self) -> Vec<f64> {
            (0..self.len).map(|i| self.get_weight_internal(i)).collect()
        }

        /// Sort the weights in place.
        ///
        /// # Arguments
        ///
        /// * `reverse` - If True, sort in descending order (default: False)
        #[pyo3(signature = (*, reverse=false))]
        pub fn sort(&mut self, reverse: bool) {
            let mut weights: Vec<f64> = (0..self.len).map(|i| self.get_weight_internal(i)).collect();
            weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if reverse {
                weights.reverse();
            }
            self.rebuild_from_weights(weights);
        }

        // =====================================================================
        // Internal testing utilities (not part of public API)
        // =====================================================================

        /// Run a chi-squared goodness-of-fit test on this sampler.
        ///
        /// This is an internal testing method, not part of the public API.
        /// The leading underscore indicates it should not be relied upon.
        #[pyo3(name = "_test_distribution")]
        #[pyo3(signature = (num_samples=10000, seed=None))]
        #[allow(clippy::items_after_statements)]
        #[allow(clippy::too_many_lines)]
        pub fn test_distribution(
            &self,
            num_samples: usize,
            seed: Option<u64>,
        ) -> PyChiSquaredResult {
            use rand::prelude::*;
            use rand_chacha::ChaCha8Rng;

            let n = self.len;

            // Get log weights from tree
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
            const MIN_EXPECTED_CHI2: f64 = 5.0;
            #[allow(clippy::cast_precision_loss)]
            let num_samples_f64 = num_samples as f64;

            let mut included_observed = Vec::new();
            let mut included_weights = Vec::new();
            let mut excluded_count = 0usize;
            let mut unexpected_samples = 0usize;

            for (i, &w) in weights.iter().enumerate() {
                let expected = (w / total_weight) * num_samples_f64;
                if expected >= MIN_EXPECTED_CHI2 {
                    included_observed.push(observed[i]);
                    included_weights.push(w);
                } else if w > 0.0 {
                    excluded_count += 1;
                } else {
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

    /// Result of a chi-squared goodness-of-fit test (internal use only).
    /// Not exported in the public module API.
    #[pyclass(name = "_ChiSquaredResult")]
    #[derive(Clone)]
    pub struct PyChiSquaredResult {
        #[pyo3(get)]
        pub chi_squared: f64,
        #[pyo3(get)]
        pub degrees_of_freedom: usize,
        #[pyo3(get)]
        pub p_value: f64,
        #[pyo3(get)]
        pub num_samples: usize,
        #[pyo3(get)]
        pub excluded_count: usize,
        #[pyo3(get)]
        pub unexpected_samples: usize,
    }

    #[pymethods]
    impl PyChiSquaredResult {
        #[must_use]
        pub fn passes(&self, alpha: f64) -> bool {
            self.p_value > alpha
        }

        fn __repr__(&self) -> String {
            format!(
                "_ChiSquaredResult(chi_squared={:.4}, df={}, p_value={:.6}, n={}, excluded={}, unexpected={})",
                self.chi_squared, self.degrees_of_freedom, self.p_value, self.num_samples,
                self.excluded_count, self.unexpected_samples
            )
        }
    }

    /// Result of a likelihood-based statistical test (internal use only).
    #[pyclass(name = "_LikelihoodTestResult")]
    #[derive(Clone)]
    pub struct PyLikelihoodTestResult {
        #[pyo3(get)]
        pub observed_log_likelihood: f64,
        #[pyo3(get)]
        pub expected_log_likelihood: f64,
        #[pyo3(get)]
        pub variance: f64,
        #[pyo3(get)]
        pub z_score: f64,
        #[pyo3(get)]
        pub p_value: f64,
        #[pyo3(get)]
        pub num_samples: usize,
    }

    #[pymethods]
    impl PyLikelihoodTestResult {
        #[must_use]
        pub fn passes(&self, alpha: f64) -> bool {
            self.p_value > alpha
        }

        fn __repr__(&self) -> String {
            format!(
                "_LikelihoodTestResult(observed={:.4}, expected={:.4}, var={:.4}, z={:.4}, p={:.6}, n={})",
                self.observed_log_likelihood, self.expected_log_likelihood,
                self.variance, self.z_score, self.p_value, self.num_samples
            )
        }
    }

    /// Run a likelihood-based statistical test on a sampler.
    ///
    /// This is a standalone function (not a method) for testing the sampler's
    /// correctness with dynamic weight updates.
    ///
    /// # Arguments
    ///
    /// * `initial_weights` - Initial list of positive weights
    /// * `num_samples` - Number of samples to take (must be >= 100)
    /// * `assignments` - List of (sample_index, weight_index, new_weight) tuples
    /// * `seed` - Optional seed for reproducibility
    ///
    /// # Returns
    ///
    /// A `_LikelihoodTestResult` with the test statistics.
    #[pyfunction]
    #[pyo3(name = "_likelihood_test")]
    #[pyo3(signature = (initial_weights, num_samples, assignments, seed=None))]
    pub fn py_likelihood_test(
        initial_weights: Vec<f64>,
        num_samples: usize,
        assignments: Vec<(usize, usize, f64)>,
        seed: Option<u64>,
    ) -> PyResult<PyLikelihoodTestResult> {
        use crate::core::{likelihood_test, Assignment};

        // Validate inputs
        if num_samples < 100 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_samples must be at least 100",
            ));
        }
        if initial_weights.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "initial_weights cannot be empty",
            ));
        }
        if !initial_weights.iter().any(|&w| w > 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "at least one weight must be positive",
            ));
        }

        // Validate assignments
        for &(sample_idx, _, weight) in &assignments {
            if sample_idx >= num_samples {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("assignment sample_index {} is >= num_samples {}", sample_idx, num_samples),
                ));
            }
            if !weight.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "assignment weight must be finite",
                ));
            }
            if weight < 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "assignment weight must be non-negative",
                ));
            }
        }

        // Convert assignments
        let rust_assignments: Vec<Assignment> = assignments
            .into_iter()
            .map(|(sample_index, weight_index, new_weight)| Assignment {
                sample_index,
                weight_index,
                new_weight,
            })
            .collect();

        // Run the test
        let result = seed.map_or_else(
            || {
                let mut rng = rand::thread_rng();
                likelihood_test(&initial_weights, num_samples, &rust_assignments, &mut rng)
            },
            |s| {
                let mut rng = ChaCha8Rng::seed_from_u64(s);
                likelihood_test(&initial_weights, num_samples, &rust_assignments, &mut rng)
            },
        );

        match result {
            Ok(r) => Ok(PyLikelihoodTestResult {
                observed_log_likelihood: r.observed_log_likelihood,
                expected_log_likelihood: r.expected_log_likelihood,
                variance: r.variance,
                z_score: r.z_score,
                p_value: r.p_value,
                num_samples: r.num_samples,
            }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    /// Iterator over weights in a `SamplerList`.
    #[pyclass]
    pub struct PyWeightIterator {
        weights: Vec<f64>,
        index: usize,
    }

    #[pymethods]
    impl PyWeightIterator {
        #[allow(clippy::missing_const_for_fn)] // pymethod cannot be const
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&mut self) -> Option<f64> {
            if self.index >= self.weights.len() {
                return None;
            }
            let weight = self.weights[self.index];
            self.index += 1;
            Some(weight)
        }
    }

    // =========================================================================
    // SamplerDict - A dict-like type with weighted random sampling
    // =========================================================================

    use std::collections::HashMap;

    /// A dictionary-like type with weighted random sampling.
    ///
    /// Keys are arbitrary hashable Python objects (stored as String for simplicity).
    /// Values are non-negative floats representing weights.
    ///
    /// The `sample()` method returns a random key with probability proportional
    /// to its weight.
    ///
    /// # Implementation
    ///
    /// Uses a `Vec<K>` for keys, `HashMap<K, usize>` for key->index lookup,
    /// and `SamplerList` for weights. Deletion uses swap-remove to maintain
    /// O(log* N) sampling performance.
    #[pyclass]
    pub struct SamplerDict {
        /// Keys stored in order (with swap-remove on delete)
        keys: Vec<String>,
        /// Maps keys to their index in the vec
        key_to_index: HashMap<String, usize>,
        /// The tree stores weights at corresponding indices
        tree: MutableTree,
        /// Internal random number generator
        rng: ChaCha8Rng,
    }

    impl SamplerDict {
        /// Get the weight at an internal index.
        fn get_weight_internal(&self, internal_idx: usize) -> f64 {
            self.tree
                .element_log_weight(internal_idx)
                .map_or(0.0, |lw| {
                    if lw == f64::NEG_INFINITY {
                        0.0
                    } else {
                        lw.exp2()
                    }
                })
        }

        /// Validate a weight value (must be non-negative and finite).
        fn validate_weight(weight: f64) -> PyResult<()> {
            if weight < 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be non-negative",
                ));
            }
            if !weight.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be finite (not infinity or NaN)",
                ));
            }
            Ok(())
        }
    }

    #[pymethods]
    impl SamplerDict {
        /// Create a new `SamplerDict`, optionally initialized with weights.
        ///
        /// # Arguments
        ///
        /// * `weights` - Optional dictionary of key-weight pairs (can be empty or None)
        /// * `seed` - Optional seed for the random number generator
        ///
        /// # Errors
        ///
        /// Returns error if any weight is negative, infinite, or NaN.
        #[new]
        #[pyo3(signature = (weights=None, *, seed=None))]
        #[allow(clippy::needless_pass_by_value)]
        pub fn new(weights: Option<HashMap<String, f64>>, seed: Option<u64>) -> PyResult<Self> {
            let rng = seed.map_or_else(ChaCha8Rng::from_entropy, ChaCha8Rng::seed_from_u64);

            let weights = weights.unwrap_or_default();

            // Validate all weights first
            for &w in weights.values() {
                Self::validate_weight(w)?;
            }

            let mut keys = Vec::with_capacity(weights.len());
            let mut key_to_index = HashMap::with_capacity(weights.len());
            let mut log_weights = Vec::with_capacity(weights.len());

            for (key, weight) in weights {
                let idx = keys.len();
                keys.push(key.clone());
                key_to_index.insert(key, idx);
                let log_weight = if weight == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    weight.log2()
                };
                log_weights.push(log_weight);
            }

            let tree = MutableTree::new(log_weights);

            Ok(Self {
                keys,
                key_to_index,
                tree,
                rng,
            })
        }

        /// Return the number of keys.
        #[allow(clippy::missing_const_for_fn)]
        fn __len__(&self) -> usize {
            self.keys.len()
        }

        /// Get the weight for a key.
        ///
        /// # Errors
        ///
        /// Returns `KeyError` if the key is not present.
        fn __getitem__(&self, key: &str) -> PyResult<f64> {
            match self.key_to_index.get(key) {
                Some(&idx) => Ok(self.get_weight_internal(idx)),
                None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    key.to_string(),
                )),
            }
        }

        /// Set the weight for a key.
        ///
        /// If the key already exists, updates its weight.
        /// If the key is new, inserts it.
        ///
        /// Setting weight to 0 keeps the key present but excludes it from sampling.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if weight is negative, infinite, or NaN.
        fn __setitem__(&mut self, key: &str, weight: f64) -> PyResult<()> {
            Self::validate_weight(weight)?;
            let log_weight = if weight == 0.0 {
                f64::NEG_INFINITY
            } else {
                weight.log2()
            };

            if let Some(&idx) = self.key_to_index.get(key) {
                // Update existing key's weight
                self.tree.update(idx, log_weight);
            } else {
                // Insert new key
                let new_idx = self.keys.len();
                self.keys.push(key.to_string());
                self.key_to_index.insert(key.to_string(), new_idx);

                // Reuse deleted slot if available, otherwise insert new
                if new_idx < self.tree.len() {
                    // Reuse a deleted slot
                    self.tree.update(new_idx, log_weight);
                } else {
                    // Insert a new element at the end
                    self.tree.insert(log_weight);
                }
            }
            Ok(())
        }

        /// Delete a key from the dictionary.
        ///
        /// Uses swap-remove: the last key is moved to the deleted position.
        ///
        /// # Errors
        ///
        /// Returns `KeyError` if the key is not present.
        fn __delitem__(&mut self, key: &str) -> PyResult<()> {
            let Some(&idx) = self.key_to_index.get(key) else {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    key.to_string(),
                ));
            };

            let last_idx = self.keys.len() - 1;

            if idx == last_idx {
                // Deleting the last element - simple pop
                self.keys.pop();
                self.key_to_index.remove(key);
                self.tree.delete(idx);
            } else {
                // Swap-remove: move last key to deleted position
                let last_key = self.keys.pop().unwrap();

                // Update the swapped key's index in the hashmap
                self.key_to_index.remove(key);
                *self.key_to_index.get_mut(&last_key).unwrap() = idx;

                // Put the last key in the deleted position
                self.keys[idx] = last_key;

                // Update tree: copy last weight to deleted position, then delete last
                let last_weight_log = self
                    .tree
                    .element_log_weight(last_idx)
                    .unwrap_or(f64::NEG_INFINITY);
                self.tree.update(idx, last_weight_log);
                self.tree.delete(last_idx);
            }
            Ok(())
        }

        /// Check if a key exists in the dictionary.
        fn __contains__(&self, key: &str) -> bool {
            self.key_to_index.contains_key(key)
        }

        /// Return an iterator over keys.
        fn __iter__(&self) -> PyKeyIterator {
            PyKeyIterator {
                keys: self.keys.clone(),
                index: 0,
            }
        }

        /// Return a list of all keys.
        fn keys(&self) -> Vec<String> {
            self.keys.clone()
        }

        /// Return a list of all weights (values).
        fn values(&self) -> Vec<f64> {
            (0..self.keys.len())
                .map(|i| self.get_weight_internal(i))
                .collect()
        }

        /// Return a list of (key, weight) tuples.
        fn items(&self) -> Vec<(String, f64)> {
            self.keys
                .iter()
                .enumerate()
                .map(|(i, k)| (k.clone(), self.get_weight_internal(i)))
                .collect()
        }

        /// Get the weight for a key, or a default value if not present.
        ///
        /// # Arguments
        ///
        /// * `key` - The key to look up
        /// * `default` - Value to return if key is not present (default: None)
        #[pyo3(signature = (key, default=None))]
        fn get(&self, key: &str, default: Option<f64>) -> Option<f64> {
            self.key_to_index
                .get(key)
                .map(|&idx| self.get_weight_internal(idx))
                .or(default)
        }

        /// Remove and return the weight for a key.
        ///
        /// # Errors
        ///
        /// Returns `KeyError` if the key is not present.
        fn pop(&mut self, key: &str) -> PyResult<f64> {
            let weight = self.__getitem__(key)?;
            self.__delitem__(key)?;
            Ok(weight)
        }

        /// Update the dictionary with key-weight pairs from another dict.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if any weight is invalid.
        #[allow(clippy::needless_pass_by_value)]
        fn update(&mut self, other: HashMap<String, f64>) -> PyResult<()> {
            for (key, weight) in other {
                self.__setitem__(&key, weight)?;
            }
            Ok(())
        }

        /// Remove all keys from the dictionary.
        fn clear(&mut self) {
            self.keys.clear();
            self.key_to_index.clear();
            self.tree = MutableTree::new(vec![]);
        }

        /// Set a key's weight if not already present.
        ///
        /// Returns the weight for the key (new or existing).
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the weight is invalid.
        fn setdefault(&mut self, key: &str, default: f64) -> PyResult<f64> {
            if let Some(&idx) = self.key_to_index.get(key) {
                Ok(self.get_weight_internal(idx))
            } else {
                self.__setitem__(key, default)?;
                Ok(default)
            }
        }

        /// Sample a random key according to the weight distribution.
        ///
        /// Returns a key with probability proportional to its weight.
        /// Keys with weight 0 are excluded from sampling.
        ///
        /// # Errors
        ///
        /// Returns error if the dictionary is empty or all weights are 0.
        pub fn sample(&mut self) -> PyResult<String> {
            if self.keys.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot sample from empty dict",
                ));
            }
            let tree = self.tree.as_tree();
            match sample(&tree, &mut self.rng) {
                Some(idx) => Ok(self.keys[idx].clone()),
                None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot sample: all weights are 0",
                )),
            }
        }

        /// Reseed the internal random number generator.
        pub fn seed(&mut self, seed: u64) {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }

        /// Return a string representation.
        fn __repr__(&self) -> String {
            let items: Vec<String> = self
                .keys
                .iter()
                .enumerate()
                .map(|(i, k)| format!("{:?}: {}", k, self.get_weight_internal(i)))
                .collect();
            format!("SamplerDict({{{}}})", items.join(", "))
        }
    }

    /// Iterator over keys in a `SamplerDict`.
    #[pyclass]
    pub struct PyKeyIterator {
        keys: Vec<String>,
        index: usize,
    }

    #[pymethods]
    impl PyKeyIterator {
        #[allow(clippy::missing_const_for_fn)]
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&mut self) -> Option<String> {
            if self.index >= self.keys.len() {
                return None;
            }
            let key = self.keys[self.index].clone();
            self.index += 1;
            Some(key)
        }
    }

    /// Python module definition
    #[pymodule]
    #[allow(clippy::missing_errors_doc)]
    pub fn dynamic_random_sampler(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
        m.add_class::<SamplerList>()?;
        m.add_class::<PyWeightIterator>()?;
        m.add_class::<SamplerDict>()?;
        m.add_class::<PyKeyIterator>()?;
        m.add_class::<PyLikelihoodTestResult>()?;
        m.add_function(wrap_pyfunction!(py_likelihood_test, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::*;
