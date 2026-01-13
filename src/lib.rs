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
    use pyo3::types::PySlice;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    use crate::core::{sample, MutableTree, DELETED_LOG_WEIGHT};

    /// A dynamic weighted random sampler that behaves like a Python list.
    ///
    /// Implements the data structure from "Dynamic Generation of Discrete Random Variates"
    /// by Matias, Vitter, and Ni (1993/2003).
    ///
    /// Internally uses soft-delete with stable indices for O(log* N) updates,
    /// but presents a list-like interface where deleted elements are hidden
    /// and indices are contiguous.
    #[pyclass]
    pub struct DynamicSampler {
        /// The mutable tree data structure (uses soft deletes internally)
        tree: MutableTree,
        /// Maps Python index -> internal tree index (only active elements)
        /// This is rebuilt when needed (after deletes change the mapping)
        index_map: Vec<usize>,
        /// Whether the index map needs rebuilding
        index_map_dirty: bool,
        /// Internal random number generator (`ChaCha8` for reproducibility)
        rng: ChaCha8Rng,
    }

    impl DynamicSampler {
        /// Rebuild the index map from the tree's current state.
        fn rebuild_index_map(&mut self) {
            self.index_map.clear();
            for i in 0..self.tree.len() {
                if !self.tree.is_deleted(i) {
                    self.index_map.push(i);
                }
            }
            self.index_map_dirty = false;
        }

        /// Get the index map, rebuilding if necessary.
        fn get_index_map(&mut self) -> &[usize] {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            &self.index_map
        }

        /// Map a Python index to an internal tree index.
        #[allow(clippy::cast_sign_loss)] // Intentional: we check the sign before casting
        fn map_index(&mut self, index: isize) -> PyResult<usize> {
            let map = self.get_index_map();
            let len = map.len();
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
            Ok(map[idx])
        }

        /// Get the weight at an internal tree index.
        fn get_weight_internal(&self, internal_idx: usize) -> f64 {
            self.tree
                .element_log_weight(internal_idx)
                .map_or(0.0, f64::exp2)
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
    impl DynamicSampler {
        /// Create a new sampler from a list of weights.
        ///
        /// Weights must be positive.
        ///
        /// # Arguments
        ///
        /// * `weights` - List of positive weights
        /// * `seed` - Optional seed for the random number generator. If None, uses entropy
        ///
        /// # Errors
        ///
        /// Returns error if weights is empty or contains non-positive values.
        #[new]
        #[pyo3(signature = (weights, seed=None))]
        #[allow(clippy::needless_pass_by_value)]
        pub fn new(weights: Vec<f64>, seed: Option<u64>) -> PyResult<Self> {
            if weights.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weights cannot be empty",
                ));
            }
            for &w in &weights {
                Self::validate_positive_weight(w)?;
            }
            let log_weights: Vec<f64> = weights.iter().map(|w| w.log2()).collect();
            let tree = MutableTree::new(log_weights);
            let index_map: Vec<usize> = (0..weights.len()).collect();
            let rng = seed.map_or_else(ChaCha8Rng::from_entropy, ChaCha8Rng::seed_from_u64);
            Ok(Self {
                tree,
                index_map,
                index_map_dirty: false,
                rng,
            })
        }

        /// Return the number of elements (excluding deleted elements via `del`).
        fn __len__(&mut self) -> usize {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            self.index_map.len()
        }

        /// Get the weight at the given index or slice.
        ///
        /// Supports negative indices and slices like Python lists.
        ///
        /// # Errors
        ///
        /// Returns error if index is out of bounds.
        #[allow(deprecated)]
        fn __getitem__(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            if let Ok(slice) = key.downcast::<PySlice>() {
                // Handle slice
                if self.index_map_dirty {
                    self.rebuild_index_map();
                }
                let len = self.index_map.len();
                let py_indices = slice_indices(slice, len)?;
                let weights: Vec<f64> = py_indices
                    .iter()
                    .map(|&i| {
                        let internal_idx = self.index_map[i];
                        self.get_weight_internal(internal_idx)
                    })
                    .collect();
                Ok(weights.into_pyobject(py)?.into_any().unbind())
            } else if let Ok(index) = key.extract::<isize>() {
                // Handle integer index
                let internal_idx = self.map_index(index)?;
                let weight = self.get_weight_internal(internal_idx);
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
        /// in the list (indices don't shift). Use `del` to actually remove.
        ///
        /// For slices, value must be an iterable of the same length as the slice.
        ///
        /// # Errors
        ///
        /// Returns error if weight is negative, infinite, NaN, or index is out of bounds.
        #[allow(deprecated)]
        fn __setitem__(
            &mut self,
            key: &Bound<'_, PyAny>,
            value: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            if let Ok(slice) = key.downcast::<PySlice>() {
                // Handle slice
                if self.index_map_dirty {
                    self.rebuild_index_map();
                }
                let len = self.index_map.len();
                let py_indices = slice_indices(slice, len)?;
                let weights: Vec<f64> = value.extract()?;
                if weights.len() != py_indices.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "attempt to assign sequence of size {} to slice of size {}",
                        weights.len(),
                        py_indices.len()
                    )));
                }
                for &w in &weights {
                    Self::validate_nonnegative_weight(w)?;
                }
                for (&py_idx, &weight) in py_indices.iter().zip(weights.iter()) {
                    let internal_idx = self.index_map[py_idx];
                    let log_weight = if weight == 0.0 {
                        f64::NEG_INFINITY
                    } else {
                        weight.log2()
                    };
                    self.tree.update(internal_idx, log_weight);
                }
                Ok(())
            } else if let Ok(index) = key.extract::<isize>() {
                // Handle integer index
                let weight: f64 = value.extract()?;
                Self::validate_nonnegative_weight(weight)?;
                let internal_idx = self.map_index(index)?;
                let log_weight = if weight == 0.0 {
                    f64::NEG_INFINITY // Tree uses NEG_INFINITY for zero weight
                } else {
                    weight.log2()
                };
                self.tree.update(internal_idx, log_weight);
                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "indices must be integers or slices",
                ))
            }
        }

        /// Delete the element at the given index or slice.
        ///
        /// Removes the element(s) and shifts subsequent indices down, like a Python list.
        /// This is different from setting weight to 0 (which keeps the element).
        ///
        /// Optimized for deleting from the end: if we're deleting a contiguous
        /// slice at the end, we truncate `index_map` directly instead of rebuilding.
        ///
        /// # Errors
        ///
        /// Returns error if index is out of bounds.
        #[allow(deprecated)]
        fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(slice) = key.downcast::<PySlice>() {
                // Handle slice - delete in reverse order to avoid index shifting issues
                if self.index_map_dirty {
                    self.rebuild_index_map();
                }
                let len = self.index_map.len();
                let mut py_indices = slice_indices(slice, len)?;

                if py_indices.is_empty() {
                    return Ok(());
                }

                // Sort in reverse order so we delete from the end first
                py_indices.sort_unstable_by(|a, b| b.cmp(a));

                // Check if we're deleting a contiguous slice from the end
                // This is O(1) and avoids the O(n) rebuild
                let is_contiguous_from_end = py_indices
                    .iter()
                    .enumerate()
                    .all(|(i, &idx)| idx == len - 1 - i);

                // Delete from tree
                for &py_idx in &py_indices {
                    let internal_idx = self.index_map[py_idx];
                    self.tree.delete(internal_idx);
                }

                if is_contiguous_from_end {
                    // Optimization: just truncate the index_map
                    self.index_map.truncate(len - py_indices.len());
                } else {
                    // General case: need full rebuild
                    self.index_map_dirty = true;
                }
                Ok(())
            } else if let Ok(index) = key.extract::<isize>() {
                // Handle integer index
                // map_index already rebuilds index_map if needed
                let internal_idx = self.map_index(index)?;
                let len = self.index_map.len();
                self.tree.delete(internal_idx);

                // Check if we're deleting the last element (optimization)
                #[allow(clippy::cast_sign_loss)]
                let py_idx = if index < 0 {
                    len - ((-index) as usize)
                } else {
                    index as usize
                };

                if py_idx == len - 1 {
                    // Deleting the last element - just pop from index_map
                    self.index_map.pop();
                } else {
                    // General case: need full rebuild
                    self.index_map_dirty = true;
                }
                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "indices must be integers or slices",
                ))
            }
        }

        /// Check if a weight value exists among active elements.
        ///
        /// Equivalent to `weight in sampler`. Checks only non-deleted elements.
        fn __contains__(&mut self, weight: f64) -> bool {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            let map = &self.index_map;
            map.iter().any(|&i| {
                let w = self.get_weight_internal(i);
                (w - weight).abs() < 1e-10
            })
        }

        /// Return an iterator over all weights (excluding deleted elements).
        fn __iter__(&mut self) -> PyWeightIterator {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            let weights: Vec<f64> = self
                .index_map
                .iter()
                .map(|&i| self.get_weight_internal(i))
                .collect();
            PyWeightIterator { weights, index: 0 }
        }

        // =====================================================================
        // Python list-like operations
        // =====================================================================

        /// Append a weight to the end.
        ///
        /// # Errors
        ///
        /// Returns error if weight is non-positive, infinite, or NaN.
        pub fn append(&mut self, weight: f64) -> PyResult<()> {
            Self::validate_positive_weight(weight)?;
            let log_weight = weight.log2();
            let new_idx = self.tree.insert(log_weight);
            // Add to index map (new element at end)
            if self.index_map_dirty {
                self.rebuild_index_map();
            } else {
                self.index_map.push(new_idx);
            }
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
                let log_weight = w.log2();
                let new_idx = self.tree.insert(log_weight);
                if !self.index_map_dirty {
                    self.index_map.push(new_idx);
                }
            }
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            Ok(())
        }

        /// Remove and return the last weight.
        ///
        /// # Errors
        ///
        /// Returns error if the sampler is empty.
        pub fn pop(&mut self) -> PyResult<f64> {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            // SAFETY: We return early if empty, so pop() is guaranteed to succeed
            let Some(internal_idx) = self.index_map.pop() else {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "pop from empty list",
                ));
            };
            let weight = self.get_weight_internal(internal_idx);
            self.tree.delete(internal_idx);
            Ok(weight)
        }

        /// Remove all elements.
        pub fn clear(&mut self) {
            // Delete all elements in the tree
            for i in 0..self.tree.len() {
                if !self.tree.is_deleted(i) {
                    self.tree.delete(i);
                }
            }
            self.index_map.clear();
            self.index_map_dirty = false;
        }

        /// Find the first index of an element with the given weight.
        ///
        /// # Errors
        ///
        /// Returns error if no element with this weight exists.
        pub fn index(&mut self, weight: f64) -> PyResult<usize> {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            for (py_idx, &internal_idx) in self.index_map.iter().enumerate() {
                let w = self.get_weight_internal(internal_idx);
                if (w - weight).abs() < 1e-10 {
                    return Ok(py_idx);
                }
            }
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{weight} is not in list"
            )))
        }

        /// Count the number of elements with the given weight.
        pub fn count(&mut self, weight: f64) -> usize {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            self.index_map
                .iter()
                .filter(|&&i| (self.get_weight_internal(i) - weight).abs() < 1e-10)
                .count()
        }

        /// Remove the first occurrence of a weight.
        ///
        /// # Errors
        ///
        /// Returns error if no element with this weight exists.
        pub fn remove(&mut self, weight: f64) -> PyResult<()> {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            // Find the python index of the weight
            let mut found_py_idx = None;
            for (py_idx, &internal_idx) in self.index_map.iter().enumerate() {
                let w = self.get_weight_internal(internal_idx);
                if (w - weight).abs() < 1e-10 {
                    found_py_idx = Some((py_idx, internal_idx));
                    break;
                }
            }
            let (_, internal_idx) = found_py_idx.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{weight} is not in list"))
            })?;
            self.tree.delete(internal_idx);
            self.index_map_dirty = true;
            Ok(())
        }

        /// Sample a random index according to the weight distribution.
        ///
        /// Returns a Python index j with probability `w_j / sum(w_i)`.
        /// Uses O(log* N) expected time.
        ///
        /// Elements with weight 0 are excluded from sampling.
        ///
        /// Uses the internal RNG. For reproducible results, create the sampler
        /// with a seed: `DynamicSampler(weights, seed=12345)`.
        ///
        /// # Errors
        ///
        /// Returns error if the sampler is empty or all elements have weight 0.
        pub fn sample(&mut self) -> PyResult<usize> {
            if self.index_map_dirty {
                self.rebuild_index_map();
            }
            if self.index_map.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot sample from empty list",
                ));
            }
            let tree = self.tree.as_tree();
            let internal_idx = sample(&tree, &mut self.rng).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot sample: all elements have weight 0",
                )
            })?;
            // Map internal index back to Python index
            self.index_map
                .iter()
                .position(|&i| i == internal_idx)
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "sampled a deleted element unexpectedly",
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

        /// Get the current state of the random number generator.
        ///
        /// Note: State persistence is not yet fully implemented. This returns an empty
        /// vector as a placeholder. For reproducibility, use construction-time seeding
        /// with the `seed` parameter.
        ///
        /// # Returns
        ///
        /// A bytes object (currently empty - placeholder for future implementation).
        #[must_use]
        #[allow(clippy::missing_const_for_fn)] // pymethod cannot be const
        pub fn getstate(&self) -> Vec<u8> {
            // Full state serialization requires exposing ChaCha8Rng internals
            // which is complex. For now, return empty and document the limitation.
            Vec::new()
        }

        /// Set the state of the random number generator.
        ///
        /// # Arguments
        ///
        /// * `state` - State bytes from a previous call to `getstate()`
        ///
        /// # Errors
        ///
        /// Returns error if the state is invalid.
        #[allow(clippy::needless_pass_by_value)]
        #[allow(clippy::unused_self)]
        pub fn setstate(&mut self, _state: Vec<u8>) -> PyResult<()> {
            // Placeholder - full state restoration requires more complex implementation
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "getstate/setstate not yet fully implemented for RNG state persistence",
            ))
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
            // We exclude low-expected elements from chi-squared but still allow samples.
            // Only truly zero-weight elements (deleted) should never be sampled.
            const MIN_EXPECTED_CHI2: f64 = 5.0; // Standard chi-squared assumption
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
                } else if w > 0.0 {
                    // Low but non-zero weight - exclude from chi-squared, sampling is valid
                    excluded_count += 1;
                } else {
                    // Zero weight (deleted element) - should never be sampled
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

    /// Iterator over weights in a `DynamicSampler`.
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

    /// Python module definition
    #[pymodule]
    #[allow(clippy::missing_errors_doc)]
    pub fn dynamic_random_sampler(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
        m.add_class::<DynamicSampler>()?;
        m.add_class::<PyChiSquaredResult>()?;
        m.add_class::<PyWeightIterator>()?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::*;
