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

    /// A dynamic weighted random sampler with O(log* N) operations.
    ///
    /// Supports efficient sampling from a discrete probability distribution
    /// with dynamically changing weights.
    #[pyclass]
    pub struct DynamicSampler {
        /// Weights stored in log space (log2 of actual weight)
        log_weights: Vec<f64>,
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
            let log_weights = weights.iter().map(|w| w.log2()).collect();
            Ok(Self { log_weights })
        }

        /// Return the number of elements.
        #[allow(clippy::missing_const_for_fn)]
        fn __len__(&self) -> usize {
            self.log_weights.len()
        }

        /// Get the weight of element at index (in original space, not log space).
        ///
        /// # Errors
        ///
        /// Returns error if index is out of bounds.
        pub fn weight(&self, index: usize) -> PyResult<f64> {
            self.log_weights
                .get(index)
                .map(|&lw| lw.exp2())
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyIndexError, _>("index out of bounds")
                })
        }

        /// Update the weight of element at index.
        ///
        /// # Errors
        ///
        /// Returns error if weight is non-positive or index is out of bounds.
        pub fn update(&mut self, index: usize, weight: f64) -> PyResult<()> {
            if weight <= 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weight must be positive",
                ));
            }
            if index >= self.log_weights.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "index out of bounds",
                ));
            }
            self.log_weights[index] = weight.log2();
            Ok(())
        }

        /// Sample a random index according to the weight distribution.
        ///
        /// Returns index j with probability `w_j / sum(w_i)`.
        ///
        /// # Errors
        ///
        /// Returns error if sampling fails (should not happen in practice).
        pub fn sample(&self) -> PyResult<usize> {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Convert to probability space for sampling
            let max_log = self
                .log_weights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = self
                .log_weights
                .iter()
                .map(|&lw| (lw - max_log).exp2())
                .collect();
            let total: f64 = weights.iter().sum();

            let mut u: f64 = rng.gen::<f64>() * total;
            for (i, &w) in weights.iter().enumerate() {
                u -= w;
                if u <= 0.0 {
                    return Ok(i);
                }
            }
            Ok(self.log_weights.len() - 1)
        }

        /// Run a chi-squared goodness-of-fit test on this sampler.
        ///
        /// Takes `num_samples` samples and tests whether the observed distribution
        /// matches the expected distribution based on weights.
        ///
        /// # Arguments
        ///
        /// * `num_samples` - Number of samples to take (default: 10000)
        ///
        /// # Returns
        ///
        /// A `PyChiSquaredResult` containing the test statistics.
        #[pyo3(signature = (num_samples=10000))]
        pub fn test_distribution(&self, num_samples: usize) -> PyChiSquaredResult {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            let n = self.log_weights.len();

            // Convert log weights to regular weights
            let max_log = self
                .log_weights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = self
                .log_weights
                .iter()
                .map(|&lw| (lw - max_log).exp2())
                .collect();
            let total_weight: f64 = weights.iter().sum();

            // Count observed occurrences by sampling
            let mut observed = vec![0usize; n];
            for _ in 0..num_samples {
                let mut u: f64 = rng.gen::<f64>() * total_weight;
                for (i, &w) in weights.iter().enumerate() {
                    u -= w;
                    if u <= 0.0 {
                        observed[i] += 1;
                        break;
                    }
                }
                // Handle floating point edge case
                if u > 0.0 {
                    observed[n - 1] += 1;
                }
            }

            // Use the core stats module for the calculation
            let result = crate::core::chi_squared_from_counts(&observed, &weights, num_samples);

            PyChiSquaredResult {
                chi_squared: result.chi_squared,
                degrees_of_freedom: result.degrees_of_freedom,
                p_value: result.p_value,
                num_samples: result.num_samples,
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
                "ChiSquaredResult(chi_squared={:.4}, df={}, p_value={:.6}, n={})",
                self.chi_squared, self.degrees_of_freedom, self.p_value, self.num_samples
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
