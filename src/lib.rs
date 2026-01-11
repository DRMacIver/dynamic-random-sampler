//! Dynamic Random Sampler - Rust implementation
//!
//! This module implements the data structure from "Dynamic Generation of Discrete
//! Random Variates" by Matias, Vitter, and Ni (1993/2003).
//!
//! The implementation will be completed inside the devcontainer.

#![allow(clippy::redundant_pub_crate)]

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
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("index out of bounds"))
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
    /// Returns index j with probability w_j / sum(w_i).
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
}

/// Python module definition
#[pymodule]
fn dynamic_random_sampler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DynamicSampler>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sampler() {
        let sampler = DynamicSampler::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(sampler.log_weights.len(), 3);
    }

    #[test]
    fn test_weight_roundtrip() {
        let sampler = DynamicSampler::new(vec![1.0, 2.0, 4.0]).unwrap();
        assert!((sampler.weight(0).unwrap() - 1.0).abs() < 1e-10);
        assert!((sampler.weight(1).unwrap() - 2.0).abs() < 1e-10);
        assert!((sampler.weight(2).unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_weights_rejected() {
        assert!(DynamicSampler::new(vec![]).is_err());
    }

    #[test]
    fn test_negative_weights_rejected() {
        assert!(DynamicSampler::new(vec![1.0, -1.0]).is_err());
    }

    #[test]
    fn test_zero_weight_rejected() {
        assert!(DynamicSampler::new(vec![1.0, 0.0]).is_err());
    }
}
