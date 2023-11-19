//! # RRT* Library
//! Implements two Rapidly-exploring Random Tree (RRT*) algorithms in Rust:
//! - PQ-RRT* (Potential field Quick RRT*)
//! - IRRT* (Informed RRT*)
//! - RRT*
//! - RRT
//!
//! ## Usage
//! Intended for use through Python (pyo3) bindings. Relies on getting ENC data from python shapely objects.
use pyo3::prelude::*;
mod common;
mod model;
mod steering;
mod utils;

pub mod enc_data;
pub mod irrt_star;
pub mod pqrrt_star;
pub mod rrt;
pub mod rrt_star;

/// The RRT* library in rust. Implements four RRT* algorithms:
/// - PQ-RRT* (Potential field Quick RRT*)
/// - Adaptive Informed RRT*
/// - Informed RRT*
/// - RRT*
/// - RRT
///
/// The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rrt_star_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::irrt_star::IRRTStar>()?;
    m.add_class::<crate::pqrrt_star::PQRRTStar>()?;
    m.add_class::<crate::rrt_star::RRTStar>()?;
    m.add_class::<crate::rrt::RRT>()?;
    Ok(())
}
