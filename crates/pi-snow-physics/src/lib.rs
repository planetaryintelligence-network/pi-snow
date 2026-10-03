//! Physical parameterisations for a one-dimensional snow cover.
//!
//! Clean-room: every parameterisation here is implemented from a published
//! source that someone read, and carries that citation in its doc comment. A
//! parameterisation whose coefficients could not be read from a source is
//! **absent** rather than reconstructed from memory. See
//! `experiments/phase-2-rust-engine-v1/citation-audit.md` for what is sourced
//! and what is not.

pub mod albedo;
pub mod aoki_albedo;
pub mod atmosphere;
pub mod grain_classification;
pub mod metamorphism;
pub mod new_snow;
pub mod settlement;
pub mod solar;
pub mod stability;
pub mod surface;
pub mod surface_layer;
pub mod surface_numerics;
pub mod thermal;
pub mod turbulence;
pub mod water;
pub mod wet_growth;
