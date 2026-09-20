//! Core state types for a one-dimensional snow cover.
//!
//! This crate holds the data model and the physical constants, and nothing that
//! evolves a state in time. Keeping the model free of physics means the I/O
//! crate can round-trip a profile without linking any parameterisation, which
//! is what lets stage 0 of the build order be tested before stage 1 exists.
//!
//! The implementation is clean-room: derived from published literature, not
//! translated from any existing snow model's source. See
//! `experiments/phase-2-rust-engine-v1/plan.md`.

pub mod constants;
pub mod state;

pub use state::{Column, Composition, Element, Microstructure};
