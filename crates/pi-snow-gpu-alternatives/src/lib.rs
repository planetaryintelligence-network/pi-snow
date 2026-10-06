//! pi-snow's column physics on a CUDA GPU, one column per thread.
//!
//! `stepper` is a CPU port of the engine loop that stops after every step; `unit`
//! loads a v3b-style unit directory; `layout` packs columns into the GPU's
//! struct-of-arrays state; `gpu` compiles and drives the kernel in
//! `kernels/pi_snow.cu`; `forcing` builds 10-minute grid forcing from hourly values.
//! See `experiments/engine-gpu-v1/plan.md`.

pub mod stepper;
pub mod unit;
pub mod batch;
pub mod forcing;
pub mod gpu;
pub mod grid;
pub mod sha256;
pub mod lockstep;
pub mod a5_flux_gate;
pub mod forcing_texture;
pub mod mo;
pub mod runner;
pub mod synth;
