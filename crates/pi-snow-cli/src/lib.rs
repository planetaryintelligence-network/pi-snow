//! The file-level interface: `pi-snow` reads the same INI, SMET and SNO the
//! reference engine does and writes PRO and SMET where it does, so the two are
//! interchangeable at the pipeline's existing seam (`_run_native`).

pub mod batch;
pub mod dirichlet_storage_diagnostic;
pub mod engine;
pub mod finite;
pub mod ledger;
pub mod meltout_candidate;
pub mod meltout_diagnostic;
pub mod unit;
