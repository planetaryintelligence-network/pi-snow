//! Per-step evidence for engine-meltout-v1's lower-boundary identity.
//!
//! The signed Dirichlet reconstruction is
//! `Qg0 = -[conductance0 * (T1+ - T0) + 0.5 * source0]`: `Qg0` is positive
//! into snow, while the bracket is positive for heat conducted out of snow.

use std::fmt::Write;

use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Default)]
pub struct MeltoutDiagnostic {
    text: String,
}

impl MeltoutDiagnostic {
    pub fn new() -> Self {
        Self {
            text: String::from(concat!(
                "timestamp_utc,elements,bottom_thickness_m,bottom_density_kg_m3,source0_W_m2,",
                "T0_K,T1_post_K,conductance0_W_m2_K,dirichlet_Qg0_W_m2,residual_Qg0_W_m2,",
                "difference_W_m2\n",
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        timestamp: DateTime<Utc>,
        elements: usize,
        thickness: f64,
        density: f64,
        source0: f64,
        t0: f64,
        t1_post: f64,
        conductance: f64,
        qg0: f64,
    ) {
        let reconstructed = -(conductance * (t1_post - t0) + 0.5 * source0);
        writeln!(
            self.text,
            "{},{elements},{thickness:.12},{density:.9},{source0:.9},{t0:.9},{t1_post:.9},{conductance:.9},{reconstructed:.9},{qg0:.9},{:.9}",
            timestamp.to_rfc3339(),
            (reconstructed - qg0).abs(),
        )
        .expect("writing to a String cannot fail");
    }

    pub fn into_text(self) -> String {
        self.text
    }
}
