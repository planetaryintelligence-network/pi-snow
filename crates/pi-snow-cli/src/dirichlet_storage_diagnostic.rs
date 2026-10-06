//! Boundary-node storage terms for the engine-meltout-v1 A4 control proof.

use std::fmt::Write;

use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Default)]
pub struct DirichletStorageDiagnostic {
    text: String,
}

impl DirichletStorageDiagnostic {
    pub fn new() -> Self {
        Self {
            text: String::from(concat!(
                "timestamp_utc,T0_pre_K,T0_post_K,C0_J_m-2_K-1,dt_s,Qreset_W_m2,",
                "discrete_Qg0_W_m2,residual_Qg0_W_m2,corrected_Qg0_W_m2,",
                "raw_difference_W_m2,corrected_difference_W_m2\n",
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        timestamp: DateTime<Utc>,
        t0_pre: f64,
        t0_post: f64,
        c0: f64,
        dt: f64,
        conductance0: f64,
        t1_post: f64,
        source0: f64,
        residual_qg0: f64,
    ) {
        let discrete_qg0 = -(conductance0 * (t1_post - t0_post) + 0.5 * source0);
        let qreset = c0 * (t0_post - t0_pre) / dt;
        let corrected_qg0 = residual_qg0 - qreset;
        let raw_difference = residual_qg0 - discrete_qg0;
        let corrected_difference = corrected_qg0 - discrete_qg0;
        writeln!(
            self.text,
            "{},{t0_pre:.9},{t0_post:.9},{c0:.9},{dt:.9},{qreset:.9},{discrete_qg0:.9},{residual_qg0:.9},{corrected_qg0:.9},{:.9},{:.9}",
            timestamp.to_rfc3339(),
            raw_difference,
            corrected_difference,
        )
        .expect("writing to a String cannot fail");
    }

    pub fn into_text(self) -> String {
        self.text
    }
}
