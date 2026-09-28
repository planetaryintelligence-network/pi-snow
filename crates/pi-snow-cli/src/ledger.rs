//! The mass ledger written under `--physics MASS_REPORT=LEDGER`.
//!
//! pi-snow's SMET carries the reference's mass columns, and those cannot close
//! its budget: vapour deposition is written nowhere, and `MS_Sublimation`
//! counts the flux asked of the top element, not what it could give. The
//! ledger records, per output row, what each stage of a step claims to have
//! moved, taken from the stage's own variables rather than from differences of
//! the column's mass, so a stage that changes mass without saying so shows up
//! as a closure residual instead of being absorbed. It changes no physics and
//! nothing in the PRO or SMET.

use std::fmt::Write;

use chrono::{DateTime, Utc};

/// Mass terms over one output interval, kg m^-2.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Solid precipitation handed to the deposition buffer.
    pub solid_precipitation: f64,
    /// Snow deposited from the buffer into the pack, in whole elements.
    pub deposited: f64,
    /// Rain added to the top element.
    pub rain_to_pack: f64,
    /// Rain on bare ground, which leaves as runoff at once.
    pub rain_to_ground: f64,
    /// Water leaving the bottom of the bucket cascade.
    pub runoff: f64,
    /// Water leaving because the bottom element melted out.
    pub drained: f64,
    /// Net vapour flux the surface balance asks of the top element (+ gain).
    pub vapour_flux: f64,
    /// Ice removed from the top element by sublimation.
    pub sublimation_applied: f64,
    /// Ice added to the top element by deposition.
    pub deposition_applied: f64,
}

impl Interval {
    /// Net mass the stages claim entered the pack: what the closure test in
    /// `experiments/engine-mass-v1/plan.md` (form PL) compares with the SWE
    /// change.
    pub fn net_to_pack(&self) -> f64 {
        self.deposited + self.rain_to_pack - self.runoff - self.drained - self.sublimation_applied
            + self.deposition_applied
    }
}

/// The ledger's rows, one per SMET row.
#[derive(Debug, Default)]
pub struct Ledger {
    text: String,
}

impl Ledger {
    pub fn new() -> Self {
        let mut text = String::new();
        text.push_str(
            "timestamp,swe,buffered,solid_precipitation,deposited,rain_to_pack,rain_to_ground,\
             runoff,drained,vapour_flux,sublimation_applied,deposition_applied,vapour_unapplied\n",
        );
        Self { text }
    }

    /// Append the row for the output at `t`, with the column's SWE and the
    /// buffer at that instant and the interval's terms.
    pub fn push(&mut self, t: DateTime<Utc>, swe: f64, buffered: f64, i: &Interval) {
        let unapplied = i.vapour_flux - (i.deposition_applied - i.sublimation_applied);
        writeln!(
            self.text,
            "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            t.format("%Y-%m-%dT%H:%M:%S"),
            swe,
            buffered,
            i.solid_precipitation,
            i.deposited,
            i.rain_to_pack,
            i.rain_to_ground,
            i.runoff,
            i.drained,
            i.vapour_flux,
            i.sublimation_applied,
            i.deposition_applied,
            unapplied,
        )
        .expect("writing to a String cannot fail");
    }

    pub fn into_text(self) -> String {
        self.text
    }
}
