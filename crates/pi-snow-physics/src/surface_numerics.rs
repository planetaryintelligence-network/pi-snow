//! A floored linearisation of the surface energy balance
//! (engine-finite-guard-v1, `--physics SURFACE_LINEARISATION=RADIATIVE_FLOOR`).
//!
//! Phase 2 (`surface::step_general`) linearises the balance as `F(T) ~ a + b T`
//! with `b` a finite difference over 0.01 K, capped at zero. Across the
//! two-height Monin-Obukhov solve's jump between its coupled and decoupled
//! branches that difference can come out shallow or zero, which takes all the
//! damping off a thin top element: on TPRUT1 (v3b, 2026-02-19 18:20 UTC) one
//! step put the surface at -361 K, the vapour pressure there is NaN, and the
//! pack was NaN to July. The budget always contains the emitted longwave,
//! whose own slope is `-4 eps sigma T^3` (`surface::budget`'s OLWR term), so
//! this variant never lets `b` be shallower than that. Everything else is
//! `surface::step_general` line for line. A numerical safeguard, ours; no
//! physics source is claimed.

use pi_snow_core::Column;
use pi_snow_core::constants::{STEFAN_BOLTZMANN, T_MELT};

use crate::surface::{Atmosphere, Budget, SurfaceParameters, budget};
use crate::thermal::{self, Conductivity, Ground};

/// As `surface::step_general`, with the floored slope.
pub fn step_general(
    column: &mut Column,
    atm: &Atmosphere,
    p: &SurfaceParameters,
    ground: Ground,
    sources: &[f64],
    dt: f64,
    k: Conductivity,
) -> Budget {
    if column.elements.is_empty() {
        return budget(atm, p, column.node_temperature[0]);
    }
    let start = column.node_temperature.clone();
    let mut t_lin = *start.last().unwrap();
    let mut used = (0.0, 0.0);
    for _ in 0..3 {
        column.node_temperature.clone_from(&start);
        used = linearise(atm, p, t_lin);
        thermal::step_general(column, ground, used.0, used.1, sources, dt, k);
        t_lin = *column.node_temperature.last().unwrap();
    }
    let mut report = budget(atm, p, t_lin);
    report.total = used.0 + used.1 * t_lin;
    report
}

/// `(a, b)` with `F(T) ~ a + b T` about `t0`, `b <= -4 eps sigma t^3`.
pub fn linearise(atm: &Atmosphere, p: &SurfaceParameters, t0: f64) -> (f64, f64) {
    const DELTA: f64 = 0.01;
    let t = t0.min(T_MELT - DELTA);
    let f0 = budget(atm, p, t).total;
    let f1 = budget(atm, p, t + DELTA).total;
    let radiative = -4.0 * p.emissivity * STEFAN_BOLTZMANN * t.powi(3);
    let b = ((f1 - f0) / DELTA).min(0.0).min(radiative);
    (f0 - b * t, b)
}
