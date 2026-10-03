//! Observed form of the Phase 2 three-pass surface/conduction iteration.

use pi_snow_core::Column;
use pi_snow_core::constants::{STEFAN_BOLTZMANN, T_MELT};

use crate::conduction_diagnostic::{self, ConductionStats};
use crate::surface::{Atmosphere, Budget, SurfaceParameters, budget};
use crate::thermal::{self, Conductivity, Ground};

#[derive(Debug, Clone, Copy)]
pub struct SurfaceIteration {
    pub iteration: usize,
    pub linearization_input_k: f64,
    pub linearization_temperature_k: f64,
    pub a_w_m2: f64,
    pub b_w_m2_k: f64,
    pub finite_difference_budget_f0_w_m2: f64,
    pub finite_difference_budget_f1_w_m2: f64,
    pub raw_finite_difference_slope_w_m2_k: f64,
    pub radiative_slope_w_m2_k: f64,
    pub solved_surface_k: f64,
    pub nonlinear_budget_w_m2: f64,
    pub surface_linearization_residual_w_m2: f64,
    pub conduction: ConductionStats,
}

/// The same three solves as `surface::step_general`, with per-iteration values
/// sent to an observer. `linearise` is the original option's exact function.
pub fn step_general(
    column: &mut Column,
    atm: &Atmosphere,
    p: &SurfaceParameters,
    ground: Ground,
    sources: &[f64],
    dt: f64,
    conductivity: Conductivity,
    linearise: fn(&Atmosphere, &SurfaceParameters, f64) -> (f64, f64),
    mut observe: impl FnMut(SurfaceIteration),
) -> Budget {
    if column.elements.is_empty() {
        return budget(atm, p, column.node_temperature[0]);
    }
    let start = column.node_temperature.clone();
    let mut t_lin = *start.last().unwrap();
    let mut used = (0.0, 0.0);
    for iteration in 1..=3 {
        column.node_temperature.clone_from(&start);
        let input = t_lin;
        let (a, b) = linearise(atm, p, input);
        let linearization_temperature = input.min(T_MELT - 0.01);
        let f0 = budget(atm, p, linearization_temperature).total;
        let f1 = budget(atm, p, linearization_temperature + 0.01).total;
        used = (a, b);
        let before = column.clone();
        thermal::step_general(column, ground, a, b, sources, dt, conductivity);
        let solved_surface_k = *column.node_temperature.last().unwrap();
        let nonlinear_budget_w_m2 = budget(atm, p, solved_surface_k).total;
        let surface_linearization_residual_w_m2 =
            nonlinear_budget_w_m2 - (a + b * solved_surface_k);
        let stats = conduction_diagnostic::summarize(
            &before,
            column,
            ground,
            a,
            b,
            sources,
            dt,
            conductivity,
        )
        .unwrap_or_default();
        let t = linearization_temperature;
        observe(SurfaceIteration {
            iteration,
            linearization_input_k: input,
            linearization_temperature_k: t,
            a_w_m2: a,
            b_w_m2_k: b,
            finite_difference_budget_f0_w_m2: f0,
            finite_difference_budget_f1_w_m2: f1,
            raw_finite_difference_slope_w_m2_k: (f1 - f0) / 0.01,
            radiative_slope_w_m2_k: -4.0 * p.emissivity * STEFAN_BOLTZMANN * t.powi(3),
            solved_surface_k,
            nonlinear_budget_w_m2,
            surface_linearization_residual_w_m2,
            conduction: stats,
        });
        t_lin = solved_surface_k;
    }
    let mut report = budget(atm, p, t_lin);
    report.total = used.0 + used.1 * t_lin;
    report
}
