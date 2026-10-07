//! Rig-only observations of the actual joint snow/soil Picard solve.
use super::coupled::CoupledReport;
use pi_snow_core::{
    Column,
    constants::{STEFAN_BOLTZMANN, T_MELT},
};
use pi_snow_physics::{
    conduction_diagnostic,
    surface::{self, Atmosphere, SurfaceParameters},
    surface_diagnostic::SurfaceIteration,
    thermal::{Conductivity, Ground},
};

pub fn iteration(
    index: usize,
    input_k: f64,
    a: f64,
    b: f64,
    before: &Column,
    after: &Column,
    report: &CoupledReport,
    atmosphere: &Atmosphere,
    parameters: &SurfaceParameters,
    sources: &[f64],
    ground_shortwave: f64,
    dt: f64,
    conductivity: Conductivity,
) -> Result<SurfaceIteration, String> {
    let t = input_k.min(T_MELT - 0.01);
    let f0 = surface::budget(atmosphere, parameters, t);
    let f1 = surface::budget(atmosphere, parameters, t + 0.01);
    let solved = *after
        .node_temperature
        .last()
        .ok_or("missing snow surface node")?;
    let nonlinear = surface::budget(atmosphere, parameters, solved);
    let stats = conduction_diagnostic::summarize(
        before,
        after,
        Ground::Flux(ground_shortwave - report.soil.top_flux_w_m2),
        a,
        b,
        sources,
        dt,
        conductivity,
    )
    .ok_or("coupled snow diagnostic shape differs")?;
    Ok(SurfaceIteration {
        iteration: index,
        linearization_input_k: input_k,
        linearization_temperature_k: t,
        a_w_m2: a,
        b_w_m2_k: b,
        finite_difference_budget_f0_w_m2: f0.total,
        finite_difference_budget_f1_w_m2: f1.total,
        raw_finite_difference_slope_w_m2_k: (f1.total - f0.total) / 0.01,
        finite_difference_budget_f0: f0,
        finite_difference_budget_f1: f1,
        radiative_slope_w_m2_k: -4.0 * parameters.emissivity * STEFAN_BOLTZMANN * t.powi(3),
        solved_surface_k: solved,
        nonlinear_budget_w_m2: nonlinear.total,
        solved_surface_budget: nonlinear,
        surface_linearization_residual_w_m2: nonlinear.total - (a + b * solved),
        conduction: stats,
    })
}
