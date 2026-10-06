//! Uncapped CryoGrid bare-soil budget and conservative implicit boundary.

use super::{Bottom, FREEZING_K, Phase, SoilColumn, StepReport, Surface};
use pi_snow_core::constants::{C_WATER, LH_SUBLIMATION, LH_VAPORISATION, STEFAN_BOLTZMANN};
use pi_snow_physics::atmosphere::{
    air_density, specific_humidity, vapour_pressure_ice, vapour_pressure_water,
};
use pi_snow_physics::surface::{Atmosphere, Budget, SurfaceParameters};
use pi_snow_physics::turbulence::{self, SurfaceLayer};

pub fn budget(atm: &Atmosphere, params: &SurfaceParameters, phase: Phase) -> Budget {
    let temperature_k = phase.temperature_k;
    let liquid = phase.liquid_fraction;
    let vapour = liquid * vapour_pressure_water(temperature_k)
        + (1.0 - liquid) * vapour_pressure_ice(temperature_k);
    let layer = SurfaceLayer {
        wind: atm.wind,
        z_wind: params.z_wind,
        z_scalar: params.z_scalar,
        z0: params.z0,
        t_air: atm.t_air,
        t_surface: temperature_k,
        pressure: atm.pressure,
        q_air: specific_humidity(
            atm.relative_humidity * vapour_pressure_water(atm.t_air),
            atm.pressure,
        ),
        q_surface: specific_humidity(vapour, atm.pressure),
    };
    let flux = turbulence::fluxes(&layer, params.scheme);
    // Existing stability depends on temperature, so a unit humidity difference
    // exposes rho/r_a even at a neutral or vapour-equilibrium surface.
    let probe = SurfaceLayer {
        q_air: 1.0,
        q_surface: 0.0,
        ..layer
    };
    let rho = air_density(atm.pressure, 0.5 * (atm.t_air + temperature_k));
    let exchange = turbulence::fluxes(&probe, params.scheme).vapour_mass / rho;
    let vapour_mass = flux.vapour_mass / (1.0 + 50.0 * exchange);
    let latent = vapour_mass * (liquid * LH_VAPORISATION + (1.0 - liquid) * LH_SUBLIMATION);
    let olwr = 0.97 * STEFAN_BOLTZMANN * temperature_k.powi(4) + 0.03 * atm.ilwr;
    let lwr_net = atm.ilwr - olwr;
    let rain = atm.rain_rate * C_WATER * (atm.t_air - temperature_k);
    Budget {
        olwr,
        lwr_net,
        sensible: flux.sensible,
        latent,
        rain,
        vapour_mass,
        total: atm.absorbed_shortwave + lwr_net + flux.sensible + latent + rain,
    }
}

pub use super::bare_scan::{Agreement, Certification, RootScan};

#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub certified: bool,
    pub stiffness_lower_bound_w_m2_k: f64,
    pub maximum_sampled_budget_slope_w_m2_k: f64,
    pub scan: Option<RootScan>,
    pub certification: Option<Certification>,
    pub column_solves: usize,
    pub budget_evaluations: usize,
    pub newton_iterations: usize,
    pub nonconvergences: usize,
}

impl Diagnostics {
    pub fn multiple_roots(&self) -> bool {
        self.scan.as_ref().is_some_and(|s| s.roots.len() > 1)
    }
}

#[derive(Debug, Clone)]
pub struct BareReport {
    pub soil: StepReport,
    pub surface_flux_w_m2: f64,
    pub flux_residual_w_m2: f64,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Clone)]
pub struct Failure {
    pub message: String,
    pub diagnostics: Diagnostics,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Failure {}

pub fn step(
    soil: &mut SoilColumn,
    dt: f64,
    bottom: Bottom,
    atm: &Atmosphere,
    params: &SurfaceParameters,
) -> Result<BareReport, Failure> {
    solve(soil, dt, bottom, |phase| budget(atm, params, phase).total)
}

pub fn pure_phase(temperature_k: f64) -> Phase {
    Phase {
        temperature_k,
        liquid_fraction: if temperature_k >= FREEZING_K {
            1.0
        } else {
            0.0
        },
        dtemperature_denthalpy: 0.0,
    }
}

/// Every trial reduces the simultaneous implicit soil rows from the same entry
/// state. Failed attempts leave the caller unchanged and retain their counters.
pub fn solve(
    soil: &mut SoilColumn,
    dt: f64,
    bottom: Bottom,
    balance: impl Fn(Phase) -> f64,
) -> Result<BareReport, Failure> {
    let mut diagnostics = Diagnostics::default();
    match solve_inner(soil, dt, bottom, &balance, &mut diagnostics) {
        Ok(mut report) => {
            report.diagnostics = diagnostics;
            Ok(report)
        }
        Err(message) => {
            diagnostics.nonconvergences = 1;
            if let Some(scan) = &mut diagnostics.scan {
                scan.assess(None);
            }
            Err(Failure {
                message,
                diagnostics,
            })
        }
    }
}

fn solve_inner(
    soil: &mut SoilColumn,
    dt: f64,
    bottom: Bottom,
    balance: &impl Fn(Phase) -> f64,
    diagnostics: &mut Diagnostics,
) -> Result<BareReport, String> {
    let entry = soil.clone();
    let trial = |q: f64, d: &mut Diagnostics| -> Result<(f64, SoilColumn, StepReport), String> {
        d.column_solves += 1;
        let mut column = entry.clone();
        let report = column.step(dt, Surface::Flux(q), bottom)?;
        d.budget_evaluations += 1;
        let residual = q - balance(column.phase(0));
        if !residual.is_finite() {
            return Err("nonfinite bare-soil balance".into());
        }
        Ok((residual, column, report))
    };
    let imposed = |temperature, d: &mut Diagnostics| -> Result<f64, String> {
        d.column_solves += 1;
        let mut column = entry.clone();
        Ok(column
            .step(dt, Surface::Temperature(temperature), bottom)?
            .top_flux_w_m2)
    };
    let (mut lo, mut hi) = (imposed(210.0, diagnostics)?, imposed(340.0, diagnostics)?);
    if trial(lo, diagnostics)?.0 > 0.0 || trial(hi, diagnostics)?.0 < 0.0 {
        return Err("bare-soil balance has no bracket in [210,340] K".into());
    }
    let lower = super::solver::surface_stiffness_lower_bound(&entry, dt, bottom);
    let mut maximum = f64::NEG_INFINITY;
    for i in 0..=1300 {
        let t = 210.0 + 0.1 * i as f64;
        // Pure-branch differences never straddle the freezing temperature.
        let (a, b) = if t < FREEZING_K {
            (t - 0.01, (t + 0.01).min(FREEZING_K - 1e-9))
        } else {
            ((t - 0.01).max(FREEZING_K + 1e-9), t + 0.01)
        };
        diagnostics.budget_evaluations += 2;
        let slope = (balance(pure_phase(b)) - balance(pure_phase(a))) / (b - a);
        if !slope.is_finite() {
            return Err("nonfinite bare-soil sampled slope".into());
        }
        maximum = maximum.max(slope);
    }
    diagnostics.stiffness_lower_bound_w_m2_k = lower;
    diagnostics.maximum_sampled_budget_slope_w_m2_k = maximum;
    let width = entry.material.water_kg_m3() * super::LATENT_J_KG * SoilColumn::node_volume(0) / dt;
    diagnostics.budget_evaluations += 103;
    let certificate = super::bare_scan::certify(balance, maximum, lower, width)?;
    diagnostics.certified = !certificate.failed.iter().any(|f| *f);
    if !diagnostics.certified {
        let scan =
            super::bare_scan::scan(&entry, dt, bottom, balance, lower, [lo, hi], &certificate)?;
        diagnostics.column_solves += 1 + scan.resolve_solves;
        diagnostics.budget_evaluations += 1301;
        diagnostics.scan = Some(scan);
    }
    diagnostics.certification = Some(certificate);
    diagnostics.budget_evaluations += 1;
    let mut q = balance(entry.phase(0)).clamp(lo, hi);
    for _ in 0..100 {
        diagnostics.newton_iterations += 1;
        let (residual, column, report) = trial(q, diagnostics)?;
        if residual.abs() <= 1e-6 {
            if let Some(scan) = &mut diagnostics.scan {
                scan.assess(Some(column.phase(0)));
            }
            *soil = column;
            return Ok(BareReport {
                soil: report,
                surface_flux_w_m2: q,
                flux_residual_w_m2: residual,
                diagnostics: Diagnostics::default(),
            });
        }
        if residual > 0.0 {
            hi = q;
        } else {
            lo = q;
        }
        let epsilon = 0.01_f64.max(q.abs() * 1e-6);
        let derivative = (trial(q + epsilon, diagnostics)?.0 - trial(q - epsilon, diagnostics)?.0)
            / (2.0 * epsilon);
        let newton = q - residual / derivative;
        q = if newton.is_finite() && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
    }
    Err("bare-soil safeguarded solve failed after 100 iterations".into())
}
