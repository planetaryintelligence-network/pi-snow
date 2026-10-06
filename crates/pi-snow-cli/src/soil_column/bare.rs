//! Uncapped CryoGrid bare-soil budget and conservative implicit boundary.

use super::{Bottom, SoilColumn, StepReport, Surface};
use pi_snow_core::constants::{C_WATER, LH_SUBLIMATION, LH_VAPORISATION, STEFAN_BOLTZMANN};
use pi_snow_physics::atmosphere::{
    air_density, specific_humidity, vapour_pressure_ice, vapour_pressure_water,
};
use pi_snow_physics::surface::{Atmosphere, Budget, SurfaceParameters};
use pi_snow_physics::turbulence::{self, SurfaceLayer};

pub fn budget(atm: &Atmosphere, params: &SurfaceParameters, temperature_k: f64) -> Budget {
    let water = temperature_k >= super::FREEZING_K;
    let vapour = if water {
        vapour_pressure_water(temperature_k)
    } else {
        vapour_pressure_ice(temperature_k)
    };
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
    let latent = vapour_mass
        * if water {
            LH_VAPORISATION
        } else {
            LH_SUBLIMATION
        };
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

#[derive(Debug, Clone, Copy)]
pub struct BareReport {
    pub soil: StepReport,
    pub surface_flux_w_m2: f64,
    pub flux_residual_w_m2: f64,
    pub monotonicity_violations: usize,
}

pub fn step(
    soil: &mut SoilColumn,
    dt: f64,
    bottom: Bottom,
    atm: &Atmosphere,
    params: &SurfaceParameters,
) -> Result<BareReport, String> {
    solve(soil, dt, bottom, |temperature| {
        budget(atm, params, temperature).total
    })
}

/// The scalar flux root is a Schur reduction of the simultaneous soil rows.
/// Each trial advances a clone from the same entry state, with no explicit
/// surface-temperature lag or update to the caller before convergence.
pub fn solve(
    soil: &mut SoilColumn,
    dt: f64,
    bottom: Bottom,
    balance: impl Fn(f64) -> f64,
) -> Result<BareReport, String> {
    let entry = soil.clone();
    let trial = |q: f64| -> Result<(f64, SoilColumn, StepReport), String> {
        let mut column = entry.clone();
        let report = column.step(dt, Surface::Flux(q), bottom)?;
        let residual = q - balance(column.phase(0).temperature_k);
        if !residual.is_finite() {
            return Err("nonfinite bare-soil balance".into());
        }
        Ok((residual, column, report))
    };
    let imposed = |temperature| -> Result<f64, String> {
        let mut column = entry.clone();
        Ok(column
            .step(dt, Surface::Temperature(temperature), bottom)?
            .top_flux_w_m2)
    };
    let (mut lo, mut hi) = (imposed(210.0)?, imposed(340.0)?);
    if trial(lo)?.0 > 0.0 || trial(hi)?.0 < 0.0 {
        return Err("bare-soil balance has no bracket in [210,340] K".into());
    }
    let k0 = entry
        .material
        .conductivity(entry.phase(0).temperature_k, entry.phase(0).liquid_fraction);
    let k1 = entry
        .material
        .conductivity(entry.phase(1).temperature_k, entry.phase(1).liquid_fraction);
    let anchor = entry
        .material
        .capacity(0.0)
        .min(entry.material.capacity(1.0))
        * SoilColumn::node_volume(0)
        / dt
        + 2.0 / (1.0 / k0 + 1.0 / k1) / 0.01;
    // Sampled finite-difference predicate over the registered temperature
    // domain. The report names a violation; it is no interval proof.
    let violation = (0..=130).any(|i| {
        let t = 210.0 + i as f64;
        (balance(t + 0.01) - balance(t - 0.01)) / 0.02 >= anchor
    });
    let mut q = balance(entry.phase(0).temperature_k).clamp(lo, hi);
    for _ in 0..100 {
        let (residual, column, report) = trial(q)?;
        if residual.abs() <= 1e-6 {
            *soil = column;
            return Ok(BareReport {
                soil: report,
                surface_flux_w_m2: q,
                flux_residual_w_m2: residual,
                monotonicity_violations: usize::from(violation),
            });
        }
        if residual > 0.0 {
            hi = q;
        } else {
            lo = q;
        }
        let epsilon = 0.01_f64.max(q.abs() * 1e-6);
        let derivative = (trial(q + epsilon)?.0 - trial(q - epsilon)?.0) / (2.0 * epsilon);
        let newton = q - residual / derivative;
        q = if newton.is_finite() && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
    }
    Err("bare-soil safeguarded solve failed after 100 iterations".into())
}
