//! The surface energy balance, and its coupling to conduction.
//!
//! The configuration sets `CHANGE_BC = false`, so the upper boundary is always
//! the energy balance: net shortwave absorbed at the surface, net longwave,
//! turbulent sensible and latent heat, and heat carried by rain. Each flux is
//! positive into the snow.
//!
//! Coupling: the balance is nonlinear in surface temperature (the T^4 of
//! emitted longwave and the stability dependence of the turbulent fluxes), so it
//! is linearised about the current surface temperature, `F(T) ~ a + b T` with
//! `b` from a finite difference, and imposed as an implicit flux boundary on the
//! conduction solve. The linearisation is refreshed and the step repeated from
//! the same starting state three times, which converges well within a
//! millikelvin at a 10-minute step. The balance is evaluated at no more than
//! the melting point: a surface cannot be warmer than 0 C, and energy that
//! would push it there is melt, which the phase-change step takes up.
//!
//! Sources: emissivity from Dozier and Warren (1982), Water Resour. Res. 18,
//! 1424-1434 (all-wave 0.985 to 0.990, independent of grain size); turbulent
//! fluxes from `turbulence`; vapour pressure and air properties from
//! `atmosphere`. The rain term, `rho_w c_w R (T_rain - T_s)` with the rain at
//! air temperature, follows Lafaysse et al. (2026) eq. 92 as read in round 1.

use pi_snow_core::Column;
use pi_snow_core::constants::{C_WATER, LH_SUBLIMATION, LH_VAPORISATION, STEFAN_BOLTZMANN, T_MELT};

use crate::atmosphere::{specific_humidity, vapour_pressure_ice, vapour_pressure_water};
use crate::thermal::{self, Conductivity, Ground};
use crate::turbulence::{self, Scheme, SurfaceLayer};

/// All-wave emissivity of snow: the midpoint of Dozier and Warren's
/// 0.985-0.990. The midpoint is our choice within their range.
pub const SNOW_EMISSIVITY: f64 = 0.9875;

/// Forcing at the surface for one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atmosphere {
    /// Air temperature, K.
    pub t_air: f64,
    /// Relative humidity with respect to liquid water (`FORCE_RH_WATER`), 0-1.
    pub relative_humidity: f64,
    /// Wind speed, m s^-1.
    pub wind: f64,
    /// Shortwave absorbed at the surface, W m^-2, already net of albedo.
    pub absorbed_shortwave: f64,
    /// Incoming longwave, W m^-2.
    pub ilwr: f64,
    /// Surface pressure, Pa.
    pub pressure: f64,
    /// Rain reaching the surface, kg m^-2 s^-1.
    pub rain_rate: f64,
}

/// Fixed surface-layer parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceParameters {
    pub emissivity: f64,
    /// Measurement heights above the snow surface, m, already reduced by the
    /// snow depth (`ADJUST_HEIGHT_OF_*_VALUE = true`).
    pub z_wind: f64,
    pub z_scalar: f64,
    pub z0: f64,
    pub scheme: Scheme,
}

/// The surface energy budget at one surface temperature, W m^-2.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Budget {
    pub olwr: f64,
    pub lwr_net: f64,
    pub sensible: f64,
    pub latent: f64,
    pub rain: f64,
    /// Water vapour exchanged, kg m^-2 s^-1, positive deposition.
    pub vapour_mass: f64,
    /// Sum of every term, including absorbed shortwave.
    pub total: f64,
}

/// Evaluate the budget with the surface at `t_surface` (capped at melting).
pub fn budget(atm: &Atmosphere, p: &SurfaceParameters, t_surface: f64) -> Budget {
    let ts = t_surface.min(T_MELT);
    let olwr = p.emissivity * STEFAN_BOLTZMANN * ts.powi(4) + (1.0 - p.emissivity) * atm.ilwr;
    let lwr_net = atm.ilwr - olwr;

    let melting = ts >= T_MELT;
    let e_surface = if melting {
        vapour_pressure_water(ts)
    } else {
        vapour_pressure_ice(ts)
    };
    let e_air = atm.relative_humidity * vapour_pressure_water(atm.t_air);
    let layer = SurfaceLayer {
        wind: atm.wind,
        z_wind: p.z_wind,
        z_scalar: p.z_scalar,
        z0: p.z0,
        t_air: atm.t_air,
        t_surface: ts,
        q_air: specific_humidity(e_air, atm.pressure),
        q_surface: specific_humidity(e_surface, atm.pressure),
        pressure: atm.pressure,
    };
    let f = turbulence::fluxes(&layer, p.scheme);
    let latent_heat = if melting {
        LH_VAPORISATION
    } else {
        LH_SUBLIMATION
    };
    let latent = latent_heat * f.vapour_mass;
    let rain = atm.rain_rate * C_WATER * (atm.t_air - ts);
    Budget {
        olwr,
        lwr_net,
        sensible: f.sensible,
        latent,
        rain,
        vapour_mass: f.vapour_mass,
        total: atm.absorbed_shortwave + lwr_net + f.sensible + latent + rain,
    }
}

/// Advance the column one step with the energy balance on top and
/// `ground_flux` (W m^-2, into the snow) at the bottom. Returns the budget at
/// the end-of-step surface temperature, which is the flux the implicit
/// boundary actually imposed.
pub fn step(
    column: &mut Column,
    atm: &Atmosphere,
    p: &SurfaceParameters,
    ground_flux: f64,
    dt: f64,
    k: Conductivity,
) -> Budget {
    step_with_sources(column, atm, p, ground_flux, &[], dt, k)
}

/// As [`step`], with shortwave (or any heat) absorbed inside the elements, W
/// m^-2, bottom element first.
pub fn step_with_sources(
    column: &mut Column,
    atm: &Atmosphere,
    p: &SurfaceParameters,
    ground_flux: f64,
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
        thermal::step_surface_linear_with_sources(
            column,
            ground_flux,
            used.0,
            used.1,
            sources,
            dt,
            k,
        );
        t_lin = *column.node_temperature.last().unwrap();
    }
    // Report the flux the last solve imposed, a + b T_end, rather than the
    // nonlinear budget at T_end: the two differ by the linearisation error, and
    // only the former balances the energy the solver actually added.
    let mut report = budget(atm, p, t_lin);
    report.total = used.0 + used.1 * t_lin;
    report
}

/// As [`step_with_sources`] with any lower boundary. Returns the surface budget
/// the solver imposed.
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

/// Linearise the budget about `t0`: returns `(a, b)` with `F(T) ~ a + b T`.
/// The slope is kept non-positive: a surface that gains energy as it warms
/// would make the implicit solve unstable, and the physical budget never does.
fn linearise(atm: &Atmosphere, p: &SurfaceParameters, t0: f64) -> (f64, f64) {
    const DELTA: f64 = 0.01;
    let t = t0.min(T_MELT - DELTA);
    let f0 = budget(atm, p, t).total;
    let f1 = budget(atm, p, t + DELTA).total;
    let b = ((f1 - f0) / DELTA).min(0.0);
    (f0 - b * t, b)
}
