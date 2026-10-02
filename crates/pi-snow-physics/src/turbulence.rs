//! Turbulent exchange between the snow surface and the air above it.
//!
//! Two schemes, one per physics arm of the comparison:
//!
//! * `MoHoltslag` (`ATMOSPHERIC_STABILITY = MO_HOLTSLAG`, the "ours" arm):
//!   Monin-Obukhov similarity iterated on the Obukhov length. Stable branch:
//!   Holtslag and De Bruin (1988), J. Appl. Meteorol. 27, 689-704, eq. 12 with
//!   a = 0.7, b = 0.75, c = 5, d = 0.35 and psi_h = psi_m (eq. 10), read in the
//!   primary paper. Unstable branch: the Businger-Dyer forms with gamma = 16,
//!   Paulson (1970), J. Appl. Meteorol. 9, 857-861, read in full.
//! * `Louis` (`ATMOSPHERIC_STABILITY = RICHARDSON`, the "uac" arm): bulk
//!   Richardson-number transfer coefficients of Louis (1979), system I, with the
//!   coefficients kappa_m 0.35, kappa_h 0.41, b 4.7, c_m 7.4, c_h 5.3, as
//!   reproduced by Louis, Tiedtke and Geleyn (1981/82, ECMWF workshop, pp.
//!   59-63). **Secondary**: the primary Louis (1979) was not obtained.
//!
//! Choices that are ours, not the sources': the scalar roughness length equals
//! the momentum roughness length; temperatures stand in for potential
//! temperatures (the difference over 2 m is about 0.02 K); there is no virtual
//! temperature correction; and with no wind there is no turbulent flux (free
//! convection is not represented). In the Louis scheme, which assumes one
//! measurement level, the scalar height is used for both the Richardson number
//! and the neutral coefficient.
//!
//! Signs: every flux is positive into the snow.

use crate::atmosphere::{CP_AIR, air_density};
use pi_snow_core::constants::{GRAVITY, VON_KARMAN};

/// Which stability treatment to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    MoHoltslag,
    Louis,
    /// `MoHoltslag` with the wind carried to the scalar height (`surface_layer::scalar_height`).
    MoHoltslagScalarHeight,
    /// `MoHoltslag` with z/L bounded as CLM5 does (`surface_layer::zeta_limited`).
    MoHoltslagZetaLimited,
}

/// The state of the surface layer at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceLayer {
    /// Wind speed at `z_wind`, m s^-1.
    pub wind: f64,
    /// Height of the wind measurement above the snow surface, m.
    pub z_wind: f64,
    /// Height of the temperature and humidity measurement above the surface, m.
    pub z_scalar: f64,
    /// Roughness length, m (used for momentum and scalars alike).
    pub z0: f64,
    /// Air and surface temperature, K.
    pub t_air: f64,
    pub t_surface: f64,
    /// Air and surface specific humidity, kg kg^-1.
    pub q_air: f64,
    pub q_surface: f64,
    /// Surface pressure, Pa.
    pub pressure: f64,
}

/// Turbulent fluxes, positive into the snow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fluxes {
    /// Sensible heat, W m^-2.
    pub sensible: f64,
    /// Water vapour, kg m^-2 s^-1; positive is deposition, negative
    /// sublimation or evaporation. The latent heat is this times the latent
    /// heat of whichever phase change applies, which the caller decides.
    pub vapour_mass: f64,
    /// Friction velocity, m s^-1.
    pub u_star: f64,
    /// Obukhov length, m; infinite when neutral. Not defined for `Louis`,
    /// which reports the sign of stability as +/- infinity only.
    pub obukhov_length: f64,
}

/// Stability function for momentum at `zeta = z / L`.
pub fn psi_m(zeta: f64) -> f64 {
    if zeta >= 0.0 {
        holtslag_de_bruin(zeta)
    } else {
        let x = (1.0 - 16.0 * zeta).powf(0.25);
        2.0 * ((1.0 + x) / 2.0).ln() + ((1.0 + x * x) / 2.0).ln() - 2.0 * x.atan()
            + std::f64::consts::FRAC_PI_2
    }
}

/// Stability function for heat and water vapour at `zeta = z / L`.
pub fn psi_h(zeta: f64) -> f64 {
    if zeta >= 0.0 {
        holtslag_de_bruin(zeta)
    } else {
        let x = (1.0 - 16.0 * zeta).powf(0.25);
        2.0 * ((1.0 + x * x) / 2.0).ln()
    }
}

/// Holtslag and De Bruin (1988) eq. 12: `-psi = a z + b (z - c/d) e^(-d z) + b c/d`.
fn holtslag_de_bruin(zeta: f64) -> f64 {
    const A: f64 = 0.7;
    const B: f64 = 0.75;
    const C: f64 = 5.0;
    const D: f64 = 0.35;
    -(A * zeta + B * (zeta - C / D) * (-D * zeta).exp() + B * C / D)
}

/// Turbulent fluxes for a surface layer under the chosen scheme.
pub fn fluxes(layer: &SurfaceLayer, scheme: Scheme) -> Fluxes {
    let mean_t = 0.5 * (layer.t_air + layer.t_surface);
    let rho = air_density(layer.pressure, mean_t);
    if layer.wind <= 0.0 {
        return Fluxes {
            sensible: 0.0,
            vapour_mass: 0.0,
            u_star: 0.0,
            obukhov_length: f64::INFINITY,
        };
    }
    match scheme {
        Scheme::MoHoltslag => monin_obukhov(layer, rho, mean_t),
        Scheme::Louis => louis(layer, rho, mean_t),
        Scheme::MoHoltslagScalarHeight => crate::surface_layer::scalar_height(layer, rho, mean_t),
        Scheme::MoHoltslagZetaLimited => crate::surface_layer::zeta_limited(layer, rho, mean_t),
    }
}

fn monin_obukhov(l: &SurfaceLayer, rho: f64, mean_t: f64) -> Fluxes {
    let dt = l.t_air - l.t_surface;
    let dq = l.q_air - l.q_surface;
    let (zw, zs, z0) = (l.z_wind, l.z_scalar, l.z0);
    // Iterate on 1/L, which is 0 at neutral and so needs no special case, with
    // under-relaxation: the plain fixed point oscillates on clear, calm nights.
    let mut inverse_l = 0.0_f64;
    let (mut u_star, mut theta_star, mut q_star) = (0.0, 0.0, 0.0);
    for _ in 0..200 {
        let dm = (zw / z0).ln() - psi_m(zw * inverse_l) + psi_m(z0 * inverse_l);
        let dh = (zs / z0).ln() - psi_h(zs * inverse_l) + psi_h(z0 * inverse_l);
        u_star = VON_KARMAN * l.wind / dm;
        theta_star = VON_KARMAN * dt / dh;
        q_star = VON_KARMAN * dq / dh;
        let target = VON_KARMAN * GRAVITY * theta_star / (u_star * u_star * mean_t);
        let next = 0.5 * (inverse_l + target);
        if (next - inverse_l).abs() <= 1e-10 * next.abs().max(1e-6) {
            inverse_l = next;
            break;
        }
        inverse_l = next;
    }
    Fluxes {
        sensible: rho * CP_AIR * u_star * theta_star,
        vapour_mass: rho * u_star * q_star,
        u_star,
        obukhov_length: if inverse_l == 0.0 {
            f64::INFINITY
        } else {
            1.0 / inverse_l
        },
    }
}

fn louis(l: &SurfaceLayer, rho: f64, mean_t: f64) -> Fluxes {
    const KAPPA_M: f64 = 0.35;
    const KAPPA_H: f64 = 0.41;
    const B: f64 = 4.7;
    const C_M: f64 = 7.4;
    const C_H: f64 = 5.3;
    let z = l.z_scalar;
    let log = ((z + l.z0) / l.z0).ln();
    let (a_m, a_h) = (KAPPA_M / log, KAPPA_H / log);
    let ri = GRAVITY * z * (l.t_air - l.t_surface) / (mean_t * l.wind * l.wind);
    let (cm, ch) = if ri >= 0.0 {
        let f = 1.0 / (1.0 + B * ri).powi(2);
        (a_m * a_m * f, a_h * a_h * f)
    } else {
        let root = (ri.abs() * (z + l.z0) / l.z0).sqrt();
        (
            a_m * a_m * (1.0 - 2.0 * B * ri / (1.0 + 2.0 * a_m * a_m * B * C_M * root)),
            a_h * a_h * (1.0 - 2.0 * B * ri / (1.0 + 2.0 * a_h * a_h * B * C_H * root)),
        )
    };
    Fluxes {
        sensible: rho * CP_AIR * ch * l.wind * (l.t_air - l.t_surface),
        vapour_mass: rho * ch * l.wind * (l.q_air - l.q_surface),
        u_star: cm.sqrt() * l.wind,
        obukhov_length: if ri > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        },
    }
}
