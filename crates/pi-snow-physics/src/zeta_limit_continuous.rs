//! `SURFACE_LAYER_NEUTRAL=CONTINUOUS` (engine-soil-spinup-v2, plan section 4.1).
//!
//! The `surface_layer::zeta_limited` solve with exactly its two limit calls
//! replaced. CLM5 eq. 2.5.49 holds z/L to 0.01..2 when it is non-negative and
//! to -100..-0.01 otherwise, so the held value jumps from +0.01 to -0.01 where
//! the surface crosses the air temperature. Sensible heat is continuous there
//! (theta* is proportional to the temperature difference) but moisture
//! exchange and latent heat step by about 1 %, which leaves a bare-ground
//! surface balance with no root inside the step.
//!
//! The limit here is `clamp(z_w / L, -100, 2)`: continuous, with `limit(0) = 0`,
//! and equal to the transplant's wherever the converged `|z/L|` is above 0.01.
//! Everything else is `surface_layer::solve` verbatim: start at 1/L = 0, 200
//! iterations, relaxation 0.5, the same stopping rule, the same heights and
//! stability functions. It is our continuous variant of our own transplant.
//! It is not canonical CLM5.

use crate::atmosphere::CP_AIR;
use crate::turbulence::{Fluxes, SurfaceLayer, psi_h, psi_m};
use pi_snow_core::constants::{GRAVITY, VON_KARMAN};

/// Bounds on z/L at the wind height (CLM5 eq. 2.5.49's outer bounds).
pub const ZETA_MIN: f64 = -100.0;
pub const ZETA_MAX: f64 = 2.0;

/// The Phase 2 two-height system with z/L at the wind height held to
/// `[ZETA_MIN, ZETA_MAX]`. `rho` and `mean_t` come from the caller as for
/// `surface_layer::zeta_limited`; the calm-wind bypass is in
/// `turbulence::fluxes`.
pub fn zeta_continuous(l: &SurfaceLayer, rho: f64, mean_t: f64) -> Fluxes {
    let dt = l.t_air - l.t_surface;
    let dq = l.q_air - l.q_surface;
    let (zw, zs, z0) = (l.z_wind, l.z_scalar, l.z0);
    let limit = |inverse_l: f64| (zw * inverse_l).clamp(ZETA_MIN, ZETA_MAX) / zw;
    let mut inverse_l = 0.0_f64;
    let (mut u_star, mut theta_star, mut q_star) = (0.0, 0.0, 0.0);
    for _ in 0..200 {
        let dm = (zw / z0).ln() - psi_m(zw * inverse_l) + psi_m(z0 * inverse_l);
        let dh = (zs / z0).ln() - psi_h(zs * inverse_l) + psi_h(z0 * inverse_l);
        u_star = VON_KARMAN * l.wind / dm;
        theta_star = VON_KARMAN * dt / dh;
        q_star = VON_KARMAN * dq / dh;
        let target = limit(VON_KARMAN * GRAVITY * theta_star / (u_star * u_star * mean_t));
        let next = limit(0.5 * (inverse_l + target));
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
