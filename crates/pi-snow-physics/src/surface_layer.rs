//! Alternative ways of solving the Monin-Obukhov surface layer
//! (engine-surface-v1, `--physics SURFACE_LAYER=...`).
//!
//! Both keep the Phase 2 stability functions (Holtslag and De Bruin 1988 for
//! stable air, Paulson 1970 for unstable; see `turbulence`), which Schlogl et
//! al. (2017, Table 5) found the closest of the published corrections to
//! eddy-covariance sensible heat over snow. What changes is the system they
//! are solved in. With wind at 10 m and scalars at 2 m (both less the snow
//! depth) the Phase 2 system has no root above a bulk Richardson number of
//! about 0.06, where the iteration runs away and the flux goes to zero
//! (engine-mass-v1; pinned by `tests/turbulence_decoupling.rs`).
//!
//! * [`scalar_height`]: the wind is carried from its own height down to the
//!   scalar height along the neutral logarithmic profile, `u(z_s) = u(z_w)
//!   ln(z_s / z0) / ln(z_w / z0)`, and the Phase 2 iteration is then run with
//!   both at the scalar height. This is **not** Schlogl, Lehning, Nishimura,
//!   Huwald, Cullen and Mott (2017), Boundary-Layer Meteorol. 165, 161-180,
//!   who solve their eqs. 1-2 (p. 164) with wind and temperature measured at
//!   one height; it borrows their single-height form for a forcing whose wind
//!   is at another height. Algebraically the only change from Phase 2 is that
//!   the stability correction on momentum between z_s and z_w is dropped (the
//!   neutral exchange coefficient is unchanged), so in stable air u* and the
//!   fluxes can only rise. At one height the same functions keep a root up to a
//!   bulk Richardson number of 1/0.7. The reduction and its use here are ours.
//! * [`zeta_limited`]: wind and scalars at their own heights as in Phase 2,
//!   with z/L at the wind height held to 0.01..2 in stable and -100..-0.01 in
//!   unstable air at every iteration, as the Community Land Model does: CLM5
//!   Technical Note, section 2.5.1, eq. 2.5.49 (and the first guess, eq.
//!   2.5.48), read on the published page. CLM pairs the limit with its own
//!   stability functions (Zeng et al. 1998); carrying the limit alone onto
//!   Holtslag and De Bruin's functions is our transplant.

use crate::turbulence::{Fluxes, SurfaceLayer, psi_h, psi_m};
use pi_snow_core::constants::{GRAVITY, VON_KARMAN};

use crate::atmosphere::CP_AIR;

/// CLM5 eq. 2.5.49's bounds on z/L.
const ZETA_STABLE: (f64, f64) = (0.01, 2.0);
const ZETA_UNSTABLE: (f64, f64) = (-100.0, -0.01);

/// One reference height: the wind reduced to the scalar height under the
/// neutral log law, then the Phase 2 iteration at that single height. A wind
/// already at or below the scalar height is used as it is (only the heights
/// are made equal), which does not occur with a 10 m wind and 2 m scalars.
pub fn scalar_height(l: &SurfaceLayer, rho: f64, mean_t: f64) -> Fluxes {
    let wind = if l.z_wind > l.z_scalar {
        l.wind * (l.z_scalar / l.z0).ln() / (l.z_wind / l.z0).ln()
    } else {
        l.wind
    };
    let at_scalar = SurfaceLayer {
        wind,
        z_wind: l.z_scalar,
        ..*l
    };
    solve(&at_scalar, rho, mean_t, None)
}

/// The Phase 2 two-height system with z/L at the wind height held to CLM5's
/// bounds.
pub fn zeta_limited(l: &SurfaceLayer, rho: f64, mean_t: f64) -> Fluxes {
    solve(l, rho, mean_t, Some(l.z_wind))
}

/// The Phase 2 iteration on 1/L (same start, relaxation and tolerance), with
/// an optional clamp of `z_ref / L` applied to every new estimate.
fn solve(l: &SurfaceLayer, rho: f64, mean_t: f64, clamp_at: Option<f64>) -> Fluxes {
    let dt = l.t_air - l.t_surface;
    let dq = l.q_air - l.q_surface;
    let (zw, zs, z0) = (l.z_wind, l.z_scalar, l.z0);
    let limit = |inverse_l: f64| match clamp_at {
        None => inverse_l,
        Some(z) => {
            let zeta = z * inverse_l;
            let (lo, hi) = if zeta >= 0.0 {
                ZETA_STABLE
            } else {
                ZETA_UNSTABLE
            };
            zeta.clamp(lo, hi) / z
        }
    };
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
