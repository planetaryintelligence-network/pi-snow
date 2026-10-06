//! A replica of `pi_snow_physics::turbulence`'s Monin-Obukhov iteration that also counts
//! its passes. The physics crate does not expose the count and is not edited, so the
//! lockstep gate (G1a) recomputes it here for the step's final surface budget; a test
//! checks that the replica's fluxes equal the crate's bit for bit.

use pi_snow_core::constants::{GRAVITY, T_MELT, VON_KARMAN};
use pi_snow_physics::atmosphere::{
    CP_AIR, air_density, specific_humidity, vapour_pressure_ice, vapour_pressure_water,
};
use pi_snow_physics::turbulence::{Scheme, SurfaceLayer, psi_h, psi_m};

/// Sensible heat, vapour flux and the number of passes the iteration took (0 when the
/// scheme is not Monin-Obukhov or there is no wind).
pub fn fluxes_counted(l: &SurfaceLayer, scheme: Scheme) -> (f64, f64, u32) {
    let mean_t = 0.5 * (l.t_air + l.t_surface);
    let rho = air_density(l.pressure, mean_t);
    if l.wind <= 0.0 || scheme != Scheme::MoHoltslag {
        return (f64::NAN, f64::NAN, 0);
    }
    let dt = l.t_air - l.t_surface;
    let dq = l.q_air - l.q_surface;
    let (zw, zs, z0) = (l.z_wind, l.z_scalar, l.z0);
    let mut inverse_l = 0.0_f64;
    let (mut u_star, mut theta_star, mut q_star) = (0.0, 0.0, 0.0);
    let mut passes = 0;
    for _ in 0..200 {
        passes += 1;
        let dm = (zw / z0).ln() - psi_m(zw * inverse_l) + psi_m(z0 * inverse_l);
        let dh = (zs / z0).ln() - psi_h(zs * inverse_l) + psi_h(z0 * inverse_l);
        u_star = VON_KARMAN * l.wind / dm;
        theta_star = VON_KARMAN * dt / dh;
        q_star = VON_KARMAN * dq / dh;
        let target = VON_KARMAN * GRAVITY * theta_star / (u_star * u_star * mean_t);
        let next = 0.5 * (inverse_l + target);
        if (next - inverse_l).abs() <= 1e-10 * next.abs().max(1e-6) {
            break;
        }
        inverse_l = next;
    }
    (rho * CP_AIR * u_star * theta_star, rho * u_star * q_star, passes)
}

/// The surface layer `surface::budget` builds at surface temperature `t_surface`.
#[allow(clippy::too_many_arguments)]
pub fn budget_layer(
    t_air: f64,
    relative_humidity: f64,
    wind: f64,
    pressure: f64,
    z_wind: f64,
    z_scalar: f64,
    z0: f64,
    t_surface: f64,
) -> SurfaceLayer {
    let ts = t_surface.min(T_MELT);
    let e_surface = if ts >= T_MELT {
        vapour_pressure_water(ts)
    } else {
        vapour_pressure_ice(ts)
    };
    let e_air = relative_humidity * vapour_pressure_water(t_air);
    SurfaceLayer {
        wind,
        z_wind,
        z_scalar,
        z0,
        t_air,
        t_surface: ts,
        q_air: specific_humidity(e_air, pressure),
        q_surface: specific_humidity(e_surface, pressure),
        pressure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_snow_physics::turbulence::fluxes;

    #[test]
    fn replica_matches_the_crate_bit_for_bit() {
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20_000 {
            let layer = budget_layer(
                240.0 + 40.0 * next(),
                0.2 + 0.8 * next(),
                0.05 + 12.0 * next(),
                70_000.0 + 10_000.0 * next(),
                0.5 + 9.5 * next(),
                0.5 + 1.5 * next(),
                0.002,
                235.0 + 40.0 * next(),
            );
            let crate_f = fluxes(&layer, Scheme::MoHoltslag);
            let (s, v, _) = fluxes_counted(&layer, Scheme::MoHoltslag);
            assert_eq!(s.to_bits(), crate_f.sensible.to_bits(), "{layer:?}");
            assert_eq!(v.to_bits(), crate_f.vapour_mass.to_bits(), "{layer:?}");
        }
    }

    #[test]
    fn a_clear_calm_night_takes_more_than_one_pass() {
        // Air 8 K warmer than a radiating surface, light wind: strongly stable.
        let layer = budget_layer(263.0, 0.8, 0.8, 75_000.0, 10.0, 2.0, 0.002, 255.0);
        let (_, _, passes) = fluxes_counted(&layer, Scheme::MoHoltslag);
        assert!(passes > 1, "passes {passes}");
    }
}
