//! Surface-layer turbulent exchange: stability functions and bulk fluxes.

use approx::assert_relative_eq;
use pi_snow_physics::atmosphere as atm;
use pi_snow_physics::turbulence::{self, Scheme, SurfaceLayer};

const KAPPA: f64 = 0.4;

fn layer(wind: f64, t_air: f64, t_surface: f64) -> SurfaceLayer {
    SurfaceLayer {
        wind,
        z_wind: 10.0,
        z_scalar: 2.0,
        z0: 0.002,
        t_air,
        t_surface,
        q_air: 0.002,
        q_surface: 0.002,
        pressure: 70_000.0,
    }
}

#[test]
fn stability_functions_vanish_at_neutral_from_both_sides() {
    // Holtslag and De Bruin (1988) eq. 12 and Paulson (1970) both give zero at
    // z/L = 0, and must meet there or the flux jumps as stratification flips.
    for f in [turbulence::psi_m, turbulence::psi_h] {
        assert_relative_eq!(f(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(f(1e-9), f(-1e-9), epsilon = 1e-7);
    }
}

#[test]
fn stable_branch_is_holtslag_de_bruin_1988_eq_12() {
    // -psi = a z + b (z - c/d) exp(-d z) + b c / d, a 0.7, b 0.75, c 5, d 0.35,
    // with psi_h = psi_m (their eq. 10).
    for zeta in [0.1_f64, 0.5, 1.0, 5.0, 10.0] {
        let expected =
            -(0.7 * zeta + 0.75 * (zeta - 5.0 / 0.35) * (-0.35 * zeta).exp() + 0.75 * 5.0 / 0.35);
        assert_relative_eq!(turbulence::psi_m(zeta), expected, max_relative = 1e-12);
        assert_relative_eq!(turbulence::psi_h(zeta), expected, max_relative = 1e-12);
    }
}

#[test]
fn unstable_branch_is_businger_dyer_with_paulsons_gamma() {
    // Paulson (1970): x = (1 - 16 z/L)^(1/4),
    // psi_m = 2 ln((1+x)/2) + ln((1+x^2)/2) - 2 atan x + pi/2,
    // psi_h = 2 ln((1+x^2)/2).
    let zeta = -0.7_f64;
    let x = (1.0 - 16.0 * zeta).powf(0.25);
    let pm = 2.0 * ((1.0 + x) / 2.0).ln() + ((1.0 + x * x) / 2.0).ln() - 2.0 * x.atan()
        + std::f64::consts::FRAC_PI_2;
    assert_relative_eq!(turbulence::psi_m(zeta), pm, max_relative = 1e-12);
    assert_relative_eq!(
        turbulence::psi_h(zeta),
        2.0 * ((1.0 + x * x) / 2.0).ln(),
        max_relative = 1e-12
    );
}

#[test]
fn neutral_conditions_give_the_log_law_exactly() {
    // Equal air and surface temperature: no buoyancy, L infinite, and
    // u* = kappa U / ln(z / z0) with no sensible heat flux.
    let f = turbulence::fluxes(&layer(5.0, 268.15, 268.15), Scheme::MoHoltslag);
    assert_relative_eq!(
        f.u_star,
        KAPPA * 5.0 / (10.0_f64 / 0.002).ln(),
        max_relative = 1e-12
    );
    assert_eq!(f.sensible, 0.0);
    assert!(f.obukhov_length.is_infinite());
}

#[test]
fn warm_air_over_cold_snow_heats_the_snow_and_stability_damps_it() {
    // Sign: positive into the snow. The neutral bulk value bounds the stable
    // flux from above, because stable stratification suppresses mixing.
    let l = layer(3.0, 268.15, 258.15);
    let f = turbulence::fluxes(&l, Scheme::MoHoltslag);
    assert!(f.sensible > 0.0);
    assert!(f.obukhov_length > 0.0, "stable");
    let rho = atm::air_density(l.pressure, 0.5 * (l.t_air + l.t_surface));
    let neutral = rho * atm::CP_AIR * KAPPA * KAPPA * 3.0 * 10.0
        / ((10.0_f64 / 0.002).ln() * (2.0_f64 / 0.002).ln());
    assert!(
        f.sensible < neutral,
        "stable {} should be below neutral {}",
        f.sensible,
        neutral
    );
}

#[test]
fn cold_air_over_a_warmer_surface_is_unstable_and_enhanced() {
    let l = layer(3.0, 263.15, 272.15);
    let f = turbulence::fluxes(&l, Scheme::MoHoltslag);
    assert!(f.sensible < 0.0 && f.obukhov_length < 0.0);
    let rho = atm::air_density(l.pressure, 0.5 * (l.t_air + l.t_surface));
    let neutral = rho * atm::CP_AIR * KAPPA * KAPPA * 3.0 * -9.0
        / ((10.0_f64 / 0.002).ln() * (2.0_f64 / 0.002).ln());
    assert!(
        f.sensible < neutral,
        "unstable {} should exceed neutral {} in magnitude",
        f.sensible,
        neutral
    );
}

#[test]
fn dry_air_sublimates_the_surface() {
    // Vapour flux follows the humidity gradient: drier air than the saturated
    // surface draws mass off the snow, negative into the snow.
    let mut l = layer(4.0, 263.15, 263.15);
    l.q_surface = atm::specific_humidity(atm::vapour_pressure_ice(263.15), l.pressure);
    l.q_air = 0.5 * l.q_surface;
    let f = turbulence::fluxes(&l, Scheme::MoHoltslag);
    assert!(f.vapour_mass < 0.0);
}

#[test]
fn louis_neutral_transfer_coefficient_is_its_von_karman_form() {
    // ECMWF (1982) reproduction of Louis (1979), system I: at Ri = 0,
    // C_h = a_h^2 with a_h = 0.41 / ln((z + z0) / z0), so H = rho cp C_h U dT
    // tends to that as dT tends to 0.
    let l = layer(5.0, 268.15 + 1e-6, 268.15);
    let f = turbulence::fluxes(&l, Scheme::Louis);
    let a_h = 0.41 / ((2.0 + 0.002) / 0.002_f64).ln();
    let rho = atm::air_density(l.pressure, 268.15);
    let expected = rho * atm::CP_AIR * a_h * a_h * 5.0 * 1e-6;
    assert_relative_eq!(f.sensible, expected, max_relative = 1e-3);
}
